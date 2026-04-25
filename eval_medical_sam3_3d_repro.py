import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_medical_sam2_3d_repro import (
    DATASET_CONFIGS,
    PRESETS,
    apply_preset,
    frame_selection,
    generate_slice_prompt,
    load_case,
    nanmean,
    per_class_dice_present,
    per_class_iou_nonempty_slices,
    read_case_list,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-style 3D SAM3 reproduction on BTCV/AMOS/MMWHS"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument(
        "--preset",
        type=str,
        default="paper_best_prompt",
        choices=sorted(PRESETS),
    )
    parser.add_argument("--frame_mode", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--prompt_mode", type=int, default=None, choices=[1, 2, 3, 5])
    parser.add_argument("--bidirectional", type=int, default=None, choices=[0, 1])
    parser.add_argument("--uniform_frame_count", type=int, default=3)
    parser.add_argument(
        "--rgb_mode",
        type=str,
        default="repeat",
        choices=["repeat", "neighbor"],
        help="repeat matches grayscale-to-3ch; neighbor uses prev/current/next slices",
    )
    parser.add_argument("--sam3_root", type=str, default="/home/why/codes/sam3")
    parser.add_argument(
        "--sam3_version",
        type=str,
        default="sam3",
        choices=["sam3"],
        help="Current adapter targets the SAM3 tracker API used for SAM2-style video tasks.",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default="",
        help="Optional local SAM3 checkpoint path. If omitted, the builder will try Hugging Face.",
    )
    parser.add_argument("--load_from_hf", type=int, default=1, choices=[0, 1])
    parser.add_argument("--box_expand", type=int, default=4)
    parser.add_argument("--point_num", type=int, default=3)
    parser.add_argument("--min_slice_area", type=int, default=16)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--offload_video_to_cpu", type=int, default=1, choices=[0, 1])
    parser.add_argument("--async_loading_frames", type=int, default=0, choices=[0, 1])
    parser.add_argument("--keep_temp_frames", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/why/logs_SAM2SSL/medical_sam3_3d_repro",
    )
    return parser.parse_args()


def build_predictor(args, device):
    sam3_root = Path(args.sam3_root).expanduser().resolve()
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))

    from sam3.model_builder import build_sam3_video_model

    checkpoint_path = None
    load_from_hf = bool(args.load_from_hf)
    if args.sam3_checkpoint:
        checkpoint_path = str(Path(args.sam3_checkpoint).expanduser().resolve())
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
        load_from_hf = False

    model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        device=str(device),
        compile=False,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    if hasattr(predictor, "fill_hole_area"):
        predictor.fill_hole_area = 0
    return predictor


def volume_to_uint8_rgb_frames(volume, rgb_mode):
    volume = volume.astype(np.float32)
    value_min = float(volume.min())
    value_max = float(volume.max())
    volume = (volume - value_min) / max(value_max - value_min, 1e-6)
    volume = np.clip(np.round(volume * 255.0), 0, 255).astype(np.uint8)

    frames = []
    depth = volume.shape[-1]
    for index in range(depth):
        if rgb_mode == "neighbor":
            prev_index = max(index - 1, 0)
            next_index = min(index + 1, depth - 1)
            rgb = np.stack(
                [volume[:, :, prev_index], volume[:, :, index], volume[:, :, next_index]],
                axis=-1,
            )
        else:
            rgb = np.repeat(volume[:, :, index][:, :, None], 3, axis=-1)
        frames.append(Image.fromarray(rgb, mode="RGB"))
    return frames


def write_frame_folder(frames, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(output_dir / f"{index:05d}.jpg", format="JPEG", quality=95)


def normalize_box_xyxy(box, width, height):
    box = np.asarray(box, dtype=np.float32).copy()
    box[[0, 2]] /= float(width)
    box[[1, 3]] /= float(height)
    return box


def normalize_points_xy(points, width, height):
    points = np.asarray(points, dtype=np.float32).copy()
    points[:, 0] /= float(width)
    points[:, 1] /= float(height)
    return points


def add_prompts(predictor, state, selected_slices, mask_volume, obj_id, args):
    valid = False
    height, width = mask_volume.shape[:2]
    for slice_idx in selected_slices:
        slice_mask = mask_volume[:, :, slice_idx]
        if int(slice_mask.sum()) < args.min_slice_area:
            continue
        prompt = generate_slice_prompt(slice_mask, args.prompt_mode, args.box_expand, args.point_num)
        if prompt["type"] == "mask":
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=int(slice_idx),
                obj_id=int(obj_id),
                mask=torch.from_numpy(prompt["mask"]),
            )
            valid = True
        elif prompt["type"] == "box":
            if prompt["box"] is None:
                continue
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(slice_idx),
                obj_id=int(obj_id),
                box=torch.from_numpy(normalize_box_xyxy(prompt["box"], width, height)),
            )
            valid = True
        else:
            if prompt["points"].shape[0] == 0:
                continue
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(slice_idx),
                obj_id=int(obj_id),
                points=torch.from_numpy(normalize_points_xy(prompt["points"], width, height)),
                labels=torch.from_numpy(prompt["labels"]),
            )
            valid = True
    return valid


def collect_scores_from_generator(class_scores, class_id, generator):
    for frame_idx, object_ids, low_res_masks, video_res_masks, obj_scores in generator:
        if video_res_masks.dim() == 4:
            frame_logits = video_res_masks[:, 0]
        elif video_res_masks.dim() == 3:
            frame_logits = video_res_masks
        else:
            raise ValueError(f"Unexpected SAM3 video_res_masks shape: {tuple(video_res_masks.shape)}")
        for obj_pos, obj_id in enumerate(object_ids):
            if int(obj_id) != int(class_id) or obj_pos >= frame_logits.shape[0]:
                continue
            class_scores[:, :, int(frame_idx)] = torch.sigmoid(frame_logits[obj_pos]).float().cpu()


def run_single_class(predictor, video_path, mask_volume, class_id, selected_slices, args):
    height, width, depth = mask_volume.shape
    anchor_idx = selected_slices[len(selected_slices) // 2] if len(selected_slices) > 1 else selected_slices[0]
    class_scores = torch.zeros((height, width, depth), dtype=torch.float32)

    if args.bidirectional:
        state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            async_loading_frames=bool(args.async_loading_frames),
        )
        valid = add_prompts(predictor, state, selected_slices, mask_volume, class_id, args)
        if valid:
            collect_scores_from_generator(
                class_scores,
                class_id,
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=int(anchor_idx),
                    max_frame_num_to_track=depth,
                    reverse=True,
                    propagate_preflight=True,
                ),
            )

        state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            async_loading_frames=bool(args.async_loading_frames),
        )
        valid = add_prompts(predictor, state, selected_slices, mask_volume, class_id, args)
        if valid:
            collect_scores_from_generator(
                class_scores,
                class_id,
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=min(int(anchor_idx) + 1, depth - 1),
                    max_frame_num_to_track=depth,
                    reverse=False,
                    propagate_preflight=True,
                ),
            )
        return class_scores

    state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        async_loading_frames=bool(args.async_loading_frames),
    )
    valid = add_prompts(predictor, state, selected_slices, mask_volume, class_id, args)
    if valid:
        collect_scores_from_generator(
            class_scores,
            class_id,
            predictor.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=depth,
                reverse=False,
                propagate_preflight=True,
            ),
        )
    return class_scores


def scores_to_labels(class_score_map, shape, num_classes):
    height, width, depth = shape
    fg_probs = torch.zeros((num_classes - 1, height, width, depth), dtype=torch.float32)
    for class_id, class_scores in class_score_map.items():
        if 1 <= int(class_id) < num_classes:
            fg_probs[int(class_id) - 1] = class_scores
    bg_prob = torch.clamp(1.0 - fg_probs.max(dim=0, keepdim=True).values, min=1e-4)
    all_probs = torch.cat([bg_prob, fg_probs], dim=0)
    all_probs = all_probs / all_probs.sum(dim=0, keepdim=True).clamp_min(1e-6)
    return torch.argmax(all_probs, dim=0).numpy().astype(np.int16)


def model_tag(args):
    if args.sam3_checkpoint:
        return Path(args.sam3_checkpoint).stem
    return args.sam3_version


def main():
    args = apply_preset(parse_args())
    frame_selection._uniform_frame_count = int(args.uniform_frame_count)
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    if args.root_path is None:
        args.root_path = dataset_cfg["root_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_predictor(args, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    temp_root = save_dir / "_tmp_frames"
    temp_root.mkdir(parents=True, exist_ok=True)

    case_ids = read_case_list(args.dataset, args.root_path, args.split)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]

    run_tag = (
        f"{args.dataset}_{model_tag(args)}_{args.preset}_F{args.frame_mode}_P{args.prompt_mode}_"
        f"D{int(args.bidirectional)}_K{int(args.uniform_frame_count)}_{args.rgb_mode}_{args.split}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    result_dir = save_dir / run_tag
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "args": vars(args),
        "dataset_config": {
            "num_classes": dataset_cfg["num_classes"],
            "class_names": dataset_cfg["class_names"],
        },
        "preset_description": PRESETS[args.preset]["description"],
        "run_tag": run_tag,
        "cases": [],
    }

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), autocast_context:
        for case_id in case_ids:
            image_np, label_np = load_case(args.dataset, args.root_path, case_id)
            frame_pils = volume_to_uint8_rgb_frames(image_np, rgb_mode=args.rgb_mode)

            if bool(args.keep_temp_frames):
                case_frame_dir = temp_root / case_id
                if not case_frame_dir.exists():
                    write_frame_folder(frame_pils, case_frame_dir)
                video_path = str(case_frame_dir)
            else:
                with tempfile.TemporaryDirectory(
                    prefix=f"{args.dataset}_{case_id}_",
                    dir=str(temp_root),
                ) as temp_dir:
                    case_frame_dir = Path(temp_dir)
                    write_frame_folder(frame_pils, case_frame_dir)
                    video_path = str(case_frame_dir)
                    case_record = evaluate_case(predictor, video_path, case_id, label_np, dataset_cfg, args)
                    summary["cases"].append(case_record)
                    print(
                        f"[{args.dataset}] case={case_id} dice={case_record['mean_dice_present']:.4f} "
                        f"iou={case_record['mean_iou_nonempty_slices']:.4f}",
                        flush=True,
                    )
                continue

            case_record = evaluate_case(predictor, video_path, case_id, label_np, dataset_cfg, args)
            summary["cases"].append(case_record)
            print(
                f"[{args.dataset}] case={case_id} dice={case_record['mean_dice_present']:.4f} "
                f"iou={case_record['mean_iou_nonempty_slices']:.4f}",
                flush=True,
            )

    class_dice_matrix = np.asarray(
        [item["class_dice_present"] for item in summary["cases"]], dtype=np.float32
    )
    class_iou_matrix = np.asarray(
        [item["class_iou_nonempty_slices"] for item in summary["cases"]], dtype=np.float32
    )
    summary["aggregate"] = {
        "mean_dice_present": nanmean([item["mean_dice_present"] for item in summary["cases"]]),
        "mean_iou_nonempty_slices": nanmean([item["mean_iou_nonempty_slices"] for item in summary["cases"]]),
        "class_mean_dice_present": np.nanmean(class_dice_matrix, axis=0).tolist(),
        "class_mean_iou_nonempty_slices": np.nanmean(class_iou_matrix, axis=0).tolist(),
    }

    json_path = result_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[SUMMARY] dataset={args.dataset} preset={args.preset} "
        f"mean_dice={summary['aggregate']['mean_dice_present']:.4f} "
        f"mean_iou={summary['aggregate']['mean_iou_nonempty_slices']:.4f}",
        flush=True,
    )
    print(f"Saved summary to {json_path}", flush=True)


def evaluate_case(predictor, video_path, case_id, label_np, dataset_cfg, args):
    num_classes = dataset_cfg["num_classes"]
    class_scores = {}
    meta = {"selected_by_class": {}}
    for class_id in range(1, num_classes):
        mask_volume = label_np == class_id
        selected_slices = frame_selection(mask_volume, args.frame_mode)
        meta["selected_by_class"][str(class_id)] = selected_slices
        if not selected_slices:
            continue
        class_scores[class_id] = run_single_class(
            predictor,
            video_path,
            mask_volume,
            class_id,
            selected_slices,
            args,
        )

    pred = scores_to_labels(class_scores, label_np.shape, num_classes)
    class_dice = per_class_dice_present(pred, label_np, num_classes)
    class_iou = per_class_iou_nonempty_slices(pred, label_np, num_classes)
    return {
        "case_id": case_id,
        "mean_dice_present": nanmean(class_dice),
        "mean_iou_nonempty_slices": nanmean(class_iou),
        "class_dice_present": class_dice,
        "class_iou_nonempty_slices": class_iou,
        "meta": meta,
    }


if __name__ == "__main__":
    main()
