import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = [
    "spleen",
    "r.kidney",
    "l.kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "portal vein and splenic vein",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Pure SAM2 prompting benchmark on BTCV")
    parser.add_argument("--root_path", type=str, default="/data/why/Datasets/BTCV/")
    parser.add_argument("--split", type=str, default="eval", choices=["eval", "test"])
    parser.add_argument("--sam2_root", type=str, default="/home/why/SSL4MIS_work3")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="/home/why/SSL4MIS_work3/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument("--prompt_type", type=str, default="box", choices=["box", "mask"])
    parser.add_argument(
        "--strategy",
        type=str,
        default="shared_multiobj",
        choices=["shared_multiobj", "per_class_loop"],
    )
    parser.add_argument("--frame_counts", type=str, default="1,2,3")
    parser.add_argument("--max_cases", type=int, default=0, help="0 means use all cases in split")
    parser.add_argument("--box_expand", type=int, default=4)
    parser.add_argument("--min_slice_area", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--merge_mode", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/why/logs_SAM2SSL/BTCV_pureSAM2_prompt_sweep",
    )
    return parser.parse_args()


def read_case_list(root_path, split):
    split_path = Path(root_path) / "split_txt" / f"{split}.txt"
    with split_path.open("r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    return case_ids


def load_case(root_path, case_id):
    case_path = Path(root_path) / "btcv_h5" / f"{case_id}.h5"
    with h5py.File(case_path, "r") as h5f:
        image = h5f["image"][:].astype(np.float32)
        label = h5f["label"][:].astype(np.int16)
    return image, label


def resolve_config_name(base_dir: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        config_path = candidate.resolve()
    else:
        matches = [
            (base_dir / value).resolve(),
            (base_dir / "sam2" / value).resolve(),
        ]
        config_path = None
        for item in matches:
            if item.exists():
                config_path = item
                break
        if config_path is None:
            return value

    sam2_pkg_dir = (base_dir / "sam2").resolve()
    try:
        return config_path.relative_to(sam2_pkg_dir).as_posix()
    except ValueError:
        return value


def build_predictor(args, device):
    sam2_root = Path(args.sam2_root).expanduser().resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor_npz

    predictor = build_sam2_video_predictor_npz(
        config_file=resolve_config_name(sam2_root, args.sam2_cfg),
        ckpt_path=str(Path(args.sam2_checkpoint).expanduser().resolve()),
        device=str(device),
        mode="eval",
    )
    predictor.eval()
    if hasattr(predictor, "fill_hole_area"):
        predictor.fill_hole_area = 0
    return predictor


def volume_to_rgb_sequence(volume):
    sequence = []
    depth = volume.shape[-1]
    for index in range(depth):
        prev_index = max(index - 1, 0)
        next_index = min(index + 1, depth - 1)
        frame = torch.stack(
            [volume[:, :, prev_index], volume[:, :, index], volume[:, :, next_index]],
            dim=0,
        )
        sequence.append(frame)
    return torch.stack(sequence, dim=0)


def preprocess_sequence(frame_sequence, input_size, device):
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    frame_sequence = F.interpolate(
        frame_sequence.float(),
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )
    frame_min = frame_sequence.amin()
    frame_max = frame_sequence.amax()
    frame_sequence = (frame_sequence - frame_min) / (frame_max - frame_min).clamp_min(1e-6)
    return ((frame_sequence.to(device) - pixel_mean) / pixel_std).contiguous()


def mask_to_box(mask_2d, box_expand):
    coords = torch.nonzero(mask_2d, as_tuple=False)
    if coords.numel() == 0:
        return None
    y_min = max(0, int(coords[:, 0].min().item()) - box_expand)
    y_max = min(mask_2d.shape[0] - 1, int(coords[:, 0].max().item()) + box_expand)
    x_min = max(0, int(coords[:, 1].min().item()) - box_expand)
    x_max = min(mask_2d.shape[1] - 1, int(coords[:, 1].max().item()) + box_expand)
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)


def unique_fg_classes(label_volume):
    class_ids = [int(x) for x in np.unique(label_volume) if int(x) > 0]
    return sorted(class_ids)


def class_slice_areas(label_volume, class_id):
    class_mask = label_volume == class_id
    return class_mask.sum(axis=(0, 1))


def select_class_frames(label_volume, class_id, frame_count, min_slice_area):
    areas = class_slice_areas(label_volume, class_id)
    valid = np.where(areas >= min_slice_area)[0]
    if valid.size == 0:
        valid = np.where(areas > 0)[0]
    if valid.size == 0:
        return []
    if valid.size <= frame_count:
        return sorted(set(int(x) for x in valid.tolist()))
    if frame_count == 1:
        return [int(valid[np.argmax(areas[valid])])]
    anchors = np.round(np.linspace(0, valid.size - 1, num=frame_count)).astype(int)
    selected = [int(valid[idx]) for idx in anchors]
    strongest = int(valid[np.argmax(areas[valid])])
    if strongest not in selected:
        weakest_pos = min(range(len(selected)), key=lambda pos: float(areas[selected[pos]]))
        selected[weakest_pos] = strongest
    return sorted(set(selected))


def choose_shared_frames(label_volume, class_ids, frame_count, min_slice_area):
    depth = label_volume.shape[-1]
    frame_class_sets = []
    frame_total_area = []
    for slice_idx in range(depth):
        curr_classes = set()
        total_area = 0
        for class_id in class_ids:
            area = int((label_volume[:, :, slice_idx] == class_id).sum())
            if area >= min_slice_area:
                curr_classes.add(class_id)
                total_area += area
        frame_class_sets.append(curr_classes)
        frame_total_area.append(total_area)

    remaining = set(class_ids)
    selected = []
    while remaining and len(selected) < frame_count:
        best_idx = None
        best_gain = (-1, -1)
        for slice_idx, class_set in enumerate(frame_class_sets):
            if slice_idx in selected:
                continue
            gain = len(class_set & remaining)
            if gain <= 0:
                continue
            key = (gain, frame_total_area[slice_idx])
            if key > best_gain:
                best_gain = key
                best_idx = slice_idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining -= frame_class_sets[best_idx]

    if not selected:
        best_idx = int(np.argmax(frame_total_area))
        selected = [best_idx]

    while len(selected) < frame_count:
        candidates = [idx for idx in range(depth) if idx not in selected]
        if not candidates:
            break
        best_idx = max(candidates, key=lambda idx: frame_total_area[idx])
        if frame_total_area[best_idx] <= 0:
            break
        selected.append(best_idx)

    return sorted(set(int(idx) for idx in selected))


def add_prompts(predictor, state, class_prompt_map, prompt_type, box_expand, min_slice_area):
    valid_prompt = False
    for class_id, prompt_item in class_prompt_map.items():
        for slice_idx in prompt_item["selected_slices"]:
            slice_mask = prompt_item["prompt_mask"][:, :, slice_idx]
            if int(slice_mask.sum().item()) < min_slice_area:
                continue
            if prompt_type == "mask":
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(slice_idx),
                    obj_id=int(class_id),
                    mask=slice_mask,
                )
            else:
                box = mask_to_box(slice_mask, box_expand)
                if box is None:
                    continue
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(slice_idx),
                    obj_id=int(class_id),
                    box=box,
                )
            valid_prompt = True
    return valid_prompt


def propagate_once(predictor, frame_sequence, class_prompt_map, merge_mode="avg", prompt_type="box", box_expand=4, min_slice_area=32, reverse=False):
    first_item = next(iter(class_prompt_map.values()))
    prompt_mask = first_item["prompt_mask"]
    height, width, depth = prompt_mask.shape
    state = predictor.init_state(images=frame_sequence, video_height=height, video_width=width)
    if not add_prompts(predictor, state, class_prompt_map, prompt_type, box_expand, min_slice_area):
        predictor.reset_state(state)
        return {}

    class_ids = sorted(int(cid) for cid in class_prompt_map.keys())
    class_index = {cid: idx for idx, cid in enumerate(class_ids)}
    class_scores = prompt_mask.new_zeros((len(class_ids), height, width, depth), dtype=torch.float32)
    prompt_slices = [s for item in class_prompt_map.values() for s in item["selected_slices"]]
    start_frame_idx = max(prompt_slices) if reverse else min(prompt_slices)

    for frame_idx, object_ids, out_mask_logits in predictor.propagate_in_video(
        state,
        start_frame_idx=int(start_frame_idx),
        max_frame_num_to_track=depth,
        reverse=reverse,
    ):
        if out_mask_logits.dim() == 4:
            frame_logits = out_mask_logits[:, 0]
        elif out_mask_logits.dim() == 3:
            frame_logits = out_mask_logits
        else:
            raise ValueError(f"Unexpected SAM2 logit shape: {tuple(out_mask_logits.shape)}")

        for obj_pos, obj_id in enumerate(object_ids):
            obj_id = int(obj_id)
            if obj_id not in class_index or obj_pos >= frame_logits.shape[0]:
                continue
            class_scores[class_index[obj_id], :, :, int(frame_idx)] = torch.sigmoid(frame_logits[obj_pos]).float()

    predictor.reset_state(state)
    return {class_id: class_scores[idx].cpu() for idx, class_id in enumerate(class_ids)}


def merge_direction_outputs(forward_map, reverse_map, merge_mode="avg"):
    merged = {}
    for class_id in sorted(set(forward_map) | set(reverse_map)):
        if class_id in forward_map and class_id in reverse_map:
            if merge_mode == "max":
                merged[class_id] = torch.maximum(forward_map[class_id], reverse_map[class_id])
            else:
                merged[class_id] = 0.5 * (forward_map[class_id] + reverse_map[class_id])
        elif class_id in forward_map:
            merged[class_id] = forward_map[class_id]
        else:
            merged[class_id] = reverse_map[class_id]
    return merged


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


def dice_per_class(pred, label, num_classes):
    dices = []
    for class_id in range(1, num_classes):
        pred_mask = pred == class_id
        gt_mask = label == class_id
        denom = pred_mask.sum() + gt_mask.sum()
        if denom == 0:
            dices.append(1.0)
        else:
            dices.append(float(2.0 * (pred_mask & gt_mask).sum() / max(denom, 1)))
    return dices


def infer_shared_multiobj(predictor, frame_sequence, label_volume, class_ids, frame_count, args):
    selected_frames = choose_shared_frames(label_volume, class_ids, frame_count, args.min_slice_area)
    class_prompt_map = {}
    covered = set()
    for class_id in class_ids:
        selected_for_class = []
        for slice_idx in selected_frames:
            if int((label_volume[:, :, slice_idx] == class_id).sum()) >= args.min_slice_area:
                selected_for_class.append(int(slice_idx))
        if not selected_for_class:
            continue
        covered.add(class_id)
        class_prompt_map[class_id] = {
            "selected_slices": selected_for_class,
            "prompt_mask": torch.from_numpy((label_volume == class_id).astype(np.float32)),
        }

    if not class_prompt_map:
        return np.zeros_like(label_volume), {
            "selected_frames": selected_frames,
            "covered_classes": [],
        }

    forward_map = propagate_once(
        predictor,
        frame_sequence,
        class_prompt_map,
        merge_mode=args.merge_mode,
        prompt_type=args.prompt_type,
        box_expand=args.box_expand,
        min_slice_area=args.min_slice_area,
        reverse=False,
    )
    reverse_map = propagate_once(
        predictor,
        frame_sequence,
        class_prompt_map,
        merge_mode=args.merge_mode,
        prompt_type=args.prompt_type,
        box_expand=args.box_expand,
        min_slice_area=args.min_slice_area,
        reverse=True,
    )
    merged = merge_direction_outputs(forward_map, reverse_map, merge_mode=args.merge_mode)
    pred = scores_to_labels(merged, label_volume.shape, len(CLASS_NAMES) + 1)
    return pred, {
        "selected_frames": selected_frames,
        "covered_classes": sorted(int(x) for x in covered),
    }


def infer_per_class_loop(predictor, frame_sequence, label_volume, class_ids, frame_count, args):
    merged_scores = {}
    selected_by_class = {}
    for class_id in class_ids:
        selected_frames = select_class_frames(label_volume, class_id, frame_count, args.min_slice_area)
        selected_by_class[int(class_id)] = selected_frames
        if not selected_frames:
            continue
        class_prompt_map = {
            int(class_id): {
                "selected_slices": selected_frames,
                "prompt_mask": torch.from_numpy((label_volume == class_id).astype(np.float32)),
            }
        }
        forward_map = propagate_once(
            predictor,
            frame_sequence,
            class_prompt_map,
            merge_mode=args.merge_mode,
            prompt_type=args.prompt_type,
            box_expand=args.box_expand,
            min_slice_area=args.min_slice_area,
            reverse=False,
        )
        reverse_map = propagate_once(
            predictor,
            frame_sequence,
            class_prompt_map,
            merge_mode=args.merge_mode,
            prompt_type=args.prompt_type,
            box_expand=args.box_expand,
            min_slice_area=args.min_slice_area,
            reverse=True,
        )
        merged = merge_direction_outputs(forward_map, reverse_map, merge_mode=args.merge_mode)
        if int(class_id) in merged:
            merged_scores[int(class_id)] = merged[int(class_id)]

    pred = scores_to_labels(merged_scores, label_volume.shape, len(CLASS_NAMES) + 1)
    return pred, {
        "selected_by_class": selected_by_class,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_predictor(args, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    case_ids = read_case_list(args.root_path, args.split)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]

    frame_counts = [int(x) for x in args.frame_counts.split(",") if x.strip()]
    run_tag = f"{args.strategy}_{args.prompt_type}_{args.split}_{'_'.join(map(str, frame_counts))}_{time.strftime('%Y%m%d_%H%M%S')}"
    result_dir = save_dir / run_tag
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "args": vars(args),
        "run_tag": run_tag,
        "cases": case_ids,
        "results": {},
    }

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for frame_count in frame_counts:
        case_metrics = []
        print(f"\n=== Strategy={args.strategy}, frames={frame_count} ===", flush=True)
        with torch.inference_mode(), autocast_context:
            for case_id in case_ids:
                image_np, label_np = load_case(args.root_path, case_id)
                class_ids = unique_fg_classes(label_np)
                volume = torch.from_numpy(image_np)
                frame_sequence = preprocess_sequence(
                    volume_to_rgb_sequence(volume),
                    input_size=args.input_size,
                    device=device,
                )

                if args.strategy == "shared_multiobj":
                    pred, meta = infer_shared_multiobj(
                        predictor, frame_sequence, label_np, class_ids, frame_count, args
                    )
                else:
                    pred, meta = infer_per_class_loop(
                        predictor, frame_sequence, label_np, class_ids, frame_count, args
                    )

                dices = dice_per_class(pred, label_np, len(CLASS_NAMES) + 1)
                mean_dice = float(np.mean(dices))
                case_record = {
                    "case_id": case_id,
                    "mean_dice": mean_dice,
                    "class_dice": dices,
                    "meta": meta,
                }
                case_metrics.append(case_record)
                print(
                    f"[{args.strategy}] case={case_id} frames={frame_count} mean_dice={mean_dice:.4f}",
                    flush=True,
                )

        mean_dice_all = float(np.mean([item["mean_dice"] for item in case_metrics])) if case_metrics else 0.0
        class_mean = (
            np.mean(np.asarray([item["class_dice"] for item in case_metrics], dtype=np.float32), axis=0).tolist()
            if case_metrics
            else [0.0] * len(CLASS_NAMES)
        )
        summary["results"][str(frame_count)] = {
            "mean_dice": mean_dice_all,
            "class_mean_dice": class_mean,
            "cases": case_metrics,
        }
        print(
            f"[SUMMARY] strategy={args.strategy} frames={frame_count} mean_dice={mean_dice_all:.4f}",
            flush=True,
        )

    json_path = result_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {json_path}", flush=True)


if __name__ == "__main__":
    main()
