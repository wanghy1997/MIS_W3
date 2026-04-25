import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


DATASET_CONFIGS = {
    "BTCV": {
        "root_path": "/data/why/Datasets/BTCV",
        "num_classes": 14,
        "class_names": [
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
        ],
        "split_candidates": [
            "{root}/split_txt/{split}.txt",
            "{root}/split_txts/{split}.txt",
        ],
    },
    "AMOS": {
        "root_path": "/data/why/Datasets/amos",
        "num_classes": 16,
        "class_names": [
            "spleen",
            "r.kidney",
            "l.kidney",
            "gallbladder",
            "esophagus",
            "liver",
            "stomach",
            "aorta",
            "ivc",
            "pancreas",
            "right adrenal gland",
            "left adrenal gland",
            "duodenum",
            "bladder",
            "prostate/uterus",
        ],
        "split_candidates": [
            "{root}/split_txt/{split}.txt",
            "{repo}/data/amos_splits/{split}.txt",
        ],
    },
    "MMWHS": {
        "root_path": "/data/why/Datasets/MMWHS",
        "num_classes": 8,
        "class_names": [
            "LVC",
            "LAC",
            "MYO",
            "RAC",
            "RVC",
            "AA",
            "PA",
        ],
        "split_candidates": [
            "{root}/split_txts/{split}.txt",
        ],
    },
}


PRESETS = {
    "paper_cost_effective": {
        "frame_mode": 2,
        "prompt_mode": 3,
        "bidirectional": True,
        "description": "center slice + one box + bidirectional propagation",
    },
    "paper_best_prompt": {
        "frame_mode": 4,
        "prompt_mode": 3,
        "bidirectional": True,
        "description": "3 uniform slices + one box per slice + bidirectional propagation",
    },
    "paper_gt_upper": {
        "frame_mode": 4,
        "prompt_mode": 5,
        "bidirectional": True,
        "description": "3 uniform slices + GT masks + bidirectional propagation",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-style 3D SAM2 reproduction on BTCV/AMOS/MMWHS"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        help="Split name used by the current dataset folder, e.g. eval/test",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="paper_best_prompt",
        choices=sorted(PRESETS),
        help="Paper-style baseline preset",
    )
    parser.add_argument("--frame_mode", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--prompt_mode", type=int, default=None, choices=[1, 2, 3, 5])
    parser.add_argument("--bidirectional", type=int, default=None, choices=[0, 1])
    parser.add_argument(
        "--uniform_frame_count",
        type=int,
        default=3,
        help="Number of uniformly selected prompt slices when frame_mode=4",
    )
    parser.add_argument(
        "--rgb_mode",
        type=str,
        default="repeat",
        choices=["repeat", "neighbor"],
        help="repeat matches grayscale-to-3ch; neighbor uses prev/current/next slices",
    )
    parser.add_argument("--sam2_root", type=str, default="/home/why/SSL4MIS_work3")
    parser.add_argument(
        "--sam2_model_id",
        type=str,
        default="",
        help="Optional official SAM2/SAM2.1 Hugging Face model id, e.g. facebook/sam2.1-hiera-large",
    )
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="/home/why/SSL4MIS_work3/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--box_expand", type=int, default=4)
    parser.add_argument("--point_num", type=int, default=3)
    parser.add_argument(
        "--prompt_refine",
        type=str,
        default="none",
        choices=["none", "box", "box_points"],
        help="Optional refinement applied after the base prompt on each conditioning slice",
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="none",
        choices=["none", "mask", "box", "box_points"],
        help="Reuse tracked predictions as dense prompts on tracked frames before continuing propagation",
    )
    parser.add_argument(
        "--feedback_threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold used to binarize tracked predictions for feedback prompts",
    )
    parser.add_argument(
        "--feedback_min_area",
        type=int,
        default=16,
        help="Minimum predicted area required before a tracked frame can be recycled as feedback",
    )
    parser.add_argument("--min_slice_area", type=int, default=16)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/why/logs_SAM2SSL/medical_sam2_3d_repro",
    )
    return parser.parse_args()


def apply_preset(args):
    preset = PRESETS[args.preset]
    if args.frame_mode is None:
        args.frame_mode = preset["frame_mode"]
    if args.prompt_mode is None:
        args.prompt_mode = preset["prompt_mode"]
    if args.bidirectional is None:
        args.bidirectional = int(preset["bidirectional"])
    return args


def resolve_split_path(dataset_name, root_path, split):
    repo_root = Path(__file__).resolve().parent
    for pattern in DATASET_CONFIGS[dataset_name]["split_candidates"]:
        candidate = Path(
            pattern.format(root=root_path, split=split, repo=repo_root)
        ).expanduser()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate split file for {dataset_name} split={split} under root={root_path}"
    )


def read_case_list(dataset_name, root_path, split):
    split_path = resolve_split_path(dataset_name, root_path, split)
    with split_path.open("r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    return case_ids


def load_case(dataset_name, root_path, case_id):
    root_path = Path(root_path)
    if dataset_name == "BTCV":
        case_path = root_path / "btcv_h5" / f"{case_id}.h5"
        with h5py.File(case_path, "r") as h5f:
            image = h5f["image"][:].astype(np.float32)
            label = h5f["label"][:].astype(np.int16)
        return image, label

    if dataset_name == "AMOS":
        image_path = root_path / "npy" / f"{case_id}_image.npy"
        label_path = root_path / "npy" / f"{case_id}_label.npy"
        image = np.load(image_path).astype(np.float32)
        label = np.load(label_path).astype(np.int16)
        image = np.clip(image, -125, 275)
        image = (image + 125.0) / 400.0
        return image, label

    if dataset_name == "MMWHS":
        image_path = root_path / "npy" / f"ct_train_{case_id}_image.npy"
        label_path = root_path / "npy" / f"ct_train_{case_id}_label.npy"
        image = np.load(image_path).astype(np.float32)
        label = np.load(label_path).astype(np.int16)
        return image, label

    raise ValueError(f"Unsupported dataset: {dataset_name}")


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
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2_video_predictor_npz

    config_file = resolve_config_name(sam2_root, args.sam2_cfg)
    checkpoint_path = str(Path(args.sam2_checkpoint).expanduser().resolve())
    if args.sam2_model_id:
        if args.sam2_model_id not in HF_MODEL_ID_TO_FILENAMES:
            raise ValueError(f"Unsupported official SAM2 model id: {args.sam2_model_id}")
        config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[args.sam2_model_id]
        config_basename = Path(config_name).name
        local_config_candidates = [
            sam2_root / "sam2" / "configs" / config_basename,
            sam2_root / "sam2" / config_name,
        ]
        local_config = next((path for path in local_config_candidates if path.exists()), None)
        if local_config is not None:
            config_file = resolve_config_name(sam2_root, str(local_config))
        else:
            config_file = resolve_config_name(sam2_root, config_name)
        local_candidates = [
            sam2_root / "MedSAM2" / "checkpoints" / checkpoint_name,
            sam2_root / "checkpoints" / checkpoint_name,
        ]
        local_checkpoint = next((path for path in local_candidates if path.exists()), None)
        if local_checkpoint is not None:
            checkpoint_path = str(local_checkpoint.resolve())
        else:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(repo_id=args.sam2_model_id, filename=checkpoint_name)

    predictor = build_sam2_video_predictor_npz(
        config_file=config_file,
        ckpt_path=checkpoint_path,
        device=str(device),
        mode="eval",
    )
    predictor.eval()
    if hasattr(predictor, "fill_hole_area"):
        predictor.fill_hole_area = 0
    return predictor


def volume_to_rgb_sequence(volume, rgb_mode="repeat"):
    sequence = []
    depth = volume.shape[-1]
    for index in range(depth):
        if rgb_mode == "neighbor":
            prev_index = max(index - 1, 0)
            next_index = min(index + 1, depth - 1)
            frame = torch.stack(
                [volume[:, :, prev_index], volume[:, :, index], volume[:, :, next_index]],
                dim=0,
            )
        else:
            frame = torch.stack(
                [volume[:, :, index], volume[:, :, index], volume[:, :, index]],
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


def connected_components(mask_2d):
    labeled, num = ndimage.label(mask_2d.astype(np.uint8))
    components = []
    for idx in range(1, num + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area <= 0:
            continue
        components.append((area, comp))
    components.sort(key=lambda x: x[0], reverse=True)
    return components


def frame_selection(mask_volume, frame_mode):
    non_zero = np.where(mask_volume.sum(axis=(0, 1)) > 0)[0].tolist()
    if not non_zero:
        return []
    if frame_mode == 1:
        return [int(non_zero[0])]
    if frame_mode == 2:
        return [int(non_zero[len(non_zero) // 2])]
    if frame_mode == 3:
        areas = mask_volume.sum(axis=(0, 1))
        return [int(np.argmax(areas))]
    if frame_mode == 4:
        frame_count = max(1, int(getattr(frame_selection, "_uniform_frame_count", 3)))
        if len(non_zero) <= frame_count:
            return [int(x) for x in non_zero]
        anchors = np.linspace(0, len(non_zero) - 1, num=frame_count).round().astype(int)
        return sorted(set(int(non_zero[idx]) for idx in anchors))
    raise ValueError(f"Unsupported frame mode: {frame_mode}")


def largest_region_center(mask_2d):
    components = connected_components(mask_2d)
    if not components:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    coords = np.argwhere(components[0][1])
    center = coords.mean(axis=0)
    point = np.array([[float(center[1]), float(center[0])]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    return point, labels


def top_region_centers(mask_2d, max_points=3):
    components = connected_components(mask_2d)[:max_points]
    if not components:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    points = []
    for _, comp in components:
        coords = np.argwhere(comp)
        center = coords.mean(axis=0)
        points.append([float(center[1]), float(center[0])])
    return np.asarray(points, dtype=np.float32), np.ones((len(points),), dtype=np.int32)


def mask_to_box(mask_2d, box_expand):
    coords = np.argwhere(mask_2d > 0)
    if coords.size == 0:
        return None
    y_min = max(0, int(coords[:, 0].min()) - box_expand)
    y_max = min(mask_2d.shape[0] - 1, int(coords[:, 0].max()) + box_expand)
    x_min = max(0, int(coords[:, 1].min()) - box_expand)
    x_max = min(mask_2d.shape[1] - 1, int(coords[:, 1].max()) + box_expand)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def generate_slice_prompt(mask_2d, prompt_mode, box_expand, point_num):
    if prompt_mode == 1:
        points, labels = largest_region_center(mask_2d)
        return {"type": "points", "points": points, "labels": labels}
    if prompt_mode == 2:
        points, labels = top_region_centers(mask_2d, max_points=point_num)
        return {"type": "points", "points": points, "labels": labels}
    if prompt_mode == 3:
        box = mask_to_box(mask_2d, box_expand)
        return {"type": "box", "box": box}
    if prompt_mode == 5:
        return {"type": "mask", "mask": mask_2d.astype(np.float32)}
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def generate_refine_prompt(mask_2d, refine_mode, box_expand, point_num):
    if refine_mode == "none":
        return None
    box = mask_to_box(mask_2d, box_expand)
    if refine_mode == "box":
        if box is None:
            return None
        return {"type": "box", "box": box}
    if refine_mode == "box_points":
        points, labels = top_region_centers(mask_2d, max_points=point_num)
        if box is None and points.shape[0] == 0:
            return None
        return {
            "type": "box_points",
            "box": box,
            "points": points,
            "labels": labels,
        }
    raise ValueError(f"Unsupported refine mode: {refine_mode}")


def apply_prompt(predictor, state, slice_idx, obj_id, prompt):
    if prompt is None:
        return False
    if prompt["type"] == "mask":
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(slice_idx),
            obj_id=int(obj_id),
            mask=torch.from_numpy(prompt["mask"]),
        )
        return True
    if prompt["type"] == "box":
        if prompt["box"] is None:
            return False
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=int(slice_idx),
            obj_id=int(obj_id),
            box=torch.from_numpy(prompt["box"]),
        )
        return True
    if prompt["type"] == "points":
        if prompt["points"].shape[0] == 0:
            return False
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=int(slice_idx),
            obj_id=int(obj_id),
            points=torch.from_numpy(prompt["points"]),
            labels=torch.from_numpy(prompt["labels"]),
        )
        return True
    if prompt["type"] == "box_points":
        kwargs = {
            "inference_state": state,
            "frame_idx": int(slice_idx),
            "obj_id": int(obj_id),
        }
        if prompt["box"] is not None:
            kwargs["box"] = torch.from_numpy(prompt["box"])
        if prompt["points"].shape[0] > 0:
            kwargs["points"] = torch.from_numpy(prompt["points"])
            kwargs["labels"] = torch.from_numpy(prompt["labels"])
        if "box" not in kwargs and "points" not in kwargs:
            return False
        predictor.add_new_points_or_box(**kwargs)
        return True
    raise ValueError(f"Unsupported prompt type: {prompt['type']}")


def add_prompts(predictor, state, selected_slices, mask_volume, obj_id, args):
    valid = False
    for slice_idx in selected_slices:
        slice_mask = mask_volume[:, :, slice_idx]
        if int(slice_mask.sum()) < args.min_slice_area:
            continue
        prompt = generate_slice_prompt(
            slice_mask, args.prompt_mode, args.box_expand, args.point_num
        )
        valid = apply_prompt(predictor, state, slice_idx, obj_id, prompt) or valid
        if args.prompt_mode == 5 and args.prompt_refine != "none":
            refine_prompt = generate_refine_prompt(
                slice_mask, args.prompt_refine, args.box_expand, args.point_num
            )
            valid = apply_prompt(predictor, state, slice_idx, obj_id, refine_prompt) or valid
    return valid


def logits_to_frame_tensor(out_mask_logits):
    if out_mask_logits.dim() == 4:
        return out_mask_logits[:, 0]
    if out_mask_logits.dim() == 3:
        return out_mask_logits
    raise ValueError(f"Unexpected SAM2 logit shape: {tuple(out_mask_logits.shape)}")


def extract_class_logit(object_ids, out_mask_logits, class_id):
    frame_logits = logits_to_frame_tensor(out_mask_logits)
    for obj_pos, obj_id in enumerate(object_ids):
        if int(obj_id) == int(class_id) and obj_pos < frame_logits.shape[0]:
            return frame_logits[obj_pos]
    return None


def collect_scores_from_generator(class_scores, class_id, generator):
    for frame_idx, object_ids, out_mask_logits in generator:
        class_logit = extract_class_logit(object_ids, out_mask_logits, class_id)
        if class_logit is None:
            continue
        class_scores[:, :, int(frame_idx)] = torch.sigmoid(class_logit).float().cpu()


def maybe_add_feedback_prompt(
    predictor, state, frame_idx, class_id, class_logit, selected_slice_set, args
):
    if args.feedback_mode == "none" or int(frame_idx) in selected_slice_set:
        return None
    feedback_mask = (
        torch.sigmoid(class_logit).detach().float().cpu().numpy() >= args.feedback_threshold
    )
    if int(feedback_mask.sum()) < args.feedback_min_area:
        return None
    if args.feedback_mode == "mask":
        return predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(frame_idx),
            obj_id=int(class_id),
            mask=torch.from_numpy(feedback_mask.astype(np.float32)),
        )
    prompt = generate_refine_prompt(
        feedback_mask.astype(np.uint8),
        args.feedback_mode,
        args.box_expand,
        args.point_num,
    )
    if prompt is None:
        return None
    if prompt["type"] == "box":
        return predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=int(frame_idx),
            obj_id=int(class_id),
            box=torch.from_numpy(prompt["box"]),
        )
    kwargs = {
        "inference_state": state,
        "frame_idx": int(frame_idx),
        "obj_id": int(class_id),
    }
    if prompt["box"] is not None:
        kwargs["box"] = torch.from_numpy(prompt["box"])
    if prompt["points"].shape[0] > 0:
        kwargs["points"] = torch.from_numpy(prompt["points"])
        kwargs["labels"] = torch.from_numpy(prompt["labels"])
    if "box" not in kwargs and "points" not in kwargs:
        return None
    return predictor.add_new_points_or_box(**kwargs)


def propagate_direction_stepwise(
    predictor,
    state,
    class_scores,
    class_id,
    selected_slices,
    start_frame_idx,
    end_frame_idx,
    reverse,
    args,
):
    if start_frame_idx < 0 or end_frame_idx < 0:
        return
    selected_slice_set = {int(x) for x in selected_slices}
    step = -1 if reverse else 1
    for frame_idx in range(int(start_frame_idx), int(end_frame_idx) + step, step):
        generator = predictor.propagate_in_video(
            state,
            start_frame_idx=int(frame_idx),
            max_frame_num_to_track=0,
            reverse=reverse,
        )
        for out_frame_idx, object_ids, out_mask_logits in generator:
            class_logit = extract_class_logit(object_ids, out_mask_logits, class_id)
            if class_logit is None:
                continue
            class_scores[:, :, int(out_frame_idx)] = torch.sigmoid(class_logit).float().cpu()
            feedback_out = maybe_add_feedback_prompt(
                predictor,
                state,
                out_frame_idx,
                class_id,
                class_logit,
                selected_slice_set,
                args,
            )
            if feedback_out is None:
                continue
            fb_frame_idx, fb_object_ids, fb_logits = feedback_out
            fb_class_logit = extract_class_logit(fb_object_ids, fb_logits, class_id)
            if fb_class_logit is None:
                continue
            class_scores[:, :, int(fb_frame_idx)] = torch.sigmoid(fb_class_logit).float().cpu()


def run_single_class(predictor, frame_sequence, mask_volume, class_id, selected_slices, args):
    height, width, depth = mask_volume.shape
    anchor_idx = selected_slices[len(selected_slices) // 2] if len(selected_slices) > 1 else selected_slices[0]
    class_scores = torch.zeros((height, width, depth), dtype=torch.float32)

    if args.bidirectional:
        if args.feedback_mode != "none":
            state = predictor.init_state(images=frame_sequence, video_height=height, video_width=width)
            valid = add_prompts(predictor, state, selected_slices, mask_volume, class_id, args)
            if valid:
                propagate_direction_stepwise(
                    predictor,
                    state,
                    class_scores,
                    class_id,
                    selected_slices,
                    start_frame_idx=int(anchor_idx),
                    end_frame_idx=0,
                    reverse=True,
                    args=args,
                )
            predictor.reset_state(state)

            state = predictor.init_state(images=frame_sequence, video_height=height, video_width=width)
            valid = add_prompts(predictor, state, selected_slices, mask_volume, class_id, args)
            if valid and int(anchor_idx) + 1 < depth:
                propagate_direction_stepwise(
                    predictor,
                    state,
                    class_scores,
                    class_id,
                    selected_slices,
                    start_frame_idx=int(anchor_idx) + 1,
                    end_frame_idx=depth - 1,
                    reverse=False,
                    args=args,
                )
            predictor.reset_state(state)
            return class_scores

        state = predictor.init_state(images=frame_sequence, video_height=height, video_width=width)
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
                ),
            )
        predictor.reset_state(state)

        state = predictor.init_state(images=frame_sequence, video_height=height, video_width=width)
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
                ),
            )
        predictor.reset_state(state)
        return class_scores

    state = predictor.init_state(images=frame_sequence, video_height=height, video_width=width)
    valid = add_prompts(predictor, state, selected_slices, mask_volume, class_id, args)
    if valid:
        if args.feedback_mode != "none":
            propagate_direction_stepwise(
                predictor,
                state,
                class_scores,
                class_id,
                selected_slices,
                start_frame_idx=0,
                end_frame_idx=depth - 1,
                reverse=False,
                args=args,
            )
        else:
            collect_scores_from_generator(
                class_scores,
                class_id,
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=depth,
                    reverse=False,
                ),
            )
    predictor.reset_state(state)
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


def per_class_dice_present(pred, label, num_classes):
    dices = []
    for class_id in range(1, num_classes):
        pred_mask = pred == class_id
        gt_mask = label == class_id
        gt_sum = int(gt_mask.sum())
        if gt_sum == 0:
            dices.append(np.nan)
            continue
        denom = pred_mask.sum() + gt_mask.sum()
        dices.append(float(2.0 * (pred_mask & gt_mask).sum() / max(int(denom), 1)))
    return dices


def per_class_iou_nonempty_slices(pred, label, num_classes):
    ious = []
    for class_id in range(1, num_classes):
        pred_mask = pred == class_id
        gt_mask = label == class_id
        slice_scores = []
        for z in range(label.shape[-1]):
            gt_slice = gt_mask[:, :, z]
            if gt_slice.sum() == 0:
                continue
            pred_slice = pred_mask[:, :, z]
            inter = np.logical_and(pred_slice, gt_slice).sum()
            union = pred_slice.sum() + gt_slice.sum() - inter
            slice_scores.append(float(inter / max(int(union), 1)))
        ious.append(float(np.mean(slice_scores)) if slice_scores else np.nan)
    return ious


def nanmean(values):
    arr = np.asarray(values, dtype=np.float32)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def main():
    args = apply_preset(parse_args())
    if args.prompt_refine != "none" and args.prompt_mode != 5:
        raise ValueError("prompt_refine currently only supports mask prompt_mode=5")
    frame_selection._uniform_frame_count = int(args.uniform_frame_count)
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    if args.root_path is None:
        args.root_path = dataset_cfg["root_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_predictor(args, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    case_ids = read_case_list(args.dataset, args.root_path, args.split)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]

    run_tag = (
        f"{args.dataset}_"
        f"{(args.sam2_model_id.split('/')[-1].replace('-', '_') if args.sam2_model_id else Path(args.sam2_checkpoint).stem)}_"
        f"{args.preset}_F{args.frame_mode}_P{args.prompt_mode}_"
        f"D{int(args.bidirectional)}_K{int(args.uniform_frame_count)}_{args.rgb_mode}_"
        f"R{args.prompt_refine}_FB{args.feedback_mode}_{args.split}_"
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
            volume = torch.from_numpy(image_np)
            frame_sequence = preprocess_sequence(
                volume_to_rgb_sequence(volume, rgb_mode=args.rgb_mode),
                input_size=args.input_size,
                device=device,
            )

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
                    frame_sequence,
                    mask_volume,
                    class_id,
                    selected_slices,
                    args,
                )

            pred = scores_to_labels(class_scores, label_np.shape, num_classes)
            class_dice = per_class_dice_present(pred, label_np, num_classes)
            class_iou = per_class_iou_nonempty_slices(pred, label_np, num_classes)
            case_record = {
                "case_id": case_id,
                "mean_dice_present": nanmean(class_dice),
                "mean_iou_nonempty_slices": nanmean(class_iou),
                "class_dice_present": class_dice,
                "class_iou_nonempty_slices": class_iou,
                "meta": meta,
            }
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


if __name__ == "__main__":
    main()
