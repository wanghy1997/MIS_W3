import argparse
import logging
import os
import random
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/data/why/Datasets/BTCV")
    parser.add_argument("--save_path", type=str, default="/data/why/logs_SAM2SSL")
    parser.add_argument("--exp", type=str, default="BTCV_SAM2TinyPublic_LoRA")
    parser.add_argument("--train_split", type=str, default="labeled_10p")
    parser.add_argument("--eval_split", type=str, default="eval")
    parser.add_argument("--max_eval_cases", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam2_root", type=str, default="/home/why/SSL4MIS_work3")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="/home/why/SSL4MIS_work3/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_clip_radius", type=int, default=12)
    parser.add_argument("--max_classes_per_volume", type=int, default=4)
    parser.add_argument("--eval_max_classes_per_volume", type=int, default=13)
    parser.add_argument("--min_voxels", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--disable_hole_filling", type=int, default=1)
    parser.add_argument("--num_condition_frames", type=int, default=1)
    parser.add_argument("--use_memory_encoder_lora", type=int, default=1)
    parser.add_argument("--use_amp", type=int, default=1)
    return parser.parse_args()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + lora_out * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__()
        if base.groups != 1:
            raise ValueError("LoRAConv2d only supports groups=1.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.scale = alpha / max(rank, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            in_channels=rank,
            out_channels=base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_up(self.dropout(self.lora_down(x)))
        return base_out + lora_out * self.scale


def inject_lora(module: nn.Module, rank: int, alpha: float, dropout: float, use_conv: bool, prefix: str):
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
            continue
        if use_conv and isinstance(child, nn.Conv2d) and child.groups == 1:
            setattr(module, child_name, LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
            continue
        replaced.extend(inject_lora(child, rank, alpha, dropout, use_conv, full_name))
    return replaced


def freeze_non_lora(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True


def count_trainable(model: nn.Module):
    total = 0
    trainable = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return trainable, total


def read_list(root_path: str, split: str):
    path = os.path.join(root_path, "split_txts", f"{split}.txt")
    return sorted(np.loadtxt(path, dtype=str).tolist())


def load_case(root_path: str, case_id: str):
    case_path = os.path.join(root_path, "btcv_h5", f"{case_id}.h5")
    with h5py.File(case_path, "r") as handle:
        image = handle["image"][:]
        label = handle["label"][:]
    return image.astype(np.float32), label.astype(np.int64)


def volume_to_rgb_sequence(volume: torch.Tensor):
    frames = []
    depth = volume.shape[-1]
    for index in range(depth):
        prev_index = max(index - 1, 0)
        next_index = min(index + 1, depth - 1)
        frame = torch.stack(
            [volume[:, :, prev_index], volume[:, :, index], volume[:, :, next_index]],
            dim=0,
        )
        frames.append(frame)
    return torch.stack(frames, dim=0)


def preprocess_sequence(frame_sequence: torch.Tensor, image_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor):
    frame_sequence = F.interpolate(
        frame_sequence.float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    frame_min = frame_sequence.amin()
    frame_max = frame_sequence.amax()
    frame_sequence = (frame_sequence - frame_min) / (frame_max - frame_min).clamp_min(1e-6)
    frame_sequence = frame_sequence.to(pixel_mean.device)
    return (frame_sequence - pixel_mean) / pixel_std


def select_classes(label_np: np.ndarray, max_classes: int, min_voxels: int):
    classes = []
    for class_id in np.unique(label_np):
        class_id = int(class_id)
        if class_id == 0:
            continue
        voxels = int((label_np == class_id).sum())
        if voxels < min_voxels:
            continue
        classes.append((voxels, class_id))
    classes.sort(reverse=True)
    return [class_id for _, class_id in classes[:max_classes]]


def pick_condition_slice(label_np: np.ndarray, class_id: int):
    binary = label_np == class_id
    slice_areas = binary.sum(axis=(0, 1))
    cond_idx = int(np.argmax(slice_areas))
    return cond_idx, binary[:, :, cond_idx].astype(np.float32)


def binary_dice_from_logits(logits: torch.Tensor, target: torch.Tensor):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def binary_dice_np(pred: np.ndarray, target: np.ndarray):
    intersection = float((pred & target).sum())
    denominator = float(pred.sum() + target.sum())
    return (2.0 * intersection + 1.0) / (denominator + 1.0)


def train_init_state(model, images, video_height, video_width):
    compute_device = model.device
    inference_state = {
        "images": images,
        "num_frames": len(images),
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": video_height,
        "video_width": video_width,
        "device": compute_device,
        "storage_device": compute_device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
        "tracking_has_started": False,
        "frames_already_tracked": {},
    }
    model._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


def train_add_new_mask(model, inference_state, frame_idx, obj_id, mask):
    obj_idx = model._obj_id_to_idx(inference_state, obj_id)
    point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
    mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=inference_state["device"])
    assert mask.dim() == 2
    mask_H, mask_W = mask.shape
    mask_inputs_orig = mask[None, None].float().to(inference_state["device"])
    if mask_H != model.image_size or mask_W != model.image_size:
        mask_inputs = F.interpolate(
            mask_inputs_orig,
            size=(model.image_size, model.image_size),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
        mask_inputs = (mask_inputs >= 0.5).float()
    else:
        mask_inputs = mask_inputs_orig

    mask_inputs_per_frame[frame_idx] = mask_inputs
    point_inputs_per_frame.pop(frame_idx, None)
    is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
    reverse = False if is_init_cond_frame else inference_state["frames_already_tracked"][frame_idx]["reverse"]
    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
    obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
    is_cond = is_init_cond_frame or model.add_all_frames_to_correct_as_cond
    storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

    current_out, _ = model._run_single_frame_inference(
        inference_state=inference_state,
        output_dict=obj_output_dict,
        frame_idx=frame_idx,
        batch_size=1,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=None,
        mask_inputs=mask_inputs,
        reverse=reverse,
        run_mem_encoder=False,
    )
    obj_temp_output_dict[storage_key][frame_idx] = current_out
    consolidated_out = model._consolidate_temp_output_across_obj(
        inference_state,
        frame_idx,
        is_cond=is_cond,
        run_mem_encoder=False,
        consolidate_at_video_res=True,
    )
    _, video_res_masks = model._get_orig_video_res_output(
        inference_state, consolidated_out["pred_masks_video_res"]
    )
    return frame_idx, inference_state["obj_ids"], video_res_masks


def train_propagate_preflight(model, inference_state):
    inference_state["tracking_has_started"] = True
    batch_size = model._get_obj_num(inference_state)
    temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
    output_dict = inference_state["output_dict"]
    consolidated_frame_inds = inference_state["consolidated_frame_inds"]

    for is_cond in [False, True]:
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        temp_frame_inds = set()
        for obj_temp_output_dict in temp_output_dict_per_obj.values():
            temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
        consolidated_frame_inds[storage_key].update(temp_frame_inds)
        for frame_idx in temp_frame_inds:
            consolidated_out = model._consolidate_temp_output_across_obj(
                inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
            )
            output_dict[storage_key][frame_idx] = consolidated_out
            model._add_output_per_object(
                inference_state, frame_idx, consolidated_out, storage_key
            )
            clear_non_cond_mem = model.clear_non_cond_mem_around_input and (
                model.clear_non_cond_mem_for_multi_obj or batch_size <= 1
            )
            if clear_non_cond_mem:
                model._clear_non_cond_mem_around_input(inference_state, frame_idx)
        for obj_temp_output_dict in temp_output_dict_per_obj.values():
            obj_temp_output_dict[storage_key].clear()

    for frame_idx in output_dict["cond_frame_outputs"]:
        output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        for frame_idx in obj_output_dict["cond_frame_outputs"]:
            obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
    for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
        consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)


def train_propagate_in_video(model, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
    train_propagate_preflight(model, inference_state)
    output_dict = inference_state["output_dict"]
    consolidated_frame_inds = inference_state["consolidated_frame_inds"]
    num_frames = inference_state["num_frames"]
    batch_size = model._get_obj_num(inference_state)
    if len(output_dict["cond_frame_outputs"]) == 0:
        raise RuntimeError("No conditioning frames are available for propagation.")

    clear_non_cond_mem = model.clear_non_cond_mem_around_input and (
        model.clear_non_cond_mem_for_multi_obj or batch_size <= 1
    )
    if start_frame_idx is None:
        start_frame_idx = min(output_dict["cond_frame_outputs"])
    if max_frame_num_to_track is None:
        max_frame_num_to_track = num_frames

    if reverse:
        end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
        processing_order = range(start_frame_idx, end_frame_idx - 1, -1) if start_frame_idx > 0 else []
    else:
        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)

    outputs = OrderedDict()
    for frame_idx in processing_order:
        if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
            if clear_non_cond_mem:
                model._clear_non_cond_mem_around_input(inference_state, frame_idx)
        elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out, pred_masks = model._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=output_dict,
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=reverse,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame_idx] = current_out
        model._add_output_per_object(inference_state, frame_idx, current_out, storage_key)
        inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
        _, video_res_masks = model._get_orig_video_res_output(inference_state, pred_masks)
        outputs[int(frame_idx)] = video_res_masks[0, 0]
    return outputs


def merge_direction_outputs(forward_outputs, reverse_outputs):
    merged = OrderedDict()
    keys = sorted(set(forward_outputs.keys()) | set(reverse_outputs.keys()))
    for frame_idx in keys:
        if frame_idx in forward_outputs and frame_idx in reverse_outputs:
            merged[frame_idx] = 0.5 * (forward_outputs[frame_idx] + reverse_outputs[frame_idx])
        elif frame_idx in forward_outputs:
            merged[frame_idx] = forward_outputs[frame_idx]
        else:
            merged[frame_idx] = reverse_outputs[frame_idx]
    return merged


def run_training_propagation(model, frames, cond_idx, cond_mask, clip_radius):
    height, width = cond_mask.shape
    max_track = clip_radius + 1

    state_fwd = train_init_state(model, frames, video_height=height, video_width=width)
    train_add_new_mask(model, state_fwd, cond_idx, obj_id=1, mask=cond_mask)
    forward_outputs = train_propagate_in_video(
        model,
        state_fwd,
        start_frame_idx=cond_idx,
        max_frame_num_to_track=max_track,
        reverse=False,
    )

    state_rev = train_init_state(model, frames, video_height=height, video_width=width)
    train_add_new_mask(model, state_rev, cond_idx, obj_id=1, mask=cond_mask)
    reverse_outputs = train_propagate_in_video(
        model,
        state_rev,
        start_frame_idx=cond_idx,
        max_frame_num_to_track=max_track,
        reverse=True,
    )
    return merge_direction_outputs(forward_outputs, reverse_outputs)


@torch.no_grad()
def run_eval_propagation(model, frames, cond_idx, cond_mask):
    height, width = cond_mask.shape

    state_fwd = model.init_state(images=frames, video_height=height, video_width=width)
    model.add_new_mask(state_fwd, cond_idx, obj_id=1, mask=cond_mask)
    forward_outputs = OrderedDict()
    for frame_idx, _obj_ids, video_res_masks in model.propagate_in_video(
        state_fwd, start_frame_idx=cond_idx, max_frame_num_to_track=len(frames), reverse=False
    ):
        forward_outputs[int(frame_idx)] = video_res_masks[0, 0].detach().cpu()
    model.reset_state(state_fwd)

    state_rev = model.init_state(images=frames, video_height=height, video_width=width)
    model.add_new_mask(state_rev, cond_idx, obj_id=1, mask=cond_mask)
    reverse_outputs = OrderedDict()
    for frame_idx, _obj_ids, video_res_masks in model.propagate_in_video(
        state_rev, start_frame_idx=cond_idx, max_frame_num_to_track=len(frames), reverse=True
    ):
        reverse_outputs[int(frame_idx)] = video_res_masks[0, 0].detach().cpu()
    model.reset_state(state_rev)
    return merge_direction_outputs(forward_outputs, reverse_outputs)


def build_model(args, device):
    sam2_root = Path(args.sam2_root).resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor_npz

    model = build_sam2_video_predictor_npz(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=str(device),
        mode="train",
    )
    if bool(args.disable_hole_filling) and hasattr(model, "fill_hole_area"):
        model.fill_hole_area = 0

    replaced = []
    replaced.extend(
        inject_lora(
            model.memory_attention,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            use_conv=False,
            prefix="memory_attention",
        )
    )
    if bool(args.use_memory_encoder_lora):
        replaced.extend(
            inject_lora(
                model.memory_encoder,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                use_conv=True,
                prefix="memory_encoder",
            )
        )
    replaced.extend(
        inject_lora(
            model.sam_mask_decoder,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            use_conv=True,
            prefix="sam_mask_decoder",
        )
    )
    freeze_non_lora(model)
    model = model.to(device)
    trainable, total = count_trainable(model)
    return model, replaced, trainable, total


def get_lora_state_dict(model: nn.Module):
    state = {}
    for key, value in model.state_dict().items():
        if "lora_" in key:
            state[key] = value.detach().cpu()
    return state


def setup_logging(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(save_dir, "log_train.txt"), mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def evaluate(model, cases, args, device, pixel_mean, pixel_std):
    model.eval()
    all_dice = []
    eval_cases = cases[: args.max_eval_cases]
    for case_id in tqdm(eval_cases, desc="eval", leave=False):
        volume_np, label_np = load_case(args.root_path, case_id)
        volume = torch.from_numpy(volume_np)
        frames = volume_to_rgb_sequence(volume)
        frames = preprocess_sequence(
            frames,
            image_size=model.image_size,
            pixel_mean=pixel_mean.to(device),
            pixel_std=pixel_std.to(device),
        ).to(device)
        class_scores = np.zeros((args.num_classes - 1,) + label_np.shape, dtype=np.float32)
        selected_classes = select_classes(
            label_np,
            max_classes=args.eval_max_classes_per_volume,
            min_voxels=args.min_voxels,
        )
        for class_id in selected_classes:
            cond_idx, cond_mask = pick_condition_slice(label_np, class_id)
            pred_logits = run_eval_propagation(model, frames, cond_idx, cond_mask)
            for frame_idx, logits in pred_logits.items():
                class_scores[class_id - 1, :, :, frame_idx] = torch.sigmoid(logits).numpy()
        pred_volume = np.argmax(
            np.concatenate(
                [
                    np.clip(1.0 - class_scores.max(axis=0, keepdims=True), 1e-4, 1.0),
                    class_scores,
                ],
                axis=0,
            ),
            axis=0,
        )
        case_dice = []
        for class_id in range(1, args.num_classes):
            pred_binary = pred_volume == class_id
            gt_binary = label_np == class_id
            case_dice.append(binary_dice_np(pred_binary, gt_binary))
        all_dice.append(case_dice)
    dice_array = np.asarray(all_dice, dtype=np.float32)
    return float(dice_array.mean()), dice_array.mean(axis=0)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_path, args.exp)
    setup_logging(save_dir)
    logging.info(str(args))
    logging.info("Using device: %s", device)

    train_cases = read_list(args.root_path, args.train_split)
    eval_cases = read_list(args.root_path, args.eval_split)
    logging.info("Train cases: %d", len(train_cases))
    logging.info("Eval cases: %d", min(len(eval_cases), args.max_eval_cases))

    model, replaced, trainable, total = build_model(args, device)
    logging.info("Injected LoRA modules: %d", len(replaced))
    logging.info("Sample LoRA targets: %s", replaced[:20])
    logging.info("Trainable params: %d / %d (%.4f%%)", trainable, total, 100.0 * trainable / max(total, 1))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp) and device.type == "cuda")
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    best_eval = -1.0
    best_path = os.path.join(save_dir, "best_lora.pt")

    for epoch in range(1, args.epochs + 1):
        # Keep the predictor in eval mode so its output structure matches the
        # inference path, while LoRA parameters still receive gradients.
        model.eval()
        epoch_losses = []
        for case_id in tqdm(train_cases, desc=f"train-{epoch}", leave=False):
            volume_np, label_np = load_case(args.root_path, case_id)
            volume = torch.from_numpy(volume_np)
            frames = volume_to_rgb_sequence(volume)
            frames = preprocess_sequence(
                frames,
                image_size=model.image_size,
                pixel_mean=pixel_mean.to(device),
                pixel_std=pixel_std.to(device),
            ).to(device)
            selected_classes = select_classes(
                label_np,
                max_classes=args.max_classes_per_volume,
                min_voxels=args.min_voxels,
            )
            if not selected_classes:
                continue
            for class_id in selected_classes:
                cond_idx, cond_mask_np = pick_condition_slice(label_np, class_id)
                cond_mask = torch.from_numpy(cond_mask_np).to(device)
                with torch.cuda.amp.autocast(enabled=bool(args.use_amp) and device.type == "cuda"):
                    pred_logits = run_training_propagation(
                        model,
                        frames,
                        cond_idx=cond_idx,
                        cond_mask=cond_mask,
                        clip_radius=args.train_clip_radius,
                    )
                    if not pred_logits:
                        continue
                    frame_indices = list(pred_logits.keys())
                    logits = torch.stack([pred_logits[idx] for idx in frame_indices], dim=0).unsqueeze(1)
                    targets = torch.stack(
                        [
                            torch.from_numpy((label_np[:, :, idx] == class_id).astype(np.float32))
                            for idx in frame_indices
                        ],
                        dim=0,
                    ).unsqueeze(1).to(device)
                    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
                    dice_loss = binary_dice_from_logits(logits, targets)
                    loss = bce_loss + dice_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logging.info("Epoch %d/%d: train_loss=%.4f", epoch, args.epochs, mean_loss)

        if epoch % args.eval_every != 0:
            continue

        mean_dice, class_dice = evaluate(model, eval_cases, args, device, pixel_mean, pixel_std)
        logging.info("Epoch %d eval mean DSC: %.4f", epoch, mean_dice)
        logging.info("Epoch %d eval class DSC: %s", epoch, np.array2string(class_dice, precision=4))

        ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}_lora.pt")
        torch.save(
            {
                "epoch": epoch,
                "args": vars(args),
                "mean_dice": mean_dice,
                "class_dice": class_dice,
                "lora_state_dict": get_lora_state_dict(model),
            },
            ckpt_path,
        )
        logging.info("Saved checkpoint to %s", ckpt_path)
        if mean_dice > best_eval:
            best_eval = mean_dice
            torch.save(
                {
                    "epoch": epoch,
                    "args": vars(args),
                    "mean_dice": mean_dice,
                    "class_dice": class_dice,
                    "lora_state_dict": get_lora_state_dict(model),
                },
                best_path,
            )
            logging.info("Saved best checkpoint to %s", best_path)


if __name__ == "__main__":
    main()
