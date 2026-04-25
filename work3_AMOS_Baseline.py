import os
import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
from utils_ori import metrics, ramps, test_amos, cube_losses, cube_utils
from dataloaders.dataset_SAM2 import *
from networks.magicnet import VNet_Magic
from loss_amos import GADice, GACE


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='AMOS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/why/Datasets/amos/npy/', help='Name of Dataset')
parser.add_argument('--save_path', type=str, default='/data/why/logs_SAM2SSL/', help='path to save')
parser.add_argument('--exp', type=str, default='SAM2SSL', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=17000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=30, help='total samples of the dataset')
parser.add_argument('--max_train_samples', type=int, default=20, help='maximum samples to train')
parser.add_argument('--max_test_samples', type=int, default=6, help='maximum samples to test')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=4, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
parser.add_argument('--use_medsam2', type=int, default=1, help='whether to use MedSAM2 as the second teacher')
parser.add_argument('--medsam2_root', type=str, default=os.path.join(SCRIPT_DIR, 'MedSAM2'), help='MedSAM2 repo root')
parser.add_argument('--medsam2_cfg', type=str, default='configs/sam2.1_hiera_t512.yaml', help='MedSAM2 config path')
parser.add_argument('--medsam2_checkpoint', type=str, default='./checkpoints/MedSAM2_latest.pt', help='MedSAM2 checkpoint path')
parser.add_argument('--medsam2_warmup', type=int, default=200, help='start using MedSAM2 after this many iterations')
parser.add_argument('--medsam2_interval', type=int, default=4, help='run MedSAM2 every N iterations')
parser.add_argument('--medsam2_blend_alpha', type=float, default=0.35, help='blend weight of MedSAM2 over EMA teacher')
parser.add_argument('--medsam2_num_classes', type=int, default=16, help='number of semantic classes for MedSAM2 teacher')
parser.add_argument('--medsam2_prompt_type', type=str, default='box', choices=['box', 'mask'], help='prompt type for MedSAM2')
parser.add_argument('--medsam2_prompt_thresh', type=float, default=0.7, help='confidence threshold used to build MedSAM2 prompts')
parser.add_argument('--medsam2_teacher_prob_thresh', type=float, default=0.6, help='confidence threshold used to rank MedSAM2 teacher voxels')
parser.add_argument('--medsam2_min_voxels', type=int, default=256, help='minimum 3D voxels for a class to be tracked by MedSAM2')
parser.add_argument('--medsam2_min_slice_area', type=int, default=32, help='minimum 2D area for a conditioning slice')
parser.add_argument('--medsam2_box_expand', type=int, default=4, help='expand MedSAM2 prompt boxes by this many pixels')
parser.add_argument('--medsam2_max_classes', type=int, default=4, help='maximum foreground classes to track per unlabeled patch')
parser.add_argument('--medsam2_num_condition_frames', type=int, default=1, help='number of key slices used as MedSAM2 conditioning frames')
parser.add_argument('--medsam2_full_volume', type=int, default=1, help='use full-volume teacher inference before cropping back to the current patch')
parser.add_argument('--medsam2_stride_xy', type=int, default=64, help='sliding-window stride in x/y for full-volume EMA inference')
parser.add_argument('--medsam2_stride_z', type=int, default=64, help='sliding-window stride in z for full-volume EMA inference')
parser.add_argument('--medsam2_cache_size', type=int, default=2, help='number of full-volume teacher cases kept in memory')
parser.add_argument('--medsam2_cache_ttl', type=int, default=40, help='refresh cached full-volume teacher outputs after this many iterations')
parser.add_argument('--medsam2_hard_weight', type=float, default=0.5, help='weight for hard MedSAM2 supervision')
parser.add_argument('--medsam2_soft_weight', type=float, default=0.25, help='weight for soft MedSAM2 distillation')
parser.add_argument('--medsam2_distill_temperature', type=float, default=1.0, help='temperature for MedSAM2 KL distillation')
args = parser.parse_args()


class MedSAM2VolumeTeacher(torch.nn.Module):
    """
    MedSAM2 teacher for 3D CT volumes or patches.

    Input:
      images: [B, 1, W, H, D]
      prompt_probs: [B, C, W, H, D]

    Output:
      probs: [B, C, W, H, D]
      labels: [B, W, H, D]
      coverage: [B, 1, W, H, D]
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_classes = int(args.medsam2_num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_type = str(args.medsam2_prompt_type).lower()
        self.prompt_thresh = float(args.medsam2_prompt_thresh)
        self.teacher_prob_thresh = float(args.medsam2_teacher_prob_thresh)
        self.min_voxels = max(1, int(args.medsam2_min_voxels))
        self.min_slice_area = max(1, int(args.medsam2_min_slice_area))
        self.box_expand = max(0, int(args.medsam2_box_expand))
        self.max_classes = max(1, int(args.medsam2_max_classes))
        self.num_condition_frames = max(1, int(args.medsam2_num_condition_frames))

        medsam2_root = Path(args.medsam2_root).expanduser().resolve()
        if not medsam2_root.exists():
            raise FileNotFoundError(f"MedSAM2 root not found: {medsam2_root}")
        if str(medsam2_root) not in sys.path:
            sys.path.insert(0, str(medsam2_root))

        try:
            from sam2.build_sam import build_sam2_video_predictor_npz
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import MedSAM2. Please install its runtime dependencies "
                "(at least hydra-core, omegaconf, iopath, tensordict) in the current environment."
            ) from exc

        config_name = self._resolve_config_name(medsam2_root, args.medsam2_cfg)
        checkpoint_path = self._resolve_file(medsam2_root, args.medsam2_checkpoint)

        self.predictor = build_sam2_video_predictor_npz(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=str(self.device),
            mode="eval",
        )
        self.predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad = False

        self.input_size = int(getattr(self.predictor, "image_size", 512))
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _resolve_file(base_dir: Path, value: str) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            if not candidate.exists():
                raise FileNotFoundError(f"File not found: {candidate}")
            return candidate

        direct = (base_dir / value).resolve()
        if direct.exists():
            return direct

        nested = (base_dir / "sam2" / value).resolve()
        if nested.exists():
            return nested

        raise FileNotFoundError(f"Cannot resolve file '{value}' under {base_dir}")

    @staticmethod
    def _resolve_config_name(base_dir: Path, value: str) -> str:
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

    def _volume_to_rgb_sequence(self, volume):
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

    def _preprocess_sequence(self, frame_sequence):
        frame_sequence = F.interpolate(
            frame_sequence.float(),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        frame_min = frame_sequence.amin()
        frame_max = frame_sequence.amax()
        frame_sequence = (frame_sequence - frame_min) / (frame_max - frame_min).clamp_min(1e-6)
        return (frame_sequence - self.pixel_mean.to(frame_sequence.device)) / self.pixel_std.to(frame_sequence.device)

    def _select_classes(self, prompt_probs):
        prompt_labels = torch.argmax(prompt_probs, dim=0)
        class_candidates = []
        for class_id in range(1, self.num_classes):
            class_mask = prompt_labels == class_id
            class_voxels = int(class_mask.sum().item())
            if class_voxels < self.min_voxels:
                continue
            confident_voxels = int((class_mask & (prompt_probs[class_id] >= self.teacher_prob_thresh)).sum().item())
            class_candidates.append((confident_voxels, class_voxels, class_id))

        class_candidates.sort(reverse=True)
        return [item[2] for item in class_candidates[: self.max_classes]]

    def _pick_key_slices(self, class_prob, class_mask):
        confident_mask = class_mask & (class_prob >= self.prompt_thresh)
        slice_area = confident_mask.sum(dim=(0, 1))
        if slice_area.max().item() < self.min_slice_area:
            confident_mask = class_mask
            slice_area = confident_mask.sum(dim=(0, 1))

        valid_slices = torch.nonzero(slice_area >= self.min_slice_area, as_tuple=False).flatten()
        if valid_slices.numel() == 0:
            return [], confident_mask

        if valid_slices.numel() <= self.num_condition_frames:
            selected = valid_slices.tolist()
        else:
            selected_scores = slice_area[valid_slices]
            _, top_indices = torch.topk(selected_scores, k=self.num_condition_frames)
            selected = valid_slices[top_indices].tolist()

        selected = sorted({int(index) for index in selected})
        return selected, confident_mask

    def _mask_to_box(self, mask_2d):
        coords = torch.nonzero(mask_2d, as_tuple=False)
        if coords.numel() == 0:
            return None

        y_min = max(0, int(coords[:, 0].min().item()) - self.box_expand)
        y_max = min(mask_2d.shape[0] - 1, int(coords[:, 0].max().item()) + self.box_expand)
        x_min = max(0, int(coords[:, 1].min().item()) - self.box_expand)
        x_max = min(mask_2d.shape[1] - 1, int(coords[:, 1].max().item()) + self.box_expand)
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

    def _add_prompts(self, inference_state, selected_slices, prompt_mask):
        valid_prompt = False
        for slice_idx in selected_slices:
            slice_mask = prompt_mask[:, :, slice_idx]
            if int(slice_mask.sum().item()) < self.min_slice_area:
                continue

            if self.prompt_type == "mask":
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=int(slice_idx),
                    obj_id=1,
                    mask=slice_mask,
                )
            else:
                box = self._mask_to_box(slice_mask)
                if box is None:
                    continue
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(slice_idx),
                    obj_id=1,
                    box=box,
                )
            valid_prompt = True
        return valid_prompt

    def _propagate_single_direction(self, frame_sequence, selected_slices, prompt_mask, reverse=False):
        height, width = prompt_mask.shape[0], prompt_mask.shape[1]
        depth = prompt_mask.shape[2]
        state = self.predictor.init_state(
            images=frame_sequence,
            video_height=height,
            video_width=width,
        )
        if not self._add_prompts(state, selected_slices, prompt_mask):
            self.predictor.reset_state(state)
            return None

        class_scores = prompt_mask.new_zeros((height, width, depth), dtype=torch.float32)
        start_frame_idx = max(selected_slices) if reverse else min(selected_slices)
        for frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
            state,
            start_frame_idx=int(start_frame_idx),
            max_frame_num_to_track=depth,
            reverse=reverse,
        ):
            class_scores[:, :, int(frame_idx)] = torch.sigmoid(out_mask_logits[0, 0]).float()

        self.predictor.reset_state(state)
        return class_scores

    @torch.inference_mode()
    def forward(self, images, prompt_probs):
        if images.ndim != 5 or images.size(1) != 1:
            raise ValueError(f"MedSAM2VolumeTeacher expects images in [B, 1, W, H, D], but got {tuple(images.shape)}")
        if prompt_probs.ndim != 5 or prompt_probs.size(1) != self.num_classes:
            raise ValueError(
                f"Prompt probabilities must be [B, {self.num_classes}, W, H, D], but got {tuple(prompt_probs.shape)}"
            )

        batch_size, _, width, height, depth = images.shape
        fg_probs = images.new_zeros((batch_size, self.num_classes - 1, width, height, depth), dtype=torch.float32)
        coverage = images.new_zeros((batch_size, 1, width, height, depth), dtype=torch.float32)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with autocast_context:
            for batch_idx in range(batch_size):
                volume = images[batch_idx, 0].detach()
                teacher_probs = prompt_probs[batch_idx].detach()
                frame_sequence = self._volume_to_rgb_sequence(volume)
                frame_sequence = self._preprocess_sequence(frame_sequence).to(self.device)

                prompt_labels = torch.argmax(teacher_probs, dim=0)
                active_classes = self._select_classes(teacher_probs)
                for class_id in active_classes:
                    class_mask = prompt_labels == class_id
                    selected_slices, prompt_mask = self._pick_key_slices(teacher_probs[class_id], class_mask)
                    if not selected_slices:
                        continue

                    scores_forward = self._propagate_single_direction(
                        frame_sequence, selected_slices, prompt_mask, reverse=False
                    )
                    scores_reverse = self._propagate_single_direction(
                        frame_sequence, selected_slices, prompt_mask, reverse=True
                    )
                    if scores_forward is None and scores_reverse is None:
                        continue

                    class_scores = None
                    if scores_forward is not None:
                        class_scores = scores_forward
                    if scores_reverse is not None:
                        class_scores = scores_reverse if class_scores is None else torch.maximum(class_scores, scores_reverse)

                    fg_probs[batch_idx, class_id - 1] = class_scores.to(images.device)
                    coverage[batch_idx, 0] = torch.maximum(
                        coverage[batch_idx, 0], (class_scores > 0.5).float().to(images.device)
                    )

        bg_prob = torch.clamp(1.0 - fg_probs.max(dim=1, keepdim=True).values, min=1e-4)
        all_probs = torch.cat([bg_prob, fg_probs], dim=1)
        all_probs = all_probs / all_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        labels = torch.argmax(all_probs, dim=1)

        return {
            "probs": all_probs,
            "labels": labels,
            "coverage": coverage,
        }


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)


def create_model(n_classes=14, cube_size=32, patchsize=96, ema=False):
    # Network definition
    net = VNet_Magic(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patchsize)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def blend_teacher_probs(ema_probs, medsam2_probs=None, coverage=None, alpha=0.0):
    if medsam2_probs is None or coverage is None or alpha <= 0:
        return ema_probs, 0.0

    coverage = coverage.float()
    blended = ema_probs * (1.0 - coverage) + ((1.0 - alpha) * ema_probs + alpha * medsam2_probs) * coverage
    blended = blended / blended.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return blended, float(coverage.mean().item())


def masked_ce_loss(logits, target, mask):
    if mask is None:
        return F.cross_entropy(logits, target.long())
    mask = mask.float()
    valid = mask.sum()
    if valid.item() <= 0:
        return logits.sum() * 0.0
    loss_map = F.cross_entropy(logits, target.long(), reduction='none')
    return (loss_map * mask).sum() / valid


def masked_kl_loss(logits, target_probs, mask, temperature=1.0):
    if mask is None:
        return F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            target_probs,
            reduction='batchmean',
        ) * (temperature ** 2)
    mask = mask.float()
    valid = mask.sum()
    if valid.item() <= 0:
        return logits.sum() * 0.0
    loss_map = F.kl_div(
        F.log_softmax(logits / temperature, dim=1),
        target_probs.clamp_min(1e-6),
        reduction='none',
    ).sum(dim=1)
    return (loss_map * mask).sum() / valid * (temperature ** 2)


def infer_full_volume_probs(model, image_np, patch_size, stride_xy, stride_z, num_classes):
    was_training = model.training
    model.eval()
    try:
        _, score_map = test_amos.test_single_case_fast(
            model,
            image_np,
            stride_xy=stride_xy,
            stride_z=stride_z,
            patch_size=patch_size,
            num_classes=num_classes,
        )
    finally:
        if was_training:
            model.train()
    return torch.from_numpy(score_map).unsqueeze(0).float().cuda()


def crop_with_pad(volume_tensor, crop_bbox, pad_width):
    if volume_tensor.dim() == 4:
        volume_tensor = volume_tensor.unsqueeze(0)
    pad_width = [int(x) for x in pad_width]
    crop_bbox = [int(x) for x in crop_bbox]
    if sum(pad_width) > 0:
        volume_tensor = F.pad(
            volume_tensor,
            (pad_width[4], pad_width[5], pad_width[2], pad_width[3], pad_width[0], pad_width[1]),
        )
    w1, w2, h1, h2, d1, d2 = crop_bbox
    return volume_tensor[:, :, w1:w2, h1:h2, d1:d2]


def get_full_volume_teacher_outputs(case_name, case_index, dataset, ema_model, medsam2_teacher, teacher_cache, iter_num):
    refresh_due = False
    if case_name in teacher_cache:
        cache_item = teacher_cache[case_name]
        refresh_due = (iter_num - cache_item['iter']) > args.medsam2_cache_ttl
        if not refresh_due:
            teacher_cache.move_to_end(case_name)
            return (
                cache_item['teacher_probs'].cuda().float(),
                cache_item['medsam2_probs'].cuda().float(),
                cache_item['coverage'].cuda().float(),
            )

    full_image = dataset.images_u[case_index]
    full_image_tensor = torch.from_numpy(full_image).unsqueeze(0).unsqueeze(0).float().cuda()
    ema_full_probs = infer_full_volume_probs(
        ema_model,
        full_image,
        patch_size=patch_size,
        stride_xy=args.medsam2_stride_xy,
        stride_z=args.medsam2_stride_z,
        num_classes=num_classes,
    )

    medsam2_full_probs = torch.zeros_like(ema_full_probs)
    medsam2_coverage = torch.zeros((1, 1) + tuple(full_image.shape), dtype=torch.float32, device=ema_full_probs.device)
    if medsam2_teacher is not None:
        medsam2_out = medsam2_teacher(full_image_tensor, ema_full_probs.detach())
        medsam2_full_probs = medsam2_out['probs']
        medsam2_coverage = medsam2_out['coverage']

    blended_teacher_probs, _ = blend_teacher_probs(
        ema_full_probs,
        medsam2_full_probs,
        medsam2_coverage,
        alpha=args.medsam2_blend_alpha,
    )

    teacher_cache[case_name] = {
        'iter': iter_num,
        'teacher_probs': blended_teacher_probs.detach().cpu().half(),
        'medsam2_probs': medsam2_full_probs.detach().cpu().half(),
        'coverage': medsam2_coverage.detach().cpu().half(),
    }
    teacher_cache.move_to_end(case_name)
    while len(teacher_cache) > max(1, int(args.medsam2_cache_size)):
        teacher_cache.popitem(last=False)

    return blended_teacher_probs, medsam2_full_probs, medsam2_coverage


def build_unlabeled_teacher_patches(sampled_batch, unlabeled_slice, dataset, ema_model, medsam2_teacher, teacher_cache, iter_num):
    case_names = sampled_batch['case_name']
    case_indices = sampled_batch['case_index'][unlabeled_slice].cpu().tolist()
    crop_bboxes = sampled_batch['crop_bbox'][unlabeled_slice].cpu().tolist()
    pad_widths = sampled_batch['pad_width'][unlabeled_slice].cpu().tolist()

    teacher_patch_list = []
    medsam2_patch_list = []
    coverage_patch_list = []
    for offset, case_index in enumerate(case_indices):
        batch_pos = offset + unlabeled_slice.start
        case_name = case_names[batch_pos]
        teacher_probs_full, medsam2_probs_full, coverage_full = get_full_volume_teacher_outputs(
            case_name=case_name,
            case_index=int(case_index),
            dataset=dataset,
            ema_model=ema_model,
            medsam2_teacher=medsam2_teacher,
            teacher_cache=teacher_cache,
            iter_num=iter_num,
        )
        teacher_patch_list.append(crop_with_pad(teacher_probs_full, crop_bboxes[offset], pad_widths[offset]))
        medsam2_patch_list.append(crop_with_pad(medsam2_probs_full, crop_bboxes[offset], pad_widths[offset]))
        coverage_patch_list.append(crop_with_pad(coverage_full, crop_bboxes[offset], pad_widths[offset]))

    teacher_patch_probs = torch.cat(teacher_patch_list, dim=0)
    medsam2_patch_probs = torch.cat(medsam2_patch_list, dim=0)
    coverage_patch = torch.cat(coverage_patch_list, dim=0)
    return teacher_patch_probs, medsam2_patch_probs, coverage_patch

def read_list(split):
    ids_list = np.loadtxt(
        os.path.join('./data/amos_splits/', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)

if args.labelnum == 4:
    # 2%, 4 labeld 
    labeled_list = read_list('labeled_2p')
    unlabeled_list = read_list('unlabeled_2p')
elif args.labelnum == 10:
    # 5%, 10 labeld 
    labeled_list = read_list('labeled_5p')
    unlabeled_list = read_list('unlabeled_5p')
else:
    print('Error labelnum!')
    os.exit()
eval_list = read_list('eval')
test_list = read_list('test')

# if args.GA:
    # snapshot_path = args.save_path + "/{}_{}_GA_{}labeled".format(args.dataset_name, args.exp, args.labelnum)
snapshot_path = args.save_path + "/{}_{}_{}labeled".format(args.dataset_name, args.exp, args.labelnum)
# else:
#     snapshot_path = args.save_path + "/{}_{}_{}labeled".format(args.dataset_name, args.exp, args.labelnum)
print(snapshot_path)

num_classes = 16
class_momentum = 0.999
patch_size = (96, 96, 96)

train_data_path = args.root_path
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs
cube_size = args.cube_size


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def config_log(snapshot_path_tmp, typename):

    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(labeled_list, unlabeled_list, eval_list, fold_id=1):
    
    snapshot_path_tmp = snapshot_path
    train_list = labeled_list + unlabeled_list    
    handler, sh = config_log(snapshot_path_tmp, 'fold' + str(fold_id))
    logging.info(str(args))

    model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0])
    ema_model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0], ema=True)
    medsam2_teacher = None
    if bool(args.use_medsam2):
        medsam2_teacher = MedSAM2VolumeTeacher(args)
        medsam2_teacher.eval()

    db_train = AMOS_fast(labeled_list, unlabeled_list,
                    base_dir=train_data_path,
                    transform=transforms.Compose([
                        RandomCrop(patch_size),
                        ToTensor(),
                    ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(len(unlabeled_list)*2))
    unlabeled_idxs = list(range(len(unlabeled_list)*2, len(unlabeled_list)*4))
    print('labeled_list: ', len(labeled_list))
    print('unlabeled_list: ', len(unlabeled_list))
    print('train_list: ', len(train_list))
    print('eval_list: ', len(eval_list))
    print('test_list: ', len(test_list))
    print(min(labeled_idxs), max(labeled_idxs), min(unlabeled_idxs), max(unlabeled_idxs))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr)

    writer = SummaryWriter(snapshot_path_tmp)
    logging.info("{} itertations per epoch".format(len(trainloader)))




    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)      


    ema_model.train()

    iter_num = 0
    best_dice_avg = 0
    metric_all_cases = None
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    loc_list = None
    dist_logger = cube_utils.OrganClassLogger(num_classes=num_classes)
    medsam2_coverage = 0.0
    teacher_cache = OrderedDict()

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            labeled_volume_batch = volume_batch[:labeled_bs]

            model.train()
            outputs = model(volume_batch)[0] # Original Model Outputs

            # Cross-image Partition-and-Recovery
            bs, c, w, h, d = volume_batch.shape
            nb_cubes = h // cube_size
            cube_part_ind, cube_rec_ind = cube_utils.get_part_and_rec_ind(volume_shape=volume_batch.shape,
                                                                          nb_cubes=nb_cubes,
                                                                          nb_chnls=16)
            img_cross_mix = volume_batch.view(bs, c, w, h, d)
            img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
            img_cross_mix = img_cross_mix.view(bs, c, w, h, d)
            

            outputs_mix, embedding = model(img_cross_mix)
            c_ = embedding.shape[1]
            pred_rec = torch.gather(embedding, dim=0, index=cube_rec_ind)
            pred_rec = pred_rec.view(bs, c_, w, h, d)
            outputs_unmix = model.forward_prediction_head(pred_rec)
            

            # Get pseudo-label from teacher model
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)[0]
                teacher_probs = F.softmax(ema_output, dim=1)
                medsam2_patch_probs = None
                medsam2_patch_coverage = None

                medsam2_coverage = 0.0
                if (
                    medsam2_teacher is not None
                    and iter_num >= args.medsam2_warmup
                ):
                    if bool(args.medsam2_full_volume):
                        teacher_probs, medsam2_patch_probs, medsam2_patch_coverage = build_unlabeled_teacher_patches(
                            sampled_batch=sampled_batch,
                            unlabeled_slice=slice(labeled_bs, None),
                            dataset=trainloader.dataset,
                            ema_model=ema_model,
                            medsam2_teacher=medsam2_teacher,
                            teacher_cache=teacher_cache,
                            iter_num=iter_num,
                        )
                        medsam2_coverage = float(medsam2_patch_coverage.mean().item())
                    elif iter_num % max(1, args.medsam2_interval) == 0:
                        medsam2_out = medsam2_teacher(unlabeled_volume_batch, teacher_probs.detach())
                        teacher_probs, medsam2_coverage = blend_teacher_probs(
                            teacher_probs,
                            medsam2_out["probs"],
                            medsam2_out["coverage"],
                            alpha=args.medsam2_blend_alpha,
                        )
                        medsam2_patch_probs = medsam2_out["probs"]
                        medsam2_patch_coverage = medsam2_out["coverage"]

                pred_value_teacher, pred_class_teacher = torch.max(teacher_probs, dim=1)

            # nt = 3, ts = 32
            # loc_list: 27 x [1, 1] (x + Wy + WHz)
            if iter_num == 0:
                loc_list = cube_utils.get_loc_mask(volume_batch, cube_size)

            # calculate some losses
            loss_seg = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            outputs_soft = F.softmax(outputs, dim=1)
            outputs_unmix_soft = F.softmax(outputs_unmix, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs])
            loss_unmix_dice = dice_loss(outputs_unmix_soft[:labeled_bs], label_batch[:labeled_bs])
            
            
            supervised_loss = (loss_seg + loss_seg_dice + loss_unmix_dice)
            count_ss = 3

            # Magic-cube Location Reasoning
            # patch_list: N=27 x [4, 1, 1, 32, 32, 32] (bs, pn, c, w, h, d)
            patch_list = cube_losses.get_patch_list(volume_batch, cube_size=cube_size)
            # idx = 27
            idx = torch.randperm(len(patch_list)).cuda()  # random
            # cube location loss
            loc_loss = 0
            feat_list = None
            if loc_list is not None:
                loc_loss, feat_list = cube_losses.cube_location_loss(model, loc_list, patch_list, idx, labeled_bs, cube_size=cube_size)

            consistency_loss = 0
            count_consist = 1

            # Within-image Partition-and-Recovery
            if feat_list is not None:
                embed_list = []
                for i in range(bs):
                    pred_tmp, embed_tmp = model.forward_decoder(feat_list[i])
                    embed_list.append(embed_tmp.unsqueeze(0))

                embed_all = torch.cat(embed_list, dim=0)
                embed_all_unmix = cube_losses.unmix_tensor(embed_all, labeled_volume_batch.shape)
                pred_all_unmix = model.forward_prediction_head(embed_all_unmix) 
                unmix_pred_soft = F.softmax(pred_all_unmix, dim=1)
                loss_lab_local_dice = dice_loss(unmix_pred_soft[:labeled_bs], label_batch[:labeled_bs])
                supervised_loss += loss_lab_local_dice
                count_ss += 1
                

            # Cube-wise Pseudo-label Blending
            pred_class_mix = None
            with torch.no_grad():
                # To store some class pixels at the beginning of training to calculate the organ-class dist
                if iter_num > 100 and feat_list is not None:
                    # Get organ-class distribution
                    current_organ_dist = dist_logger.get_class_dist().cuda()  # (1, C)
                    # Normalize
                    current_organ_dist = current_organ_dist ** (1. / args.T_dist)
                    current_organ_dist = current_organ_dist / current_organ_dist.sum()
                    current_organ_dist = current_organ_dist / current_organ_dist.max()


                    weight_map = current_organ_dist[pred_class_teacher].unsqueeze(1).repeat(1, num_classes, 1, 1, 1)

                    unmix_pl = cube_losses.get_mix_pl(model, feat_list, volume_batch.shape, bs - labeled_bs)
                    unmix_pl_soft = F.softmax(unmix_pl, dim=1)
                    unlab_pl_mix_soft = (1. - weight_map) * teacher_probs + weight_map * unmix_pl_soft
                    unlab_pl_mix_soft = unlab_pl_mix_soft / unlab_pl_mix_soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    _, pred_class_mix = torch.max(unlab_pl_mix_soft, dim=1)

                    # pr_class: 2x96**3, 1
                    conf, pr_class = torch.max(unlab_pl_mix_soft.detach(), dim=1)
                    dist_logger.append_class_list(pr_class.view(-1, 1))

                elif feat_list is not None:
                    conf, pr_class = torch.max(teacher_probs.detach(), dim=1)
                    dist_logger.append_class_list(pr_class.view(-1, 1))


            if iter_num % 20 == 0 and len(dist_logger.class_total_pixel_store):
                dist_logger.update_class_dist()

            consistency_weight = get_current_consistency_weight(iter_num // 350)
            # debiase the pseudo-label: blend ema and unmixed_within pseudo-label
            if pred_class_mix is None:
                consistency_loss_unmix = dice_loss(outputs_unmix_soft[labeled_bs:], pred_class_teacher)
            else:
                consistency_loss_unmix = dice_loss(outputs_unmix_soft[labeled_bs:], pred_class_mix)

            consistency_loss += consistency_loss_unmix
            medsam2_hard_loss = torch.tensor(0.0, device=volume_batch.device)
            medsam2_soft_loss = torch.tensor(0.0, device=volume_batch.device)
            if medsam2_patch_probs is not None and medsam2_patch_coverage is not None:
                medsam2_mask = medsam2_patch_coverage[:, 0]
                medsam2_labels = torch.argmax(medsam2_patch_probs, dim=1)
                medsam2_hard_loss = (
                    masked_ce_loss(outputs[labeled_bs:], medsam2_labels, medsam2_mask)
                    + masked_ce_loss(outputs_unmix[labeled_bs:], medsam2_labels, medsam2_mask)
                ) / 2.0
                medsam2_soft_loss = (
                    masked_kl_loss(
                        outputs[labeled_bs:],
                        medsam2_patch_probs,
                        medsam2_mask,
                        temperature=args.medsam2_distill_temperature,
                    )
                    + masked_kl_loss(
                        outputs_unmix[labeled_bs:],
                        medsam2_patch_probs,
                        medsam2_mask,
                        temperature=args.medsam2_distill_temperature,
                    )
                ) / 2.0

            supervised_loss /= count_ss
            consistency_loss /= count_consist

            # Final Loss
            loss = (
                supervised_loss
                + 0.1 * loc_loss
                + consistency_weight * consistency_loss
                + args.medsam2_hard_weight * medsam2_hard_loss
                + args.medsam2_soft_weight * medsam2_soft_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            if iter_num % 100 == 0:
                logging.info('Fold {}, iteration {}: loss: {:.4f}, '
                             'cons_dist: {:.4f}, loss_weight: {:4f}, '
                             'loss_loc: {:.4f}, medsam2_cov: {:.4f}, '
                             'medsam2_hard: {:.4f}, medsam2_soft: {:.4f}'.format(fold_id, iter_num,
                                                       loss,
                                                       consistency_loss,
                                                       consistency_weight,
                                                       0.1 * loc_loss,
                                                       medsam2_coverage,
                                                       medsam2_hard_loss,
                                                       medsam2_soft_loss))
            lr_ = base_lr * (1.0 - iter_num / 70000) ** 0.9 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            if iter_num % 1000 == 0:

                model.eval()
                dice_all, std_all, metric_all_cases = test_amos.validation_all_case_fast(model,
                                                                                    num_classes=num_classes,
                                                                                    base_dir=train_data_path,
                                                                                    image_list=eval_list,
                                                                                    patch_size=patch_size,
                                                                                    stride_xy=90,
                                                                                    stride_z=80)
                dice_avg = dice_all.mean()

                logging.info('iteration {}, '
                             'average DSC: {:.4f}, '
                             'spleen: {:.4f}, '
                             'r.kidney: {:.4f}, '
                             'l.kidney: {:.4f}, '
                             'gallbladder: {:.4f}, '
                             'esophagus: {:.4f}, '
                             'liver: {:.4f}, '
                             'stomach: {:.4f}, '
                             'aorta: {:.4f}, '
                             'inferior vena cava: {:.4f}'
                             'pancreas: {:.4f}, '
                             'right adrenal gland: {:.4f}, '
                             'left adrenal gland: {:.4f}, '
                             'duodenum: {:.4f}, '
                             'bladder: {:.4f}, '
                             'prostate/uterus: {:.4f}'
                             .format(iter_num,
                                     dice_avg,
                                     dice_all[0],
                                     dice_all[1],
                                     dice_all[2],
                                     dice_all[3],
                                     dice_all[4],
                                     dice_all[5],
                                     dice_all[6],
                                     dice_all[7],
                                     dice_all[8],
                                     dice_all[9],
                                     dice_all[10],
                                     dice_all[11],
                                     dice_all[12],
                                     dice_all[13],
                                     dice_all[14]))

                if dice_avg > best_dice_avg:
                    best_dice_avg = dice_avg
                    best_model_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}_best.pth'.format(str(iter_num).zfill(5), str(best_dice_avg)[:8]))
                    torch.save(model.state_dict(), best_model_path)
                    logging.info("save best model to {}".format(best_model_path))
                else:
                    save_mode_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}.pth'.format(str(iter_num).zfill(5), str(dice_avg)[:8]))
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                
                model.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
    
    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases, best_model_path


if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle(f"AMOS_{args.labelnum}")
    import stat
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.chmod(snapshot_path, stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
    if os.path.exists(snapshot_path + '/GALoss'):
        shutil.rmtree(snapshot_path + '/GALoss')
    shutil.copytree('.', snapshot_path + '/GALoss', shutil.ignore_patterns(['.git', '__pycache__']))


    metric_final, best_model_path = train(labeled_list, unlabeled_list, eval_list)
    
    save_best_path = best_model_path
    model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0])
    model.load_state_dict(torch.load(save_best_path))
    model.eval()
    _, _, metric_final = test_amos.validation_all_case(model, num_classes=num_classes, base_dir=train_data_path, image_list=test_list, patch_size=patch_size, stride_xy=32, stride_z=16)

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_save_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_save_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}, \n'
                 'spleen: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'r.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'l.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'gallbladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'esophagus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'liver: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'stomach: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'aorta: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'ivc: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'pancreas: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'right adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'Left adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'duodenum: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'bladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, \n'
                 'prostate/uterus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean(),
                         metric_mean[0][0], metric_std[0][0], metric_mean[1][0], metric_std[1][0], metric_mean[2][0], metric_std[2][0], metric_mean[3][0], metric_std[3][0],
                         metric_mean[0][1], metric_std[0][1], metric_mean[1][1], metric_std[1][1], metric_mean[2][1], metric_std[2][1], metric_mean[3][1], metric_std[3][1],
                         metric_mean[0][2], metric_std[0][2], metric_mean[1][2], metric_std[1][2], metric_mean[2][2], metric_std[2][2], metric_mean[3][2], metric_std[3][2],
                         metric_mean[0][3], metric_std[0][3], metric_mean[1][3], metric_std[1][3], metric_mean[2][3], metric_std[2][3], metric_mean[3][3], metric_std[3][3],
                         metric_mean[0][4], metric_std[0][4], metric_mean[1][4], metric_std[1][4], metric_mean[2][4], metric_std[2][4], metric_mean[3][4], metric_std[3][4],
                         metric_mean[0][5], metric_std[0][5], metric_mean[1][5], metric_std[1][5], metric_mean[2][5], metric_std[2][5], metric_mean[3][5], metric_std[3][5],
                         metric_mean[0][6], metric_std[0][6], metric_mean[1][6], metric_std[1][6], metric_mean[2][6], metric_std[2][6], metric_mean[3][6], metric_std[3][6],
                         metric_mean[0][7], metric_std[0][7], metric_mean[1][7], metric_std[1][7], metric_mean[2][7], metric_std[2][7], metric_mean[3][7], metric_std[3][7],
                         metric_mean[0][8], metric_std[0][8], metric_mean[1][8], metric_std[1][8], metric_mean[2][8], metric_std[2][8], metric_mean[3][8], metric_std[3][8],
                         metric_mean[0][9], metric_std[0][9], metric_mean[1][9], metric_std[1][9], metric_mean[2][9], metric_std[2][9], metric_mean[3][9], metric_std[3][9],
                         metric_mean[0][10], metric_std[0][10], metric_mean[1][10], metric_std[1][10], metric_mean[2][10], metric_std[2][10], metric_mean[3][10], metric_std[3][10],
                         metric_mean[0][11], metric_std[0][11], metric_mean[1][11], metric_std[1][11], metric_mean[2][11], metric_std[2][11], metric_mean[3][11], metric_std[3][11],
                         metric_mean[0][12], metric_std[0][12], metric_mean[1][12], metric_std[1][12], metric_mean[2][12], metric_std[2][12], metric_mean[3][12], metric_std[3][12],
                         metric_mean[0][13], metric_std[0][13], metric_mean[1][13], metric_std[1][13], metric_mean[2][13], metric_std[2][13], metric_mean[3][13], metric_std[3][13],
                         metric_mean[0][14], metric_std[0][14], metric_mean[1][14], metric_std[1][14], metric_mean[2][14], metric_std[2][14], metric_mean[3][14], metric_std[3][14]))

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
