import os
import re
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
from utils_ori import ramps, cube_losses, cube_utils
from utils import test_amos, test_util
from dataloaders.dataset_SAM2 import *

from networks.magicnet import VNet_Magic
from loss_amos import GADice, GACE


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MMWHS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/why/Datasets/MMWHS/', help='Name of Dataset')
parser.add_argument('--save_path', type=str, default='/data/why/logs_SAM2SSL/', help='path to save')
parser.add_argument('--exp', type=str, default='SAM2SSL_reciprocal', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=35000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=90, help='total samples of the dataset')
parser.add_argument('--max_train_samples', type=int, default=66, help='maximum samples to train')
parser.add_argument('--max_test_samples', type=int, default=22, help='maximum samples to test')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=10, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
parser.add_argument('--use_medsam2', type=int, default=1, help='whether to use MedSAM2 as the second teacher')
parser.add_argument('--medsam2_root', type=str, default=os.path.join(SCRIPT_DIR, 'MedSAM2'), help='MedSAM2 repo root')
parser.add_argument('--medsam2_cfg', type=str, default='sam2_hiera_b+.yaml', help='MedSAM2 config path')
parser.add_argument('--medsam2_checkpoint', type=str, default='./checkpoints/sam2_hiera_base_plus.pt', help='MedSAM2 checkpoint path')
parser.add_argument('--medsam2_warmup', type=int, default=3000, help='start using MedSAM2 after this many iterations')
parser.add_argument('--medsam2_interval', type=int, default=50, help='run MedSAM2 every N iterations')
parser.add_argument('--medsam2_blend_alpha', type=float, default=0.18, help='blend weight of MedSAM2 over EMA teacher')
parser.add_argument('--medsam2_num_classes', type=int, default=8, help='number of semantic classes for MedSAM2 teacher')
parser.add_argument('--medsam2_prompt_type', type=str, default='mask', choices=['box', 'mask'], help='prompt type for MedSAM2')
parser.add_argument('--medsam2_prompt_thresh', type=float, default=0.75, help='confidence threshold used to build MedSAM2 prompts')
parser.add_argument('--medsam2_teacher_prob_thresh', type=float, default=0.75, help='confidence threshold used to rank MedSAM2 teacher voxels')
parser.add_argument('--medsam2_min_voxels', type=int, default=1000, help='minimum 3D voxels for a class to be tracked by MedSAM2')
parser.add_argument('--medsam2_min_slice_area', type=int, default=100, help='minimum 2D area for a conditioning slice')
parser.add_argument('--medsam2_box_expand', type=int, default=4, help='expand MedSAM2 prompt boxes by this many pixels')
parser.add_argument('--medsam2_max_classes', type=int, default=7, help='maximum foreground classes to track per unlabeled patch')
parser.add_argument('--medsam2_num_condition_frames', type=int, default=10, help='number of key slices used as MedSAM2 conditioning frames')
parser.add_argument('--medsam2_full_volume', type=int, default=0, help='use full-volume teacher inference before cropping back to the current patch')
parser.add_argument('--medsam2_stride_xy', type=int, default=64, help='sliding-window stride in x/y for full-volume EMA inference')
parser.add_argument('--medsam2_stride_z', type=int, default=64, help='sliding-window stride in z for full-volume EMA inference')
parser.add_argument('--medsam2_cache_size', type=int, default=32, help='number of full-volume teacher cases kept in memory')
parser.add_argument('--medsam2_cache_ttl', type=int, default=100000, help='refresh cached full-volume teacher outputs after this many iterations')
parser.add_argument('--medsam2_hard_weight', type=float, default=0.05, help='weight for hard MedSAM2 supervision')
parser.add_argument('--medsam2_soft_weight', type=float, default=0.02, help='weight for soft MedSAM2 distillation')
parser.add_argument('--medsam2_distill_temperature', type=float, default=1.0, help='temperature for MedSAM2 KL distillation')
parser.add_argument('--medsam2_main_teacher_blend', type=int, default=0, help='blend trusted SAM2 voxels into the main EMA teacher')
parser.add_argument('--medsam2_safe_ema_thresh', type=float, default=0.75, help='minimum EMA confidence for trusted SAM2 supervision')
parser.add_argument('--medsam2_safe_sam2_thresh', type=float, default=0.75, help='minimum SAM2 confidence for trusted supervision')
parser.add_argument('--medsam2_safe_min_coverage', type=float, default=0.002, help='minimum valid SAM2 coverage ratio per patch')
parser.add_argument('--medsam2_safe_max_coverage', type=float, default=0.12, help='maximum valid SAM2 coverage ratio per patch')
parser.add_argument('--medsam2_safe_min_trusted_voxels', type=int, default=512, help='minimum number of trusted SAM2 voxels before applying auxiliary losses')
parser.add_argument('--medsam2_foreground_only', type=int, default=1, help='trust only foreground voxels when distilling from SAM2')
parser.add_argument('--medsam2_rgb_mode', type=str, default='repeat', choices=['repeat', 'neighbor'], help='RGB conversion mode for SAM2 frame sequences')
parser.add_argument('--medsam2_prompt_use_unmix', type=int, default=1, help='blend within-image pseudo labels into the small-model prompt sent to SAM2')
parser.add_argument('--medsam2_prompt_unmix_alpha', type=float, default=0.35, help='how much unmix pseudo labels influence SAM2 prompting')
parser.add_argument('--medsam2_prompt_min_conf', type=float, default=0.70, help='minimum joint confidence before unmix pseudo labels are injected into SAM2 prompts')
parser.add_argument('--resume_path', type=str, default='', help='checkpoint path for resuming training')
parser.add_argument('--resume_iter', type=int, default=-1, help='override iteration number when resuming')
parser.add_argument('--resume_best_dice', type=float, default=-1.0, help='override best dice when resuming')
parser.add_argument('--disable_medsam2_hole_filling', type=int, default=1, help='disable SAM2 hole filling to avoid custom op warnings')
parser.add_argument('--eval_only', type=int, default=0, help='only run dual eval/test on the given checkpoint and exit')
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
        if bool(getattr(args, "disable_medsam2_hole_filling", 0)) and hasattr(self.predictor, "fill_hole_area"):
            self.predictor.fill_hole_area = 0
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
            if str(getattr(self.args, "medsam2_rgb_mode", "repeat")).lower() == "neighbor":
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
            anchor_positions = torch.linspace(
                0,
                valid_slices.numel() - 1,
                steps=self.num_condition_frames,
                device=valid_slices.device,
            ).round().long()
            selected = [int(valid_slices[idx].item()) for idx in anchor_positions]

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


def create_model(n_classes=8, cube_size=32, patchsize=96, ema=False):
    # Network definition
    net = VNet_Magic(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patchsize)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def build_trusted_sam2_mask(ema_probs, medsam2_probs=None, coverage=None):
    if medsam2_probs is None or coverage is None:
        return None, 0.0, 0.0, 0.0

    coverage_mask = coverage[:, 0] > 0.5
    ema_conf, ema_labels = torch.max(ema_probs, dim=1)
    sam2_conf, sam2_labels = torch.max(medsam2_probs, dim=1)

    agreement_mask = ema_labels.eq(sam2_labels)
    if bool(args.medsam2_foreground_only):
        agreement_mask = agreement_mask & (sam2_labels > 0)

    confidence_mask = (ema_conf >= args.medsam2_safe_ema_thresh) & (sam2_conf >= args.medsam2_safe_sam2_thresh)
    coverage_ratio = coverage_mask.float().flatten(1).mean(dim=1)
    size_gate = ((coverage_ratio >= args.medsam2_safe_min_coverage) & (coverage_ratio <= args.medsam2_safe_max_coverage))
    size_gate = size_gate.view(-1, 1, 1, 1)

    trusted_mask = coverage_mask & agreement_mask & confidence_mask & size_gate
    agreement_ratio = float((coverage_mask & agreement_mask).float().mean().item())
    trusted_ratio = float(trusted_mask.float().mean().item())
    coverage_mean = float(coverage_mask.float().mean().item())
    return trusted_mask.float(), coverage_mean, trusted_ratio, agreement_ratio


def safe_blend_teacher_probs(ema_probs, medsam2_probs=None, trusted_mask=None, alpha=0.0):
    if medsam2_probs is None or trusted_mask is None or alpha <= 0:
        return ema_probs, 0.0

    trusted_mask = trusted_mask.float().unsqueeze(1)
    blended = ema_probs * (1.0 - trusted_mask) + ((1.0 - alpha) * ema_probs + alpha * medsam2_probs) * trusted_mask
    blended = blended / blended.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return blended, float(trusted_mask[:, 0].mean().item())


def build_small_teacher_probs(ema_probs, unmix_probs=None):
    if unmix_probs is None or not bool(args.medsam2_prompt_use_unmix):
        return ema_probs

    alpha = float(args.medsam2_prompt_unmix_alpha)
    if alpha <= 0:
        return ema_probs

    ema_conf, ema_labels = torch.max(ema_probs, dim=1)
    unmix_conf, unmix_labels = torch.max(unmix_probs, dim=1)
    agree_mask = ema_labels.eq(unmix_labels)
    confidence_mask = (
        (ema_conf >= float(args.medsam2_prompt_min_conf))
        & (unmix_conf >= float(args.medsam2_prompt_min_conf))
    )
    merge_mask = (agree_mask & confidence_mask).float().unsqueeze(1)
    fused_probs = ((1.0 - alpha) * ema_probs + alpha * unmix_probs)
    fused_probs = fused_probs / fused_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    mixed = ema_probs * (1.0 - merge_mask) + fused_probs * merge_mask
    mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return mixed


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


def infer_resume_iter_from_path(path):
    if not path:
        return 0
    match = re.search(r'iter_(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else 0


def infer_resume_dice_from_path(path):
    if not path:
        return 0.0
    match = re.search(r'dice_([0-9.]+)', os.path.basename(path))
    if not match:
        return 0.0
    try:
        return float(match.group(1).rstrip('.'))
    except ValueError:
        return 0.0


def load_resume_state(model, ema_model, optimizer, resume_path):
    checkpoint = torch.load(resume_path, map_location='cpu')
    resume_iter = 0
    resume_best_dice = 0.0
    loaded_training_state = isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
    if loaded_training_state:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_model_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            ema_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_iter = int(checkpoint.get('iter_num', 0))
        resume_best_dice = float(checkpoint.get('best_dice_avg', 0.0))
    else:
        model.load_state_dict(checkpoint)
        ema_model.load_state_dict(checkpoint)

    if args.resume_iter >= 0:
        resume_iter = args.resume_iter
    elif loaded_training_state and resume_iter <= 0:
        resume_iter = infer_resume_iter_from_path(resume_path)
    elif not loaded_training_state:
        resume_iter = 0

    if args.resume_best_dice >= 0:
        resume_best_dice = args.resume_best_dice
    elif resume_best_dice <= 0:
        resume_best_dice = infer_resume_dice_from_path(resume_path)

    return resume_iter, resume_best_dice, loaded_training_state


def save_training_state(path, model, ema_model, optimizer, iter_num, best_dice_avg):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter_num': int(iter_num),
            'best_dice_avg': float(best_dice_avg),
        },
        path,
    )


def get_full_volume_teacher_outputs(case_name, case_index, dataset, ema_model, medsam2_teacher, teacher_cache, iter_num):
    refresh_due = False
    if case_name in teacher_cache:
        cache_item = teacher_cache[case_name]
        refresh_due = (iter_num - cache_item['iter']) > args.medsam2_cache_ttl
        if not refresh_due:
            teacher_cache.move_to_end(case_name)
            return (
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

    teacher_cache[case_name] = {
        'iter': iter_num,
        'medsam2_probs': medsam2_full_probs.detach().cpu().half(),
        'coverage': medsam2_coverage.detach().cpu().half(),
    }
    teacher_cache.move_to_end(case_name)
    while len(teacher_cache) > max(1, int(args.medsam2_cache_size)):
        teacher_cache.popitem(last=False)

    return medsam2_full_probs, medsam2_coverage


def build_unlabeled_sam2_patches(sampled_batch, unlabeled_slice, dataset, ema_model, medsam2_teacher, teacher_cache, iter_num):
    case_names = sampled_batch['case_name']
    case_indices = sampled_batch['case_index'][unlabeled_slice].cpu().tolist()
    crop_bboxes = sampled_batch['crop_bbox'][unlabeled_slice].cpu().tolist()
    pad_widths = sampled_batch['pad_width'][unlabeled_slice].cpu().tolist()

    medsam2_patch_list = []
    coverage_patch_list = []
    for offset, case_index in enumerate(case_indices):
        batch_pos = offset + unlabeled_slice.start
        case_name = case_names[batch_pos]
        medsam2_probs_full, coverage_full = get_full_volume_teacher_outputs(
            case_name=case_name,
            case_index=int(case_index),
            dataset=dataset,
            ema_model=ema_model,
            medsam2_teacher=medsam2_teacher,
            teacher_cache=teacher_cache,
            iter_num=iter_num,
        )
        medsam2_patch_list.append(crop_with_pad(medsam2_probs_full, crop_bboxes[offset], pad_widths[offset]))
        coverage_patch_list.append(crop_with_pad(coverage_full, crop_bboxes[offset], pad_widths[offset]))

    medsam2_patch_probs = torch.cat(medsam2_patch_list, dim=0)
    coverage_patch = torch.cat(coverage_patch_list, dim=0)
    return medsam2_patch_probs, coverage_patch


def read_list(split):
    ids_list = np.loadtxt(
        os.path.join(args.root_path, 'split_txts', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


if args.labelnum == 10:
    labeled_list = read_list('labeled_10p')
    unlabeled_list = read_list('unlabeled_10p')
elif args.labelnum == 50:
    labeled_list = read_list('labeled_50p')
    unlabeled_list = read_list('unlabeled_50p')
else:
    raise ValueError(f'Unsupported labelnum: {args.labelnum}')

eval_list = read_list('test')
test_list = read_list('test')

snapshot_path = args.save_path + "/{}_{}_{}labeled".format(args.dataset_name, args.exp, args.labelnum)
print(snapshot_path)

CLASS_NAMES = ['LVC', 'LAC', 'MYO', 'RAC', 'RVC', 'AA', 'PA']

num_classes = 8
class_momentum = 0.999
patch_size = (96, 96, 96)

train_data_path = args.root_path
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs
cube_size = args.cube_size


def log_validation_summary(iter_num, dice_avg, dice_all):
    parts = [f"iteration {iter_num}, average DSC: {dice_avg:.4f}"]
    parts.extend(f"{name}: {score:.4f}" for name, score in zip(CLASS_NAMES, dice_all))
    logging.info(', '.join(parts))


def log_metric_summary(metric_mean, metric_std):
    lines = [
        'Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}'.format(
            metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean()
        )
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(
            '{}: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'.format(
                name,
                metric_mean[0][idx], metric_std[0][idx],
                metric_mean[1][idx], metric_std[1][idx],
                metric_mean[2][idx], metric_std[2][idx],
                metric_mean[3][idx], metric_std[3][idx],
            )
        )
    logging.info('\n'.join(lines))


def _resize_label_like_eval(label_np):
    label_t = torch.FloatTensor(label_np).unsqueeze(0).unsqueeze(0)
    label_t = F.interpolate(label_t, size=(160, 160, 80), mode='nearest').int()
    return label_t.squeeze().cpu().numpy().astype(np.int8)


def evaluate_small_and_sam2_on_cases(model, medsam2_teacher, image_list, stride_xy, stride_z):
    was_training = model.training
    model.eval()
    small_total_metric = []
    sam2_total_metric = []
    try:
        for case_idx in tqdm(image_list):
            image_path = os.path.join(train_data_path, 'npy', f'ct_train_{case_idx}_image.npy')
            label_path = os.path.join(train_data_path, 'npy', f'ct_train_{case_idx}_label.npy')
            image = np.load(image_path).astype(np.float32)
            gt_mask = np.load(label_path).astype(np.int16)

            small_prediction, score_map = test_amos.test_single_case_fast(
                model,
                image,
                stride_xy,
                stride_z,
                patch_size,
                num_classes=num_classes,
            )

            full_image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            small_probs = torch.from_numpy(score_map).unsqueeze(0).float().cuda()
            sam2_out = medsam2_teacher(full_image_tensor, small_probs.detach())
            sam2_prediction = sam2_out['labels'][0].detach().cpu().numpy().astype(np.int8)

            gt_eval = _resize_label_like_eval(gt_mask)
            small_eval = _resize_label_like_eval(small_prediction)
            sam2_eval = _resize_label_like_eval(sam2_prediction)

            small_case_metric = np.zeros((4, num_classes - 1))
            sam2_case_metric = np.zeros((4, num_classes - 1))
            for class_id in range(1, num_classes):
                small_case_metric[:, class_id - 1] = test_util.cal_metric(small_eval == class_id, gt_eval == class_id)
                sam2_case_metric[:, class_id - 1] = test_util.cal_metric(sam2_eval == class_id, gt_eval == class_id)

            small_total_metric.append(np.expand_dims(small_case_metric, axis=0))
            sam2_total_metric.append(np.expand_dims(sam2_case_metric, axis=0))
    finally:
        if was_training:
            model.train()

    small_all_metric = np.concatenate(small_total_metric, axis=0)
    sam2_all_metric = np.concatenate(sam2_total_metric, axis=0)
    return {
        'small_metric_all': small_all_metric,
        'sam2_metric_all': sam2_all_metric,
        'small_dice': np.mean(small_all_metric, axis=0)[0],
        'sam2_dice': np.mean(sam2_all_metric, axis=0)[0],
        'small_std': np.std(small_all_metric, axis=0)[0],
        'sam2_std': np.std(sam2_all_metric, axis=0)[0],
    }


def log_dual_summary(tag, iter_num, dual_metrics):
    small_avg = float(np.mean(dual_metrics['small_dice']))
    sam2_avg = float(np.mean(dual_metrics['sam2_dice']))
    parts = [
        f"{tag} iteration {iter_num}",
        f"small_avg_DSC: {small_avg:.4f}",
        f"sam2_prompted_avg_DSC: {sam2_avg:.4f}",
    ]
    for name, s_score, t_score in zip(CLASS_NAMES, dual_metrics['small_dice'], dual_metrics['sam2_dice']):
        parts.append(f"{name}: small {float(s_score):.4f} / sam2 {float(t_score):.4f}")
    logging.info(', '.join(parts))




if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def config_log(snapshot_path_tmp, typename, append=False):

    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="a" if append else "w")
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
    handler, sh = config_log(snapshot_path_tmp, 'fold' + str(fold_id), append=bool(args.resume_path))
    logging.info(str(args))

    model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0])
    ema_model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0], ema=True)
    medsam2_teacher = None
    if bool(args.use_medsam2):
        medsam2_teacher = MedSAM2VolumeTeacher(args)
        medsam2_teacher.eval()

    db_train = MMWHS(train_list,
                    base_dir=train_data_path,
                    transform=transforms.Compose([
                        RandomCrop(patch_size),
                        ToTensor(),
                    ]))

    labeled_idxs = list(range(0, len(labeled_list)))
    unlabeled_idxs = list(range(len(labeled_list), len(train_list)))
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

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

    dice_loss = GADice()
    ce_loss = GACE(k=10, gama=0.5)      


    ema_model.train()

    iter_num = 0
    best_dice_avg = 0
    best_model_path = args.resume_path if args.resume_path else None
    metric_all_cases = None
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    loc_list = None
    dist_logger = cube_utils.OrganClassLogger(num_classes=num_classes)
    medsam2_coverage = 0.0
    teacher_cache = OrderedDict()
    full_volume_warning_logged = False

    if args.resume_path:
        if not os.path.exists(args.resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_path}")
        iter_num, best_dice_avg, loaded_training_state = load_resume_state(model, ema_model, optimizer, args.resume_path)
        logging.info(
            "Resumed training from %s at iter %d with best dice %.4f",
            args.resume_path,
            iter_num,
            best_dice_avg,
        )
        if not loaded_training_state:
            logging.info(
                "Resume checkpoint is a raw model state dict without optimizer/EMA state; "
                "treating this run as fine-tuning from iter 0 unless --resume_iter is explicitly set."
            )
        if iter_num == 0:
            model.eval()
            dice_all, _, _ = test_util.validation_all_case_MMWHS(
                model,
                num_classes=num_classes,
                base_dir=train_data_path,
                image_list=eval_list,
                patch_size=patch_size,
                stride_xy=90,
                stride_z=80,
            )
            logging.info(
                "Resume checkpoint validation before further training: average DSC %.4f",
                float(dice_all.mean()),
            )
            model.train()

    writer = SummaryWriter(snapshot_path_tmp, purge_step=iter_num if args.resume_path else None)
    logging.info("{} itertations per epoch".format(len(trainloader)))

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
            cube_inds = cube_utils.get_part_and_rec_ind(
                volume_shape=volume_batch.shape,
                nb_cubes=nb_cubes,
                nb_chnls=16,
            )
            if len(cube_inds) == 3:
                cube_part_ind, cube_rec_ind, _ = cube_inds
            else:
                cube_part_ind, cube_rec_ind = cube_inds
            img_cross_mix = volume_batch.view(bs, c, w, h, d)
            img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
            img_cross_mix = img_cross_mix.view(bs, c, w, h, d)
            

            outputs_mix, embedding = model(img_cross_mix)
            c_ = embedding.shape[1]
            pred_rec = torch.gather(embedding, dim=0, index=cube_rec_ind)
            pred_rec = pred_rec.view(bs, c_, w, h, d)
            outputs_unmix = model.forward_prediction_head(pred_rec)
            outputs_soft = F.softmax(outputs, dim=1)
            outputs_unmix_soft = F.softmax(outputs_unmix, dim=1)
            

            # Get pseudo-label from teacher model
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)[0]
                ema_teacher_probs = F.softmax(ema_output, dim=1)
                small_teacher_probs = build_small_teacher_probs(
                    ema_teacher_probs,
                    outputs_unmix_soft[labeled_bs:].detach() if outputs_unmix_soft.size(0) > labeled_bs else None,
                )
                teacher_probs = small_teacher_probs
                medsam2_patch_probs = None
                medsam2_patch_coverage = None
                medsam2_trusted_mask = None
                medsam2_trusted_ratio = 0.0
                medsam2_agreement_ratio = 0.0
                medsam2_blend_ratio = 0.0

                medsam2_coverage = 0.0
                if (
                    medsam2_teacher is not None
                    and iter_num >= args.medsam2_warmup
                ):
                    can_use_full_volume = (
                        bool(args.medsam2_full_volume)
                        and hasattr(trainloader.dataset, 'images_u')
                        and all(
                            key in sampled_batch
                            for key in ('case_name', 'case_index', 'crop_bbox', 'pad_width')
                        )
                    )
                    if bool(args.medsam2_full_volume) and not can_use_full_volume and not full_volume_warning_logged:
                        logging.info(
                            "MMWHS safeSAM2 is falling back to patch-level SAM2 because the baseline loader "
                            "does not provide full-volume cache fields."
                        )
                        full_volume_warning_logged = True

                    if can_use_full_volume:
                        medsam2_patch_probs, medsam2_patch_coverage = build_unlabeled_sam2_patches(
                            sampled_batch=sampled_batch,
                            unlabeled_slice=slice(labeled_bs, None),
                            dataset=trainloader.dataset,
                            ema_model=ema_model,
                            medsam2_teacher=medsam2_teacher,
                            teacher_cache=teacher_cache,
                            iter_num=iter_num,
                        )
                    elif iter_num % max(1, args.medsam2_interval) == 0:
                        medsam2_out = medsam2_teacher(unlabeled_volume_batch, small_teacher_probs.detach())
                        medsam2_patch_probs = medsam2_out["probs"]
                        medsam2_patch_coverage = medsam2_out["coverage"]

                    if medsam2_patch_probs is not None and medsam2_patch_coverage is not None:
                        medsam2_trusted_mask, medsam2_coverage, medsam2_trusted_ratio, medsam2_agreement_ratio = build_trusted_sam2_mask(
                            small_teacher_probs,
                            medsam2_patch_probs,
                            medsam2_patch_coverage,
                        )
                        if bool(args.medsam2_main_teacher_blend):
                            teacher_probs, medsam2_blend_ratio = safe_blend_teacher_probs(
                                small_teacher_probs,
                                medsam2_patch_probs,
                                medsam2_trusted_mask,
                                alpha=args.medsam2_blend_alpha,
                            )
                        else:
                            teacher_probs = small_teacher_probs

                pred_value_teacher, pred_class_teacher = torch.max(teacher_probs, dim=1)

            # nt = 3, ts = 32
            # loc_list: 27 x [1, 1] (x + Wy + WHz)
            if iter_num == 0:
                loc_list = cube_utils.get_loc_mask(volume_batch, cube_size)

            # calculate some losses
            loss_seg = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
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
                loc_loss, feat_list = cube_losses.cube_location_loss(
                    model,
                    loc_list,
                    patch_list,
                    idx,
                    labeled_bs,
                    cube_size=cube_size,
                )

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
                    unlab_pl_mix = (1. - weight_map) * ema_output + weight_map * unmix_pl
                    unlab_pl_mix_soft = F.softmax(unlab_pl_mix, dim=1)
                    _, pred_class_mix = torch.max(unlab_pl_mix_soft, dim=1)

                    # pr_class: 2x96**3, 1
                    conf, pr_class = torch.max(unlab_pl_mix_soft.detach(), dim=1)
                    dist_logger.append_class_list(pr_class.view(-1, 1))

                elif feat_list is not None:
                    conf, pr_class = torch.max(F.softmax(ema_output, dim=1).detach(), dim=1)
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
            if medsam2_patch_probs is not None and medsam2_trusted_mask is not None:
                medsam2_mask = medsam2_trusted_mask
                if medsam2_mask.sum().item() >= max(1, int(args.medsam2_safe_min_trusted_voxels)):
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
            lr_scale = max(0.0, 1.0 - float(iter_num) / float(max_iterations))
            lr_ = base_lr * (lr_scale ** 0.9)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if iter_num % 100 == 0:
                logging.info('Fold {}, iteration {}: loss: {:.4f}, '
                             'lr: {:.6f}, cons_dist: {:.4f}, loss_weight: {:4f}, '
                             'loss_loc: {:.4f}, medsam2_cov: {:.4f}, medsam2_trust: {:.4f}, '
                             'medsam2_agree: {:.4f}, medsam2_blend: {:.4f}, '
                             'medsam2_hard: {:.4f}, medsam2_soft: {:.4f}'.format(fold_id, iter_num,
                                                       loss,
                                                       lr_,
                                                       consistency_loss,
                                                       consistency_weight,
                                                       0.1 * loc_loss,
                                                       medsam2_coverage,
                                                       medsam2_trusted_ratio,
                                                       medsam2_agreement_ratio,
                                                       medsam2_blend_ratio,
                                                       medsam2_hard_loss,
                                                       medsam2_soft_loss))
            if iter_num >= 400 and iter_num % 500 == 0:

                model.eval()
                dice_all, std_all, metric_all_cases = test_util.validation_all_case_MMWHS(model,
                                                                                    num_classes=num_classes,
                                                                                    base_dir=train_data_path,
                                                                                    image_list=eval_list,
                                                                                    patch_size=patch_size,
                                                                                    stride_xy=90,
                                                                                    stride_z=80)
                dice_avg = dice_all.mean()

                log_validation_summary(iter_num, dice_avg, dice_all)
                if medsam2_teacher is not None:
                    dual_eval = evaluate_small_and_sam2_on_cases(
                        model,
                        medsam2_teacher,
                        eval_list,
                        stride_xy=90,
                        stride_z=80,
                    )
                    log_dual_summary('eval', iter_num, dual_eval)
                    np.save(
                        os.path.join(snapshot_path_tmp, f'dual_eval_iter_{str(iter_num).zfill(5)}_small.npy'),
                        dual_eval['small_metric_all'],
                    )
                    np.save(
                        os.path.join(snapshot_path_tmp, f'dual_eval_iter_{str(iter_num).zfill(5)}_sam2.npy'),
                        dual_eval['sam2_metric_all'],
                    )

                if dice_avg > best_dice_avg:
                    best_dice_avg = dice_avg
                    best_model_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}_best.pth'.format(str(iter_num).zfill(5), str(best_dice_avg)[:8]))
                    torch.save(model.state_dict(), best_model_path)
                    logging.info("save best model to {}".format(best_model_path))
                else:
                    save_mode_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}.pth'.format(str(iter_num).zfill(5), str(dice_avg)[:8]))
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                save_training_state(
                    os.path.join(snapshot_path_tmp, 'latest_training_state.pth'),
                    model,
                    ema_model,
                    optimizer,
                    iter_num,
                    best_dice_avg,
                )
                
                model.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
    
    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases, best_model_path


if __name__ == "__main__":

    import stat
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.chmod(snapshot_path, stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
    if os.path.exists(snapshot_path + '/GALoss'):
        shutil.rmtree(snapshot_path + '/GALoss')
    shutil.copytree('.', snapshot_path + '/GALoss', shutil.ignore_patterns(['.git', '__pycache__']))

    if bool(args.eval_only):
        handler, sh = config_log(snapshot_path, 'eval_only')
        logging.info(str(args))
        model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0])
        if not args.resume_path:
            raise ValueError('--eval_only requires --resume_path')
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
        model.eval()
        medsam2_teacher = None
        if bool(args.use_medsam2):
            medsam2_teacher = MedSAM2VolumeTeacher(args)
            medsam2_teacher.eval()

        eval_dice, eval_std, eval_metric = test_util.validation_all_case_MMWHS(
            model,
            num_classes=num_classes,
            base_dir=train_data_path,
            image_list=eval_list,
            patch_size=patch_size,
            stride_xy=90,
            stride_z=80,
        )
        log_validation_summary(0, float(eval_dice.mean()), eval_dice)

        test_dice, test_std, test_metric = test_util.validation_all_case_MMWHS(
            model,
            num_classes=num_classes,
            base_dir=train_data_path,
            image_list=test_list,
            patch_size=patch_size,
            stride_xy=32,
            stride_z=16,
        )
        logging.info("test small-model average DSC %.4f", float(test_dice.mean()))

        if medsam2_teacher is not None:
            dual_eval = evaluate_small_and_sam2_on_cases(
                model,
                medsam2_teacher,
                eval_list,
                stride_xy=90,
                stride_z=80,
            )
            dual_test = evaluate_small_and_sam2_on_cases(
                model,
                medsam2_teacher,
                test_list,
                stride_xy=32,
                stride_z=16,
            )
            log_dual_summary('eval', 0, dual_eval)
            log_dual_summary('test', 0, dual_test)
            np.save(os.path.join(snapshot_path, 'eval_only_dual_eval_small.npy'), dual_eval['small_metric_all'])
            np.save(os.path.join(snapshot_path, 'eval_only_dual_eval_sam2.npy'), dual_eval['sam2_metric_all'])
            np.save(os.path.join(snapshot_path, 'eval_only_dual_test_small.npy'), dual_test['small_metric_all'])
            np.save(os.path.join(snapshot_path, 'eval_only_dual_test_sam2.npy'), dual_test['sam2_metric_all'])

        logging.getLogger().removeHandler(handler)
        logging.getLogger().removeHandler(sh)
        raise SystemExit(0)


    metric_final, best_model_path = train(labeled_list, unlabeled_list, eval_list)
    
    save_best_path = best_model_path
    model = create_model(n_classes=num_classes, cube_size=cube_size, patchsize=patch_size[0])
    model.load_state_dict(torch.load(save_best_path))
    model.eval()
    _, _, metric_final = test_util.validation_all_case_MMWHS(model, num_classes=num_classes, base_dir=train_data_path, image_list=test_list, patch_size=patch_size, stride_xy=32, stride_z=16)
    dual_test = None
    if bool(args.use_medsam2):
        medsam2_teacher = MedSAM2VolumeTeacher(args)
        medsam2_teacher.eval()
        dual_test = evaluate_small_and_sam2_on_cases(
            model,
            medsam2_teacher,
            test_list,
            stride_xy=32,
            stride_z=16,
        )

    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_save_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_save_path, metric_final)
    if dual_test is not None:
        np.save(os.path.join(snapshot_path, f'dual_test_small_{args.dataset_name}_{args.exp}.npy'), dual_test['small_metric_all'])
        np.save(os.path.join(snapshot_path, f'dual_test_sam2_{args.dataset_name}_{args.exp}.npy'), dual_test['sam2_metric_all'])

    handler, sh = config_log(snapshot_path, 'total_metric')
    log_metric_summary(metric_mean, metric_std)
    if dual_test is not None:
        log_dual_summary('test', -1, dual_test)

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

"""

CUDA_VISIBLE_DEVICES=1 python work3_BTCV_Baseline.py \
  --labelnum 10 \
  --max_iteration 35000 \
  --use_medsam2 1 \
  --medsam2_num_classes 14 \
  --medsam2_warmup 3000 \
  --medsam2_interval 50 \
  --medsam2_cache_size 32 \
  --medsam2_cache_ttl 100000 \
  --medsam2_blend_alpha 0.10 \
  --medsam2_hard_weight 0.10 \
  --medsam2_soft_weight 0.05 \
  --medsam2_prompt_thresh 0.90 \
  --medsam2_teacher_prob_thresh 0.90 \
  --medsam2_min_voxels 1000 \
  --medsam2_min_slice_area 100 \
  --medsam2_max_classes 2

  
"""
