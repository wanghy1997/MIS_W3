#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-medsam2}"
conda activate "${CONDA_ENV}"

EXP_NAME="${EXP_NAME:-MMWHS_SAM2SSL_bp_studentPromptRefine_20260425}"
CKPT_PATH="${CKPT_PATH:-/data/why/logs/GA_MMWHS_GA_10_fromscratch_20260420/iter_05500_dice_0.802967_best.pth}"

python work3_MMWHS_Baseline_safeSAM2.py \
  --exp "${EXP_NAME}" \
  --labelnum 10 \
  --eval_only 1 \
  --eval_checkpoint "${CKPT_PATH}" \
  --resume_path "${CKPT_PATH}" \
  --resume_iter 0 \
  --resume_best_dice 0.802967 \
  --medsam2_root /home/why/SSL4MIS_work3/MedSAM2 \
  --medsam2_cfg sam2_hiera_b+.yaml \
  --medsam2_checkpoint /home/why/SSL4MIS_work3/MedSAM2/checkpoints/sam2_hiera_base_plus.pt \
  --medsam2_prompt_type mask \
  --medsam2_num_condition_frames 5 \
  --medsam2_rgb_mode repeat \
  --medsam2_enable_train 0 \
  --medsam2_test_refine 1 \
  --medsam2_test_blend_with_base 2 \
  --medsam2_test_conf_thresh 0.00 \
  --medsam2_test_min_coverage 0.002 \
  --medsam2_test_max_coverage 1.0 \
  --medsam2_prompt_thresh 0.5 \
  --medsam2_teacher_prob_thresh 0.5 \
  --medsam2_min_voxels 128 \
  --medsam2_min_slice_area 32 \
  --medsam2_max_classes 7 \
  --base_lr 0.001 \
  --labeled_bs 2 \
  --batch_size 4
