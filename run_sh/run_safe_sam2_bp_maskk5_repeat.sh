#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-SSL38}"
conda activate "${CONDA_ENV}"

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
  echo "Usage: $0 {BTCV|MMWHS|AMOS}"
  exit 1
fi

EXP_NAME="${EXP_NAME:-SAM2SSL_safe_bp_maskK5_repeat_20260424}"

COMMON_ARGS=(
  --exp "${EXP_NAME}"
  --labelnum 10
  --medsam2_root /home/why/SSL4MIS_work3/MedSAM2
  --medsam2_cfg sam2_hiera_b+.yaml
  --medsam2_checkpoint /home/why/SSL4MIS_work3/MedSAM2/checkpoints/sam2_hiera_base_plus.pt
  --medsam2_prompt_type mask
  --medsam2_num_condition_frames 5
  --medsam2_rgb_mode repeat
)

case "${DATASET}" in
  BTCV)
    python work3_BTCV_Baseline_safeSAM2.py \
      "${COMMON_ARGS[@]}" \
      --max_iteration 30000 \
      --labeled_bs 1 \
      --batch_size 2 \
      --base_lr 0.01 \
      --resume_path /data/why/logs/GA_BTCV_GA_10_base0326/iter_19500_dice_0.620036_best.pth \
      --resume_iter 19500 \
      --resume_best_dice 0.620036
    ;;
  MMWHS)
    python work3_MMWHS_Baseline_safeSAM2.py \
      "${COMMON_ARGS[@]}" \
      --max_iteration 12000 \
      --labeled_bs 2 \
      --batch_size 4 \
      --base_lr 0.001 \
      --medsam2_full_volume 0 \
      --medsam2_blend_alpha 0.18 \
      --medsam2_prompt_thresh 0.75 \
      --medsam2_teacher_prob_thresh 0.75 \
      --medsam2_max_classes 4 \
      --medsam2_main_teacher_blend 1 \
      --resume_path /data/why/logs/GA_MMWHS_GA_10_fromscratch_20260420/iter_05500_dice_0.802967_best.pth \
      --resume_iter 0 \
      --resume_best_dice 0.802967
    ;;
  AMOS)
    python work3_AMOS_Baseline_safeSAM2.py \
      "${COMMON_ARGS[@]}" \
      --base_lr 0.001 \
      --resume_path /data/why/logs_SAM2SSL/AMOS_GA_base_fromscratch_20260421_double3090_10labeled/iter_15000_dice_0.480311_best.pth \
      --resume_iter 0 \
      --resume_best_dice 0.480311
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}"
    exit 1
    ;;
esac
