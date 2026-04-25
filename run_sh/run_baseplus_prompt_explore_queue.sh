#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_baseplus_prompt_explore_queue.sh <BTCV|MMWHS|AMOS> <conda_env> [extra args...]}"
CONDA_ENV="${2:?usage: run_baseplus_prompt_explore_queue.sh <BTCV|MMWHS|AMOS> <conda_env> [extra args...]}"
shift 2
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_medical_sam2_3d_repro.py"
SAVE_DIR="/data/why/logs_SAM2SSL/medical_sam2_3d_repro"
MODEL_ID="facebook/sam2-hiera-base-plus"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "Unable to locate conda.sh under ${HOME}/miniconda3"
  exit 1
fi

run_variant() {
  local variant_name="$1"
  shift
  echo "[$(date '+%F %T')] Starting ${DATASET} ${variant_name}"
  python "${EVAL_SCRIPT}" \
    --dataset "${DATASET}" \
    --split eval \
    --preset paper_best_prompt \
    --prompt_mode 5 \
    --uniform_frame_count 10 \
    --rgb_mode repeat \
    --input_size 1024 \
    --sam2_model_id "${MODEL_ID}" \
    --save_dir "${SAVE_DIR}" \
    "$@" \
    "${EXTRA_ARGS[@]}"
  echo "[$(date '+%F %T')] Finished ${DATASET} ${variant_name}"
}

run_variant "mask_K10" \
  --prompt_refine none \
  --feedback_mode none

run_variant "mask_box_K10" \
  --prompt_refine box \
  --feedback_mode none

run_variant "mask_box_K10_feedback_mask" \
  --prompt_refine box \
  --feedback_mode mask \
  --feedback_threshold 0.5 \
  --feedback_min_area 16
