#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <DATASET> <ENV_NAME>"
  exit 1
fi

DATASET="$1"
ENV_NAME="$2"

ROOT_DIR="/home/why/SSL4MIS_work3"
PYTHON_BIN="/home/why/miniconda3/bin/conda"
SCRIPT_PATH="${ROOT_DIR}/eval_medical_sam2_3d_repro.py"
SAVE_DIR="/data/why/logs_SAM2SSL/medical_sam2_3d_repro"

run_one() {
  local label="$1"
  shift
  echo ""
  echo "=== ${DATASET} :: ${label} :: $(date '+%F %T') ==="
  "${PYTHON_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python "${SCRIPT_PATH}" --dataset "${DATASET}" --split eval --preset paper_best_prompt \
    --save_dir "${SAVE_DIR}" "$@"
}

# Prompt-type sweep for multi-organ behavior
run_one "P1_point_repeat_K5" --prompt_mode 1 --uniform_frame_count 5 --rgb_mode repeat
run_one "P2_multipoint_repeat_K5" --prompt_mode 2 --uniform_frame_count 5 --rgb_mode repeat
run_one "P3_box_neighbor_K5" --prompt_mode 3 --uniform_frame_count 5 --rgb_mode neighbor
run_one "P5_mask_repeat_K5" --prompt_mode 5 --uniform_frame_count 5 --rgb_mode repeat

# Prompt-budget saturation after the prompt-type sweep
run_one "P3_box_repeat_K7" --prompt_mode 3 --uniform_frame_count 7 --rgb_mode repeat

