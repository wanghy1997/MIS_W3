#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset: BTCV|MMWHS|AMOS> <sam3_checkpoint> [env_name]" >&2
  exit 1
fi

DATASET="$1"
SAM3_CKPT="$2"
ENV_NAME="${3:-sam3_py312}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

python eval_medical_sam3_3d_repro.py \
  --dataset "${DATASET}" \
  --preset paper_gt_upper \
  --split eval \
  --rgb_mode repeat \
  --uniform_frame_count 5 \
  --sam3_root /home/why/codes/sam3 \
  --sam3_checkpoint "${SAM3_CKPT}" \
  --load_from_hf 0 \
  --save_dir /data/why/logs_SAM2SSL/medical_sam3_3d_repro
