#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:-/home/why/SSL4MIS_work3/MedSAM2/checkpoints}"
SLEEP_SECONDS="${2:-30}"

FILES=(
  "sam2.1_hiera_tiny.pt"
  "sam2.1_hiera_small.pt"
  "sam2.1_hiera_base_plus.pt"
  "sam2.1_hiera_large.pt"
  "sam2_hiera_tiny.pt"
  "sam2_hiera_small.pt"
  "sam2_hiera_base_plus.pt"
  "sam2_hiera_large.pt"
)

while true; do
  missing=0
  for file in "${FILES[@]}"; do
    if [ ! -f "${CHECKPOINT_DIR}/${file}" ]; then
      echo "WAITING ${file}"
      missing=1
      break
    fi
  done
  if [ "${missing}" -eq 0 ]; then
    echo "ALL_CHECKPOINTS_READY ${CHECKPOINT_DIR}"
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

