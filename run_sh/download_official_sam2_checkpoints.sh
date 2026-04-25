#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

OUT_DIR="${1:-/home/why/SSL4MIS_work3/MedSAM2/checkpoints}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

declare -A EXPECTED_SIZE=(
  ["sam2.1_hiera_tiny.pt"]=156008466
  ["sam2.1_hiera_small.pt"]=184416285
  ["sam2.1_hiera_base_plus.pt"]=323606802
  ["sam2.1_hiera_large.pt"]=898083611
  ["sam2_hiera_tiny.pt"]=155906050
  ["sam2_hiera_small.pt"]=184309650
  ["sam2_hiera_base_plus.pt"]=323493298
  ["sam2_hiera_large.pt"]=897952466
)

FILES=(
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt sam2.1_hiera_tiny.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt sam2.1_hiera_small.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt sam2.1_hiera_base_plus.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt sam2.1_hiera_large.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt sam2_hiera_tiny.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt sam2_hiera_small.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt sam2_hiera_base_plus.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt sam2_hiera_large.pt"
)

for item in "${FILES[@]}"; do
  url="${item% *}"
  file="${item#* }"
  expected_size="${EXPECTED_SIZE[${file}]}"
  if [ -f "${file}" ]; then
    current_size="$(stat -c%s "${file}")"
    if [ "${current_size}" = "${expected_size}" ]; then
      echo "EXISTS ${file}"
      continue
    fi
    echo "RESUME ${file} ${current_size}/${expected_size}"
  else
    echo "DOWNLOADING ${file}"
  fi
  curl -L --fail --retry 3 -C - -o "${file}" "${url}"
  echo "DONE ${file}"
done
