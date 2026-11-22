#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${SAM_VIT_LARGE_DIR}" "${TORCH_HOME}/hub/checkpoints"

echo "Caching ImageNet ResNet-101 weights."
"${PYTHON_BIN}" - <<'PY'
from torchvision.models import ResNet101_Weights, resnet101

resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
print("ResNet-101 ImageNet weights: PASS")
PY

DOWNLOAD_ARGS=(
  --model facebook/sam-vit-large
  --output-dir "${SAM_VIT_LARGE_DIR}"
)
if [[ "${FORCE_ASSET_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--force)
fi

echo "Downloading facebook/sam-vit-large from Hugging Face."
"${PYTHON_BIN}" tools/rsprompter/download_hf_sam_pretrain_ckpt.py "${DOWNLOAD_ARGS[@]}"

require_file "${SAM_VIT_LARGE_DIR}/config.json"
require_file "${SAM_VIT_LARGE_DIR}/pytorch_model.bin"

if [[ ! -f "${SAM_VIT_LARGE_CKPT}" || "${FORCE_ASSET_CONVERSION:-0}" == "1" ]]; then
  echo "Creating the bilayer SAM checkpoint."
  "${PYTHON_BIN}" tools/convert_model_weights.py \
    "${SAM_VIT_LARGE_DIR}/pytorch_model.bin" \
    "${SAM_VIT_LARGE_CKPT}"
else
  echo "Converted SAM checkpoint already exists; skipping conversion."
fi
require_file "${SAM_VIT_LARGE_CKPT}"

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

sam_dir = Path(os.environ["SAM_VIT_LARGE_DIR"])
paths = (
    sam_dir / "config.json",
    sam_dir / "pytorch_model.bin",
    Path(os.environ["SAM_VIT_LARGE_CKPT"]),
)
for path in paths:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Invalid SAM asset: {path}")
    print(f"{path}: {path.stat().st_size / (1024**3):.2f} GiB")
print("SAM ViT-L download and bilayer conversion: PASS")
PY

echo "Next: bash scripts/run_pidray30_primary60.sh"
