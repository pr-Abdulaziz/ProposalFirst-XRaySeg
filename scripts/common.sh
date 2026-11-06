#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# The pinned PIDray/PIDray-A runtime needs Python 3.10 or 3.11. On cloud images
# setup.sh creates this project-local environment when the base Python is newer.
if [[ -n "${XRAY_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${XRAY_PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XRAY_PROJECT_ROOT="${PROJECT_ROOT}"
export PIP_CACHE_DIR="${XRAY_PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
export HF_HOME="${XRAY_HF_CACHE_DIR:-${PROJECT_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/pretrained/torch}"

export PIDRAY_DATA_ROOT="${PIDRAY_DATA_ROOT:-${PROJECT_ROOT}/data/PIDray}"
export OCC_PIDRAY_DATA_ROOT="${OCC_PIDRAY_DATA_ROOT:-${PIDRAY_DATA_ROOT}}"
export PIDRAYA_ANN_ROOT="${PIDRAYA_ANN_ROOT:-${PROJECT_ROOT}/data/PIDray-A/annotations}"
export PIXRAY_DATA_ROOT="${PIXRAY_DATA_ROOT:-${PROJECT_ROOT}/data/PIXray}"
export PIXRAYA_ANN_ROOT="${PIXRAYA_ANN_ROOT:-${PROJECT_ROOT}/data/PIXray-A/annotations}"
export SAM_VIT_LARGE_DIR="${SAM_VIT_LARGE_DIR:-${PROJECT_ROOT}/pretrained/sam_vit_large}"
export SAM_VIT_LARGE_CKPT="${SAM_VIT_LARGE_CKPT:-${SAM_VIT_LARGE_DIR}/pytorch_model_new.bin}"
export XRAY_BATCH_SIZE_PER_GPU="${XRAY_BATCH_SIZE_PER_GPU:-2}"
export XRAY_EFFECTIVE_BATCH_SIZE="${XRAY_EFFECTIVE_BATCH_SIZE:-16}"
export XRAY_NUM_WORKERS="${XRAY_NUM_WORKERS:-4}"

mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}
