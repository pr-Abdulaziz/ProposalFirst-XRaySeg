#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1 && [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bootstrap_python="${PYTHON_BIN}"
python_supported="$("${bootstrap_python}" -c 'import sys; print(int(sys.version_info[:2] in {(3, 10), (3, 11)}))')"
venv_python="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${venv_python}" ]]; then
  echo "Creating the project-local .venv."
  if [[ "${python_supported}" == "1" ]] \
      && "${bootstrap_python}" -m venv --help >/dev/null 2>&1; then
    "${bootstrap_python}" -m venv "${PROJECT_ROOT}/.venv"
  else
    echo "A local Python 3.10/3.11 venv is unavailable; using uv with Python 3.10."
    if ! command -v uv >/dev/null 2>&1; then
      echo "Installing uv to create the Python 3.10 environment."
      if "${bootstrap_python}" -m pip --version >/dev/null 2>&1; then
        "${bootstrap_python}" -m pip install "uv==0.6.14"
      elif command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/0.6.14/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
      else
        echo "Python pip and curl are unavailable; cannot install uv." >&2
        exit 1
      fi
    fi
    uv venv --seed --python 3.10 "${PROJECT_ROOT}/.venv"
  fi
fi

PYTHON_BIN="${venv_python}"
python_supported="$("${PYTHON_BIN}" -c 'import sys; print(int(sys.version_info[:2] in {(3, 10), (3, 11)}))')"
if [[ "${python_supported}" != "1" ]]; then
  echo "The existing .venv is not Python 3.10/3.11; recreating it with Python 3.10."
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv to create the Python 3.10 environment."
    if "${bootstrap_python}" -m pip --version >/dev/null 2>&1; then
      "${bootstrap_python}" -m pip install "uv==0.6.14"
    elif command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/0.6.14/install.sh | sh
      export PATH="${HOME}/.local/bin:${PATH}"
    else
      echo "Python pip and curl are unavailable; cannot install uv." >&2
      exit 1
    fi
  fi
  uv venv --clear --seed --python 3.10 "${PROJECT_ROOT}/.venv"
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
fi

echo "Installing the pinned PIDray segmentation runtime into: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
"${PYTHON_BIN}" -m pip install openmim==0.3.9
MIM_BIN="$(dirname "${PYTHON_BIN}")/mim"
if [[ ! -x "${MIM_BIN}" ]]; then
  echo "OpenMIM installed, but the mim command is not available." >&2
  exit 1
fi
"${MIM_BIN}" install "mmengine==0.10.7"
"${MIM_BIN}" install "mmcv==2.1.0"
"${PYTHON_BIN}" -m pip install -r requirements/cloud.txt
# The vendored MMDetection source is imported directly from the project root
# through PYTHONPATH (set in common.sh).  Its legacy setup.py does not support
# editable installation, and no package installation is required for training.

"${PYTHON_BIN}" - <<'PY'
import sys

import mmcv
import mmengine
import torch
import torchvision
from mmcv.ops import roi_align  # noqa: F401

expected = {
    "torch": "2.1.2",
    "torchvision": "0.16.2",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
}
actual = {
    "torch": torch.__version__.split("+")[0],
    "torchvision": torchvision.__version__.split("+")[0],
    "mmcv": mmcv.__version__,
    "mmengine": mmengine.__version__,
}
if actual != expected:
    raise RuntimeError(f"Version mismatch: expected {expected}, found {actual}")
if torch.version.cuda != "11.8":
    raise RuntimeError(f"Expected CUDA 11.8 PyTorch build, found {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a CUDA GPU.")

print(f"python={sys.version.split()[0]}")
print(f"versions={actual}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"gpu_memory={gpu_memory_gib:.2f} GiB")
if gpu_memory_gib < 47:
    print(
        "WARNING: default batch 4 may not fit this GPU. Before running every "
        "experiment, export XRAY_BATCH_SIZE_PER_GPU=2 and "
        "XRAY_EFFECTIVE_BATCH_SIZE=16. The smoke tests will verify the choice."
    )
print("MMCV compiled operators: PASS")
PY

echo "Environment setup: PASS"
echo "Next: bash scripts/prepare_assets.sh"
