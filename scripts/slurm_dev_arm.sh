#!/usr/bin/env bash
# Launch one PIXray development arm.
#
#   sbatch --job-name=dev_A scripts/slurm_dev_arm.sh <config> [--resume]
#
# The guards below exist because a development run that touched the official
# test split would silently invalidate the whole comparison, and because a
# stale or edited split would invalidate it just as quietly. Both are checked
# before a single GPU second is spent.
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --output=output/dev_campaign/2026-09-12/train_%x_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1

# The development split was set INACTIVE on 2026-09-12 in favour of the original
# official train/test split. Running an arm from it would produce numbers that
# are not comparable to methods 01-08, so it takes an explicit opt-in.
if [[ "${XRAY_ALLOW_DEV_SPLIT:-0}" != "1" ]]; then
  echo "ABORT: the development split is inactive (see splits/pixray_dev_v1/INACTIVE.md)."
  echo "       Set XRAY_ALLOW_DEV_SPLIT=1 to override deliberately."
  exit 4
fi

CONFIG="${1:?usage: sbatch scripts/slurm_dev_arm.sh <config> [--resume]}"
shift || true

echo "node=$(hostname) job=${SLURM_JOB_ID} config=${CONFIG} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON_BIN}" scripts/verify_dev_split.py || { echo "ABORT: split verification failed"; exit 2; }

"${PYTHON_BIN}" - "${CONFIG}" <<'GUARD' || exit 3
import sys
from mmengine.config import Config
cfg = Config.fromfile(sys.argv[1])
problems = []
if cfg.get("test_dataloader") is not None:
    problems.append("config still defines a test_dataloader")
if cfg.get("test_cfg") is not None:
    problems.append("config still defines a test loop")
if cfg.get("load_from") is not None:
    problems.append(f"config loads weights from {cfg.load_from!r}; development "
                    "arms must start from ImageNet, not a full-training checkpoint")
ann = cfg.train_dataloader["dataset"]["ann_file"]
if not ann.endswith("dev_train.json"):
    problems.append(f"training annotations are {ann}, not the development split")
if problems:
    print("ABORT: " + "; ".join(problems))
    sys.exit(1)
print(f"guard passed: train={ann}")
print(f"guard passed: val={cfg.val_dataloader['dataset']['ann_file']}")
GUARD

"${PYTHON_BIN}" tools/train.py "${CONFIG}" "$@"
status=$?
echo "train exit=${status}  end=$(date -Is)"
exit "${status}"
