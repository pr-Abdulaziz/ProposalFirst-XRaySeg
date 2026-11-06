#!/usr/bin/env bash
# Evaluate one full-split arm's FINAL checkpoint on the official test split.
#
#   sbatch scripts/slurm_eval_full_split.sh <config> <checkpoint>
#
# The frozen rule is the final epoch, so this refuses any checkpoint that is not
# epoch_60.pth: picking a different one after seeing a number is exactly the
# failure the freeze exists to prevent.
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --output=output/full_split_campaign/2026-09-12/eval_%x_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1
export XRAY_NUM_WORKERS=2

CONFIG="${1:?usage: sbatch scripts/slurm_eval_full_split.sh <config> <checkpoint>}"
CKPT="${2:?usage: sbatch scripts/slurm_eval_full_split.sh <config> <checkpoint>}"

echo "node=$(hostname) job=${SLURM_JOB_ID} config=${CONFIG} ckpt=${CKPT} start=$(date -Is)"

if [[ "$(basename "${CKPT}")" != "epoch_60.pth" ]]; then
  echo "ABORT: the frozen checkpoint rule is the final epoch (epoch_60.pth); got $(basename "${CKPT}")"
  exit 3
fi
[[ -f "${CKPT}" ]] || { echo "ABORT: ${CKPT} does not exist"; exit 3; }

"${PYTHON_BIN}" - "${CONFIG}" <<'GUARD' || exit 3
import hashlib, json, sys
from pathlib import Path
freeze = json.loads(Path("output/full_split_campaign/2026-09-12/freeze.json").read_text())
digest = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
match = [a for a, s in freeze["arms"].items() if s["config_sha256"] == digest]
if not match:
    print("ABORT: config does not match any frozen arm hash; it was edited after the freeze")
    sys.exit(1)
for path, expected in freeze["annotation_sha256"].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        print(f"ABORT: {path} has changed since the freeze")
        sys.exit(1)
print(f"guard passed: arm {match[0]}, annotations unchanged since the freeze")
GUARD

WORK_DIR="$(dirname "${CKPT}")/test"
mkdir -p "${WORK_DIR}"
# Predictions are written out so the result manifest can hash them, the way
# Method 03's work_dirs/pixray_p2/test/test/predictions.pkl is hashed.
"${PYTHON_BIN}" tools/test.py "${CONFIG}" "${CKPT}" \
  --work-dir "${WORK_DIR}" --out "${WORK_DIR}/predictions.pkl"
status=$?
echo "eval exit=${status}  end=$(date -Is)"
exit "${status}"
