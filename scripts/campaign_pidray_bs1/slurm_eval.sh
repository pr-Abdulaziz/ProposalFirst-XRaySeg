#!/usr/bin/env bash
# Evaluate one frozen PIDray-30 run on the three released test splits.
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/run_guard.py gpu-check || { echo "ABORT: GPU not clean"; exit 4; }
MID="${1:?method id}"
SEED="${2:?seed}"
echo "node=$(hostname) job=${SLURM_JOB_ID} method=${MID} seed=${SEED} start=$(date -Is)"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/evaluate_run.py "${MID}" "${SEED}"
rc=$?
echo "eval exit=${rc} end=$(date -Is)"
exit ${rc}
