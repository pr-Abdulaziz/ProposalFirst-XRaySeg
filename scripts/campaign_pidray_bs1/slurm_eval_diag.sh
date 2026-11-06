#!/usr/bin/env bash
# Evaluate one PIDray diagnostic run on easy / hard / hidden with the corrected category mapping.
#   sbatch scripts/campaign_pidray_bs1/slurm_eval_diag.sh <slug> <seed>
#SBATCH --partition=main
#SBATCH --exclude=issnode1
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
echo "node=$(hostname) job=${SLURM_JOB_ID} slug=$1 seed=$2 start=$(date -Is)"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/evaluate_diag.py "$1" "$2"
rc=$?
echo "eval exit=${rc} end=$(date -Is)"
exit ${rc}
