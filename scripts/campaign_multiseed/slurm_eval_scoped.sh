#!/usr/bin/env bash
# Evaluate one seed-42 run under the scope amendment (summaries/scope_amendment_seed42.json).
# Submitted by scope_amendment_seed42.py --submit-evals after all seed-42 training.
#
#   sbatch scripts/campaign_multiseed/slurm_eval_scoped.sh <method id> <seed>
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode2
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
# The GPU must be clean (no compute process, <500 MiB in use) before anything runs.
"${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check || { echo "ABORT: GPU not clean"; exit 4; }

MID="${1:?method id}"
SEED="${2:?seed}"
echo "node=$(hostname) job=${SLURM_JOB_ID} method=${MID} seed=${SEED} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" scripts/campaign_multiseed/evaluate_scoped.py "${MID}" "${SEED}"
rc=$?
echo "eval exit=${rc} end=$(date -Is)"
exit ${rc}
