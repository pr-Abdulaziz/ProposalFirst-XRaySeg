#!/usr/bin/env bash
#   sbatch scripts/revision_2026_11/slurm_eval_ablation.sh <09|09M> <seed> [variants...]
#SBATCH --partition=main
#SBATCH --exclude=node,node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -uo pipefail
cd .
source scripts/common.sh
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check || { echo "ABORT: GPU not clean"; exit 4; }
echo "node=$(hostname) job=${SLURM_JOB_ID} args=$* start=$(date -Is)"
"${PYTHON_BIN}" scripts/revision_2026_11/evaluate_ablation.py "$@"
rc=$?
echo "exit=${rc} end=$(date -Is)"
exit ${rc}
