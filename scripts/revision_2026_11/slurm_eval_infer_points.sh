#!/usr/bin/env bash
#   sbatch scripts/revision_2026_11/slurm_eval_infer_points.sh <09|09M> <seed> <infer_points>
#SBATCH --partition=main
#SBATCH --exclude=node,node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -uo pipefail
cd .
source scripts/common.sh
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check || { echo "ABORT: GPU not clean"; exit 4; }
echo "node=$(hostname) job=${SLURM_JOB_ID} method=$1 seed=$2 points=$3 start=$(date -Is)"
"${PYTHON_BIN}" scripts/revision_2026_11/evaluate_infer_points.py "$1" "$2" "$3"
rc=$?
echo "exit=${rc} end=$(date -Is)"
exit ${rc}
