#!/usr/bin/env bash
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=00:40:00
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

echo "--- 1. CPU regression suites ---"
"${PYTHON_BIN}" tests/test_zero_id_compat.py; echo "zero_id tests exit=$?"

echo "--- 2. build both SAM models on GPU and run one forward pass ---"
"${PYTHON_BIN}" scripts/smoke_sam_geometry.py; echo "smoke exit=$?"
echo "end=$(date -Is)"
