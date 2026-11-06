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
"${PYTHON_BIN}" tests/test_method_08_contract.py; echo "contract exit=$?"
"${PYTHON_BIN}" scripts/smoke_method_08.py; echo "smoke exit=$?"
echo "end=$(date -Is)"
