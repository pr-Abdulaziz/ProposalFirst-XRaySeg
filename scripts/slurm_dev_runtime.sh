#!/usr/bin/env bash
#SBATCH --job-name=xr_devrt
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --output=output/dev_campaign/2026-09-12/runtime_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
"${PYTHON_BIN}" scripts/measure_dev_runtime.py
echo "runtime exit=$?  end=$(date -Is)"
