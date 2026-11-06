#!/usr/bin/env bash
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=00:30:00
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1
echo "node=$(hostname) job=${SLURM_JOB_ID}"
"${PYTHON_BIN}" scripts/trace_method03_baseline.py; echo "trace exit=$?"
