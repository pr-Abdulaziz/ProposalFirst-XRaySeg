#!/usr/bin/env bash
# Isolate why method 08 diverges: same config, one variable changed at a time.
#SBATCH --job-name=pid_diag08
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=output/pidray_bs2_e20_2026-09-21/preflight/slurm_diag08_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
echo "--- A: campaign protocol (expected to diverge near iteration 291) ---"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/preflight.py 08 --iters 800
echo "--- B: fp32, AMP disabled ---"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/preflight.py 08 --iters 800 --fp32
echo "--- C: fp16 with the MMDetection default 500-iteration warmup ---"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/preflight.py 08 --iters 800 --warmup 500
echo "end=$(date -Is)"
