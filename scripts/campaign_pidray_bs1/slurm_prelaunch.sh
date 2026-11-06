#!/usr/bin/env bash
# Everything that must be green before the ten-run campaign is launched:
#   1. method 08 diagnostic, one variable changed at a time
#   2. stability probes for 02, 04 and 07, which have never run at batch 2
#SBATCH --job-name=pid_prelaunch
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=output/pidray_bs2_e20_2026-09-21/preflight/slurm_prelaunch_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"

echo "### 08-A campaign protocol (expected to diverge near iteration 291)"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/preflight.py 08 --iters 800
echo "### 08-B fp32, AMP disabled"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/preflight.py 08 --iters 800 --fp32
echo "### 08-C fp16 with the MMDetection default 500-iteration warmup"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/preflight.py 08 --iters 800 --warmup 500

for m in 02 04 07; do
  echo "### stability probe ${m}"
  "${PYTHON_BIN}" scripts/campaign_pidray_bs1/preflight.py "${m}" --iters 3000
  echo "=== preflight ${m} exit=$? $(date -Is)"
done
echo "end=$(date -Is)"
