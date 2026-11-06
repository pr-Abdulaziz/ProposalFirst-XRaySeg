#!/usr/bin/env bash
# Inference-geometry preflight for campaign methods (uses each method's preflight checkpoint).
#
#   sbatch scripts/campaign_multiseed/slurm_preflight_geometry.sh 03 09M 09 01 02 04 08 05 06
#SBATCH --job-name=ms_geometry
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=output/pixray_multiseed_bs4_e50_2026-09-14/preflight/slurm_geometry_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is) methods=$*"
status=0
for m in "$@"; do
  "${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check || { echo "=== geometry ${m} exit=4 (GPU not clean)"; status=1; continue; }
  "${PYTHON_BIN}" scripts/campaign_multiseed/preflight_inference_geometry.py "${m}"
  rc=$?
  echo "=== geometry ${m} exit=${rc} $(date -Is)"
  [[ ${rc} -ne 0 ]] && status=1
done
echo "end=$(date -Is) status=${status}"
exit ${status}
