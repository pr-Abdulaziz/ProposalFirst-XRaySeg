#!/usr/bin/env bash
# PIDray-30 profiling at batch 1 for the ten comparison methods (measurement only).
#
#   sbatch scripts/campaign_pidray_bs1/slurm_profile_bs1.sh 01 02 03 04 07 08 09 09M 05 06
#
# issnode1 is the login node (foreign GPU users outside SLURM); issnode5 is drained.
#SBATCH --job-name=pid_bs1_profile
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=output/pidray_bs2_e20_2026-09-21/profile/slurm_profile_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is) methods=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
status=0
for m in "$@"; do
  "${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check \
    || { echo "=== profile ${m} exit=4 (GPU not clean)"; status=1; continue; }
  "${PYTHON_BIN}" scripts/campaign_pidray_bs1/profile_bs1.py "${m}"
  rc=$?
  echo "=== profile ${m} exit=${rc} $(date -Is)"
  [[ ${rc} -ne 0 ]] && status=1
done
echo "end=$(date -Is) status=${status}"
exit ${status}
