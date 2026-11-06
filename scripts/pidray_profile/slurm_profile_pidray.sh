#!/usr/bin/env bash
# PIDray time/memory profiling at batch 2 for the comparison methods (measurement only).
#
#   sbatch scripts/pidray_profile/slurm_profile_pidray.sh 06 05 03 09M 09 01 02 04 08 07
#
# issnode1 is the login node (foreign GPU users outside SLURM); issnode2 is drained.
#SBATCH --job-name=pidray_profile
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=output/pidray_profile_2026-09-14/slurm_profile_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is) methods=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
status=0
for m in "$@"; do
  if [[ "${PROFILE_ALLOW_SHARED_GPU:-0}" == "1" ]]; then
    # User-approved exception for this measurement: record the shared GPU state instead of aborting.
    echo "=== GPU state before ${m} (shared GPU allowed): $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) | processes: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  else
    "${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check || { echo "=== profile ${m} exit=4 (GPU not clean)"; status=1; continue; }
  fi
  "${PYTHON_BIN}" scripts/pidray_profile/profile_pidray.py "${m}"
  rc=$?
  echo "=== profile ${m} exit=${rc} $(date -Is)"
  [[ ${rc} -ne 0 ]] && status=1
done
echo "end=$(date -Is) status=${status}"
exit ${status}
