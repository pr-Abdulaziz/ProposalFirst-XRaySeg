#!/usr/bin/env bash
#   sbatch scripts/campaign_pidray_bs1/slurm_rehearsal.sh 01 09 06 05 07
#SBATCH --job-name=pid_bs1_rehearsal
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=output/pidray_bs2_e20_2026-09-21/rehearsal/slurm_rehearsal_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is) methods=$*"
status=0
for m in "$@"; do
  "${PYTHON_BIN}" scripts/campaign_pidray_bs1/dress_rehearsal.py "${m}"
  rc=$?
  echo "=== rehearsal ${m} exit=${rc} $(date -Is)"
  [[ ${rc} -ne 0 ]] && status=1
done
echo "end=$(date -Is) status=${status}"
exit ${status}
