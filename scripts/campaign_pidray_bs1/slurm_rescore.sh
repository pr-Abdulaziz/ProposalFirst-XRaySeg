#!/usr/bin/env bash
# Re-score every PIDray-30 prediction file with the correct label -> category-id mapping (CPU only).
#SBATCH --partition=main
#SBATCH --exclude=issnode1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/rescore_predictions.py
rc=$?
echo "rescore exit=${rc} end=$(date -Is)"
exit ${rc}
