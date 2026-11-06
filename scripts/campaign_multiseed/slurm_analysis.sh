#!/usr/bin/env bash
# Extended evaluation (errors, confusion matrices, mask quality, PR curves, paired comparisons). CPU only.
#SBATCH --job-name=ms_analysis
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=output/pixray_multiseed_bs4_e50_2026-09-14/analysis/slurm_analysis_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
"${PYTHON_BIN}" scripts/campaign_multiseed/analyze_errors_quality.py --workers 8
rc=$?
echo "end=$(date -Is) exit=${rc}"
exit ${rc}
