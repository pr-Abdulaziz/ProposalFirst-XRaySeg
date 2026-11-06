#!/usr/bin/env bash
# Two-level bootstrap, permutation tests, strata, then figures. CPU only.
#SBATCH --job-name=ms_statistics
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=output/pixray_multiseed_bs4_e50_2026-09-14/statistics/slurm_statistics_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
"${PYTHON_BIN}" scripts/campaign_multiseed/statistics_campaign.py --B 2000 --J 5000 || { echo "statistics FAILED"; exit 1; }
"${PYTHON_BIN}" scripts/campaign_multiseed/figures_campaign.py || { echo "figures FAILED"; exit 1; }
echo "end=$(date -Is) status=0"
