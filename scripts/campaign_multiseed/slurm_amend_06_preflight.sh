#!/usr/bin/env bash
# Amendment 06 (activation checkpointing): equivalence test, preflight on the amended config, finalize.
#SBATCH --job-name=ms06_ckpt_preflight
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=output/pixray_multiseed_bs4_e50_2026-09-14/protocol/amendments/06_activation_checkpointing/slurm_preflight_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
"${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check || { echo "ABORT: GPU not clean"; exit 4; }
"${PYTHON_BIN}" scripts/campaign_multiseed/check_checkpoint_equivalence.py || { echo "=== equivalence FAIL"; exit 1; }
"${PYTHON_BIN}" scripts/campaign_multiseed/run_guard.py gpu-check || { echo "ABORT: GPU not clean"; exit 4; }
PREFLIGHT_CONFIG="$(pwd)/output/pixray_multiseed_bs4_e50_2026-09-14/protocol/amendments/06_activation_checkpointing/seed_42.py" \
PREFLIGHT_OUT_SUFFIX=_checkpointed "${PYTHON_BIN}" scripts/campaign_multiseed/preflight.py 06 || { echo "=== preflight FAIL"; exit 1; }
"${PYTHON_BIN}" scripts/campaign_multiseed/amend_06_checkpointing.py --finalize || exit 1
echo "end=$(date -Is) status=0"
