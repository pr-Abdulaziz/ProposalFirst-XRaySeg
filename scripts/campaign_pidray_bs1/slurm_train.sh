#!/usr/bin/env bash
# Train one frozen PIDray-30 batch-1 run. Submitted by submit_campaign.py, which
# sets the job name, time limit and log path.
#
#   sbatch scripts/campaign_pidray_bs1/slurm_train.sh <method id> <seed> [--resume]
#
# issnode1 is the login node (foreign GPU users outside SLURM); issnode5 is drained.
#SBATCH --partition=main
#SBATCH --exclude=issnode1,issnode5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
set +e
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/run_guard.py gpu-check || { echo "ABORT: GPU not clean"; exit 4; }

MID="${1:?method id}"
SEED="${2:?seed}"
RESUME="${3:-}"
echo "node=$(hostname) job=${SLURM_JOB_ID} method=${MID} seed=${SEED} ${RESUME} start=$(date -Is)"

"${PYTHON_BIN}" scripts/campaign_pidray_bs1/run_guard.py train-start "${MID}" "${SEED}" ${RESUME} || exit 3
CFG="$("${PYTHON_BIN}" -c "import sys; sys.path.insert(0, '.'); from scripts.campaign_pidray_bs1.campaign_common import resolved_config; print(resolved_config('${MID}', ${SEED}))")"
"${PYTHON_BIN}" scripts/campaign_multiseed/train_run.py "${CFG}" ${RESUME}
rc=$?
echo "train exit=${rc} end=$(date -Is)"
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/run_guard.py train-end "${MID}" "${SEED}" "${rc}"
exit $?
