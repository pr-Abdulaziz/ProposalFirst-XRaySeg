#!/usr/bin/env bash
# Train one run of the PIDray gradient-isolation diagnostic from its resolved config.
# Same entry point as the frozen campaign (scripts/campaign_multiseed/train_run.py), without the
# campaign manifest guard (the diagnostic is not part of the frozen manifest). Status events are
# appended to output/pidray_diag_gradiso_2026-09-26/runs/<slug>/seed_<seed>/status.jsonl.
#
#   sbatch scripts/campaign_pidray_bs1/slurm_train_diag.sh <slug> <seed>
#SBATCH --partition=main
#SBATCH --exclude=issnode1
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
SLUG="${1:?slug}"
SEED="${2:?seed}"
DIAG=output/pidray_diag_gradiso_2026-09-26
CFG="${DIAG}/protocol/resolved_configs/${SLUG}/seed_${SEED}.py"
STATUS="${DIAG}/runs/${SLUG}/seed_${SEED}/status.jsonl"
mkdir -p "$(dirname "${STATUS}")"
event() { echo "{\"event\": \"$1\", \"job\": \"${SLURM_JOB_ID}\", \"node\": \"$(hostname)\", \"time\": $(date +%s)$2}" >> "${STATUS}"; }
"${PYTHON_BIN}" scripts/campaign_pidray_bs1/run_guard.py gpu-check || { event gpu_check_failed ""; exit 4; }
echo "node=$(hostname) job=${SLURM_JOB_ID} config=${CFG} start=$(date -Is)"
event train_started ", \"config_sha256\": \"$(sha256sum "${CFG}" | cut -d' ' -f1)\""
"${PYTHON_BIN}" scripts/campaign_multiseed/train_run.py "${CFG}"
rc=$?
event train_end ", \"exit_code\": ${rc}"
echo "train exit=${rc} end=$(date -Is)"
exit ${rc}
