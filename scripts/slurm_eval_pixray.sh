#!/usr/bin/env bash
# Evaluate one or more completed PIXray experiments on a single GPU node.
#
# Evaluation is deliberately kept separate from training: the primary runner
# only executes the train stage, so finished checkpoints carry no metrics until
# this script runs. Experiments are evaluated sequentially because they all
# append to one Excel workbook.
#
#   EXPERIMENTS="P3 P4" sbatch scripts/slurm_eval_pixray.sh
#   EXPERIMENTS="P3" REPORT=output/reports/pixray/eval_smoke.xlsx \
#       sbatch scripts/slurm_eval_pixray.sh
set -euo pipefail

: "${EXPERIMENTS:?EXPERIMENTS was not provided, for example EXPERIMENTS=\"P3 P4\"}"

cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh

export XRAY_BATCH_SIZE_PER_GPU=2
export XRAY_EFFECTIVE_BATCH_SIZE=16
export XRAY_NUM_WORKERS=2
export PYTHONUNBUFFERED=1

report="${REPORT:-output/reports/pixray/pixray_primary60.xlsx}"

echo "Experiments: ${EXPERIMENTS}"
echo "Report:      ${report}"
echo "Node:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID:-none}"
echo "Started:     $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Sequential by construction: one process, experiments consumed in order, so no
# two evaluations ever write to the workbook at the same time.
# --continue-on-error: one experiment failing must not abort the other ten.
# Several evaluation paths have never run against a live model, so a single
# failure is a real possibility; failures are listed at the end of the log.
"${PYTHON_BIN}" scripts/run_pixray_experiments.py evaluate \
  --experiments ${EXPERIMENTS} \
  --skip-preflight \
  --continue-on-error \
  --keep-predictions \
  --error-samples 20 \
  --report "${report}"

echo "EVALUATION: PASS (${EXPERIMENTS})"
echo "Finished: $(date -Is)"
