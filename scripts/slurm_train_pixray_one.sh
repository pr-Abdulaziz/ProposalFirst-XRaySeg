#!/usr/bin/env bash
# issnode1 is the login node: SLURM advertises it as idle, but interactive
# processes outside the scheduler hold its GPU, so allocations there die with
# CUDA OOM in under a minute (jobs 10346-10348 on 2026-09-05). issnode2 has been
# drained since 2026-09-03. These directives must stay above the first
# executable line or sbatch stops parsing and silently ignores them.
#SBATCH --exclude=issnode1,issnode2
# Without an explicit request SLURM asks for the node's full 192900M, which
# exceeds the student QOS ceiling of 160G and leaves the job pending forever on
# QOSMaxMemoryPerJob. These are the values every completed PIXray run used.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

set -euo pipefail

: "${EXPERIMENT:?EXPERIMENT was not provided}"

cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh

export XRAY_BATCH_SIZE_PER_GPU=2
export XRAY_EFFECTIVE_BATCH_SIZE=16
export XRAY_NUM_WORKERS=2
export PYTHONUNBUFFERED=1

echo "Experiment: $EXPERIMENT"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Started: $(date -Is)"
nvidia-smi

# RESUME=1 continues an interrupted run from its last checkpoint. Without it
# the runner refuses to touch a work directory that already holds checkpoints,
# which is what protects a completed run from being silently restarted.
resume_flag=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_flag=(--resume)
  echo "Resuming from the last checkpoint in work_dirs"
fi

"${PYTHON_BIN}" scripts/run_pixray_experiments.py train \
  --experiments "$EXPERIMENT" "${resume_flag[@]}"

echo "$EXPERIMENT TRAINING: PASS"
echo "Finished: $(date -Is)"
