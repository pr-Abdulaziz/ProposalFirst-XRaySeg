#!/usr/bin/env bash
# Watch a set of PIXray training jobs and recover ones that die mid-run.
#
# The configs set resume=False and SLURM does not restart a job that fails, so
# an interrupted run is simply lost: it leaves checkpoints but no
# training_summary.json, the queue moves on, and nothing says so. Over a
# multi-day chain on shared nodes that is easy to miss.
#
# A run is considered dead when its job has left the queue, no
# training_summary.json exists, and checkpoints are present. It is then
# resubmitted with RESUME=1, at most MAX_RETRIES times, so a genuine code fault
# cannot loop forever.
#
#   EXPERIMENTS="B5 B6 B1 B2 B3 B4" nohup scripts/watch_pixray_baselines.sh &
#   EXPERIMENTS="B5" AUTO_RESUBMIT=0 scripts/watch_pixray_baselines.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

EXPERIMENTS="${EXPERIMENTS:?EXPERIMENTS was not provided, for example EXPERIMENTS=\"B5 B6\"}"
INTERVAL="${INTERVAL:-300}"
MAX_RETRIES="${MAX_RETRIES:-2}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
DEADLINE_HOURS="${DEADLINE_HOURS:-72}"
LOG="${LOG:-output/logs/watchdog_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "${LOG}")"
declare -A RETRIES
declare -A REPORTED
for e in ${EXPERIMENTS}; do RETRIES[$e]=0; REPORTED[$e]=""; done

say() { printf '%s  %s\n' "$(date -Is)" "$*" | tee -a "${LOG}"; }

work_dir_for() {
  case "$1" in
    S3-P) echo "work_dirs/pixray_s3_p" ;;
    S3-MBoost) echo "work_dirs/pixray_s3_mboost" ;;
    *) echo "work_dirs/pixray_$(echo "$1" | tr '[:upper:]' '[:lower:]')" ;;
  esac
}

say "watching: ${EXPERIMENTS}"
say "interval ${INTERVAL}s, max ${MAX_RETRIES} retries each, auto-resubmit=${AUTO_RESUBMIT}"

deadline=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
while (( $(date +%s) < deadline )); do
  outstanding=0
  for e in ${EXPERIMENTS}; do
    dir="$(work_dir_for "$e")"
    if [[ -f "${dir}/training_summary.json" ]]; then
      if [[ "${REPORTED[$e]}" != "done" ]]; then
        say "${e}: COMPLETE"
        REPORTED[$e]="done"
      fi
      continue
    fi
    outstanding=$((outstanding + 1))

    # still queued or running: nothing to do
    if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qx "pixray-${e}"; then
      if [[ "${REPORTED[$e]}" != "active" ]]; then
        say "${e}: in the queue"
        REPORTED[$e]="active"
      fi
      continue
    fi

    # left the queue with no summary
    ckpts=$(ls "${dir}"/*.pth 2>/dev/null | wc -l)
    if (( ckpts == 0 )); then
      if [[ "${REPORTED[$e]}" != "nostart" ]]; then
        say "${e}: NOT IN QUEUE and no checkpoints -- it never started, or failed immediately. Not resubmitting; check its log."
        REPORTED[$e]="nostart"
      fi
      continue
    fi

    # A job the user cancelled must not be resurrected. Only an involuntary
    # end is an interruption worth recovering from; CANCELLED means a person
    # (or an admin) stopped it deliberately, and resubmitting would fight them.
    state=$(sacct -n -X -j "$(sacct -n -X --name="pixray-${e}" -u "${USER}" -o JobID -S now-7days | tail -1 | tr -d ' ')" -o State 2>/dev/null | head -1 | awk '{print $1}')
    if [[ "${state}" == CANCELLED* ]]; then
      if [[ "${REPORTED[$e]}" != "cancelled" ]]; then
        say "${e}: CANCELLED deliberately (${ckpts} checkpoints). Not resubmitting. Resume it yourself with:  RESUME=1 EXPERIMENT=${e} sbatch scripts/slurm_train_pixray_one.sh"
      fi
      REPORTED[$e]="cancelled"
      continue
    fi

    # Announce a death once, not once per poll.
    if [[ "${REPORTED[$e]}" != "dead" && "${REPORTED[$e]}" != "exhausted" ]]; then
      last=$(ls -t "${dir}"/epoch_*.pth 2>/dev/null | head -1)
      say "${e}: DIED MID-RUN (state=${state:-unknown}, ${ckpts} checkpoints, latest ${last##*/}, no training_summary.json)"
    fi
    if [[ "${AUTO_RESUBMIT}" != "1" ]]; then
      if [[ "${REPORTED[$e]}" != "dead" ]]; then
        say "${e}: auto-resubmit disabled. Recover with:  RESUME=1 EXPERIMENT=${e} sbatch scripts/slurm_train_pixray_one.sh"
      fi
      REPORTED[$e]="dead"
      continue
    fi
    if (( RETRIES[$e] >= MAX_RETRIES )); then
      if [[ "${REPORTED[$e]}" != "exhausted" ]]; then
        say "${e}: already retried ${RETRIES[$e]} times; leaving it alone. This looks like a real fault, not an interruption."
      fi
      REPORTED[$e]="exhausted"
      continue
    fi
    RETRIES[$e]=$(( RETRIES[$e] + 1 ))
    jid=$(sbatch --parsable \
      --job-name="pixray-${e}" \
      --output="output/logs/pixray_${e}_%N_%j.log" \
      --export=ALL,EXPERIMENT="${e}",RESUME=1 \
      scripts/slurm_train_pixray_one.sh 2>&1)
    say "${e}: resubmitted with RESUME=1 as job ${jid} (retry ${RETRIES[$e]}/${MAX_RETRIES})"
    REPORTED[$e]="active"
  done

  if (( outstanding == 0 )); then
    say "all watched experiments have training_summary.json"
    # Submitting the evaluation from here rather than as a SLURM dependency on
    # the last training job: a dependency would fire while a resubmitted run was
    # still going, because the recovery job is not the one it depends on.
    if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qx "pixray-eval"; then
      # An evaluation is already queued or running -- typically one submitted
      # up front with a dependency on the last training job. Submitting a
      # second would append duplicate rows to the workbook.
      say "an evaluation job is already in the queue; not submitting another"
      exit 0
    fi
    if [[ "${EVAL_ON_COMPLETE:-0}" == "1" ]]; then
      jid=$(EXPERIMENTS="${EXPERIMENTS}" sbatch --parsable \
        --job-name="pixray-eval" \
        --cpus-per-task=8 --mem=32G --gres=gpu:1 --partition=main \
        --exclude=issnode1,issnode2 \
        --output="output/logs/pixray_eval_%N_%j.log" \
        --export=ALL,EXPERIMENTS="${EXPERIMENTS}" \
        scripts/slurm_eval_pixray.sh 2>&1)
      say "submitted evaluation for ${EXPERIMENTS} as job ${jid}"
    else
      say "evaluation not submitted (EVAL_ON_COMPLETE is not 1). Run:  EXPERIMENTS=\"${EXPERIMENTS}\" sbatch scripts/slurm_eval_pixray.sh"
    fi
    exit 0
  fi
  sleep "${INTERVAL}"
done
say "deadline of ${DEADLINE_HOURS}h reached; exiting"
