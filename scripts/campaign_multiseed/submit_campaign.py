#!/usr/bin/env python
"""Submit the frozen campaign through SLURM.

Training: one job per approved run, in the manifest's fixed order. Training jobs are
independent of each other; the account's QOS allows one running job, so the queue
runs them one at a time in submission order.

Evaluation: nothing is evaluated until every approved training run has finished
successfully (user instruction, 2026-09-14). With --submit, every evaluation job is
submitted with --dependency=afterok on all training jobs and --kill-on-invalid-dep=yes,
so one failed training cancels every pending evaluation; evaluate_run.py re-checks
that every run recorded train_complete. If evaluations were cancelled, resolve the
failed run, then submit them with --submit-evals (requires every run complete).

    python scripts/campaign_multiseed/submit_campaign.py --dry-run
    python scripts/campaign_multiseed/submit_campaign.py --submit
    python scripts/campaign_multiseed/submit_campaign.py --submit-evals
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    MANIFEST, OUT, append_jsonl, run_dir, write_once)


def slurm_time(hours: float) -> str:
    h = min(7 * 24, max(24, math.ceil(hours)))
    return f"{h // 24}-{h % 24:02d}:00:00"


def sbatch(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def train_complete(mid: str, seed: int) -> bool:
    st = run_dir(mid, seed) / "status.jsonl"
    return st.is_file() and any(json.loads(x)["event"] == "train_complete" for x in st.read_text().splitlines())


def submit_evals(approved: list[dict], dependency: str | None) -> None:
    for entry in approved:
        mid, seed, key = entry["method"], entry["seed"], entry["run_key"]
        rd = run_dir(mid, seed)
        cmd = ["sbatch", "--parsable", f"--job-name=ev{mid}_s{seed}", "--time=04:00:00",
               f"--output={rd}/slurm_eval_%j.log"]
        if dependency:
            cmd += [f"--dependency={dependency}", "--kill-on-invalid-dep=yes"]
        cmd += ["scripts/campaign_multiseed/slurm_eval.sh", mid, str(seed)]
        eid = sbatch(cmd)
        record = dict(run_key=key, eval_job=eid, eval_command=cmd, dependency=dependency,
                      submitted=time.strftime("%Y-%m-%dT%H:%M:%S"))
        write_once(rd / f"eval_submission_job{eid}.json", record)
        append_jsonl(rd / "status.jsonl", dict(event="eval_submitted", eval_job=eid, time=time.time()))
        append_jsonl(OUT / "summaries" / "job_list.jsonl", record)
        print(f"[{entry['index']:02d}] {key}: eval {eid}" + (" (after all training)" if dependency else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--submit-evals", action="store_true")
    args = ap.parse_args()
    man = json.loads(MANIFEST.read_text())
    if man.get("status") != "frozen":
        raise SystemExit("ABORT: manifest is not frozen")
    approved = [e for e in man["run_order"] if man["runs"][e["run_key"]]["launchable"]]

    if args.submit_evals:
        missing = [e["run_key"] for e in approved if not train_complete(e["method"], e["seed"])]
        if missing:
            raise SystemExit(f"ABORT: {len(missing)} training runs not complete: {missing}")
        submit_evals(approved, None)
        return 0

    total_hours, train_jobs = 0.0, []
    for entry in approved:
        mid, seed, key = entry["method"], entry["seed"], entry["run_key"]
        rd = run_dir(mid, seed)
        projected = man["preflight"][mid]["timing"]["projected_train_hours"]
        total_hours += projected
        limit = slurm_time(2.0 * projected + 4)
        cmd = ["sbatch", "--parsable", f"--job-name=ms{mid}_s{seed}", f"--time={limit}",
               f"--output={rd}/slurm_train_%j.log",
               "scripts/campaign_multiseed/slurm_train.sh", mid, str(seed)]
        if (rd / "submission.json").exists():
            print(f"skip {key}: already submitted ({rd / 'submission.json'})")
            continue
        if args.dry_run:
            print(f"[{entry['index']:02d}] {key:45s} time={limit} projected={projected:.1f} h")
            continue
        rd.mkdir(parents=True, exist_ok=True)
        tid = sbatch(cmd)
        train_jobs.append(tid)
        record = dict(run_key=key, order_index=entry["index"], train_job=tid, train_command=cmd,
                      time_limit=limit, projected_train_hours=projected, submitted=time.strftime("%Y-%m-%dT%H:%M:%S"))
        write_once(rd / "submission.json", record)
        append_jsonl(rd / "status.jsonl", dict(event="submitted", train_job=tid, time=time.time()))
        append_jsonl(OUT / "summaries" / "job_list.jsonl", record)
        print(f"[{entry['index']:02d}] {key}: train {tid} ({limit})")
    print(f"projected sequential training time: {total_hours:.0f} GPU-hours (~{total_hours / 24:.1f} days) plus evaluation")
    if args.dry_run:
        print(f"evaluation: {len(approved)} jobs, each held until all {len(approved)} training jobs succeed")
        return 0
    if len(train_jobs) != len(approved):
        print("some runs were already submitted; evaluations not submitted here. Use --submit-evals after all complete.")
        return 0
    submit_evals(approved, "afterok:" + ":".join(train_jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
