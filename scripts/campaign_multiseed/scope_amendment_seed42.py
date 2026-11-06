#!/usr/bin/env python
"""Record the seed-42 scope amendment (user instruction, 2026-09-15) and submit its evaluations.

The user stopped the campaign at seed 42: the running seed-123 job was interrupted, and
every seed-123/2026 training job plus every evaluation job was cancelled. The
frozen manifest stays untouched. This script writes, once each:
  summaries/scoped_manifest_seed42.json   frozen manifest with only seed-42 runs launchable
  summaries/scope_amendment_seed42.json   what changed, why, cancelled jobs, hashes
and marks each cancelled run in its status.jsonl. With --submit-evals it submits one
evaluation per seed-42 run. Each job waits on afterok for the remaining seed-42 training
jobs, so no evaluation starts until all seed-42 training has succeeded.

    python scripts/campaign_multiseed/scope_amendment_seed42.py --record --cancelled <job ids...>
    python scripts/campaign_multiseed/scope_amendment_seed42.py --submit-evals
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    MANIFEST, OUT, append_jsonl, run_dir, sha256, write_once)

SCOPED = OUT / "summaries" / "scoped_manifest_seed42.json"
AMENDMENT = OUT / "summaries" / "scope_amendment_seed42.json"
SEEDS = [42]


def events(mid: str, seed: int) -> list[dict]:
    st = run_dir(mid, seed) / "status.jsonl"
    return [json.loads(x) for x in st.read_text().splitlines()] if st.is_file() else []


def record(cancelled: list[str]) -> None:
    man = json.loads(MANIFEST.read_text())
    scoped = copy.deepcopy(man)
    out_of_scope = []
    for key, run in scoped["runs"].items():
        if run["seed"] not in SEEDS and run["launchable"]:
            run["launchable"] = False
            run["out_of_scope"] = "seed-42-only amendment (user instruction, 2026-09-15)"
            done = any(e["event"] == "train_complete" for e in events(run["method"], run["seed"]))
            out_of_scope.append(dict(run_key=key, training_completed_before_amendment=done))
    scoped["run_order"] = [e for e in scoped["run_order"] if e["seed"] in SEEDS]
    scoped["scope_amendment"] = str(AMENDMENT.relative_to(ROOT))
    write_once(SCOPED, scoped)
    jobs = {}
    for line in (OUT / "summaries" / "job_list.jsonl").read_text().splitlines():
        r = json.loads(line)
        for k in ("train_job", "eval_job"):
            if r.get(k) in cancelled:
                jobs[r[k]] = dict(run_key=r["run_key"], kind=k)
    for jid, info in jobs.items():
        mid_slug, seed = info["run_key"].split("/seed_")
        rd = OUT / "runs" / mid_slug / f"seed_{seed}"
        append_jsonl(rd / "status.jsonl", dict(event=f"{info['kind'].split('_')[0]}_cancelled_by_user", job=jid,
                                               reason="seed-42-only amendment", time=time.time()))
    amend = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        instruction="User (2026-09-15): complete the work on seed 42 only; interrupt the current training "
                    "(04 seed 123 at epoch 48-49 of 50) and continue from Method 07.",
        seeds_in_scope=SEEDS,
        unchanged="every frozen setting, resolved config, source file and the frozen manifest itself",
        cancelled_jobs=jobs,
        out_of_scope_runs=out_of_scope,
        reporting="single seed (42): per-method values, no mean/SD; paired differences are single-seed "
                  "and must not be read as seed-robust",
        evaluation_policy="no evaluation starts until every seed-42 training run has completed",
        frozen_manifest_sha256=sha256(MANIFEST),
        scoped_manifest=str(SCOPED.relative_to(ROOT)),
        scoped_manifest_sha256=sha256(SCOPED))
    write_once(AMENDMENT, amend)
    print(json.dumps(dict(cancelled=len(jobs), out_of_scope=len(out_of_scope),
                          in_scope=[e["run_key"] for e in scoped["run_order"]]), indent=1))


def submit_evals() -> None:
    scoped = json.loads(SCOPED.read_text())
    pending = []
    for e in scoped["run_order"]:
        evs = events(e["method"], e["seed"])
        if not any(x["event"] == "train_complete" for x in evs):
            pending.append(json.loads((run_dir(e["method"], e["seed"]) / "submission.json").read_text())["train_job"])
    dep = ["--dependency=afterok:" + ":".join(pending), "--kill-on-invalid-dep=yes"] if pending else []
    for e in scoped["run_order"]:
        mid, seed, key = e["method"], e["seed"], e["run_key"]
        rd = run_dir(mid, seed)
        cmd = ["sbatch", "--parsable", f"--job-name=ev{mid}_s{seed}", "--time=04:00:00",
               f"--output={rd}/slurm_eval_%j.log", *dep,
               "scripts/campaign_multiseed/slurm_eval_scoped.sh", mid, str(seed)]
        eid = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
        rec = dict(run_key=key, eval_job=eid, eval_command=cmd, waits_for_training_jobs=pending,
                   submitted=time.strftime("%Y-%m-%dT%H:%M:%S"))
        write_once(rd / f"eval_submission_job{eid}.json", rec)
        append_jsonl(rd / "status.jsonl", dict(event="eval_submitted", eval_job=eid, time=time.time()))
        append_jsonl(OUT / "summaries" / "job_list.jsonl", rec)
        print(f"{key}: eval {eid}" + (f" after {len(pending)} training jobs" if pending else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--cancelled", nargs="*", default=[])
    ap.add_argument("--submit-evals", action="store_true")
    a = ap.parse_args()
    if a.record:
        record(a.cancelled)
    if a.submit_evals:
        submit_evals()
    return 0


if __name__ == "__main__":
    sys.exit(main())
