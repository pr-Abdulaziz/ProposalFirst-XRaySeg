#!/usr/bin/env python
"""Restore seeds 123 and 2026 after the seed-42-only amendment (user instruction, 2026-09-16).

The user asked to train the remaining runs for three seeds: methods 04, 07, 08, 09, 09M, 05 and 06
for seeds 123 and 2026 (01, 02, 03 already completed for those seeds before the amendment).
Nothing frozen changes: each run uses its frozen resolved config through slurm_train.sh and the
train-start guard. 04 seed 123 was interrupted after epoch 49 and resumes from its own
last_checkpoint (--resume; update accounting and RNG state are restored from the checkpoint).
Every other run starts fresh; its work directory must not exist yet.

Evaluation: no seed-123/2026 run is evaluated until every one of the 30 approved runs has completed.
Each evaluation job depends on afterok for every training job still queued (the new jobs plus any
queued seed-42 training) and evaluate_run.py re-checks all 30 runs against the frozen manifest.
Seed-42 evaluations already submitted under scope_amendment_seed42.json are left unchanged.

    python scripts/campaign_multiseed/restore_three_seeds.py --dry-run
    python scripts/campaign_multiseed/restore_three_seeds.py --submit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    MANIFEST, OUT, append_jsonl, run_dir, sha256, write_once)
from scripts.campaign_multiseed.submit_campaign import slurm_time, train_complete  # noqa: E402

METHODS = ("04", "07", "08", "09", "09M", "05", "06")
SEEDS = (123, 2026)
AMENDMENT = OUT / "summaries" / "scope_amendment_restore_3seeds.json"


def sbatch(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    man = json.loads(MANIFEST.read_text())
    if AMENDMENT.exists():
        raise SystemExit(f"ABORT: {AMENDMENT} exists; already submitted")

    plan, problems = [], []
    for e in man["run_order"]:
        mid, seed, key = e["method"], e["seed"], e["run_key"]
        if mid not in METHODS or seed not in SEEDS:
            continue
        if train_complete(mid, seed):
            problems.append(f"{key} is already complete")
            continue
        wd = Path(man["runs"][key]["work_dir"])
        resume = (wd / "last_checkpoint").is_file()
        if wd.exists() and not resume:
            problems.append(f"{key}: work_dir exists without last_checkpoint; inspect before training")
        projected = man["preflight"][mid]["timing"]["projected_train_hours"]
        plan.append(dict(index=e["index"], method=mid, seed=seed, run_key=key, resume=resume,
                         time_limit=slurm_time(2.0 * projected + 4), projected_train_hours=projected))
    if problems:
        raise SystemExit("ABORT:\n" + "\n".join(problems))
    queued = subprocess.run(["squeue", "-h", "-u", "ahalqahtani", "-o", "%i %j"], capture_output=True,
                            text=True, check=True).stdout.split("\n")
    queued_train = [l.split()[0] for l in queued if l.strip() and l.split()[1].startswith("ms")]
    for p in plan:
        print(f"[{p['index']:02d}] {p['run_key']:45s} {'RESUME' if p['resume'] else 'fresh '} time={p['time_limit']}")
    print(f"{len(plan)} training runs; queued campaign training jobs the evaluations will also wait for: {queued_train}")
    if a.dry_run:
        return 0

    new_jobs = []
    for p in plan:
        mid, seed = p["method"], p["seed"]
        rd = run_dir(mid, seed)
        cmd = ["sbatch", "--parsable", f"--job-name=ms{mid}_s{seed}", f"--time={p['time_limit']}",
               f"--output={rd}/slurm_train_%j.log", "scripts/campaign_multiseed/slurm_train.sh", mid, str(seed)]
        if p["resume"]:
            cmd.append("--resume")
        jid = sbatch(cmd)
        new_jobs.append(jid)
        rec = dict(p, train_job=jid, train_command=cmd, submitted=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   reason="restore three seeds (user instruction 2026-09-16)")
        write_once(rd / f"resubmission_job{jid}.json", rec)
        append_jsonl(rd / "status.jsonl", dict(event="resubmitted", train_job=jid, resume=p["resume"], time=time.time()))
        append_jsonl(OUT / "summaries" / "job_list.jsonl", rec)
        print(f"{p['run_key']}: train {jid}{' (resume)' if p['resume'] else ''}")

    dep = "afterok:" + ":".join(queued_train + new_jobs)
    evals = []
    for e in man["run_order"]:
        mid, seed, key = e["method"], e["seed"], e["run_key"]
        if seed not in SEEDS:
            continue
        rd = run_dir(mid, seed)
        cmd = ["sbatch", "--parsable", f"--job-name=ev{mid}_s{seed}", "--time=04:00:00",
               f"--output={rd}/slurm_eval_%j.log", f"--dependency={dep}", "--kill-on-invalid-dep=yes",
               "scripts/campaign_multiseed/slurm_eval.sh", mid, str(seed)]
        eid = sbatch(cmd)
        evals.append(dict(run_key=key, eval_job=eid))
        rec = dict(run_key=key, eval_job=eid, eval_command=cmd, dependency=dep,
                   submitted=time.strftime("%Y-%m-%dT%H:%M:%S"))
        write_once(rd / f"eval_submission_job{eid}.json", rec)
        append_jsonl(rd / "status.jsonl", dict(event="eval_submitted", eval_job=eid, time=time.time()))
        append_jsonl(OUT / "summaries" / "job_list.jsonl", rec)
        print(f"{key}: eval {eid} (after all training)")

    write_once(AMENDMENT, dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        instruction="User (2026-09-16): train the remaining runs to complete three seeds: 04, 07, 08, 09, 09M, "
                    "05, 06 x seeds 123 and 2026.",
        seeds_in_scope=[42, 123, 2026],
        supersedes_for_reporting="summaries/scope_amendment_seed42.json (seed-42 evaluations kept as submitted)",
        unchanged="every frozen setting, resolved config and source file; the frozen manifest itself",
        training_runs=[dict(p, train_job=j) for p, j in zip(plan, new_jobs)],
        evaluation_jobs=evals, evaluation_dependency=dep,
        evaluation_policy="seed-123/2026 evaluations start only after all 30 approved training runs have completed",
        frozen_manifest_sha256=sha256(MANIFEST)))
    print(f"recorded {AMENDMENT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
