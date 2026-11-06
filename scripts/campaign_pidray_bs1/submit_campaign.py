#!/usr/bin/env python
"""Submit the PIDray-30 batch-1 campaign to SLURM, in the fixed run order.

Training jobs are chained with ``--dependency=afterok`` so they run one after the
other and a failure stops the chain instead of silently continuing. The QOS
allows one running job per account, so the chain is also what the scheduler would
do anyway; the dependency makes the order explicit and auditable.

Each job's time limit comes from the batch-1 profile in
output/<campaign>/profile/, with a margin, capped at the partition limit.
Runs are pinned round-robin to CAMPAIGN_NODES (user instruction, 2026-09-21).
That spreads where the work lands; it cannot make runs concurrent, because QOS
student permits one running job per account. --no-pin lets SLURM choose instead,
which avoids waiting for a specific node that another user is occupying.
Evaluations are submitted separately, after every training run has completed.

    python scripts/campaign_pidray_bs1/submit_campaign.py [--dry] [--from <method>] [--no-pin]
    python scripts/campaign_pidray_bs1/submit_campaign.py --submit-evals [--dry]
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

PROFILE = C.OUT / "profile"
PARTITION_LIMIT_H = 168          # main: 7-00:00:00
# Full runs measured 1.4-1.65x the probe projection (per-iteration cost rises as the
# RPN and cascade produce more positive RoIs), so limits are 2x the projection + 2 h.
MARGIN = 2.0
LOGS = C.OUT / "logs"


def projected_hours(mid: str) -> float | None:
    """Hours for a full run, from whichever measurement this campaign has.

    The stability probes run the frozen config itself, so their per-iteration
    time is the better estimate; the standalone profiler is the fallback.
    """
    iters = C.iters_per_epoch() * C.MAX_EPOCHS
    probe = C.OUT / "preflight" / f"{C.BY_ID[mid].slug}.json"
    if probe.is_file():
        d = json.loads(probe.read_text())
        if d.get("status") == "PASS" and d.get("variant") in (None, ["campaign protocol"]):
            return d["probe"]["s_per_iter"] * iters / 3600
    path = PROFILE / f"{C.BY_ID[mid].slug}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())["timing"]["cycle_s"]["mean"] * iters / 3600


def time_limit(hours: float | None) -> tuple[str, float]:
    if hours is None:
        return "168:00:00", PARTITION_LIMIT_H
    want = min(math.ceil(hours * MARGIN + 2), PARTITION_LIMIT_H)
    return f"{want}:00:00", want


def sbatch(args: list[str], dry: bool) -> str | None:
    print(" ".join(args), flush=True)
    if dry:
        return None
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"sbatch failed: {out.stderr.strip()}")
    return out.stdout.strip().split()[-1]


def submit_training(dry: bool, start_from: str | None) -> list[dict]:
    manifest = json.loads(C.MANIFEST.read_text())
    LOGS.mkdir(parents=True, exist_ok=True)
    order = C.run_order()
    if start_from:
        ids = [m for m, _ in order]
        order = order[ids.index(start_from):]
    submitted, previous = [], None
    pin = "--no-pin" not in sys.argv
    for index, (mid, seed) in enumerate(order):
        run = manifest["runs"][C.run_key(mid, seed)]
        if not run["launchable"]:
            continue
        wd = Path(run["work_dir"])
        if wd.exists() and list(wd.glob("*.pth")):
            raise SystemExit(f"{wd} already holds checkpoints; inspect that attempt first")
        hours = projected_hours(mid)
        limit, limit_h = time_limit(hours)
        node = C.CAMPAIGN_NODES[index % len(C.CAMPAIGN_NODES)] if pin else None
        args = ["sbatch", f"--job-name=pid{mid}_s{seed}", f"--time={limit}",
                f"--output={LOGS}/train_{mid}_s{seed}_%j.log"]
        if node:
            args.append(f"--nodelist={node}")
        if previous:
            args.append(f"--dependency=afterok:{previous}")
        args += ["scripts/campaign_pidray_bs1/slurm_train.sh", mid, str(seed)]
        job = sbatch(args, dry)
        submitted.append(dict(method=mid, seed=seed, job=job, time_limit=limit,
                              projected_hours=hours, limit_hours=limit_h,
                              node=node, depends_on=previous))
        previous = job or "DRY"
    return submitted


def submit_evals(dry: bool) -> list[dict]:
    manifest = json.loads(C.MANIFEST.read_text())
    LOGS.mkdir(parents=True, exist_ok=True)
    out = []
    for mid, seed in C.run_order():
        if not manifest["runs"][C.run_key(mid, seed)]["launchable"]:
            continue
        args = ["sbatch", f"--job-name=pev{mid}_s{seed}", "--time=06:00:00",
                f"--output={LOGS}/eval_{mid}_s{seed}_%j.log",
                "scripts/campaign_pidray_bs1/slurm_eval.sh", mid, str(seed)]
        out.append(dict(method=mid, seed=seed, job=sbatch(args, dry)))
    return out


def main() -> int:
    C.require_leakage_policy()
    if not C.MANIFEST.is_file():
        raise SystemExit("freeze the campaign manifest first")
    dry = "--dry" in sys.argv
    start_from = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    if "--submit-evals" in sys.argv:
        records = submit_evals(dry)
        kind = "evaluations"
    else:
        records = submit_training(dry, start_from)
        kind = "training"
        total = sum(r["projected_hours"] or 0 for r in records)
        print(f"\nprojected {total:.0f} GPU-hours ({total / 24:.1f} days) for {len(records)} runs",
              flush=True)
    if not dry:
        C.append_jsonl(C.OUT / "submissions.jsonl",
                       dict(kind=kind, records=records, order=[r["method"] for r in records]))
    print(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
