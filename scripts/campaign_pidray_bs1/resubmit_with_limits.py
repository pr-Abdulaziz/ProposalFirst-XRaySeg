#!/usr/bin/env python
"""Resubmit the queued PIDray runs with realistic time limits (2026-09-22).

Measured full runs take 1.4-1.65x the stability-probe projection (01: 5.37 h vs
3.82 h; 02: ~8.0 h vs 4.92 h). The probes timed the first 1,000 iterations,
before the RPN and cascade produce many positive RoIs, so per-iteration cost
rises by ~19% over a run and then levels off. The original limits (1.35x + 1 h)
would time out every remaining run. Users cannot raise a submitted job's limit,
so the pending chain is cancelled and resubmitted; nothing about training changes.

Run 02 (job RUNNING_02) is projected to finish within a minute or two of its 8 h
limit, so a resume job is queued with afternotok: it runs only if 02 fails or
times out, and is cancelled automatically if 02 completes. The next run depends
on 02 OR the resume succeeding.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_pidray_bs1.submit_campaign import projected_hours  # noqa: E402

RUNNING_02 = "12481"
PENDING = ["12482", "12483", "12484", "12485", "12486", "12487", "12488", "12489"]
FACTOR, PAD_H = 2.0, 2          # covers the observed 1.65x with margin
LOGS = C.OUT / "logs"
TRAIN = "scripts/campaign_pidray_bs1/slurm_train.sh"


def sbatch(args):
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"sbatch failed: {out.stderr.strip()}\n{' '.join(args)}")
    return out.stdout.strip().split()[-1]


def main() -> int:
    subprocess.run(["scancel", *PENDING], check=True)
    time.sleep(2)
    left = subprocess.run(["squeue", "-h", "-j", ",".join(PENDING), "-o", "%i"],
                          capture_output=True, text=True).stdout.split()
    if left:
        raise SystemExit(f"still queued after scancel: {left}")

    resume = sbatch(["sbatch", "--job-name=pid02_s42_resume", "--time=3:00:00",
                     f"--output={LOGS}/train_02_s42_resume_%j.log",
                     f"--dependency=afternotok:{RUNNING_02}", "--kill-on-invalid-dep=yes",
                     TRAIN, "02", "42", "--resume"])
    records = [dict(method="02", role="resume only if 12481 fails or times out", job=resume,
                    time_limit="3:00:00", depends_on=f"afternotok:{RUNNING_02}")]

    order = [m for m, _ in C.run_order()]
    todo = order[order.index("03"):]
    previous = f"afterok:{RUNNING_02}?afterok:{resume}"
    for index, mid in enumerate(todo, start=order.index("03")):
        hours = projected_hours(mid)
        limit_h = min(math.ceil(hours * FACTOR + PAD_H), 168)
        node = C.CAMPAIGN_NODES[index % len(C.CAMPAIGN_NODES)]
        job = sbatch(["sbatch", f"--job-name=pid{mid}_s42", f"--time={limit_h}:00:00",
                      f"--output={LOGS}/train_{mid}_s42_%j.log", f"--nodelist={node}",
                      f"--dependency={previous}", TRAIN, mid, "42"])
        records.append(dict(method=mid, job=job, node=node, time_limit=f"{limit_h}:00:00",
                            probe_projection_h=round(hours, 2), depends_on=previous))
        previous = f"afterok:{job}"
    C.append_jsonl(C.OUT / "submissions.jsonl", dict(
        kind="resubmission_time_limits", time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cancelled=PENDING, factor=FACTOR, pad_hours=PAD_H, records=records,
        reason="measured runs take 1.4-1.65x the probe projection; limits were 1.35x + 1 h"))
    print(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
