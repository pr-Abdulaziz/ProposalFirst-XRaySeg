#!/usr/bin/env python
"""Turn the batch-1 profile into the campaign schedule table.

Projections use the campaign's own iteration count (the de-leaked training set),
not the profiler's, and state the time limit each run will be submitted with.

    python scripts/campaign_pidray_bs1/summarize_profile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_pidray_bs1.submit_campaign import time_limit  # noqa: E402


def main() -> int:
    iters = C.iters_per_epoch() * C.MAX_EPOCHS
    rows, missing = [], []
    for mid, _ in C.run_order():
        path = C.OUT / "profile" / f"{C.BY_ID[mid].slug}.json"
        if not path.is_file():
            missing.append(mid)
            continue
        d = json.loads(path.read_text())
        hours = d["timing"]["cycle_s"]["mean"] * iters / 3600
        limit, limit_h = time_limit(hours)
        rows.append(dict(
            method=mid, name=C.BY_ID[mid].name, s_per_iter=d["timing"]["cycle_s"]["mean"],
            train_hours=hours, time_limit=limit, limit_hours=limit_h,
            fits_partition_limit=limit_h <= 168 and hours * 1.35 + 1 <= 168,
            eval_hours=d["inference"]["projected_hours_all_test_images"],
            peak_train_gb=d["memory_train"]["peak_allocated_mb"] / 1024,
            peak_any_gb=d["peak_allocated_mb"] / 1024,
            peak_fraction=d["peak_allocated_fraction"],
            parameters=d["parameters"]["total"], losses_finite=d["all_losses_finite"]))
    total_train = sum(r["train_hours"] for r in rows)
    total_eval = sum(r["eval_hours"] for r in rows)
    summary = dict(
        campaign=C.CAMPAIGN, batch=C.PHYSICAL_BATCH, epochs=C.MAX_EPOCHS, seeds=list(C.SEEDS),
        train_images=C.train_images(), iterations_per_run=iters, methods_profiled=len(rows),
        missing=missing, total_train_hours=total_train, total_train_days=total_train / 24,
        total_eval_hours=total_eval, grand_total_hours=total_train + total_eval,
        longest_run=max(rows, key=lambda r: r["train_hours"])["method"] if rows else None,
        max_peak_gb=max((r["peak_any_gb"] for r in rows), default=0.0),
        any_run_over_partition_limit=[r["method"] for r in rows if not r["fits_partition_limit"]],
        rows=rows)
    (C.OUT / "profile" / "schedule.json").write_text(json.dumps(summary, indent=1))
    w = max(len(r["name"]) for r in rows) if rows else 10
    print(f"{'id':>4}  {'method':<{w}}  {'s/iter':>7}  {'train h':>8}  {'limit':>9}  "
          f"{'eval h':>6}  {'peak GB':>7}")
    for r in rows:
        print(f"{r['method']:>4}  {r['name']:<{w}}  {r['s_per_iter']:7.3f}  {r['train_hours']:8.1f}  "
              f"{r['time_limit']:>9}  {r['eval_hours']:6.2f}  {r['peak_any_gb']:7.2f}")
    print(f"\ntotal training {total_train:.0f} GPU-h ({total_train / 24:.1f} days), "
          f"evaluation {total_eval:.1f} h, {iters:,} iterations per run")
    if missing:
        print(f"not yet profiled: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
