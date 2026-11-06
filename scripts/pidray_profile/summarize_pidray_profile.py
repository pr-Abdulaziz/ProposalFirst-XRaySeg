#!/usr/bin/env python
"""Tabulate PIDray batch-2 profiling results (output/pidray_profile_2026-09-14/*.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.campaign_common import METHODS  # noqa: E402

OUT = ROOT / "output/pidray_profile_2026-09-14"


def main() -> int:
    L = ["# PIDray profiling at physical batch 2 (measurement only)", "",
         "Model and 512-px pipeline from each method's PIXray comparison config; data layer replaced with PIDray-30 "
         "(23,074 training images including empty ones; PIDray-A occluder masks for 06 and 07; 12 classes). "
         "Batch 2 x accumulation 8, 2 workers, one RTX 4090. Time per iteration is the mean wall time including data "
         "loading on 400 random training images (first 20 iterations excluded). Projected training = mean x 11,537 "
         "iterations x 60 epochs. Evaluation projection covers forward passes and loading for all 47,573 test images, "
         "without COCO scoring.", "",
         "| Method | s/iteration | of which data | h/epoch | Train h (60 ep) | Peak GB train | Peak GB stress | Peak GB infer | "
         "Peak % of GPU | Eval h (forward) | Other users' GPU processes |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    total = 0.0
    for m in METHODS:
        f = OUT / f"{m.slug}.json"
        if not f.is_file():
            L.append(f"| {m.id} {m.name} | missing | | | | | | | | | |")
            continue
        r = json.loads(f.read_text())
        t, p = r["timing"], r["projection"]
        total += p["train_hours_60_epochs"]
        L.append(f"| {m.id} {m.name} | {t['cycle_s']['mean']:.3f} | {t['data_wait_s']['mean']:.3f} | "
                 f"{p['hours_per_epoch']:.2f} | {p['train_hours_60_epochs']:.1f} | "
                 f"{r['memory_train']['peak_allocated_mb']/1024:.2f} | {r['memory_stress']['peak_allocated_mb']/1024:.2f} | "
                 f"{r['inference']['peak_allocated_mb']/1024:.2f} | {100*r['peak_allocated_fraction']:.0f}% | "
                 f"{r['inference']['projected_hours_all_test_images']:.2f} | "
                 f"{'yes (' + r['node'] + ')' if any(v['foreign_compute_processes'] for v in r.get('gpu_state', {}).values()) else 'none seen'} |")
    L += ["", f"Sum of projected training time, one seed per method: {total:.0f} GPU-hours ({total/24:.1f} days sequential)."]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
