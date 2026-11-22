#!/usr/bin/env python
"""Summarise the inference-time ablations (plus the A1 196-point runs) against the full refiner and the Parent.

Two-level paired bootstrap identical to the campaign's (RNG seed 20260919, seeds then images).
Writes output/<campaign>/revision_2026-11/ablation/summary.json and prints a table.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO  # noqa: E402

from scripts.campaign_multiseed.analyze_errors_quality import predictions_path  # noqa: E402
from scripts.campaign_multiseed.campaign_common import ANN, OUT, SEEDS, run_key  # noqa: E402
from scripts.campaign_multiseed.coco_resample import APCache, ap_weighted, self_check  # noqa: E402
from scripts.campaign_multiseed.statistics_campaign import RNG_SEED  # noqa: E402

B = 2000
REV = OUT / "revision_2026-11"
VARIANTS = ["off", "level56", "laststage", "random784", "pts196", "pts392", "pts1568"]
KEYS = ("AP", "AP50", "AP75")


def variant_pred(mid, seed, v):
    d = REV / ("a1_infer196" if v == "pts196" else f"ablation/{v}") / run_key(mid, seed)
    f = sorted(glob.glob(str(d / "predictions_job*.pkl")))
    if len(f) != 1:
        raise FileNotFoundError(f"{mid} s{seed} {v}: {f}")
    return Path(f[0])


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN["test"]))
    paths = {("03", s): predictions_path("03", s) for s in SEEDS}
    for mid in ("09", "09M"):
        for s in SEEDS:
            paths[(mid, s)] = predictions_path(mid, s)
            for v in VARIANTS:
                paths[(f"{mid}:{v}", s)] = variant_pred(mid, s, v)
    N = None
    rng = np.random.default_rng(RNG_SEED)
    pt, bt = {}, {}
    W = seed_draw = None
    for (arm, s), p in paths.items():
        c = APCache(gt, p, "segm")
        assert self_check(c) < 1e-9, (arm, s)
        if W is None:
            N = len(c.img_ids)
            W = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(float)
            seed_draw = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
        one, rep = ap_weighted(c, np.ones((1, N))), ap_weighted(c, W)
        for k in KEYS:
            pt.setdefault((arm, k), {})[s] = 100 * one[k][0]
            bt.setdefault((arm, k), {})[s] = 100 * rep[k]
    arms = sorted({a for a, _ in pt})

    def arr(arm, k):
        return np.array([pt[(arm, k)][s] for s in SEEDS]), np.stack([bt[(arm, k)][s] for s in SEEDS])

    def contrast(a, b, k):
        pa, ba = arr(a, k); pb, bb = arr(b, k)
        d = pa - pb
        db = (ba - bb)[seed_draw.T, np.arange(B)].mean(axis=0)
        return dict(per_seed=d.tolist(), delta=float(d.mean()),
                    ci95=[float(x) for x in np.percentile(db, [2.5, 97.5])])

    out = dict(design=dict(B=B, rng_seed=RNG_SEED, images=N, seeds=list(SEEDS)), arms={}, contrasts={})
    for a in arms:
        out["arms"][a] = {k: dict(per_seed=arr(a, k)[0].tolist(), mean=float(arr(a, k)[0].mean())) for k in KEYS}
    print(f"{'arm':18s} {'AP':>6s} {'AP75':>6s} | vs full refiner [95% CI]        | vs Parent [95% CI]")
    for mid in ("09", "09M"):
        for v in [None] + VARIANTS:
            a = mid if v is None else f"{mid}:{v}"
            row = {}
            if v is not None:
                row["vs_full"] = {k: contrast(a, mid, k) for k in KEYS}
            row["vs_parent"] = {k: contrast(a, "03", k) for k in KEYS}
            out["contrasts"][a] = row
            f = row.get("vs_full", {}).get("AP")
            p = row["vs_parent"]["AP"]
            ftxt = f"{f['delta']:+.2f} [{f['ci95'][0]:+.2f}, {f['ci95'][1]:+.2f}]" if f else " " * 22
            print(f"{a:18s} {out['arms'][a]['AP']['mean']:6.2f} {out['arms'][a]['AP75']['mean']:6.2f} | "
                  f"{ftxt:31s} | {p['delta']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]")
    print(f"{'03 (Parent)':18s} {out['arms']['03']['AP']['mean']:6.2f} {out['arms']['03']['AP75']['mean']:6.2f}")
    (REV / "ablation" / "summary.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
