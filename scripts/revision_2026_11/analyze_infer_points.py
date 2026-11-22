#!/usr/bin/env python
"""Compare refiners evaluated at 196 vs the published 784 inference points.

Uses the campaign's two-level bootstrap (same RNG seed and draw order as statistics_campaign.py).
Writes output/<campaign>/revision_2026-11/a1_infer196/summary.json.
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
REV = OUT / "revision_2026-11" / "a1_infer196"


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN["test"]))
    arms = {}
    for s in SEEDS:
        arms[("03", s)] = predictions_path("03", s)
        for mid in ("09", "09M"):
            arms[(mid, s)] = predictions_path(mid, s)
            new = sorted(glob.glob(str(REV / run_key(mid, s) / "predictions_job*.pkl")))
            if len(new) != 1:
                raise SystemExit(f"expected one 196-point prediction file for {mid} s{s}, found {new}")
            arms[(mid + "@196", s)] = Path(new[0])
    caches = {}
    for k, p in arms.items():
        c = APCache(gt, p, "segm")
        assert self_check(c) < 1e-9, k
        caches[k] = c
    N, S = len(caches[("03", SEEDS[0])].img_ids), len(SEEDS)
    rng = np.random.default_rng(RNG_SEED)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(float)
    seed_draw = rng.integers(0, S, size=(B, S))
    names = sorted({k[0] for k in arms})
    pt, keys = {}, ("AP", "AP50", "AP75")
    bt = {}
    for n in names:
        for key in keys:
            pt[(n, key)] = np.array([100 * ap_weighted(caches[(n, s)], np.ones((1, N)))[key][0] for s in SEEDS])
        r = [ap_weighted(caches[(n, s)], W) for s in SEEDS]
        for key in keys:
            bt[(n, key)] = np.stack([100 * x[key] for x in r])                      # [S, B]

    def two_level(a):
        return a[seed_draw.T, np.arange(B)].mean(axis=0)

    out = dict(design=dict(B=B, rng_seed=RNG_SEED, images=N, seeds=list(SEEDS),
                           note="CI: two-level paired bootstrap (seeds, then images), percentile"),
               arms={}, contrasts={})
    for n in names:
        out["arms"][n] = {k: dict(per_seed=pt[(n, k)].tolist(), mean=float(pt[(n, k)].mean()),
                                  sd=float(pt[(n, k)].std(ddof=1))) for k in keys}
    for a, b in [("09@196", "09"), ("09M@196", "09M"), ("09@196", "03"), ("09M@196", "03"),
                 ("09M@196", "09@196"), ("09", "03"), ("09M", "03"), ("09M", "09")]:
        row = {}
        for k in keys:
            d = pt[(a, k)] - pt[(b, k)]
            db = two_level(bt[(a, k)] - bt[(b, k)])
            row[k] = dict(per_seed=d.tolist(), delta=float(d.mean()),
                          ci95=[float(x) for x in np.percentile(db, [2.5, 97.5])],
                          ci90=[float(x) for x in np.percentile(db, [5, 95])])
        out["contrasts"][f"{a} - {b}"] = row
        m = row["AP"]
        print(f"{a:8s} - {b:7s}: {m['delta']:+.2f} [{m['ci95'][0]:+.2f}, {m['ci95'][1]:+.2f}] "
              f"seeds {' / '.join(f'{x:+.2f}' for x in m['per_seed'])}")
    for n in names:
        print(f"{n:8s} mask AP {out['arms'][n]['AP']['mean']:.2f} +- {out['arms'][n]['AP']['sd']:.2f}  "
              f"AP50 {out['arms'][n]['AP50']['mean']:.1f}  AP75 {out['arms'][n]['AP75']['mean']:.1f}")
    (REV / "summary.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
