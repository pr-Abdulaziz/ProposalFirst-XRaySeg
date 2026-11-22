#!/usr/bin/env python
"""Why are the Table 2 percentile intervals off-centre?

Regenerates the campaign's two-level bootstrap replicates for mask AP exactly (same RNG seed, same draw
order as statistics_campaign.py), checks that they reproduce the published 95% intervals, then reports
for every experiment and for the H1/H2 differences:
  - bootstrap mean and bias (mean of replicates - point estimate)
  - percentile (published), basic and BCa 95% intervals
  - BCa acceleration from a leave-one-image-out jackknife (same image removed in all seeds)
  - image-only (seeds fixed) and seed-only (images fixed) percentile intervals.
Writes output/<campaign>/revision_2026-11/a3_bootstrap/intervals.json. Read-only otherwise.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO  # noqa: E402

from scripts.campaign_multiseed.analyze_errors_quality import predictions_path  # noqa: E402
from scripts.campaign_multiseed.campaign_common import ANN, METHODS, OUT, SEEDS  # noqa: E402
from scripts.campaign_multiseed.coco_resample import APCache, ap_weighted, self_check  # noqa: E402
from scripts.campaign_multiseed.statistics_campaign import RNG_SEED  # noqa: E402

B = 2000
DEST = OUT / "revision_2026-11" / "a3_bootstrap"
PAIRS = [("09", "03"), ("09M", "03"), ("09M", "09")]


def pct(x, lo=2.5, hi=97.5):
    return [float(v) for v in np.percentile(x, [lo, hi])]


def bca(boot, theta, jack):
    z0 = norm.ppf(np.clip(np.mean(boot < theta), 1e-6, 1 - 1e-6))
    d = jack.mean() - jack
    acc = float(np.sum(d ** 3) / (6.0 * np.sum(d ** 2) ** 1.5))
    out = []
    for q in (0.025, 0.975):
        z = norm.ppf(q)
        out.append(float(np.percentile(boot, 100 * norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))))
    return out, float(z0), acc


def summarise(theta, boot, jack, img_only, seed_only):
    p = pct(boot)
    return dict(point=float(theta), boot_mean=float(boot.mean()), bias=float(boot.mean() - theta),
                boot_sd=float(boot.std()), percentile=p, basic=[2 * theta - p[1], 2 * theta - p[0]],
                bca=(r := bca(boot, theta, jack))[0], bca_z0=r[1], bca_accel=r[2],
                frac_below_point=float(np.mean(boot < theta)),
                image_only_percentile=pct(img_only), image_only_bias=float(img_only.mean() - theta),
                seed_only_range=[float(seed_only.min()), float(seed_only.max())])


def main() -> int:
    t0 = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN["test"]))
    ids = [m.id for m in METHODS]
    caches = {}
    for mid in ids:
        for s in SEEDS:
            c = APCache(gt, predictions_path(mid, s), "segm")
            assert self_check(c) < 1e-9
            caches[(mid, s)] = c
    N, S = len(caches[(ids[0], SEEDS[0])].img_ids), len(SEEDS)
    rng = np.random.default_rng(RNG_SEED)          # identical draw order to statistics_campaign.py
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(float)
    seed_draw = rng.integers(0, S, size=(B, S))
    jackW = 1.0 - np.eye(N)
    print(f"caches ({time.time() - t0:.0f} s)", flush=True)

    point, boot, jack = {}, {}, {}
    for mid in ids:
        point[mid] = np.array([100 * ap_weighted(caches[(mid, s)], np.ones((1, N)))["AP"][0] for s in SEEDS])
        boot[mid] = np.stack([100 * ap_weighted(caches[(mid, s)], W)["AP"] for s in SEEDS])        # [S, B]
        jack[mid] = np.stack([np.concatenate([100 * ap_weighted(caches[(mid, s)], jackW[i:i + 300])["AP"]
                                              for i in range(0, N, 300)]) for s in SEEDS])      # [S, N]
        print(f"{mid} ({time.time() - t0:.0f} s)", flush=True)

    published = json.loads((OUT / "statistics" / "statistics.json").read_text())["methods"]
    multisets = list(combinations_with_replacement(range(S), S))

    def two_level(arr):                       # [S, B] -> [B]: resample seeds, then images
        return arr[seed_draw.T, np.arange(B)].mean(axis=0)

    res = dict(design=dict(B=B, rng_seed=RNG_SEED, images=N, seeds=list(SEEDS),
                           seed_multisets=len(multisets), statistic="mean over seeds of COCO mask AP"),
               methods={}, differences={})
    for mid in ids:
        r = summarise(point[mid].mean(), two_level(boot[mid]), jack[mid].mean(axis=0), boot[mid].mean(axis=0),
                      np.array([point[mid][list(m)].mean() for m in multisets]))
        r["published_ci95"] = published[mid]["mask_AP"]["ci95"]
        r["reproduces_published"] = bool(np.allclose(r["percentile"], r["published_ci95"], atol=1e-9))
        res["methods"][mid] = r
    for a, b in PAIRS:
        d = boot[a] - boot[b]
        res["differences"][f"{a}-{b}"] = summarise(
            (point[a] - point[b]).mean(), two_level(d), (jack[a] - jack[b]).mean(axis=0), d.mean(axis=0),
            np.array([(point[a] - point[b])[list(m)].mean() for m in multisets]))
    (DEST / "intervals.json").write_text(json.dumps(res, indent=1))
    for k, r in list(res["methods"].items()) + list(res["differences"].items()):
        print(f"{k:5s} pt {r['point']:6.2f} bias {r['bias']:+.2f} pct {np.round(r['percentile'], 2)} "
              f"basic {np.round(r['basic'], 2)} bca {np.round(r['bca'], 2)} img-only bias {r['image_only_bias']:+.2f} "
              f"repro {r.get('reproduces_published', '-')}", flush=True)
    return 0 if all(r["reproduces_published"] for r in res["methods"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
