#!/usr/bin/env python
"""Statistics for the multi-seed campaign: per-method summaries, paired comparisons, per-class and
per-IoU effects, and difficulty strata, with two-level bootstrap intervals and permutation tests.

Design (fixed in this file; see the Methodology, Section "Statistical analysis"):
- Point estimates: official COCOeval on each run; method value = mean over the three seeds.
- Two-level paired bootstrap, B replicates: resample the three seeds with replacement, and the 1,486
  test images with replacement (one image resample shared by all methods and seeds in a replicate);
  the statistic is recomputed from cached COCO matches (exactly reproduces official AP at unit
  weights). Percentile intervals: 95% (and 90% for equivalence).
- Paired permutation test, J replicates: per test image, the two methods' detections are swapped
  with probability 1/2 (same swap pattern for every seed); statistic = mean over seeds of the AP
  difference; two-sided p = (1 + #|perm| >= |obs|) / (1 + J).
- Equivalence (09 vs 09M): TOST with margin DELTA mask-AP points: equivalent if the 90% interval lies
  inside (-DELTA, +DELTA).
- Holm-Bonferroni within each hypothesis family at alpha = 0.05.
- Strata: image crowding by AP on image subsets; object occlusion and thinness by object-level
  measures (detection coverage at score >= 0.5 with box IoU >= 0.5, mean mask IoU of matched objects,
  mask recall at IoU >= 0.75) from the analysis step, bootstrapped over images.

    python scripts/campaign_multiseed/statistics_campaign.py [--B 2000] [--J 5000]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.analyze_errors_quality import predictions_path  # noqa: E402
from scripts.campaign_multiseed.campaign_common import ANN, BY_ID, METHODS, OUT, SEEDS  # noqa: E402
from scripts.campaign_multiseed.coco_resample import APCache, UnionCache, ap_weighted, self_check  # noqa: E402

ST = OUT / "statistics"
AN = OUT / "analysis"
RNG_SEED = 20260919
DELTA = 1.0          # equivalence margin, mask AP points
ALPHA = 0.05
COMPARISONS = [      # (A, B, family, hypothesis, label)
    ("09", "03", "primary", "H1", "Transformer refiner vs parent"),
    ("09M", "03", "primary", "H2", "MLP refiner vs parent"),
    ("09M", "09", "equivalence", "H3", "MLP vs transformer refiner"),
    ("02", "01", "architecture", "H6", "Cascade vs Mask R-CNN"),
    ("03", "02", "architecture", "H6", "AttFPN vs FPN"),
    ("04", "03", "architecture", "H6", "Recall-enhanced RPN"),
    ("08", "03", "architecture", "H6", "PIDray R0/R1 gate"),
    ("07", "04", "occluder", "H7", "Bilayer FCN + occluder masks"),
    ("06", "05", "occluder", "H7", "Occ bilayer SAM + occluder masks"),
]
CROWD_BINS = [("1-2 objects", 1, 2), ("3-4 objects", 3, 4), ("5+ objects", 5, 99)]
OCC_BINS = [("no occluder", -1.0, 0.0), ("light (<=0.25)", 0.0, 0.25), ("heavy (>0.25)", 0.25, 1.01)]
FILL_BINS = [("thin (<0.30)", 0.0, 0.30), ("medium (0.30-0.60)", 0.30, 0.60), ("compact (>=0.60)", 0.60, 1.01)]


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def ci(x, level=0.95):
    lo, hi = np.percentile(x, [50 * (1 - level), 100 - 50 * (1 - level)])
    return [float(lo), float(hi)]


def object_attributes(gt: COCO):
    """Per test object: image id, occluder covered fraction (PIXray-A), mask/box fill ratio."""
    occ = json.loads(Path(ANN["occ_test"]).read_text())
    occ_by = {}
    for a in occ["annotations"]:
        occ_by.setdefault(a["image_id"], []).append(a)
    attrs = {}
    for iid in gt.getImgIds():
        info = gt.loadImgs(iid)[0]; h, w = info["height"], info["width"]
        anns = gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=None))
        cands = occ_by.get(iid, [])
        for a in anns:
            m = gt.annToMask(a).astype(bool)
            fill = float(m.sum()) / max(a["bbox"][2] * a["bbox"][3], 1e-6)
            match = [o for o in cands if o["category_id"] == a["category_id"]
                     and np.allclose(o["bbox"], a["bbox"], atol=0.05)]
            if not match:
                cov = np.nan                          # no PIXray-A counterpart found
            else:
                bo = [p for p in (match[0].get("bg_object_segmentation") or []) if len(p) >= 6]
                if bo:
                    om = mask_util.decode(mask_util.merge(mask_util.frPyObjects(bo, h, w))).astype(bool)
                    cov = float(np.logical_and(m, om).sum()) / max(m.sum(), 1)
                else:
                    cov = -0.5                        # no occluder annotated
            attrs[a["id"]] = dict(image_id=iid, cov=cov, fill=fill)
    return attrs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--J", type=int, default=5000)
    a = ap.parse_args()
    t0 = time.time()
    ST.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN["test"]))
    ids = [m.id for m in METHODS]
    caches = {t: {} for t in ("segm", "bbox")}
    for t in caches:
        for mid in ids:
            for s in SEEDS:
                c = APCache(gt, predictions_path(mid, s), t)
                err = self_check(c)
                assert err < 1e-9, f"{mid} s{s} {t}: cache does not reproduce official AP ({err})"
                caches[t][(mid, s)] = c
    img_ids = caches["segm"][(ids[0], SEEDS[0])].img_ids
    N, S = len(img_ids), len(SEEDS)
    cat_names = [c["name"] for c in gt.loadCats(caches["segm"][(ids[0], SEEDS[0])].cat_ids)]
    print(f"caches built and verified ({time.time() - t0:.0f} s)", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=a.B).astype(float)          # [B, N]
    seed_draw = rng.integers(0, S, size=(a.B, S))                                    # [B, S]
    ones = np.ones((1, N))

    # point estimates and bootstrap replicates: [method][type] -> arrays
    point, boot = {}, {}
    for t in ("segm", "bbox"):
        for mid in ids:
            for si, s in enumerate(SEEDS):
                c = caches[t][(mid, s)]
                pt, bt = ap_weighted(c, ones), ap_weighted(c, W)
                point.setdefault((t, mid), []).append({k: v[0] for k, v in pt.items()})
                boot.setdefault((t, mid), []).append(bt)
    print(f"bootstrap replicates done ({time.time() - t0:.0f} s)", flush=True)

    def seed_mean_boot(t, mid, key):
        arr = np.stack([b[key] for b in boot[(t, mid)]], axis=0)                     # [S, B, ...]
        return np.mean(arr[seed_draw.T, np.arange(a.B)], axis=0)                      # [B, ...]

    methods_out = {}
    for mid in ids:
        rec = dict(name=BY_ID[mid].name)
        for t, lab in (("segm", "mask"), ("bbox", "box")):
            for key in ("AP", "AP50", "AP75"):
                vals = [100 * p[key] for p in point[(t, mid)]]
                b = 100 * seed_mean_boot(t, mid, key)
                rec[f"{lab}_{key}"] = dict(per_seed=dict(zip(map(str, SEEDS), vals)), mean=float(np.mean(vals)),
                                           sd=float(np.std(vals, ddof=1)), ci95=ci(b), boot_sd=float(np.std(b)))
        pc_point = np.mean([p["per_class"] for p in point[("segm", mid)]], axis=0)
        rec["mask_per_class"] = {n: float(100 * v) for n, v in zip(cat_names, pc_point)}
        methods_out[mid] = rec

    comps = []
    for A, Bm, fam, hyp, lab in COMPARISONS:
        row = dict(a=A, b=Bm, family=fam, hypothesis=hyp, label=lab)
        for t, tl in (("segm", "mask"), ("bbox", "box")):
            for key in ("AP", "AP50", "AP75"):
                d_seed = [100 * (pa[key] - pb[key]) for pa, pb in zip(point[(t, A)], point[(t, Bm)])]
                db = 100 * (seed_mean_boot(t, A, key) - seed_mean_boot(t, Bm, key))
                row[f"{tl}_{key}"] = dict(delta=float(np.mean(d_seed)), per_seed=dict(zip(map(str, SEEDS), d_seed)),
                                          ci95=ci(db), ci90=ci(db, 0.90), boot_sd=float(np.std(db)))
        # H5: is the gain larger at strict IoU?
        g75 = seed_mean_boot("segm", A, "AP75") - seed_mean_boot("segm", Bm, "AP75")
        g50 = seed_mean_boot("segm", A, "AP50") - seed_mean_boot("segm", Bm, "AP50")
        obs = row["mask_AP75"]["delta"] - row["mask_AP50"]["delta"]
        row["mask_gain_AP75_minus_AP50"] = dict(delta=obs, ci95=ci(100 * (g75 - g50)))
        # per-class deltas (mask)
        pcA = seed_mean_boot("segm", A, "per_class"); pcB = seed_mean_boot("segm", Bm, "per_class")
        dpt = np.mean([pa["per_class"] - pb["per_class"] for pa, pb in zip(point[("segm", A)], point[("segm", Bm)])], axis=0)
        row["mask_per_class"] = {n: dict(delta=float(100 * dpt[k]), ci95=ci(100 * (pcA[:, k] - pcB[:, k])))
                                 for k, n in enumerate(cat_names)}
        # permutation test on mask AP (same swap pattern for all seeds)
        swaps = rng.integers(0, 2, size=(a.J, N)).astype(float)
        perm = np.zeros(a.J)
        for s in SEEDS:
            u = UnionCache(caches["segm"][(A, s)], caches["segm"][(Bm, s)])
            for lo in range(0, a.J, 500):
                pa, pb = u.ap_pair(swaps[lo:lo + 500])
                perm[lo:lo + 500] += (pa - pb) / S
        obs_d = row["mask_AP"]["delta"] / 100
        row["mask_AP"]["p_perm"] = float((1 + np.sum(np.abs(perm) >= abs(obs_d) - 1e-12)) / (1 + a.J))
        if fam == "equivalence":
            lo90, hi90 = row["mask_AP"]["ci90"]
            row["mask_AP"]["tost_margin"] = DELTA
            row["mask_AP"]["equivalent"] = bool(-DELTA < lo90 and hi90 < DELTA)
        comps.append(row)
        print(f"{lab}: d={row['mask_AP']['delta']:+.2f} CI95 {row['mask_AP']['ci95']} p={row['mask_AP']['p_perm']:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    for fam in ("primary", "architecture", "occluder"):
        fam_rows = {i: r["mask_AP"]["p_perm"] for i, r in enumerate(comps) if r["family"] == fam}
        for i, p in holm(fam_rows).items():
            comps[i]["mask_AP"]["p_holm"] = p
            comps[i]["mask_AP"]["significant"] = bool(p < ALPHA)

    # AP by IoU threshold (mask) for every method, and deltas vs parent
    by_iou = {}
    for mid in ids:
        per_t = []
        for ti in range(10):
            vals = []
            for c in (caches["segm"][(mid, s)] for s in SEEDS):
                q = ap_weighted_threshold(c, ones, ti)
                vals.append(100 * q[0])
            per_t.append(float(np.mean(vals)))
        by_iou[mid] = per_t

    # crowding strata (image subsets)
    n_obj = np.array([len(gt.getAnnIds(imgIds=i, iscrowd=None)) for i in img_ids])
    strata = {"crowding": {}, "occlusion": {}, "thinness": {}}
    for lab, lo, hi in CROWD_BINS:
        sel = ((n_obj >= lo) & (n_obj <= hi)).astype(float)
        res = {}
        for mid in ids:
            pts = [100 * ap_weighted(caches["segm"][(mid, s)], sel[None, :])["AP"][0] for s in SEEDS]
            bt = np.stack([100 * ap_weighted(caches["segm"][(mid, s)], W[:400] * sel[None, :])["AP"] for s in SEEDS])
            res[mid] = dict(mean=float(np.mean(pts)), per_seed=pts, boot=bt.mean(0).tolist())
        strata["crowding"][lab] = dict(images=int(sel.sum()), objects=int((n_obj * sel).sum()),
                                       methods={m: {k: v for k, v in r.items() if k != "boot"} for m, r in res.items()},
                                       delta_vs_parent={m: dict(delta=res[m]["mean"] - res["03"]["mean"],
                                                                ci95=ci(np.array(res[m]["boot"]) - np.array(res["03"]["boot"])))
                                                        for m in ids if m != "03"})
    print(f"crowding strata done ({time.time() - t0:.0f} s)", flush=True)

    # object-level strata from the analysis step
    attrs = object_attributes(gt)
    img_index = {i: k for k, i in enumerate(img_ids)}
    obj_ids = np.array(sorted(attrs))
    obj_img = np.array([img_index[attrs[o]["image_id"]] for o in obj_ids])
    cov = np.array([attrs[o]["cov"] for o in obj_ids]); fill = np.array([attrs[o]["fill"] for o in obj_ids])
    per_obj = {}
    for mid in ids:
        per_seed = []
        for s in SEEDS:
            arr = np.load(AN / "runs" / BY_ID[mid].slug / f"seed_{s}_per_gt.npy")
            d = {int(r[0]): r[1] for r in arr}
            iou = np.array([d.get(int(o), np.nan) for o in obj_ids])
            per_seed.append(iou)
        per_obj[mid] = np.stack(per_seed)                     # [S, O] mask IoU (NaN = not detected)
    Wobj = W[:, obj_img]                                       # [B, O]

    def obj_stats(iou_s, sel, w):
        det = ~np.isnan(iou_s)
        iouz = np.where(det, iou_s, 0.0)
        ws = w * sel
        cov_ = (ws * det).sum(-1) / np.maximum(ws.sum(-1), 1e-12)
        miou = (ws * iouz).sum(-1) / np.maximum((ws * det).sum(-1), 1e-12)
        r75 = (ws * (iouz >= 0.75)).sum(-1) / np.maximum(ws.sum(-1), 1e-12)
        return cov_, miou, r75

    for kind, bins, values in (("occlusion", OCC_BINS, cov), ("thinness", FILL_BINS, fill)):
        for lab, lo, hi in bins:
            if kind == "occlusion" and lo < 0:
                sel = (values == -0.5).astype(float)                      # no occluder
            elif kind == "occlusion":
                sel = ((values > lo) & (values <= hi)).astype(float)      # covered fraction in (lo, hi]
            else:
                sel = ((values >= lo) & (values < hi)).astype(float)      # fill ratio in [lo, hi)
            entry = dict(objects=int(sel.sum()), methods={}, delta_vs_parent={})
            vals = {}
            for mid in ids:
                pts = [obj_stats(per_obj[mid][si], sel, np.ones_like(sel)) for si in range(S)]
                bts = [obj_stats(per_obj[mid][si][None, :], sel[None, :], Wobj) for si in range(S)]
                vals[mid] = [np.mean([p[k] for p in pts]) for k in range(3)], [np.mean([b[k] for b in bts], 0) for k in range(3)]
                entry["methods"][mid] = dict(coverage=100 * vals[mid][0][0], mean_iou=100 * vals[mid][0][1],
                                             mask_recall_075=100 * vals[mid][0][2])
            for mid in ids:
                if mid == "03":
                    continue
                entry["delta_vs_parent"][mid] = {
                    name: dict(delta=float(100 * (vals[mid][0][k] - vals["03"][0][k])),
                               ci95=ci(100 * (vals[mid][1][k] - vals["03"][1][k])))
                    for k, name in enumerate(("coverage", "mean_iou", "mask_recall_075"))}
            strata[kind][lab] = entry
    print(f"object strata done ({time.time() - t0:.0f} s)", flush=True)

    out = dict(design=dict(B=a.B, J=a.J, rng_seed=RNG_SEED, equivalence_margin=DELTA, alpha=ALPHA,
                           seeds=list(SEEDS), images=N, bins=dict(crowding=CROWD_BINS, occlusion=OCC_BINS, thinness=FILL_BINS),
                           note="bin edges fixed in code before stratum results were computed"),
               classes=cat_names, methods=methods_out, comparisons=comps, mask_ap_by_iou=by_iou, strata=strata,
               runtime_s=time.time() - t0)
    (ST / "statistics.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {ST / 'statistics.json'} ({time.time() - t0:.0f} s)")
    return 0


def ap_weighted_threshold(cache, w, ti):
    """Mask AP at a single IoU threshold index (0 = 0.50 ... 9 = 0.95) for weights w [R, N]."""
    from scripts.campaign_multiseed.coco_resample import _precision_table
    R, K = w.shape[0], len(cache.cats)
    q = np.full((R, K, 101), np.nan)
    for k, c in enumerate(cache.cats):
        if c is None:
            continue
        npig = w @ c.n_gt
        q[:, k] = _precision_table(c.tp[ti], c.fp[ti], w[:, c.img], npig)
    return np.nanmean(q, axis=(1, 2))


if __name__ == "__main__":
    sys.exit(main())
