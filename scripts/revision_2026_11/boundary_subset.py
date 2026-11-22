#!/usr/bin/env python
"""Describe the matched-object subset behind the boundary-IoU test (Table 4, H4).

Re-uses the campaign's per-object analysis files (analysis/runs/<slug>/seed_<s>_per_gt.npy: gt id, mask IoU,
Dice, boundary IoU) and the same matching rule as results_section.py: an object enters the paired test when
both experiments matched it in every seed (same class, score >= 0.5, box IoU >= 0.5). Boundary band width is
round(0.02 x image diagonal) pixels. Writes revision_2026-11/r2c5_boundary_subset.json.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.campaign_common import ANN, OUT, SEEDS, BY_ID  # noqa: E402

gt = json.loads(Path(ANN["test"]).read_text())
imgs = {i["id"]: i for i in gt["images"]}
anns = {a["id"]: a for a in gt["annotations"]}
obj = np.array(sorted(anns))
box_area = np.array([anns[o]["area"] for o in obj])                     # PIXray 'area' = box area
side = np.array([min(anns[o]["bbox"][2], anns[o]["bbox"][3]) for o in obj])
band = np.array([round(0.02 * np.hypot(imgs[anns[o]["image_id"]]["height"], imgs[anns[o]["image_id"]]["width"])) for o in obj])
size = np.where(box_area < 32 ** 2, "small", np.where(box_area < 96 ** 2, "medium", "large"))

def load(m):
    iou, biou = [], []
    for s in SEEDS:
        arr = np.load(OUT / "analysis/runs" / BY_ID[m].slug / f"seed_{s}_per_gt.npy")
        d = {int(r[0]): (r[1], r[3]) for r in arr}
        iou.append([d[o][0] if o in d else np.nan for o in obj]); biou.append([d[o][1] if o in d else np.nan for o in obj])
    return np.array(iou), np.array(biou)

I, BI = {}, {}
for m in ("03", "09", "09M"):
    I[m], BI[m] = load(m)
res = dict(n_objects=len(obj), band_px=dict(min=int(band.min()), median=float(np.median(band)), max=int(band.max())),
           size_bins_all={k: int((size == k).sum()) for k in ("small", "medium", "large")}, comparisons={})
for a in ("09", "09M"):
    both = (~np.isnan(I[a]) & ~np.isnan(I["03"])).all(0)
    dbi = np.nan_to_num(BI[a] - BI["03"]).mean(0); di = np.nan_to_num(I[a] - I["03"]).mean(0)
    excl = ~both
    never_parent = np.isnan(I["03"]).all(0)
    r = dict(matched=int(both.sum()), excluded=int(excl.sum()),
             excluded_missed_by_parent_all_seeds=int(never_parent.sum()),
             excluded_by_size={k: int((excl & (size == k)).sum()) for k in ("small", "medium", "large")},
             matched_by_size={k: int((both & (size == k)).sum()) for k in ("small", "medium", "large")},
             delta_biou_by_size={k: 100 * float(dbi[both & (size == k)].mean()) for k in ("small", "medium", "large")},
             delta_iou_by_size={k: 100 * float(di[both & (size == k)].mean()) for k in ("small", "medium", "large")},
             band_ge_half_side_share=float(((band >= side / 2) & both).sum() / both.sum()),
             band_ge_half_side_small=int(((band >= side / 2) & both & (size == "small")).sum()),
             parent_mean_biou_matched=100 * float(np.nanmean(BI["03"], 0)[both].mean()),
             parent_mean_iou_matched=100 * float(np.nanmean(I["03"], 0)[both].mean()))
    res["comparisons"][f"{a}-03"] = r
(OUT / "revision_2026-11" / "r2c5_boundary_subset.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
