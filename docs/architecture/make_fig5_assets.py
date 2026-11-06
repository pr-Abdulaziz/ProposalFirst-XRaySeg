#!/usr/bin/env python
"""Input and output thumbnails for Figure 5, from real data.

Input : PIXray test image 101627037.png (one pistol), cropped to the bag.
Output: the same crop with the predicted mask of the MLP point-refinement
        variant (method 09M, seed 42, PIXray campaign) overlaid -- the
        highest-scoring detection, mask IoU 0.949 with the ground truth.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "assets"
NAME = "101627037.png"
PRED = ROOT / ("work_dirs/pixray_multiseed_bs4_e50_2026-09-14/m09m_point_refine_mlp/"
               "seed_42/test/predictions_job11811.pkl")
RED = np.array([214, 39, 40], float)

gt = json.loads((ROOT / "data/PIXray/annotations/test.json").read_text())
img_id = next(i["id"] for i in gt["images"] if i["file_name"] == NAME)
pred = next(s for s in pickle.load(open(PRED, "rb")) if int(s["img_id"]) == img_id)
p = pred["pred_instances"]
k = int(np.argmax(p["scores"]))
rle = dict(p["masks"][k])
rle["counts"] = rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]
mask = mu.decode(rle).astype(bool)

im = np.asarray(Image.open(ROOT / "data/PIXray/test" / NAME).convert("RGB")).astype(float)
content = (im < 200).any(axis=2)
cols = np.where(content.mean(axis=0) > 0.05)[0]
rows = np.where(content.mean(axis=1) > 0.05)[0]
pad = 6
x0, x1 = max(cols.min() - pad, 0), min(cols.max() + pad, im.shape[1])
y0, y1 = max(rows.min() - pad, 0), min(rows.max() + pad, im.shape[0])

over = im.copy()
over[mask] = 0.45 * im[mask] + 0.55 * RED
def shifts(m):
    q = np.pad(m, 1)
    return q[:-2, 1:-1], q[2:, 1:-1], q[1:-1, :-2], q[1:-1, 2:]


eroded = mask.copy()
for s_ in shifts(mask):
    eroded &= s_
edge = mask & ~eroded                       # 1-pixel inner contour
for s_ in shifts(edge):                     # thicken to ~2 pixels for print
    edge = edge | (s_ & mask)
over[edge] = RED

for tag, arr in (("input", im), ("output", over)):
    t = Image.fromarray(arr[y0:y1, x0:x1].clip(0, 255).astype(np.uint8))
    t.thumbnail((560, 560), Image.LANCZOS)
    t.save(OUT / f"fig5_{tag}_pixray_test_{NAME}", optimize=True)
print(f"crop x {x0}-{x1}, y {y0}-{y1}; score {float(p['scores'][k]):.3f}; label {int(p['labels'][k])}")
