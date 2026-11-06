#!/usr/bin/env python
"""Dataset figure: one PIXray training image per class (15 panels).

Selection rule (no manual picking): for each class, one image drawn at random (fixed seed 2026, for
reproducibility) from the training images that contain exactly one annotated instance of that class
and no other object, so the class of each panel is unambiguous. The whole image is shown, letterboxed to a common panel; the instance's released
polygon is drawn as a translucent fill with an outline. Nothing is edited.

    python scripts/figures/make_dataset_class_examples.py
Writes output/figures_png/dataset/fig_pixray_class_examples.{png,pdf,json}.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data/PIXray/annotations/train.json"
IMG_DIR = ROOT / "data/PIXray/train"
OUT = ROOT / "output/figures_png/dataset"
PANEL_W, PANEL_H = 1000, 560  # common canvas (px), same 448-high aspect family as PIXray
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})


def main() -> None:
    d = json.loads(ANN.read_text())
    cats = sorted(d["categories"], key=lambda c: c["id"])
    images = {i["id"]: i for i in d["images"]}
    per_img = Counter(a["image_id"] for a in d["annotations"])
    by_cat = defaultdict(list)
    for a in d["annotations"]:
        by_cat[a["category_id"]].append(a)
    test = json.loads((ROOT / "data/PIXray/annotations/test.json").read_text())
    test_count = Counter(a["category_id"] for a in test["annotations"])

    chosen = []
    rng = random.Random(2026)
    for c in cats:
        anns = by_cat[c["id"]]
        box_areas = [a["bbox"][2] * a["bbox"][3] for a in anns]
        med = float(np.median(box_areas))
        singles = [a for a in anns if per_img[a["image_id"]] == 1]
        a = rng.choice(sorted(singles, key=lambda x: x["id"]))
        chosen.append(dict(category_id=c["id"], name=c["name"], image_id=a["image_id"],
                           file_name=images[a["image_id"]]["file_name"], annotation_id=a["id"],
                           width=images[a["image_id"]]["width"], height=images[a["image_id"]]["height"],
                           box=a["bbox"], class_median_box_area=med, single_instance_candidates=len(singles),
                           train_instances=len(anns), test_instances=test_count[c["id"]], ann=a))

    cmap = plt.get_cmap("tab20").colors
    OUT.mkdir(parents=True, exist_ok=True)

    def draw(ax, row, arr, s, ox, oy, lw):
        ax.imshow(arr)
        col = cmap[(row["category_id"] * 2) % 20]
        for poly in row["ann"]["segmentation"]:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2) * s + [ox, oy]
            ax.add_patch(Polygon(pts, closed=True, facecolor=(*col, 0.15), edgecolor="none"))
            ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor="white", linewidth=lw * 2))
            ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=col, linewidth=lw))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        ax.set_title(f"{row['category_id']}. {row['name'].replace('_', ' ')}", fontsize=6.6, pad=2)

    # (a) whole images, letterboxed to a common panel
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 2.85), gridspec_kw=dict(wspace=0.04, hspace=0.22))
    for ax, row in zip(axes.flat, chosen):
        img = Image.open(IMG_DIR / row["file_name"]).convert("RGB")
        w, h = img.size
        s = min(PANEL_W / w, PANEL_H / h)
        nw, nh = round(w * s), round(h * s)
        canvas = Image.new("RGB", (PANEL_W, PANEL_H), (255, 255, 255))
        ox, oy = (PANEL_W - nw) // 2, (PANEL_H - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BICUBIC), (ox, oy))
        draw(ax, row, np.asarray(canvas), s, ox, oy, 0.6)
        ax.set_xlim(0, PANEL_W)
        ax.set_ylim(PANEL_H, 0)
    fig.savefig(OUT / "fig_pixray_class_examples.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_pixray_class_examples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) close-ups: square crop around the same instance (box + 35% margin), original pixels
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.55), gridspec_kw=dict(wspace=0.05, hspace=0.20))
    for ax, row in zip(axes.flat, chosen):
        img = np.asarray(Image.open(IMG_DIR / row["file_name"]).convert("RGB"))
        x, y, w, h = row["box"]
        side = max(w, h) * 1.35
        cx, cy = x + w / 2, y + h / 2
        draw(ax, row, img, 1.0, 0, 0, 0.9)
        ax.set_xlim(cx - side / 2, cx + side / 2)
        ax.set_ylim(cy + side / 2, cy - side / 2)
        ax.set_facecolor("white")
    fig.savefig(OUT / "fig_pixray_class_examples_closeup.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_pixray_class_examples_closeup.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    for row in chosen:
        row.pop("ann")
    (OUT / "fig_pixray_class_examples.json").write_text(json.dumps(dict(rule=__doc__.strip(), panels=chosen), indent=1))
    for row in chosen:
        print(f"{row['category_id']:2d} {row['name']:16s} image {row['image_id']:5d} {row['file_name']} "
              f"{row['width']}x{row['height']} candidates {row['single_instance_candidates']}")


if __name__ == "__main__":
    main()
