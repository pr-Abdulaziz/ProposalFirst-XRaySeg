#!/usr/bin/env python
"""Qualitative and per-instance comparison suite from the FINAL PIXray campaign
(pixray_multiseed_bs4_e50_2026-09-14). Every row is picked by a fixed rule; nothing is hand-chosen.

Figures, written to output/pixray_multiseed_bs4_e50_2026-09-14/figures/qualitative/:
  fig_qualitative_final         main paper   GT + 03, 07, 08, 09, 09M; rows I1-I3, F1-F3
  fig_qualitative_all_methods   supplement   GT + all ten methods; rows I1-I3, F1-F5
  fig_boundary_error_maps       supplement   pixel errors (correct / missed / extra) for I1-I3, F1
  fig_per_class_representative  supplement   one representative object per class (median gain)
  fig_gain_by_size_shape        supplement   IoU change vs object size and elongation, all seeds

Protocol (identical to analyze_errors_quality.py, so every number equals the paper's mask-quality
statistics): predictions with score >= 0.50; one-to-one same-class matching at box IoU >= 0.50;
the value under a panel is the matched mask IoU of the focus object, or why it has none. Panels
draw exactly those >= 0.50 masks. Displayed seed: 42. Selections must also hold in seeds 123 and
2026, so no row is a single-seed accident.

Candidates: GT with box >= 24 px on both sides in images holding <= 6 GT. Rules use the five key
methods (03, 07, 08, 09, 09M); I1-I3, F1, F3 also require a match by all five in all three seeds.
  I1 Improvement      : largest min(09,09M) - max(03,07,08), > 0 in every seed, ours >= 0.70
  I2 Improvement, hard: same, among the 4 lowest-AP classes where 09M's class AP beats 03's
  I3 Improvement      : same, new image and new class
  F1 Regression       : both variants below their own base (03) in every seed; most negative
                        max(09,09M) - 03 in seed 42
  F2 Missed by all    : GT unmatched by all five in every seed; largest box
  F3 Hardest class    : Dart GT with the lowest best-of-five IoU in seed 42 (shared failure)
  F4 Class confusion  : GT neither variant matches in any seed, both giving a wrong-class detection
                        on it in seed 42 (box IoU >= 0.5, not another object's own detection)
  F5 False positive   : 09M seed-42 detection (score >= 0.5, box >= 16 px) overlapping no GT
                        (box IoU < 0.1); highest score
  Per class           : object matched by 03/09/09M in every seed whose seed-mean IoU change
                        (09M - 03) is closest to its class median: representative, not best
Ties: lower image id, then annotation id. Reads predictions only.

    python scripts/campaign_multiseed/make_qualitative_final.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mu  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CAMP = ROOT / "output/pixray_multiseed_bs4_e50_2026-09-14"
WORK = ROOT / "work_dirs/pixray_multiseed_bs4_e50_2026-09-14"
OUT = CAMP / "figures/qualitative"
SEEDS, SHOW, SCORE = (42, 123, 2026), 42, 0.50
SLUG = {"01": "m01_mask_rcnn_r101_fpn", "02": "m02_cascade_r101_fpn", "03": "m03_cascade_r101_attfpn",
        "04": "m04_cascade_attfpn_recall_rpn", "05": "m05_rsprompter_sam_lora", "06": "m06_occ_bilayer_sam",
        "07": "m07_bilayer_fcn_recall_rpn", "08": "m08_pidray_r0r1", "09": "m09_point_refine_transformer",
        "09M": "m09m_point_refine_mlp"}
KEY_COLS = [("03", "Parent\n(Cascade + AttFPN)"), ("07", "Bilayer FCN\n(best prior)"),
            ("08", "Parent +\nPIDray R0/R1"), ("09", "Ours:\ntransformer refiner"),
            ("09M", "Ours:\nMLP refiner")]
ALL_COLS = [("01", "Mask R-CNN"), ("02", "Cascade\n(FPN)"), ("03", "Cascade +\nAttFPN (base)"),
            ("04", "+ Recall\nRPN"), ("05", "RSPrompter\n(SAM-LoRA)"), ("06", "Occ bilayer\nSAM"),
            ("07", "Bilayer FCN\n(best prior)"), ("08", "PIDray\nR0/R1"), ("09", "Ours:\nTransformer"),
            ("09M", "Ours:\nMLP")]
KEY = [m for m, _ in KEY_COLS]
OURS, BASES = ("09", "09M"), ("03", "07", "08")
PALETTE = {"Bat": "#e6194B", "Battery": "#42d4f4", "Dart": "#f032e6", "Fireworks": "#f58231", "Gun": "#911eb4",
           "Hammer": "#9A6324", "Knife": "#3cb44b", "Lighter": "#808000", "Pliers": "#ffe119",
           "Pressure_vessel": "#000075", "Razor_blade": "#800000", "Saw_blade": "#469990",
           "Scissors": "#bfef45", "Screwdriver": "#dcbeff", "Wrench": "#4363d8"}
ERR = {"correct": (0.55, 0.55, 0.55, 0.55), "missed": (0.12, 0.37, 0.84, 0.85), "extra": (0.89, 0.10, 0.11, 0.85)}
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})


def rle(m):
    return dict(m, counts=m["counts"].encode()) if isinstance(m["counts"], str) else m


def box_iou(a_xywh, b_xyxy):
    ax2, ay2 = a_xywh[0] + a_xywh[2], a_xywh[1] + a_xywh[3]
    iw = max(0.0, min(ax2, b_xyxy[2]) - max(a_xywh[0], b_xyxy[0]))
    ih = max(0.0, min(ay2, b_xyxy[3]) - max(a_xywh[1], b_xyxy[1]))
    inter = iw * ih
    return inter / (a_xywh[2] * a_xywh[3] + (b_xyxy[2] - b_xyxy[0]) * (b_xyxy[3] - b_xyxy[1]) - inter + 1e-9)


def xyxy_to_xywh(b):
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


class Data:
    def __init__(self):
        self.coco = COCO(str(ROOT / "data/PIXray/annotations/test.json"))
        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.cname = {c["id"]: c["name"] for c in cats}
        self.ccol = {c["id"]: matplotlib.colors.to_rgb(PALETTE[c["name"]]) for c in cats}
        self.q = {}
        for m, slug in SLUG.items():
            for s in SEEDS:
                a = np.load(CAMP / "analysis/runs" / slug / f"seed_{s}_per_gt.npy")
                self.q[(m, s)] = {int(r[0]): (float(r[1]), float(r[2]), float(r[3])) for r in a}
        self.ap = {}
        for m, slug in SLUG.items():
            paths = sorted((CAMP / "evaluations" / slug).glob("seed_*/metrics_fullprec_*.json"))
            per = [json.loads(p.read_text())["segm"]["per_class"] for p in paths]
            self.ap[m] = {c: float(np.mean([x[c]["AP"] for x in per])) for c in self.cname.values()}
        self.preds = {}
        for m, slug in SLUG.items():
            p = next((WORK / slug / f"seed_{SHOW}/test").glob("predictions_job*.pkl"))
            self.preds[m] = {s["img_id"]: s["pred_instances"] for s in pickle.load(open(p, "rb"))}
        self.img_cache = {}

    def v(self, m, gid, s=SHOW, k=0):
        r = self.q[(m, s)].get(gid)
        return None if r is None else r[k]

    def dets(self, m, iid):
        p = self.preds[m].get(iid)
        if p is None:
            return []
        sc, lb, bx = np.asarray(p["scores"]), np.asarray(p["labels"]), np.asarray(p["bboxes"])
        return [dict(i=int(i), score=float(sc[i]), label=int(lb[i]), box=bx[i]) for i in np.where(sc >= SCORE)[0]]

    def mask(self, m, iid, i):
        return mu.decode(rle(self.preds[m][iid]["masks"][i])).astype(bool)

    def image(self, iid):
        if iid not in self.img_cache:
            info = self.coco.loadImgs(iid)[0]
            self.img_cache[iid] = np.asarray(Image.open(ROOT / "data/PIXray/test" / info["file_name"]).convert("RGB"))
        return self.img_cache[iid]

    def why_unmatched(self, m, a):
        others = [g for g in self.coco.loadAnns(self.coco.getAnnIds(imgIds=a["image_id"])) if g["id"] != a["id"]]
        wrong = [(box_iou(a["bbox"], d["box"]), d["label"]) for d in self.dets(m, a["image_id"])
                 if d["label"] != a["category_id"] and box_iou(a["bbox"], d["box"]) >= 0.5
                 and not any(g["category_id"] == d["label"] and box_iou(g["bbox"], d["box"]) >= 0.5 for g in others)]
        if wrong:
            return f"wrong class\n({self.cname[max(wrong)[1]]})"
        near = [d for d in self.dets(m, a["image_id"]) if d["label"] == a["category_id"] and 0.1 <= box_iou(a["bbox"], d["box"]) < 0.5]
        return "poor\nlocalisation" if near else "missed"

    def matched_mask(self, m, a):
        """The prediction the protocol matched to GT a (same class, box IoU >= 0.5, best box IoU)."""
        cands = [d for d in self.dets(m, a["image_id"]) if d["label"] == a["category_id"] and box_iou(a["bbox"], d["box"]) >= 0.5]
        if not cands or self.v(m, a["id"]) is None:
            return None
        gm = self.coco.annToMask(a).astype(bool)
        target = self.v(m, a["id"])
        best = None
        for d in sorted(cands, key=lambda d: -box_iou(a["bbox"], d["box"])):
            pm = self.mask(m, a["image_id"], d["i"])
            iou = (gm & pm).sum() / max((gm | pm).sum(), 1)
            if abs(iou - target) < 1e-4:
                return pm
            best = best if best is not None else pm
        return best


# ------------------------------------------------------------------ selection
def select(D: Data) -> dict:
    coco, cname = D.coco, D.cname
    mean_ap = {c: np.mean([D.ap[m][c] for m in SLUG]) for c in cname.values()}
    hardest = sorted(mean_ap, key=mean_ap.get)[:4]
    hard_gain = [c for c in hardest if D.ap["09M"][c] > D.ap["03"][c]]
    anns = [a for a in coco.loadAnns(coco.getAnnIds(iscrowd=None))
            if a["bbox"][2] >= 24 and a["bbox"][3] >= 24 and len(coco.getAnnIds(imgIds=a["image_id"])) <= 6]
    ok5 = [a for a in anns if all(D.v(m, a["id"], s) is not None for m in KEY for s in SEEDS)]
    margin = lambda a, s: min(D.v(k, a["id"], s) for k in OURS) - max(D.v(k, a["id"], s) for k in BASES)  # noqa: E731
    key = lambda t: (-t[0], t[1]["image_id"], t[1]["id"])  # noqa: E731
    imp = sorted([(margin(a, SHOW), a) for a in ok5
                  if all(margin(a, s) > 0 for s in SEEDS) and min(D.v(k, a["id"]) for k in OURS) >= 0.70], key=key)
    i1 = imp[0][1]
    i2 = next(a for _, a in imp if cname[a["category_id"]] in hard_gain and a["image_id"] != i1["image_id"])
    i3 = next(a for _, a in imp if a["image_id"] not in {i1["image_id"], i2["image_id"]}
              and a["category_id"] not in {i1["category_id"], i2["category_id"]})
    reg = sorted([(D.v("03", a["id"]) - max(D.v(k, a["id"]) for k in OURS), a) for a in ok5
                  if all(max(D.v(k, a["id"], s) for k in OURS) < D.v("03", a["id"], s) for s in SEEDS)], key=key)
    f1 = reg[0][1]
    missed = sorted([(a["bbox"][2] * a["bbox"][3], a) for a in anns
                     if all(D.v(m, a["id"], s) is None for m in KEY for s in SEEDS)], key=key)
    f2 = missed[0][1]
    dart = sorted([(-max(D.v(m, a["id"]) for m in KEY), a) for a in ok5 if cname[a["category_id"]] == "Dart"], key=key)
    f3 = dart[0][1]
    used = {x["image_id"] for x in (i1, i2, i3, f1, f2, f3)}
    conf = sorted([(a["bbox"][2] * a["bbox"][3], a) for a in anns if a["image_id"] not in used
                   and all(D.v(k, a["id"], s) is None for k in OURS for s in SEEDS)
                   and all(D.why_unmatched(k, a).startswith("wrong") for k in OURS)], key=key)
    f4 = conf[0][1] if conf else None
    fps = []
    for iid, p in sorted(D.preds["09M"].items()):
        gts = coco.loadAnns(coco.getAnnIds(imgIds=iid))
        if len(gts) > 6 or iid in used:
            continue
        for d in D.dets("09M", iid):
            b = xyxy_to_xywh(d["box"])
            if min(b[2], b[3]) >= 16 and all(box_iou(g["bbox"], d["box"]) < 0.1 for g in gts):
                fps.append((d["score"], dict(image_id=iid, id=-d["i"], bbox=b, fp=True, score=d["score"], label=d["label"])))
    fps.sort(key=key)
    f5 = fps[0][1] if fps else None
    rows = dict(I1=("Improvement", i1), I2=("Improvement\n(hard class)", i2), I3=("Improvement", i3),
                F1=("Failure:\nrefinement regression", f1), F2=("Failure:\nmissed by all", f2),
                F3=("Limitation:\nhardest class", f3))
    if f4:
        rows["F4"] = ("Failure:\nclass confusion", f4)
    if f5:
        rows["F5"] = ("Unmatched detection\n(no GT here)", f5)
    counts = dict(improvement=len(imp), regression=len(reg), missed_by_all=len(missed), dart=len(dart),
                  class_confusion=len(conf), false_positive=len(fps))
    return dict(rows=rows, hardest=hardest, hard_gain=hard_gain, counts=counts)


# ------------------------------------------------------------------ drawing helpers
def crop_box(bbox, W, H, pad=0.40, aspect=0.62):
    x, y, w, h = bbox
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = x + w / 2, y + h / 2
    x1, x2 = int(max(0, cx - side / 2)), int(min(W, cx + side / 2))
    y1, y2 = int(max(0, cy - side * aspect / 2)), int(min(H, cy + side * aspect / 2))
    return x1, y1, x2, y2


def overlay(ax, img, crop, masks, ccol, present=None, lw=0.9):
    x1, y1, x2, y2 = crop
    ax.imshow(img[y1:y2, x1:x2], interpolation="nearest")
    for mk, lab in masks:
        c = mk[y1:y2, x1:x2]
        if not c.any():
            continue
        if present is not None:
            present.add(lab)
        ov = np.zeros((*c.shape, 4)); ov[c] = (*ccol[lab], 0.30)
        ax.imshow(ov, interpolation="nearest")
        ax.contour(c.astype(float), levels=[0.5], colors=[ccol[lab]], linewidths=lw).set_path_effects(
            [pe.withStroke(linewidth=lw + 1.2, foreground="white")])


def focus_outline(ax, item, crop, D):
    x1, y1 = crop[0], crop[1]
    if item.get("fp"):
        b = item["bbox"]
        ax.add_patch(Rectangle((b[0] - x1, b[1] - y1), b[2], b[3], fill=False, ec="black", lw=0.6, ls=(0, (2.0, 1.4))))
    else:
        f = D.coco.annToMask(item).astype(float)[crop[1]:crop[3], crop[0]:crop[2]]
        ax.contour(f, levels=[0.5], colors=["black"], linewidths=0.55, linestyles=[(0, (2.0, 1.4))])


def finish(ax, thick):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.3 if thick else 0.4)


def legend(fig, axes, D, present, extra=(), fs=5.4, ncol=8, gap=0.045):
    fig.canvas.draw()
    y0 = min(ax.get_position().y0 for ax in np.atleast_1d(axes).ravel() if ax.get_visible())
    handles = [Patch(facecolor=(*D.ccol[l], 0.30), edgecolor=D.ccol[l], label=D.cname[l]) for l in sorted(present)]
    handles += list(extra)
    fig.legend(handles=handles, loc="upper center", ncol=min(ncol, len(handles)), frameon=False, fontsize=fs,
               bbox_to_anchor=(0.5, y0 - gap), handlelength=1.1, columnspacing=0.9)


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=400)
    plt.close(fig)


# ------------------------------------------------------------------ figure 1 and 2: comparison grids
def comparison_grid(D, rows, cols, name, width=7.0, row_h=0.93, title_fs=5.6, lab_fs=5.6, short=False):
    fig, axes = plt.subplots(len(rows), len(cols) + 1, figsize=(width, row_h * len(rows) + 0.35),
                             gridspec_kw=dict(wspace=0.05, hspace=0.45))
    facts, present = [], set()
    for r, (tag, a) in enumerate(rows):
        iid = a["image_id"]; info = D.coco.loadImgs(iid)[0]; img = D.image(iid)
        crop = crop_box(a["bbox"], info["width"], info["height"])
        gts = D.coco.loadAnns(D.coco.getAnnIds(imgIds=iid, iscrowd=None))
        ax = axes[r, 0]
        overlay(ax, img, crop, [(D.coco.annToMask(g).astype(bool), g["category_id"]) for g in gts], D.ccol, present)
        focus_outline(ax, a, crop, D); finish(ax, False)
        name_ = D.cname[a["label"]] if a.get("fp") else D.cname[a["category_id"]]
        gt_txt = "no object\nannotated" if a.get("fp") else (name_.replace("_", "\n") if short else f"{name_} (focus)")
        ax.set_xlabel(gt_txt, fontsize=lab_fs, labelpad=1.5, linespacing=1.0)
        ax.set_ylabel(f"{tag}\nimage {iid}", fontsize=lab_fs - 0.3, labelpad=2, linespacing=1.05)
        if r == 0:
            ax.set_title("Ground truth\n", fontsize=title_fs, linespacing=1.1)
        vals = {} if a.get("fp") else {m: D.v(m, a["id"]) for m, _ in cols}
        best = max((x for x in vals.values() if x is not None), default=None)
        row = dict(row=tag.replace("\n", " "), image_id=iid, file=info["file_name"], crop_xyxy=list(crop), panels={})
        if not a.get("fp"):
            row.update(annotation_id=a["id"], category=name_,
                       mask_iou_all_seeds={m: [D.v(m, a["id"], s) for s in SEEDS] for m, _ in cols},
                       boundary_iou_seed42={m: D.v(m, a["id"], k=2) for m, _ in cols})
        for c, (m, label) in enumerate(cols, start=1):
            ax = axes[r, c]
            ds = D.dets(m, iid)
            overlay(ax, img, crop, [(D.mask(m, iid, d["i"]), d["label"]) for d in ds], D.ccol, present)
            focus_outline(ax, a, crop, D); finish(ax, m in OURS)
            if a.get("fp"):
                hit = [d for d in ds if box_iou(a["bbox"], d["box"]) >= 0.5]
                txt = ((f"detection\n{hit[0]['score']:.2f}" if short else f"{D.cname[hit[0]['label']]}\nscore {hit[0]['score']:.2f}")
                       if hit else "none")
                row["panels"][m] = txt.replace("\n", " ")
                weight = "normal"
            else:
                val = vals[m]
                txt = f"IoU {val:.3f}" if val is not None else D.why_unmatched(m, a)
                row["panels"][m] = txt.replace("\n", " ")
                weight = "bold" if val is not None and val == best else "normal"
            ax.set_xlabel(txt, fontsize=lab_fs, labelpad=1.5, linespacing=1.0, weight=weight)
            if r == 0:
                ax.set_title(label, fontsize=title_fs, linespacing=1.1, weight="bold" if m in OURS else "normal")
        facts.append(row)
    legend(fig, axes, D, present, [Line2D([], [], color="black", lw=0.6, ls=(0, (2.0, 1.4)), label="focus outline")],
           fs=lab_fs)
    save(fig, name)
    return facts


# ------------------------------------------------------------------ figure 3: pixel error maps
def error_maps(D, rows, name):
    cols = [("03", "Cascade + AttFPN\n(base)"), ("07", "Bilayer FCN\n(best prior)"),
            ("09", "Ours:\nTransformer"), ("09M", "Ours:\nMLP")]
    fig, axes = plt.subplots(len(rows), len(cols) + 2, figsize=(7.0, 1.05 * len(rows) + 0.35),
                             gridspec_kw=dict(wspace=0.05, hspace=0.45))
    facts = []
    for r, (tag, a) in enumerate(rows):
        iid = a["image_id"]; info = D.coco.loadImgs(iid)[0]; img = D.image(iid)
        crop = crop_box(a["bbox"], info["width"], info["height"], pad=0.12, aspect=0.70)
        x1, y1, x2, y2 = crop
        gm = D.coco.annToMask(a).astype(bool)
        faded = (0.45 * img[y1:y2, x1:x2] + 0.55 * 255).astype(np.uint8)
        axes[r, 0].imshow(img[y1:y2, x1:x2], interpolation="nearest"); finish(axes[r, 0], False)
        axes[r, 0].set_ylabel(f"{tag}\nimage {iid}", fontsize=5.3, labelpad=2, linespacing=1.05)
        axes[r, 0].set_xlabel(f"{D.cname[a['category_id']]}", fontsize=5.6, labelpad=1.5)
        overlay(axes[r, 1], img, crop, [(gm, a["category_id"])], D.ccol); finish(axes[r, 1], False)
        axes[r, 1].set_xlabel(f"{int(gm.sum())} px", fontsize=5.6, labelpad=1.5)
        if r == 0:
            axes[r, 0].set_title("Image\n", fontsize=5.6); axes[r, 1].set_title("Ground truth\n(focus object)", fontsize=5.6)
        row = dict(row=tag.replace("\n", " "), image_id=iid, annotation_id=a["id"], gt_pixels=int(gm.sum()), methods={})
        for c, (m, label) in enumerate(cols, start=2):
            ax = axes[r, c]
            ax.imshow(faded, interpolation="nearest")
            pm = D.matched_mask(m, a)
            if pm is None:
                ax.set_xlabel("no match", fontsize=5.6)
            else:
                g, p = gm[y1:y2, x1:x2], pm[y1:y2, x1:x2]
                ov = np.zeros((*g.shape, 4))
                ov[g & p] = ERR["correct"]; ov[g & ~p] = ERR["missed"]; ov[~g & p] = ERR["extra"]
                ax.imshow(ov, interpolation="nearest")
                miss, extra = int((gm & ~pm).sum()), int((~gm & pm).sum())
                iou, biou = D.v(m, a["id"]), D.v(m, a["id"], k=2)
                ax.set_xlabel(f"IoU {iou:.3f}\n−{miss} / +{extra} px", fontsize=5.4, labelpad=1.5, linespacing=1.0,
                              weight="bold" if iou == max(x for x in (D.v(k, a["id"]) for k, _ in cols) if x is not None) else "normal")
                row["methods"][m] = dict(iou=iou, boundary_iou=biou, missed_px=miss, extra_px=extra)
            finish(ax, m in OURS)
            if r == 0:
                ax.set_title(label, fontsize=5.6, linespacing=1.1, weight="bold" if m in OURS else "normal")
        facts.append(row)
    fig.canvas.draw()
    y0 = min(ax.get_position().y0 for ax in axes.ravel())
    fig.legend(handles=[Patch(color=ERR[k][:3], alpha=ERR[k][3], label=l) for k, l in
                        (("correct", "correct (in both)"), ("missed", "− missed (GT only)"), ("extra", "+ extra (prediction only)"))],
               loc="upper center", ncol=3, frameon=False, fontsize=5.6, bbox_to_anchor=(0.5, y0 - 0.05))
    save(fig, name)
    return facts


# ------------------------------------------------------------------ figure 4: per-class representatives
def per_class(D, name):
    coco = D.coco
    picks = []
    for cid, cname in sorted(D.cname.items()):
        cand = [a for a in coco.loadAnns(coco.getAnnIds(catIds=[cid], iscrowd=None))
                if a["bbox"][2] >= 24 and a["bbox"][3] >= 24
                and all(D.v(m, a["id"], s) is not None for m in ("03", "09", "09M") for s in SEEDS)]
        if not cand:
            continue
        d = {a["id"]: np.mean([D.v("09M", a["id"], s) - D.v("03", a["id"], s) for s in SEEDS]) for a in cand}
        med = float(np.median(list(d.values())))
        a = min(cand, key=lambda a: (abs(d[a["id"]] - med), a["image_id"], a["id"]))
        picks.append((cname, a, dict(n=len(cand), median_gain=med, share_gain=float(np.mean([x > 0 for x in d.values()])))))
    cols = [("03", "Base (03)"), ("09", "Transformer (09)"), ("09M", "MLP (09M)")]
    half = (len(picks) + 1) // 2
    fig, axes = plt.subplots(half, 9, figsize=(7.0, 0.80 * half + 0.35),
                             gridspec_kw=dict(wspace=0.05, hspace=0.50, width_ratios=[1, 1, 1, 1, 0.18, 1, 1, 1, 1]))
    for r in range(half):
        axes[r, 4].set_visible(False)
    present, facts = set(), []
    for k, (cname, a, st) in enumerate(picks):
        r, off = (k, 0) if k < half else (k - half, 5)
        iid = a["image_id"]; info = coco.loadImgs(iid)[0]; img = D.image(iid)
        crop = crop_box(a["bbox"], info["width"], info["height"], pad=0.25, aspect=0.75)
        gts = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None))
        ax = axes[r, off]
        overlay(ax, img, crop, [(coco.annToMask(g).astype(bool), g["category_id"]) for g in gts], D.ccol, present, lw=0.7)
        focus_outline(ax, a, crop, D); finish(ax, False)
        ax.set_ylabel(f"{cname.replace('_', ' ')}\nimage {iid}", fontsize=5.0, labelpad=1.5, linespacing=1.05)
        ax.set_xlabel(f"n={st['n']} · med Δ {st['median_gain']:+.3f}", fontsize=4.7, labelpad=1.2)
        vals = {m: D.v(m, a["id"]) for m, _ in cols}
        best = max(vals.values())
        for j, (m, label) in enumerate(cols, start=1):
            ax = axes[r, off + j]
            overlay(ax, img, crop, [(D.mask(m, iid, d["i"]), d["label"]) for d in D.dets(m, iid)], D.ccol, present, lw=0.7)
            focus_outline(ax, a, crop, D); finish(ax, m in OURS)
            ax.set_xlabel(f"IoU {vals[m]:.3f}", fontsize=4.9, labelpad=1.2, weight="bold" if vals[m] == best else "normal")
            if r == 0:
                ax.set_title(label, fontsize=5.3, weight="bold" if m in OURS else "normal")
        if r == 0:
            axes[r, off].set_title("Ground truth", fontsize=5.3)
        facts.append(dict(category=cname, image_id=iid, annotation_id=a["id"], **st,
                          mask_iou_all_seeds={m: [D.v(m, a["id"], s) for s in SEEDS] for m, _ in cols}))
    for k in range(len(picks), 2 * half):
        for j in range(4):
            axes[k - half, 5 + j].set_visible(False)
    legend(fig, axes, D, present, [Line2D([], [], color="black", lw=0.6, ls=(0, (2.0, 1.4)), label="focus outline")],
           fs=5.2, gap=0.04)
    save(fig, name)
    return facts


# ------------------------------------------------------------------ figure 5: gain vs size / elongation
def gain_by_shape(D, name):
    coco = D.coco
    attr = {}
    for a in coco.loadAnns(coco.getAnnIds(iscrowd=None)):
        area = float(mu.area(coco.annToRLE(a)))
        pts = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in a["segmentation"]]).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        attr[a["id"]] = (np.sqrt(area), max(w, h) / max(min(w, h), 1.0))
    bins = dict(size=([0, 32, 96, np.inf], ["small\n(<32²)", "medium\n(32²–96²)", "large\n(≥96²)"]),
                elong=([1, 2, 4, 8, np.inf], ["<2", "2–4", "4–8", "≥8"]))
    res = {}
    for kind, (edges, labels) in bins.items():
        idx = 0 if kind == "size" else 1
        res[kind] = {}
        for ours in OURS:
            per_bin = []
            for b in range(len(labels)):
                seed_means, better, worse, n = [], [], [], []
                for s in SEEDS:
                    ids = [g for g in D.q[(ours, s)] if g in D.q[("03", s)] and edges[b] <= attr[g][idx] < edges[b + 1]]
                    dlt = np.array([D.q[(ours, s)][g][0] - D.q[("03", s)][g][0] for g in ids])
                    seed_means.append(dlt.mean()); better.append((dlt >= 0.05).mean()); worse.append((dlt <= -0.05).mean()); n.append(len(ids))
                per_bin.append(dict(label=labels[b].replace("\n", " "), n=int(np.mean(n)), mean=float(np.mean(seed_means)),
                                    sd=float(np.std(seed_means, ddof=1)), seed_means=[float(x) for x in seed_means],
                                    better_ge_005=float(np.mean(better)), worse_ge_005=float(np.mean(worse))))
            res[kind][ours] = per_bin
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw=dict(wspace=0.28))
    col = {"09": "#6a51a3", "09M": "#d94801"}
    for ax, (kind, title) in zip(axes, (("size", "(a) object size (√mask area, px)"), ("elong", "(b) elongation (long / short side)"))):
        labels = bins[kind][1]; x = np.arange(len(labels)); w = 0.36
        for j, ours in enumerate(OURS):
            pb = res[kind][ours]
            mean = np.array([p["mean"] for p in pb]) * 100
            ax.bar(x + (j - 0.5) * w, mean, w, color=col[ours], alpha=0.85,
                   label="Ours: Transformer (09)" if ours == "09" else "Ours: MLP (09M)")
            for i, p in enumerate(pb):
                ax.scatter([x[i] + (j - 0.5) * w] * 3, np.array(p["seed_means"]) * 100, s=4, color="black", zorder=3, lw=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x); ax.tick_params(axis="y", labelsize=6)
        ax.set_xticklabels([f"{l}\nn={p['n']}" for l, p in zip(labels, res[kind]["09M"])], fontsize=5.8)
        ax.set_ylabel("mask IoU change vs base (03), ×100", fontsize=6)
        ax.set_title(title, fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    axes[0].legend(fontsize=5.8, frameon=False, loc="upper left")
    save(fig, name)
    return res


def failure_stats(D):
    gt = D.coco.getAnnIds(iscrowd=None); n = len(gt); allm = list(SLUG)
    q = {"gt_total": n}
    for ours in OURS:
        per = []
        for s in SEEDS:
            common = [g for g in gt if g in D.q[(ours, s)] and g in D.q[("03", s)]]
            d = np.array([D.q[(ours, s)][g][0] - D.q[("03", s)][g][0] for g in common])
            per.append(dict(common=len(common), mean_delta=float(d.mean()), better_ge_005=float((d >= 0.05).mean()),
                            worse_ge_005=float((d <= -0.05).mean()), worse_ge_010=float((d <= -0.10).mean())))
        q[f"{ours}_vs_03"] = {k: [p[k] for p in per] for k in per[0]}
    q["both_variants_below_base_in_all_seeds"] = sum(
        all(g in D.q[(m, s)] for m in ("03", "09", "09M") for s in SEEDS)
        and all(max(D.q[("09", s)][g][0], D.q[("09M", s)][g][0]) < D.q[("03", s)][g][0] for s in SEEDS) for g in gt)
    q["missed_per_method_mean_over_seeds"] = {m: float(np.mean([n - len(D.q[(m, s)]) for s in SEEDS])) for m in allm}
    q["missed_by_all_ten_in_all_seeds"] = sum(all(g not in D.q[(m, s)] for m in allm for s in SEEDS) for g in gt)
    q["missed_by_key_five_in_all_seeds"] = sum(all(g not in D.q[(m, s)] for m in KEY for s in SEEDS) for g in gt)
    return q


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    sel = select(D)
    R = sel["rows"]
    main_rows = [R[k] for k in ("I1", "I2", "I3", "F1", "F2", "F3")]
    all_rows = [R[k] for k in ("I1", "I2", "I3", "F1", "F2", "F3", "F4", "F5") if k in R]
    out = dict(rule=__doc__.strip(), seed_shown=SHOW, seeds_required=list(SEEDS), score_threshold=SCORE,
               hardest_classes=sel["hardest"], hard_classes_with_09M_gain=sel["hard_gain"], candidates=sel["counts"])
    out["main"] = comparison_grid(D, main_rows, KEY_COLS, "fig_qualitative_final")
    out["all_methods"] = comparison_grid(D, all_rows, ALL_COLS, "fig_qualitative_all_methods",
                                         width=7.2, row_h=0.74, title_fs=4.9, lab_fs=4.9, short=True)
    out["error_maps"] = error_maps(D, [R[k] for k in ("I1", "I2", "I3", "F1")], "fig_boundary_error_maps")
    out["per_class"] = per_class(D, "fig_per_class_representative")
    out["gain_by_size_shape"] = gain_by_shape(D, "fig_gain_by_size_shape")
    (OUT / "failure_statistics.json").write_text(json.dumps(failure_stats(D), indent=1))
    (OUT / "qualitative_final_rows.json").write_text(json.dumps(out, indent=1, default=float))
    print("candidates:", sel["counts"])
    for r in out["all_methods"]:
        print(r["row"], r["image_id"], r.get("category", ""), r["panels"])
    for c in out["per_class"]:
        print(f"{c['category']:16s} n={c['n']:4d} median Δ={c['median_gain']:+.4f} share>0={c['share_gain']:.2f}")
    for kind, v in out["gain_by_size_shape"].items():
        for ours, pb in v.items():
            print(kind, ours, [(p["label"], p["n"], round(100 * p["mean"], 2), round(100 * p["worse_ge_005"], 1)) for p in pb])
    return 0


if __name__ == "__main__":
    sys.exit(main())
