#!/usr/bin/env python
"""Publication figures for the multi-seed campaign (static PNG 300 dpi + PDF).

Reads statistics/statistics.json, analysis/runs/*/seed_*.json and the per-run COCO scores. Colour use:
our two refiners carry the only categorical hues (MLP refiner = blue, transformer refiner = orange,
reference palette slots 1-2); the parent model is dark neutral grey and every other baseline light
grey, so identity never depends on more than two hues. Diverging maps use blue <-> red with a grey
midpoint; magnitudes use a single blue ramp. Every multi-series figure has a legend or direct labels.

    python scripts/campaign_multiseed/figures_campaign.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.campaign_common import BY_ID, METHODS, OUT, SEEDS, eval_dir  # noqa: E402

ST, AN, FIG = OUT / "statistics", OUT / "analysis", OUT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
PARENT, OTHER = "#52514e", "#a9a79f"
COLOR = {"09M": BLUE, "09": ORANGE, "03": PARENT}
LS = {"07": "-", "08": "--", "04": ":", "01": ":", "05": "--", "06": "-."}   # grey baselines told apart by line style
SHORT = {"01": "Mask R-CNN", "02": "Cascade FPN", "03": "Parent (Cascade + AttFPN)", "04": "Recall RPN",
         "05": "RSPrompter (SAM)", "06": "Occ bilayer SAM", "07": "Bilayer FCN", "08": "PIDrayNet (adapted)",
         "09": "Transformer refiner (ours)", "09M": "MLP refiner (ours)"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
OLD = ROOT / "old_output/reports/pixray_supplementary_evaluation_2026-09-13/data/coco_prf.json"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight"})


def col(mid):
    return COLOR.get(mid, OTHER)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def load():
    S = json.loads((ST / "statistics.json").read_text())
    runs = {}
    for m in METHODS:
        for s in SEEDS:
            runs[(m.id, s)] = json.loads((AN / "runs" / m.slug / f"seed_{s}.json").read_text())
    scores = {}
    for m in METHODS:
        for s in SEEDS:
            f = sorted(glob.glob(str(eval_dir(m.id, s) / "metrics_fullprec_job*.json")))[-1]
            scores[(m.id, s)] = json.loads(Path(f).read_text())
    return S, runs, scores


def fig_method_ap(S):
    M = S["methods"]
    order = sorted(M, key=lambda k: M[k]["mask_AP"]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, key, title in ((axes[0], "mask_AP", "Mask AP (%)"), (axes[1], "box_AP", "Box AP (%)")):
        for y, mid in enumerate(order):
            r = M[mid][key]
            ax.plot(r["ci95"], [y, y], color=col(mid), lw=1.6, solid_capstyle="round")
            ax.scatter(list(r["per_seed"].values()), [y] * len(r["per_seed"]), s=12, facecolor="white",
                       edgecolor=col(mid), lw=0.8, zorder=3)
            ax.scatter([r["mean"]], [y], s=26, color=col(mid), zorder=4, edgecolor="white", lw=0.6)
        ax.set_title(title, loc="left", color=INK)
        grid(ax)
    axes[0].set_yticks(range(len(order)), [SHORT[m] for m in order])
    for lab in axes[0].get_yticklabels():
        lab.set_color(INK if lab.get_text() in (SHORT["09"], SHORT["09M"], SHORT["03"]) else INK2)
    h = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, label="single seed"),
         plt.Line2D([], [], marker="o", ls="-", color=INK2, label="mean over 3 seeds, 95% CI")]
    axes[1].legend(handles=h, loc="lower right", frameon=False)
    fig.suptitle("Accuracy of the ten methods (PIXray test, 3 seeds; CI: two-level bootstrap over seeds and images)",
                 x=0.01, ha="left", fontsize=8.5, color=INK)
    save(fig, "fig_s1_method_ap")


def fig_method_ap_bars(S):
    """Bar-chart version of S1: horizontal bars from zero (mean over 3 seeds), 95% CI whiskers, seeds as dots."""
    M = S["methods"]
    order = sorted(M, key=lambda k: M[k]["mask_AP"]["mean"])          # best at the top
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, key, title in ((axes[0], "mask_AP", "Mask AP (%)"), (axes[1], "box_AP", "Box AP (%)")):
        for i, mid in enumerate(order):
            r = M[mid][key]
            ax.barh(i, r["mean"], height=0.66, color=col(mid), edgecolor="white", linewidth=1.0, zorder=2)
            ax.hlines(i, r["ci95"][0], r["ci95"][1], color=INK, lw=1.0, zorder=4)
            ax.vlines(r["ci95"], i - 0.13, i + 0.13, color=INK, lw=1.0, zorder=4)
            ax.annotate(f"{r['mean']:.1f} ± {r['sd']:.2f}", (r["ci95"][1], i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=INK)
        ax.set_xlim(0, 95)
        ax.set_title(title, loc="left", color=INK)
        grid(ax)
    axes[0].set_yticks(y, [SHORT[m] for m in order])
    for lab in axes[0].get_yticklabels():
        lab.set_color(INK if lab.get_text() in (SHORT["09"], SHORT["09M"], SHORT["03"]) else INK2)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
         ((BLUE, "MLP refiner (ours)"), (ORANGE, "Transformer refiner (ours)"), (PARENT, "Parent"), (OTHER, "Other baselines"))]
    h += [plt.Line2D([], [], color=INK, lw=1.0, label="95% CI (seeds + images)")]
    fig.legend(handles=h, frameon=False, fontsize=6.3, ncol=5, loc="lower center", bbox_to_anchor=(0.55, -0.07))
    fig.suptitle("Mask and box AP of the ten methods (PIXray test; bar = mean over 3 seeds, label = mean ± seed SD)", x=0.01, ha="left",
                 fontsize=8.5, color=INK)
    save(fig, "fig_s1_method_ap_bars")


def fig_method_ap_columns(S):
    """Column-chart version of S1: methods along the horizontal axis, bars from zero, 95% CI whiskers."""
    M = S["methods"]
    order = sorted(M, key=lambda k: -M[k]["mask_AP"]["mean"])          # best on the left
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw=dict(hspace=0.12))
    for ax, key, title in ((axes[0], "mask_AP", "Mask AP (%)"), (axes[1], "box_AP", "Box AP (%)")):
        for i, mid in enumerate(order):
            r = M[mid][key]
            ax.bar(i, r["mean"], width=0.64, color=col(mid), edgecolor="white", linewidth=1.0, zorder=2)
            ax.vlines(i, r["ci95"][0], r["ci95"][1], color=INK, lw=1.0, zorder=4)
            ax.hlines(r["ci95"], i - 0.1, i + 0.1, color=INK, lw=1.0, zorder=4)
            ax.annotate(f"{r['mean']:.1f}", (i, r["ci95"][1]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK)
        ax.set_ylim(0, 95)
        ax.set_ylabel(title)
        grid(ax, "y")
    labels = [SHORT[m].replace("Parent (Cascade + AttFPN)", "Parent") for m in order]
    axes[1].set_xticks(x, labels, rotation=35, ha="right", fontsize=6.8)
    for lab in axes[1].get_xticklabels():
        lab.set_color(INK if lab.get_text() in (SHORT["09"], SHORT["09M"], "Parent") else INK2)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
         ((BLUE, "MLP refiner (ours)"), (ORANGE, "Transformer refiner (ours)"), (PARENT, "Parent (Cascade + AttFPN)"),
          (OTHER, "Other baselines"))]
    h += [plt.Line2D([], [], color=INK, lw=1.0, label="95% CI (seeds + images)")]
    fig.legend(handles=h, frameon=False, fontsize=6.3, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Mask and box AP of the ten methods (PIXray test; bar = mean over 3 seeds)", x=0.01, y=1.0,
                 ha="left", fontsize=8.5, color=INK)
    save(fig, "fig_s1_method_ap_columns")


def fig_method_ap_columns_zoom(S, y0=40.0):
    """Column chart with the value axis starting at y0 (not zero); the break is marked on the axis and in the title."""
    M = S["methods"]
    order = sorted(M, key=lambda k: -M[k]["mask_AP"]["mean"])
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw=dict(hspace=0.12))
    for ax, key, title in ((axes[0], "mask_AP", "Mask AP (%)"), (axes[1], "box_AP", "Box AP (%)")):
        for i, mid in enumerate(order):
            r = M[mid][key]
            ax.bar(i, r["mean"] - y0, bottom=y0, width=0.64, color=col(mid), edgecolor="white", linewidth=1.0, zorder=2)
            ax.vlines(i, r["ci95"][0], r["ci95"][1], color=INK, lw=1.0, zorder=4)
            ax.hlines(r["ci95"], i - 0.1, i + 0.1, color=INK, lw=1.0, zorder=4)
            ax.annotate(f"{r['mean']:.1f}", (i, r["ci95"][1]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK)
        ax.set_ylim(y0, 88)
        ax.set_ylabel(title)
        grid(ax, "y")
        # axis-break mark at the bottom of the value axis
        kw = dict(transform=ax.transAxes, color=INK2, clip_on=False, lw=0.8)
        ax.plot([-0.012, 0.012], [0.015, 0.045], **kw)
        ax.plot([-0.012, 0.012], [0.045, 0.075], **kw)
    labels = [SHORT[m].replace("Parent (Cascade + AttFPN)", "Parent") for m in order]
    axes[1].set_xticks(x, labels, rotation=35, ha="right", fontsize=6.8)
    for lab in axes[1].get_xticklabels():
        lab.set_color(INK if lab.get_text() in (SHORT["09"], SHORT["09M"], "Parent") else INK2)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
         ((BLUE, "MLP refiner (ours)"), (ORANGE, "Transformer refiner (ours)"), (PARENT, "Parent (Cascade + AttFPN)"),
          (OTHER, "Other baselines"))]
    h += [plt.Line2D([], [], color=INK, lw=1.0, label="95% CI (seeds + images)")]
    fig.legend(handles=h, frameon=False, fontsize=6.3, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"Mask and box AP of the ten methods (PIXray test; bar = mean over 3 seeds; value axis starts at {y0:.0f}%)",
                 x=0.01, y=1.0, ha="left", fontsize=8.5, color=INK)
    save(fig, "fig_s1_method_ap_columns_from40")


def fig_forest(S):
    C = S["comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True, gridspec_kw=dict(wspace=0.1))
    labels = [f"{c['label']}  ({c['a']} - {c['b']})" for c in C][::-1]
    for ax, key, title in ((axes[0], "mask_AP", "Δ mask AP (points)"), (axes[1], "mask_AP75", "Δ mask AP75 (points)")):
        for y, c in enumerate(C[::-1]):
            r = c[key]
            ours = c["a"] in ("09", "09M")
            colr = col(c["a"]) if ours else PARENT
            if c["family"] == "equivalence" and key == "mask_AP":
                m = r.get("tost_margin", 1.0)
                ax.add_patch(plt.Rectangle((-m, y - 0.35), 2 * m, 0.7, color="#f0efec", zorder=0, lw=0))
            ax.plot(r["ci95"], [y, y], color=colr, lw=1.6, solid_capstyle="round")
            ax.scatter(list(r["per_seed"].values()), [y] * 3, s=10, facecolor="white", edgecolor=colr, lw=0.7, zorder=3)
            ax.scatter([r["delta"]], [y], s=24, color=colr, zorder=4, edgecolor="white", lw=0.6)
            if key == "mask_AP":
                p = r.get("p_holm", r.get("p_perm"))
                ptxt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                if c["family"] == "equivalence":
                    ptxt = f"p<0.001; within ±{r.get('tost_margin', 1.0):.1f}" if r.get("equivalent") else ptxt
                if r["ci95"][0] <= 0 <= r["ci95"][1]:
                    ptxt += "; CI incl. 0"
                ax.annotate(ptxt, (max(r["ci95"][1], r["delta"]), y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=6, color=INK2)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_title(title, loc="left")
        grid(ax)
    axes[0].set_xlim(-1.1, 3.6)
    axes[0].set_yticks(range(len(labels)), labels)
    fig.suptitle("Paired differences, mean over 3 seeds with 95% CI (seeds + images); hollow = single seeds",
                 x=0.01, ha="left", fontsize=8, color=INK)
    fig.text(0.01, -0.02, "p: paired permutation test over test images, Holm-adjusted within family (reflects test-set "
             "sampling only; the CI also includes seed variation). Shaded band: ±1.0 equivalence margin (TOST).",
             fontsize=6, color=INK2)
    save(fig, "fig_s2_paired_differences")


def fig_iou(S, scores):
    thr = np.arange(0.50, 0.951, 0.05)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    base = np.array([scores[("03", s)]["segm"]["ap_by_iou"] for s in SEEDS]) * 100
    for mid in ("09M", "09", "07", "08", "04"):
        d = np.array([scores[(mid, s)]["segm"]["ap_by_iou"] for s in SEEDS]) * 100 - base
        c = col(mid)
        ax.fill_between(thr, d.min(0), d.max(0), color=c, alpha=0.15 if mid in COLOR else 0.10, lw=0)
        ax.plot(thr, d.mean(0), color=c, lw=2 if mid in COLOR else 1.0, marker="o" if mid in COLOR else None, ms=3,
                ls=LS.get(mid, "-"), label=SHORT[mid])
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Δ mask AP vs parent (points)")
    grid(ax, "y")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Gain over the parent by IoU threshold (band: range over seeds)", loc="left", fontsize=7.5)
    save(fig, "fig_s3_delta_by_iou")


def fig_class_heatmap(S, runs):
    C = {(c["a"], c["b"]): c for c in S["comparisons"]}
    fills = json.loads((ST / "statistics.json").read_text())["classes"]
    classes = S["classes"]
    rows = [("09M", "03"), ("09", "03"), ("08", "03"), ("04", "03")]
    mat = np.array([[C[r]["mask_per_class"][k]["delta"] for k in classes] for r in rows])
    sig = np.array([[not (C[r]["mask_per_class"][k]["ci95"][0] <= 0 <= C[r]["mask_per_class"][k]["ci95"][1])
                     for k in classes] for r in rows])
    order = np.argsort([S["methods"]["03"]["mask_per_class"][k] for k in classes])       # hardest -> easiest
    mat, sig = mat[:, order], sig[:, order]
    cls = [classes[i].replace("_", " ") for i in order]
    lim = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    im = ax.imshow(mat, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}{'*' if sig[i, j] else ''}", ha="center", va="center", fontsize=6,
                    color=INK)
    ax.set_xticks(range(len(cls)), cls, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), [SHORT[a] for a, _ in rows])
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Δ mask AP vs parent", fontsize=7)
    ax.set_title("Per-class Δ mask AP vs parent (classes ordered hardest → easiest; * 95% CI excludes 0)",
                 loc="left", fontsize=7.5)
    save(fig, "fig_s4_per_class_delta")


def fig_mask_quality(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.22))
    for mid in ("03", "09", "09M"):
        arrs = [np.load(AN / "runs" / BY_ID[mid].slug / f"seed_{s}_per_gt.npy") for s in SEEDS]
        v = np.concatenate(arrs)
        for ax, k, lab in ((axes[0], 1, "Mask IoU of matched objects"), (axes[1], 3, "Boundary IoU of matched objects")):
            x = np.sort(v[:, k]); y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, color=col(mid), lw=1.6, label=f"{SHORT[mid]} (mean {100 * x.mean():.1f})")
            ax.set_xlabel(lab); ax.set_xlim(0.5, 1.0)
    for ax in axes:
        ax.set_ylabel("Cumulative share of objects")
        grid(ax, "both")
        ax.legend(frameon=False, loc="upper left", fontsize=6.3)
    fig.suptitle("Mask quality distributions (all three seeds pooled; curves further right = better)",
                 x=0.01, ha="left", fontsize=8)
    save(fig, "fig_s5_mask_quality_ecdf")


def fig_confusion(runs):
    classes = runs[("03", 42)]["classes"]
    names = [c.replace("_", " ") for c in classes]

    def mean_conf(mid):
        return np.mean([np.array(runs[(mid, s)]["errors"]["confusion"], float) for s in SEEDS], axis=0)

    A, B = mean_conf("03"), mean_conf("09M")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, M, title in ((axes[0], A, "Parent"), (axes[1], B, "MLP refiner (ours)")):
        R = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
        ax.imshow(R, cmap=SEQ, vmin=0, vmax=1)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if i != j and M[i, j] >= 1:
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=4.6, color=INK)
        ax.set_title(title, fontsize=7.5, loc="left")
    D = B - A
    lim = max(np.abs(D).max(), 1)
    im = axes[2].imshow(D, cmap=DIVERGING.reversed(), vmin=-lim, vmax=lim)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if abs(D[i, j]) >= 1:
                axes[2].text(j, i, f"{D[i, j]:+.0f}", ha="center", va="center", fontsize=4.6, color=INK)
    axes[2].set_title("MLP refiner − parent (red: more)", fontsize=7.5, loc="left")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    for ax in axes:
        ax.set_xticks(range(len(names) + 1), names + ["missed"], rotation=90, fontsize=5.2)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(names) + 1), names + ["background"], fontsize=5.2)
    axes[0].set_ylabel("Ground truth"); axes[1].set_xlabel("Prediction")
    fig.suptitle("Confusion matrices (score ≥ 0.5, box IoU ≥ 0.5), mean over 3 seeds: shade = row share, "
                 "numbers = off-diagonal counts", x=0.01, ha="left", fontsize=7.5)
    save(fig, "fig_s6_confusion_matrices")


def fig_errors(runs):
    kinds = [("duplicate", "Duplicate"), ("wrong_class", "Wrong class"), ("poor_localization", "Poor localisation"),
             ("other", "Other"), ("background", "Background"), ("missed", "Missed")]
    ids = [m.id for m in METHODS]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.2, 2.8), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, (k, lab) in zip(axes, kinds):
        for y, mid in enumerate(ids[::-1]):
            v = [runs[(mid, s)]["errors"]["counts"][k] for s in SEEDS]
            ax.plot([min(v), max(v)], [y, y], color=col(mid), lw=1.4)
            ax.scatter([np.mean(v)], [y], s=16, color=col(mid), zorder=3)
        ax.set_title(lab, fontsize=7, loc="left")
        grid(ax)
    axes[0].set_yticks(range(len(ids)), [SHORT[m] for m in ids[::-1]], fontsize=6)
    fig.suptitle("Detection errors at score ≥ 0.5 (count per 4,508 objects; dot = mean, line = range over seeds)",
                 x=0.01, ha="left", fontsize=7.5)
    save(fig, "fig_s7_error_types")


def fig_strata(S):
    panels = [("crowding", None, "Δ mask AP"), ("occlusion", "mask_recall_075", "Δ mask recall @ IoU 0.75"),
              ("thinness", "mask_recall_075", "Δ mask recall @ IoU 0.75")]
    short = {"1-2 objects": "1–2\nobjects", "3-4 objects": "3–4\nobjects", "5+ objects": "5+\nobjects",
             "no occluder": "none", "light (<=0.25)": "light\n≤ 0.25", "heavy (>0.25)": "heavy\n> 0.25",
             "thin (<0.30)": "thin\n< 0.30", "medium (0.30-0.60)": "medium\n0.30–0.60", "compact (>=0.60)": "compact\n≥ 0.60"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.32))
    for ax, (kind, metric, ylab) in zip(axes, panels):
        bins = list(S["strata"][kind])
        x = np.arange(len(bins))
        for off, mid in ((-0.12, "09"), (0.12, "09M")):
            vals, los, his = [], [], []
            for b in bins:
                d = S["strata"][kind][b]["delta_vs_parent"][mid]
                r = d if metric is None else d[metric]
                vals.append(r["delta"]); los.append(r["ci95"][0]); his.append(r["ci95"][1])
            ax.vlines(x + off, los, his, color=col(mid), lw=1.6)
            ax.scatter(x + off, vals, color=col(mid), s=22, zorder=3, label=SHORT[mid])
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(x, [f"{short.get(b, b)}\nn={S['strata'][kind][b].get('objects', '')}" for b in bins], fontsize=6)
        ax.set_xlim(-0.5, len(bins) - 0.5)
        ax.set_title({"crowding": "Crowding (objects per image)", "occlusion": "Occlusion (covered fraction)",
                      "thinness": "Thinness (mask / box fill)"}[kind], loc="left", fontsize=7.5)
        ax.set_ylabel(f"{ylab} vs parent (points)", fontsize=6.5)
        grid(ax, "y")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=6.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Gain over the parent by difficulty group (mean over 3 seeds, 95% bootstrap CI)",
                 x=0.01, y=1.03, ha="left", fontsize=8)
    save(fig, "fig_s8_difficulty_strata")


def fig_pr(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, gridspec_kw=dict(wspace=0.08))
    rp = np.array(runs[("03", 42)]["pr_curves"]["recall_points"])
    for ax, key, title in ((axes[0], "iou_0.50", "Mask PR, IoU 0.50"), (axes[1], "iou_0.75", "Mask PR, IoU 0.75")):
        for mid in ("05", "06", "01", "03", "09", "09M"):
            curves = np.array([[np.nan if v is None else v for v in runs[(mid, s)]["pr_curves"]["segm"][key]] for s in SEEDS])
            ax.plot(rp, np.nanmean(curves, 0), color=col(mid), lw=1.6 if mid in COLOR else 0.9,
                    ls=LS.get(mid, "-") if mid not in COLOR else "-", label=SHORT[mid])
        ax.set_xlabel("Recall"); ax.set_title(title, loc="left", fontsize=7.5); grid(ax, "both")
        ax.set_xlim(0.4, 1.0); ax.set_ylim(0.3, 1.01)
    axes[0].set_ylabel("Precision (class-averaged)")
    axes[1].legend(frameon=False, fontsize=6, loc="lower left")
    save(fig, "fig_s9_pr_curves")


def fig_old_vs_new(S):
    old = json.loads(OLD.read_text())["methods"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for mid in S["methods"]:
        o = 100 * old[mid]["segm"]["stats"]["AP"]; n = S["methods"][mid]["mask_AP"]["mean"]
        ax.plot([0, 1], [o, n], color=col(mid), lw=1.6 if mid in COLOR else 0.9, marker="o", ms=3, zorder=3 if mid in COLOR else 2)
    new = {m: S["methods"][m]["mask_AP"]["mean"] for m in S["methods"]}
    groups = [(["09M"], "MLP refiner (ours)"), (["09"], "Transformer refiner (ours)"),
              (["07", "03", "08", "04", "02"], "Parent and detector variants\n(02, 03, 04, 07, 08)"),
              (["01"], "Mask R-CNN"), (["05", "06"], "SAM-based (05, 06)")]
    for mids, lab in groups:
        y = float(np.mean([new[m] for m in mids]))
        ax.annotate(lab, (1, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6,
                    color=INK if mids[0] in COLOR else INK2)
    ax.set_xticks([0, 1], ["Earlier run\n60 ep, batch 16, 1 seed", "This campaign\n50 ep, batch 4, 3-seed mean"])
    ax.set_xlim(-0.2, 1.6); ax.set_ylabel("Mask AP (%)"); grid(ax, "y")
    ax.set_title("Mask AP under the earlier and the current protocol", loc="left", fontsize=7.5)
    save(fig, "fig_s10_old_vs_new_protocol")


def fig_variance(S):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for mid, r in S["methods"].items():
        ax.scatter(r["mask_AP"]["sd"], r["mask_AP"]["boot_sd"], color=col(mid), s=22, zorder=3)
        if mid == "04":
            ax.annotate("Recall RPN", (r["mask_AP"]["sd"], r["mask_AP"]["boot_sd"]), xytext=(5, 7),
                        textcoords="offset points", fontsize=5.8, color=INK2)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
                       ((BLUE, "MLP refiner (ours)"), (ORANGE, "Transformer refiner (ours)"), (PARENT, "Parent"),
                        (OTHER, "Other baselines"))], frameon=False, fontsize=6, loc="lower right")
    lim = max(max(r["mask_AP"]["sd"], r["mask_AP"]["boot_sd"]) for r in S["methods"].values()) * 1.15
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.7, ls="--")
    ax.annotate("equal contribution", (lim * 0.62, lim * 0.62), xytext=(4, -8), textcoords="offset points",
                fontsize=5.8, color=INK2, rotation=33)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Seed-to-seed SD of mask AP (points)"); ax.set_ylabel("Bootstrap SD of 3-seed mean (points)")
    ax.set_title("Uncertainty: test-set sampling (y) vs seed (x)", loc="left", fontsize=7.5)
    grid(ax, "both")
    save(fig, "fig_s11_uncertainty_sources")


def main() -> int:
    S, runs, scores = load()
    fig_method_ap(S); fig_method_ap_bars(S); fig_method_ap_columns(S); fig_method_ap_columns_zoom(S); fig_forest(S); fig_iou(S, scores); fig_class_heatmap(S, runs); fig_mask_quality(runs)
    fig_confusion(runs); fig_errors(runs); fig_strata(S); fig_pr(runs); fig_old_vs_new(S); fig_variance(S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
