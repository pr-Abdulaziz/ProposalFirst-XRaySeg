#!/usr/bin/env python
"""Fig. 6 redrawn with BCa intervals. Same layout as results_section.py's fig_seeds;
only the 95% intervals change, from percentile to BCa (a3_bootstrap/intervals.json). Per-seed points come
from the official scorer files."""
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parents[2] / "output/pixray_multiseed_bs4_e50_2026-09-14"
REV = OUT / "revision_2026-11"
SLUG = {"03": "m03_cascade_r101_attfpn", "09": "m09_point_refine_transformer", "09M": "m09m_point_refine_mlp"}
SEEDS = (42, 123, 2026)
iv = json.loads((REV / "a3_bootstrap/intervals.json").read_text())["methods"]
ap = {(m, s): 100 * json.load(open(glob.glob(str(OUT / f"evaluations/{SLUG[m]}/seed_{s}/metrics_fullprec_*.json"))[0]))["segm"]["stats"]["AP"]
      for m in SLUG for s in SEEDS}
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 7})
col = {"03": "#555555", "09": "#d95f02", "09M": "#1f6fb2"}
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for s in SEEDS:
    ys = [ap[(m, s)] for m in ("03", "09", "09M")]
    ax.plot([0, 1, 2], ys, color="#bbbbbb", lw=0.8, zorder=1)
    ax.scatter([0, 1, 2], ys, s=10, color="white", edgecolor="#888888", lw=0.7, zorder=2)
for x, m in enumerate(("03", "09", "09M")):
    r = iv[m]; mean = r["point"]; lo, hi = r["bca"]
    assert abs(mean - sum(ap[(m, s)] for s in SEEDS) / 3) < 1e-9
    ax.errorbar(x + 0.12, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", ms=4, color=col[m], capsize=2, lw=1.2, zorder=3)
    print(f"{m}: mean {mean:.2f} BCa [{lo:.2f}, {hi:.2f}]")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Parent", "Transformer\nrefiner", "MLP\nrefiner"])
ax.set_ylabel("mask AP (%)"); ax.set_xlim(-0.4, 2.5)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5", lw=0.5)
for ext in ("pdf", "png"):
    fig.savefig(REV / f"figures/fig6_seeds_bca.{ext}", bbox_inches="tight", dpi=300)
