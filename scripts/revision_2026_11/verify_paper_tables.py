#!/usr/bin/env python
"""Check the submitted manuscript's Tables 5, A1, A3, A4, A5 and 6 against the frozen campaign outputs.

Paper values are transcribed from version 2 of the manuscript. Each value is recomputed from raw files
(metrics_fullprec_*.json, analysis/runs/*.json, eval_record_*.json, runs/*/status.jsonl) and from the
campaign statistics (statistics.json, results_numbers.json), then compared at the paper's printed rounding.
Writes revision_2026-11/table_verification.json and prints every mismatch.
"""
import glob
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pixray_multiseed_bs4_e50_2026-09-14"
SEEDS = (42, 123, 2026)
ORDER = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "09M"]
SLUG = {"01": "m01_mask_rcnn_r101_fpn", "02": "m02_cascade_r101_fpn", "03": "m03_cascade_r101_attfpn",
        "04": "m04_cascade_attfpn_recall_rpn", "05": "m05_rsprompter_sam_lora", "06": "m06_occ_bilayer_sam",
        "07": "m07_bilayer_fcn_recall_rpn", "08": "m08_pidray_r0r1", "09": "m09_point_refine_transformer",
        "09M": "m09m_point_refine_mlp"}
NAMES = ["Mask R-CNN", "Cascade", "Parent", "Recall RPN", "RSPrompter", "Occ SAM", "Bilayer FCN", "R0/R1",
         "Transformer", "MLP"]

# ---------------- paper values (manuscript version 2) ----------------
T5 = {  # class: values in ORDER
    "Dart": [37.0, 37.4, 38.0, 38.1, 20.0, 20.2, 38.8, 37.8, 37.7, 37.7],
    "Saw_blade": [43.2, 47.8, 48.2, 47.8, 38.7, 39.0, 48.5, 48.5, 47.8, 52.5],
    "Knife": [53.1, 56.3, 56.5, 56.2, 45.4, 45.7, 56.7, 56.3, 57.0, 57.9],
    "Screwdriver": [56.7, 59.6, 60.4, 59.8, 52.7, 53.1, 60.6, 60.0, 63.7, 63.1],
    "Pliers": [59.0, 61.7, 61.8, 61.0, 51.1, 51.1, 61.6, 61.3, 62.5, 66.1],
    "Lighter": [66.6, 68.5, 68.7, 68.2, 59.0, 58.9, 68.9, 68.7, 69.7, 69.2],
    "Razor_blade": [69.6, 69.9, 70.1, 70.3, 61.2, 61.3, 70.3, 69.9, 70.0, 70.3],
    "Wrench": [67.0, 70.8, 71.3, 71.0, 66.2, 66.5, 71.8, 71.3, 73.2, 74.2],
    "Scissors": [73.7, 75.7, 75.7, 75.1, 62.1, 63.1, 75.8, 75.6, 77.1, 77.3],
    "Fireworks": [74.8, 76.5, 77.2, 76.7, 73.9, 73.8, 76.6, 76.8, 78.4, 78.3],
    "Battery": [79.0, 80.0, 80.4, 80.0, 74.0, 73.7, 80.4, 80.0, 80.4, 80.4],
    "Gun": [83.3, 84.7, 84.9, 84.7, 80.5, 80.9, 85.0, 84.8, 86.1, 86.2],
    "Hammer": [84.2, 85.8, 85.4, 85.8, 78.4, 78.5, 85.5, 85.4, 86.5, 86.3],
    "Pressure_vessel": [86.3, 89.2, 89.2, 89.0, 84.8, 85.2, 89.6, 89.2, 91.9, 92.5],
    "Bat": [88.7, 89.2, 89.7, 89.8, 88.8, 88.9, 89.6, 89.6, 94.7, 95.1],
}
T5_N = {"Dart": 323, "Saw_blade": 287, "Knife": 330, "Screwdriver": 287, "Pliers": 264, "Lighter": 265,
        "Razor_blade": 271, "Wrench": 349, "Scissors": 290, "Fireworks": 282, "Battery": 360, "Gun": 292,
        "Hammer": 322, "Pressure_vessel": 297, "Bat": 289}
# Table A1: class: (T delta, T lo, T hi, T p, T holm, M delta, M lo, M hi, M p, M holm); p '<0.001' -> None
A1 = {
    "Dart": (-0.3, -1.6, 0.7, "0.576", "1", -0.3, -1.0, 0.6, "0.49", "1"),
    "Saw_blade": (-0.4, -2.3, 1.3, "0.683", "1", 4.3, 2.7, 5.7, "<0.001", "<0.001"),
    "Knife": (0.5, -0.6, 1.6, "0.414", "1", 1.3, 0.0, 2.5, "0.0309", "0.216"),
    "Screwdriver": (3.3, 1.8, 4.6, "<0.001", "<0.001", 2.7, 1.4, 3.9, "<0.001", "<0.001"),
    "Pliers": (0.7, -1.2, 2.3, "0.43", "1", 4.2, 3.2, 5.5, "<0.001", "<0.001"),
    "Lighter": (1.0, 0.2, 1.9, "0.0222", "0.155", 0.5, -0.4, 1.4, "0.321", "1"),
    "Razor_blade": (-0.1, -1.0, 0.8, "0.858", "1", 0.3, -0.9, 1.5, "0.669", "1"),
    "Wrench": (1.9, 0.8, 2.9, "<0.001", "0.00542", 2.8, 1.9, 3.7, "<0.001", "<0.001"),
    "Scissors": (1.5, 0.7, 2.4, "<0.001", "0.00643", 1.6, 0.9, 2.4, "<0.001", "<0.001"),
    "Fireworks": (1.3, 0.3, 2.4, "0.0167", "0.134", 1.1, 0.0, 2.3, "0.0614", "0.307"),
    "Battery": (0.0, -0.7, 0.8, "0.999", "1", 0.1, -0.7, 1.0, "0.848", "1"),
    "Gun": (1.2, 0.3, 1.9, "0.00329", "0.0296", 1.3, 0.5, 2.1, "0.00168", "0.0135"),
    "Hammer": (1.0, 0.4, 1.6, "0.00101", "0.0101", 0.9, -0.1, 1.5, "0.0385", "0.231"),
    "Pressure_vessel": (2.7, 1.6, 3.8, "<0.001", "<0.001", 3.3, 2.3, 4.4, "<0.001", "<0.001"),
    "Bat": (5.0, 4.2, 6.0, "<0.001", "<0.001", 5.4, 4.7, 6.2, "<0.001", "<0.001"),
}
A3 = [[44.2, 54.9, 68.6, 57.3, 66.6, 69.4], [42.7, 58.2, 70.1, 58.0, 67.7, 70.8], [43.8, 59.1, 69.5, 60.9, 67.9, 70.1],
      [45.5, 58.7, 69.8, 62.0, 68.0, 70.4], [32.3, 47.7, 67.1, 43.6, 57.6, 67.7], [33.7, 48.0, 64.5, 46.3, 57.5, 65.1],
      [43.7, 59.0, 70.0, 62.3, 68.3, 70.7], [41.1, 58.8, 69.9, 58.6, 67.8, 70.6], [46.5, 59.8, 72.0, 57.6, 68.5, 72.6],
      [46.2, 60.5, 72.6, 58.8, 68.6, 73.2]]
A4 = [[94.7, 93.3, 94.0, 82.9, 81.6, 82.3], [96.9, 93.0, 94.9, 86.6, 83.0, 84.8], [97.2, 93.3, 95.2, 86.9, 83.4, 85.1],
      [96.8, 93.6, 95.2, 86.0, 83.1, 84.5], [89.4, 92.9, 91.1, 71.2, 73.9, 72.5], [89.4, 92.9, 91.1, 71.1, 73.9, 72.5],
      [96.7, 93.7, 95.2, 86.4, 83.7, 85.1], [97.2, 93.3, 95.2, 86.7, 83.2, 84.9], [97.3, 93.1, 95.1, 86.9, 83.2, 85.0],
      [97.5, 93.3, 95.4, 88.0, 84.3, 86.1]]
A5 = [[94.2, 85.3, 91.7, 42.5], [93.7, 86.0, 92.2, 45.5], [94.0, 86.1, 92.2, 45.6], [94.2, 86.0, 92.1, 45.1],
      [95.1, 81.5, 89.2, 26.4], [94.9, 81.6, 89.3, 26.8], [94.4, 86.1, 92.2, 45.5], [93.9, 86.1, 92.2, 45.4],
      [93.7, 86.7, 92.6, 49.8], [93.8, 87.2, 92.9, 50.3]]
T6 = [(63038634, 1.46, 33.7, 4.0), (96066744, 2.71, 26.1, 1.2), (97103679, 2.99, 24.6, 0.3), (97107534, 3.07, 23.9, 3.2),
      (332667534, 4.80, 14.4, 0.3), (336725874, 7.21, 14.3, 1.6), (105173841, 4.14, 21.0, 0.3),
      (98684224, 2.88, 23.5, 0.2), (97562062, 5.22, 22.3, 2.8), (97561806, 4.87, 23.3, 2.8)]


def rnd(x, nd):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def fmt_p(p):
    """Reproduce the paper's p formatting: <0.001, else 3 significant figures; 1 for p >= 0.9995 when Holm."""
    if p < 0.001:
        return "<0.001"
    if p >= 0.9995:
        return "1"
    s = f"{p:.3g}"
    return s


mism = []


def check(table, what, paper, computed, nd):
    if isinstance(paper, str):
        ok = paper == computed
    else:
        ok = rnd(computed, nd) == paper or (paper == 0.0 and rnd(computed, nd) == 0.0)
    if not ok:
        mism.append(dict(table=table, item=what, paper=paper, computed=computed))
    return ok


def load(pattern):
    return json.loads(open(glob.glob(str(pattern))[0]).read())


def main():
    stats = json.loads((OUT / "statistics/statistics.json").read_text())
    rn = json.loads((OUT / "reports/results_discussion_2026-09-25/data/results_numbers.json").read_text())
    fp = {(m, s): load(OUT / f"evaluations/{SLUG[m]}/seed_{s}/metrics_fullprec_*.json") for m in ORDER for s in SEEDS}
    er = {(m, s): load(OUT / f"evaluations/{SLUG[m]}/seed_{s}/eval_record_*.json") for m in ORDER for s in SEEDS}
    an = {(m, s): json.loads((OUT / f"analysis/runs/{SLUG[m]}/seed_{s}.json").read_text()) for m in ORDER for s in SEEDS}
    gt = json.loads((ROOT / "data/PIXray/annotations/test.json").read_text())
    cid = {c["name"]: c["id"] for c in gt["categories"]}
    n_checked = 0

    # ---- Table 5: per-class AP, raw per-seed per_class from the scorer ----
    for cls, vals in T5.items():
        n = sum(1 for a in gt["annotations"] if a["category_id"] == cid[cls])
        check("T5", f"{cls} n", T5_N[cls], n, 0); n_checked += 1
        for m, pv in zip(ORDER, vals):
            pcs = []
            for s in SEEDS:
                pc = fp[(m, s)]["segm"]["per_class"]
                v = pc[cls] if isinstance(pc, dict) and cls in pc else (pc[str(cid[cls])] if isinstance(pc, dict) else pc[cid[cls]])
                pcs.append(100 * (v["AP"] if isinstance(v, dict) else v))
            check("T5", f"{cls} {NAMES[ORDER.index(m)]}", pv, float(np.mean(pcs)), 1); n_checked += 1

    # ---- Table A1: per-class deltas / CIs (statistics.json) and p (results_numbers.json) ----
    comp = {(c["a"], c["b"]): c for c in stats["comparisons"]}
    for cls, row in A1.items():
        for off, a, key in ((0, "09", "09-03"), (5, "09M", "09M-03")):
            c = comp[(a, "03")]["mask_per_class"][cls]
            check("A1", f"{cls} {a} delta", row[off], c["delta"], 1)
            check("A1", f"{cls} {a} lo", row[off + 1], c["ci95"][0], 1)
            check("A1", f"{cls} {a} hi", row[off + 2], c["ci95"][1], 1)
            check("A1", f"{cls} {a} p", row[off + 3], fmt_p(rn["per_class_p"][key][cls]), 0)
            check("A1", f"{cls} {a} holm", row[off + 4], fmt_p(rn["per_class_p_holm"][key][cls]), 0)
            n_checked += 5

    # ---- Table A3: size AP / AR (official stats, 3-seed mean) ----
    keys = ["AP_small", "AP_medium", "AP_large", "AR_small", "AR_medium", "AR_large"]
    for m, row in zip(ORDER, A3):
        for k, pv in zip(keys, row):
            v = np.mean([100 * fp[(m, s)]["segm"]["stats"][k] for s in SEEDS])
            check("A3", f"{NAMES[ORDER.index(m)]} {k}", pv, v, 1); n_checked += 1

    # ---- Table A4: P/R/F1 at score 0.50, IoU 0.50 and 0.75 ----
    for m, row in zip(ORDER, A4):
        vals = []
        for t in ("0.50", "0.75"):
            for k in ("P", "R", "F1"):
                vals.append(np.mean([100 * fp[(m, s)]["segm"][f"prf_iou_{t}"]["at_0.50"][k] for s in SEEDS]))
        for lab, pv, v in zip(["P50", "R50", "F1_50", "P75", "R75", "F1_75"], row, vals):
            check("A4", f"{NAMES[ORDER.index(m)]} {lab}", pv, v, 1); n_checked += 1

    # ---- Table A5: coverage, mean IoU, Dice, share IoU >= 0.90 ----
    for m, row in zip(ORDER, A5):
        q = lambda k: np.mean([100 * an[(m, s)]["mask_quality"][k] for s in SEEDS])
        for lab, pv, k in zip(["coverage", "mean IoU", "Dice", ">=0.90"], row,
                              ["coverage", "mean_iou", "mean_dice", "iou_ge_0_90"]):
            check("A5", f"{NAMES[ORDER.index(m)]} {lab}", pv, q(k), 1); n_checked += 1

    # ---- Table 6: parameters (A5 count), training hours (status.jsonl), throughput (eval records) ----
    params = json.loads((OUT / "revision_2026-11/a5_parameters.json").read_text())
    for m, (tot, th, ips, sd) in zip(ORDER, T6):
        check("T6", f"{NAMES[ORDER.index(m)]} params", tot, params[m]["total"], 0)
        hours = []
        for s in SEEDS:
            ev = [json.loads(x) for x in (OUT / f"runs/{SLUG[m]}/seed_{s}/status.jsonl").read_text().splitlines()]
            start, t = None, 0.0
            for e in ev:
                if e["event"] == "train_started":
                    start = e["time"]
                elif e["event"] == "train_complete" and start is not None:
                    t += e["time"] - start; start = None
            hours.append(t / 3600)
        check("T6", f"{NAMES[ORDER.index(m)]} train h", th, float(np.mean(hours)), 2)
        v = [er[(m, s)]["efficiency"]["images_per_second"] for s in SEEDS]
        check("T6", f"{NAMES[ORDER.index(m)]} img/s", ips, float(np.mean(v)), 1)
        check("T6", f"{NAMES[ORDER.index(m)]} img/s SD", sd, float(np.std(v, ddof=1)), 1)
        n_checked += 4

    # ---- text claims about per-class counts (Section 5.3) ----
    for key, a, pos_claim, sig_claim in (("09M-03", "09M", 14, 8), ("09-03", "09", 11, 7)):
        d = comp[(a, "03")]["mask_per_class"]
        pos = sum(1 for c in d if d[c]["delta"] > 0)
        sig = sum(1 for c in d if rn["per_class_p_holm"][key][c] < 0.05 and d[c]["delta"] > 0)
        check("text", f"{a} classes positive", pos_claim, pos, 0)
        check("text", f"{a} classes Holm-significant", sig_claim, sig, 0)
        n_checked += 2

    res = dict(checked=n_checked, mismatches=mism)
    (OUT / "revision_2026-11/table_verification.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"checked {n_checked} values; mismatches: {len(mism)}")
    for x in mism:
        print("  ", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
