#!/usr/bin/env python
"""Numbers, tables and statistical tests for the PIXray Results-and-Discussion document.

Reads only the frozen outputs of the PIXray campaign (statistics/statistics.json, analysis/,
evaluations/, runs/, figures/qualitative/) and writes the LaTeX tables and macros of
reports/results_discussion_2026-09-25/. No prediction is re-evaluated; every number in the
document is produced here.

Tests (all paired; sample sizes are written next to every result):
  AP-level      from statistics_campaign.py: mean over S = 3 seeds; 95% two-level bootstrap interval
                (B = 2,000; seeds and the N = 1,486 test images resampled); paired image-level
                permutation test (J = 5,000; smallest attainable p = 1/5,001); Holm within family;
                TOST (90% interval inside +-1 AP) for the equivalence hypothesis.
  seed-level    one-sample t-test of the three per-seed differences (df = 2), a deliberately
                conservative check that treats the training seed as the only replication unit.
  object-level  per object, the difference is averaged over the three seeds; 95% cluster-bootstrap
                interval (images resampled, B = 2,000) and a cluster sign-flip permutation test
                (all objects of an image flip together, J = 5,000). n objects and n images reported.
  interaction   gain(heavy occlusion) - gain(no occluder) in mask recall at IoU 0.75, cluster
                bootstrap over images; two-sided bootstrap p.
  from-interval where only a 95% bootstrap interval was stored (AP75-AP50 gain, per-class deltas),
                p is a normal approximation z = delta / (width / 3.92), labelled as such.

    python scripts/campaign_multiseed/results_section.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.campaign_common import ANN, BY_ID, OUT, SEEDS  # noqa: E402
from scripts.campaign_multiseed.statistics_campaign import (FILL_BINS, OCC_BINS, RNG_SEED, holm,  # noqa: E402
                                                            object_attributes)

REP = OUT / "reports/results_discussion_2026-09-25"
TAB, DAT = REP / "tables", REP / "data"
B, J = 2000, 5000
IDS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "09M"]
SHORT = {"01": "Mask R-CNN", "02": "Cascade Mask R-CNN", "03": "Cascade + AttFPN (Parent)",
         "04": "Parent + recall RPN", "05": "RSPrompter (SAM + LoRA)", "06": "Occ bilayer SAM$^{\\dagger}$",
         "07": "Bilayer FCN$^{\\dagger}$", "08": "Parent + PIDray R0/R1", "09": "\\textbf{Ours: transformer refiner}",
         "09M": "\\textbf{Ours: MLP refiner}"}
PAIR = {("09", "03"): "Transformer refiner vs Parent", ("09M", "03"): "MLP refiner vs Parent",
        ("09M", "09"): "MLP vs transformer refiner", ("02", "01"): "Cascade vs Mask R-CNN",
        ("03", "02"): "AttFPN vs FPN", ("04", "03"): "Recall RPN vs Parent", ("08", "03"): "R0/R1 gate vs Parent",
        ("07", "04"): "Occluder masks (Bilayer FCN)", ("06", "05"): "Occluder masks (SAM bilayer)"}
FAMILY_NAME = {"primary": "H1", "equivalence": "H2", "architecture": "H4", "occluder": "H5"}
MACROS: dict[str, str] = {}
WORD = {"01": "One", "02": "Two", "03": "Three", "04": "Four", "05": "Five", "06": "Six", "07": "Seven",
        "08": "Eight", "09": "Nine", "09M": "NineM"}
THR = {50: "Fifty", 55: "FiftyFive", 60: "Sixty", 65: "SixtyFive", 70: "Seventy", 75: "SeventyFive",
       80: "Eighty", 85: "EightyFive", 90: "Ninety", 95: "NinetyFive"}


def W(*ids):
    """Method ids -> letters for LaTeX macro names (09M -> NineM, 03 -> Three)."""
    return "".join(WORD[i] for i in ids)


def mac(name, value):
    MACROS[name] = value
    return value


def f2(x, d=2, sign=False):
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "$-$")


def fp(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci_txt(ci, d=2):
    return f"[{f2(ci[0], d, True)}, {f2(ci[1], d, True)}]"


def p_from_ci(delta, ci):
    se = (ci[1] - ci[0]) / 3.92
    return float(2 * sps.norm.sf(abs(delta) / se)) if se > 0 else float("nan")


def seed_t(per_seed):
    x = np.asarray(list(per_seed), float)
    t, p = sps.ttest_1samp(x, 0.0)
    return float(t), float(p)


def write(name, text):
    """Shrink a tabular to the line width only when it is wider (\\fitwidth is defined in main.tex)."""
    text = text.replace("\\begin{tabular}", "\\fitwidth{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")
    (TAB / name).write_text(text)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True); DAT.mkdir(parents=True, exist_ok=True)
    S = json.loads((OUT / "statistics/statistics.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN["test"]))
    img_ids = sorted(gt.getImgIds()); N = len(img_ids)
    anns = gt.loadAnns(gt.getAnnIds(iscrowd=None)); n_obj = len(anns)
    cats = sorted(gt.loadCats(gt.getCatIds()), key=lambda c: c["id"]); cname = [c["name"] for c in cats]
    per_class_n = {c["name"]: len(gt.getAnnIds(catIds=[c["id"]], iscrowd=None)) for c in cats}
    mac("NImages", f"{N:,}"); mac("NObjects", f"{n_obj:,}"); mac("NClasses", str(len(cats)))
    mac("NSeeds", str(len(SEEDS))); mac("NBoot", f"{B:,}"); mac("NPerm", f"{J:,}"); mac("PMin", f"{1 / (J + 1):.4f}")

    # ------------------------------------------------ per-run COCO stats, efficiency, training time
    full, eff, train_h, analysis = {}, {}, {}, {}
    for m in IDS:
        slug = BY_ID[m].slug
        for s in SEEDS:
            d = next((OUT / "evaluations" / slug / f"seed_{s}").glob("metrics_fullprec_*.json"))
            full[(m, s)] = json.loads(d.read_text())
            rec = json.loads(next((OUT / "evaluations" / slug / f"seed_{s}").glob("eval_record_*.json")).read_text())
            eff[(m, s)] = rec["efficiency"]
            analysis[(m, s)] = json.loads((OUT / "analysis/runs" / slug / f"seed_{s}.json").read_text())
            ev = [json.loads(x) for x in (OUT / "runs" / slug / f"seed_{s}" / "status.jsonl").read_text().splitlines()]
            start, tot = None, 0.0
            for e in ev:
                if e["event"] == "train_started":
                    start = e["time"]
                elif e["event"] == "train_complete" and start is not None:
                    tot += e["time"] - start; start = None
            train_h[(m, s)] = tot / 3600

    def mean(fn, m):
        return float(np.mean([fn(m, s) for s in SEEDS]))

    # ------------------------------------------------ Table: main results
    rows = []
    for m in IDS:
        r = S["methods"][m]
        st = lambda k, t="segm": mean(lambda mm, s: 100 * full[(mm, s)][t]["stats"][k], m)  # noqa: E731
        rows.append(dict(
            m=m, ap=r["mask_AP"]["mean"], sd=r["mask_AP"]["sd"], ci=r["mask_AP"]["ci95"], ap50=r["mask_AP50"]["mean"],
            ap75=r["mask_AP75"]["mean"], aps=st("AP_small"), apm=st("AP_medium"), apl=st("AP_large"), ar=st("AR100"),
            box=r["box_AP"]["mean"], box_sd=r["box_AP"]["sd"],
            ips=mean(lambda mm, s: eff[(mm, s)]["images_per_second"], m),
            mem=mean(lambda mm, s: eff[(mm, s)]["peak_allocated_mb"], m) / 1024,
            th=mean(lambda mm, s: train_h[(mm, s)], m)))
    best = {k: max(r[k] for r in rows) for k in ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar", "box")}
    B_ = lambda r, k, d=1: (f"\\textbf{{{f2(r[k], d)}}}" if abs(r[k] - best[k]) < 1e-9 else f2(r[k], d))  # noqa: E731
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.6pt}",
             r"\caption{Results on the PIXray test set (mean of three seeds; CI: 95\% two-level bootstrap). "
             r"AP$_{S/M/L}$ use PIXray's box-area field. Throughput: end-to-end, batch 1, RTX~4090. "
             r"$^{\dagger}$Extra occluder supervision. Bold: best.}\label{tab:main}",
             r"\begin{tabular}{@{}lcccccccccccc@{}}\toprule",
             r"& \multicolumn{8}{c}{Mask} & Box & \multicolumn{3}{c}{Cost} \\ \cmidrule(lr){2-9}\cmidrule(lr){10-10}\cmidrule(l){11-13}",
             r"Experiment & AP $\pm$ SD & 95\% CI & AP$_{50}$ & AP$_{75}$ & AP$_S$ & AP$_M$ & AP$_L$ & AR$_{100}$ & AP & img/s & GPU mem (GB) & train (h) \\ \midrule"]
    for r in rows:
        lines.append(f"{SHORT[r['m']]} & {B_(r, 'ap', 2)} $\\pm$ {f2(r['sd'])} & {ci_txt(r['ci'], 1).replace('+', '')} & "
                     f"{B_(r, 'ap50')} & {B_(r, 'ap75')} & {B_(r, 'aps')} & {B_(r, 'apm')} & {B_(r, 'apl')} & {B_(r, 'ar')} & "
                     f"{B_(r, 'box')} & {f2(r['ips'], 1)} & {f2(r['mem'], 2)} & {f2(r['th'], 1)} \\\\")
        if r["m"] in ("02", "04", "06", "08"):
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    write("tab_main.tex", "\n".join(lines) + "\n")
    R = {r["m"]: r for r in rows}
    for m in IDS:
        k = W(m)
        mac(f"AP{k}", f2(R[m]["ap"])); mac(f"SD{k}", f2(R[m]["sd"])); mac(f"IPS{k}", f2(R[m]["ips"], 1))
        mac(f"APL{k}", f2(R[m]["apl"], 1)); mac(f"APS{k}", f2(R[m]["aps"], 1)); mac(f"APM{k}", f2(R[m]["apm"], 1))
        mac(f"AR{k}", f2(R[m]["ar"], 1)); mac(f"BOX{k}", f2(R[m]["box"], 1)); mac(f"TH{k}", f2(R[m]["th"], 1))
        mac(f"APfifty{k}", f2(R[m]["ap50"], 1)); mac(f"APsev{k}", f2(R[m]["ap75"], 1)); mac(f"MEM{k}", f2(R[m]["mem"], 2))

    # ------------------------------------------------ object-level data
    attrs = object_attributes(gt)
    obj = np.array(sorted(a["id"] for a in anns))
    img_pos = {i: k for k, i in enumerate(img_ids)}
    obj_img = np.array([img_pos[gt.anns[o]["image_id"]] for o in obj])
    cov = np.array([attrs[o]["cov"] for o in obj]); fill = np.array([attrs[o]["fill"] for o in obj])
    IOU = {}; BIOU = {}
    for m in IDS:
        iou_s, biou_s = [], []
        for s in SEEDS:
            arr = np.load(OUT / "analysis/runs" / BY_ID[m].slug / f"seed_{s}_per_gt.npy")
            d = {int(r[0]): (r[1], r[3]) for r in arr}
            iou_s.append([d[o][0] if o in d else np.nan for o in obj]); biou_s.append([d[o][1] if o in d else np.nan for o in obj])
        IOU[m] = np.array(iou_s); BIOU[m] = np.array(biou_s)            # [S, O], NaN = not matched

    rng = np.random.default_rng(RNG_SEED)
    Wimg = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(float)     # [B, N] image resampling weights
    flips = rng.choice([-1.0, 1.0], size=(J, N))                             # [J, N] image-level sign flips

    def cluster_test(d, sel):
        """d: per-object values (seed-averaged differences), sel: bool mask. Mean, 95% CI, p (sign-flip)."""
        d, oi = d[sel], obj_img[sel]
        per_img = np.bincount(oi, weights=d, minlength=N); cnt = np.bincount(oi, minlength=N).astype(float)
        obs = per_img.sum() / cnt.sum()
        bt = (Wimg @ per_img) / np.maximum(Wimg @ cnt, 1e-12)
        pm = (flips @ per_img) / cnt.sum()
        p = float((1 + np.sum(np.abs(pm) >= abs(obs) - 1e-15)) / (1 + J))
        return dict(mean=100 * float(obs), ci95=[100 * float(x) for x in np.percentile(bt, [2.5, 97.5])], p=p,
                    n_obj=int(sel.sum()), n_img=int((cnt > 0).sum()))

    def obj_compare(a, b):
        det_a, det_b = ~np.isnan(IOU[a]), ~np.isnan(IOU[b])
        both = (det_a & det_b).all(0)
        out = dict(
            coverage=cluster_test((det_a.astype(float) - det_b.astype(float)).mean(0), np.ones(len(obj), bool)),
            iou=cluster_test(np.nan_to_num(IOU[a] - IOU[b]).mean(0), both),
            biou=cluster_test(np.nan_to_num(BIOU[a] - BIOU[b]).mean(0), both),
            recall75=cluster_test(((np.nan_to_num(IOU[a]) >= 0.75).astype(float) - (np.nan_to_num(IOU[b]) >= 0.75)).mean(0),
                                  np.ones(len(obj), bool)))
        out["share_improved"] = float(((np.nan_to_num(IOU[a] - IOU[b]).mean(0))[both] > 0).mean())
        return out

    # ------------------------------------------------ hypothesis table (AP-level + seed-level)
    comp = {(c["a"], c["b"]): c for c in S["comparisons"]}
    OBJ = {k: obj_compare(*k) for k in comp}
    hyp_rows = []
    order = [("09", "03"), ("09M", "03"), ("09M", "09"), ("02", "01"), ("03", "02"), ("04", "03"), ("08", "03"),
             ("07", "04"), ("06", "05")]
    fam_of = {("09", "03"): "H1", ("09M", "03"): "H1", ("09M", "09"): "H2", ("02", "01"): "H4", ("03", "02"): "H4",
              ("04", "03"): "H4", ("08", "03"): "H4", ("07", "04"): "H5", ("06", "05"): "H5"}
    for k in order:
        c = comp[k]; m = c["mask_AP"]
        t, pt = seed_t(m["per_seed"].values())
        robust = m["ci95"][0] > 0 or m["ci95"][1] < 0
        if k == ("09M", "09"):
            eq = -1 < m["ci90"][0] and m["ci90"][1] < 1
            verdict = "equivalent (TOST)" if eq else "not equivalent"
        elif m.get("p_holm", 1) < 0.05 and robust:
            verdict = "supported" if m["delta"] > 0 else "significant decrease"
        elif m.get("p_holm", 1) < 0.05:
            verdict = "mixed"
        else:
            verdict = "no effect"
        hyp_rows.append(dict(key=k, fam=fam_of[k], label=c["label"], delta=m["delta"], per_seed=list(m["per_seed"].values()),
                             ci95=m["ci95"], ci90=m["ci90"], p_perm=m["p_perm"], p_holm=m.get("p_holm"), t=t, p_t=pt,
                             verdict=verdict))
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.4pt}",
             r"\caption{Planned paired comparisons on mask AP (points). $\Delta$ is the mean of the three per-seed "
             r"differences (shown individually). Sample: $S=3$ seeds $\times$ $N=" + MACROS["NImages"] + r"$ test images "
             r"(" + MACROS["NObjects"] + r" objects), identical for both arms. 95\% CI: two-level bootstrap (seeds and "
             r"images, $B=" + MACROS["NBoot"] + r"$). $p_\text{perm}$: paired image-level permutation test ($J=" +
             MACROS["NPerm"] + r"$, smallest attainable $p=" + MACROS["PMin"] + r"$); $p_\text{Holm}$: Holm-adjusted within "
             r"the family (H1: $m=2$; H4: $m=4$; H5: $m=2$). $t(2)$: one-sample $t$-test on the three per-seed "
             r"differences (seed as the only replication unit). H2 is an equivalence test: equivalent if the 90\% CI "
             r"lies inside $\pm1$ AP. \emph{Mixed}: significant by permutation but the CI includes 0.}\label{tab:hyp}",
             r"\begin{tabular}{@{}llccccccl@{}}\toprule",
             r"H & Comparison & $\Delta$ AP & per seed (42/123/2026) & 95\% CI & $p_\text{perm}$ & $p_\text{Holm}$ & $t(2)$, $p$ & Verdict \\ \midrule"]
    for r in hyp_rows:
        a, b = r["key"]
        vtxt = r["verdict"] + (" (90\\% CI " + ci_txt(r["ci90"]) + ")" if r["key"] == ("09M", "09") else "")
        lines.append(f"{r['fam']} & {r['label']} ({a}$-${b}) & {f2(r['delta'], 2, True)} & "
                     f"{'/'.join(f2(x, 2, True) for x in r['per_seed'])} & {ci_txt(r['ci95'])} & "
                     f"{fp(r['p_perm'])} & {fp(r['p_holm'])} & {f2(r['t'], 1)}, {fp(r['p_t'])} & "
                     f"{vtxt} \\\\")
        if r["key"] in (("09M", "03"), ("09M", "09"), ("08", "03")):
            lines.append(r"\addlinespace[2pt]")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_hypotheses.tex", "\n".join(lines) + "\n")
    for r in hyp_rows:
        a, b = r["key"]; key = W(a, b)
        mac(f"D{key}", f2(r["delta"], 2, True)); mac(f"CI{key}", ci_txt(r["ci95"])); mac(f"CInine{key}", ci_txt(r["ci90"]))
        mac(f"P{key}", fp(r["p_holm"] if r["p_holm"] is not None else r["p_perm"])); mac(f"T{key}", f2(r["t"], 1))
        mac(f"PT{key}", fp(r["p_t"])); mac(f"SEEDS{key}", " / ".join(f2(x, 2, True) for x in r["per_seed"]))
        c = comp[r["key"]]
        for kk, nm in (("mask_AP50", "fifty"), ("mask_AP75", "sev"), ("box_AP", "box")):
            mac(f"D{nm}{key}", f2(c[kk]["delta"], 2, True)); mac(f"CI{nm}{key}", ci_txt(c[kk]["ci95"]))

    # ------------------------------------------------ H3: mechanism (strict IoU, boundary, detection unchanged)
    by = S["mask_ap_by_iou"]; thr = [0.50 + 0.05 * i for i in range(10)]
    h3 = []
    for a in ("09", "09M"):
        c = comp[(a, "03")]; g = c["mask_gain_AP75_minus_AP50"]; o = OBJ[(a, "03")]
        h3 += [(a, "AP$_{75}$ gain $-$ AP$_{50}$ gain", "3$\\times$" + MACROS["NImages"] + " img.", g["delta"], g["ci95"],
                p_from_ci(g["delta"], g["ci95"]), "CI$\\to$z", "$>0$"),
               (a, "boundary IoU, matched objects", f"{o['biou']['n_obj']:,} obj. / {o['biou']['n_img']:,} img.",
                o["biou"]["mean"], o["biou"]["ci95"], o["biou"]["p"], "sign-flip", "$>0$"),
               (a, "mask IoU, matched objects", f"{o['iou']['n_obj']:,} obj. / {o['iou']['n_img']:,} img.",
                o["iou"]["mean"], o["iou"]["ci95"], o["iou"]["p"], "sign-flip", "$>0$"),
               (a, "mask AP$_{50}$", "3$\\times$" + MACROS["NImages"] + " img.", c["mask_AP50"]["delta"],
                c["mask_AP50"]["ci95"], p_from_ci(c["mask_AP50"]["delta"], c["mask_AP50"]["ci95"]), "CI$\\to$z", "$=0$"),
               (a, "detection coverage", f"{o['coverage']['n_obj']:,} obj. / {N:,} img.",
                o["coverage"]["mean"], o["coverage"]["ci95"], o["coverage"]["p"], "sign-flip", "$=0$")]
    ph = holm({i: r[5] for i, r in enumerate(h3) if r[7] == "$>0$"})
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
             r"\caption{H3 (mechanism): refinement should improve strict-IoU and boundary quality while leaving "
             r"detection unchanged. Differences to the parent (03) in points. Object-level rows average each object's "
             r"difference over the three seeds and use objects matched by both methods in every seed (mask and boundary "
             r"IoU) or all objects (coverage); intervals are cluster bootstrap over images ($B=" + MACROS["NBoot"] +
             r"$), $p$ from a sign-flip permutation test in which all objects of an image flip together ($J=" +
             MACROS["NPerm"] + r"$). Rows marked CI$\to$z derive $p$ from the stored two-level bootstrap interval "
             r"(normal approximation). $p_\text{Holm}$ over the six directional tests ($>0$). Boundary IoU uses a band of "
             r"$2\%$ of the image diagonal.}\label{tab:h3}",
             r"\begin{tabular}{@{}lllcccccc@{}}\toprule",
             r"Arm & Measure & Sample & $\Delta$ & 95\% CI & Test & $p$ & $p_\text{Holm}$ & Expected \\ \midrule"]
    for i, r in enumerate(h3):
        nm = "09 transformer" if r[0] == "09" else "09M MLP"
        lines.append(f"{nm} & {r[1]} & {r[2]} & {f2(r[3], 2, True)} & {ci_txt(r[4])} & {r[6]} & {fp(r[5])} & "
                     f"{fp(ph.get(i)) if i in ph else '--'} & {r[7]} \\\\")
        if i == 4:
            lines.append(r"\addlinespace[2pt]")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_h3.tex", "\n".join(lines) + "\n")
    for i, r in enumerate(h3):
        k = W(r[0]) + ["gain", "biou", "iou", "fifty", "cov"][i % 5]
        mac(f"Hthree{k}", f2(r[3], 2, True)); mac(f"HthreeCI{k}", ci_txt(r[4])); mac(f"HthreeP{k}", fp(ph.get(i, r[5])))
    for a in ("09", "09M"):
        k = W(a)
        d = [by[a][i] - by["03"][i] for i in range(10)]
        for i, t in enumerate(thr):
            mac(f"DIoU{k}{THR[int(round(t * 100))]}", f2(d[i], 2, True))
        mac(f"Share{k}", f"{100 * OBJ[(a, '03')]['share_improved']:.0f}")
        mac(f"NObjBoth{k}", f"{OBJ[(a, '03')]['iou']['n_obj']:,}")

    # ------------------------------------------------ matched-object quality + errors per method
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
             r"\caption{Object-level quality at score $\ge0.5$ (one-to-one same-class matching at box IoU $\ge0.5$; "
             r"mean over three seeds; " + MACROS["NObjects"] + r" objects). Coverage: share of objects detected. IoU, "
             r"Dice and boundary IoU over matched objects; $\ge$0.75 and $\ge$0.90: share of matched objects above that "
             r"mask IoU. Error counts per run (TIDE-style, box IoU): missed objects, wrong class, background false "
             r"positives.}\label{tab:quality}",
             r"\begin{tabular}{@{}lccccccccc@{}}\toprule",
             r"Method & Cov. & IoU & Dice & BIoU & $\ge$.75 & $\ge$.90 & missed & wrong cls & bkg FP \\ \midrule"]
    Q = {}
    for m in IDS:
        mq = lambda k: mean(lambda mm, s: analysis[(mm, s)]["mask_quality"][k], m)  # noqa: E731
        er = lambda k: mean(lambda mm, s: analysis[(mm, s)]["errors"]["counts"][k], m)  # noqa: E731
        Q[m] = dict(cov=100 * mq("coverage"), iou=100 * mq("mean_iou"), dice=100 * mq("mean_dice"),
                    biou=100 * mq("mean_boundary_iou"), r75=100 * mq("iou_ge_0_75"), r90=100 * mq("iou_ge_0_90"),
                    missed=er("missed"), wrong=er("wrong_class"), bkg=er("background"))
    bestq = {k: (max if k not in ("missed", "wrong", "bkg") else min)(Q[m][k] for m in IDS) for k in Q["03"]}
    for m in IDS:
        q = Q[m]
        cell = lambda k, d=1: (f"\\textbf{{{f2(q[k], d)}}}" if abs(q[k] - bestq[k]) < 1e-9 else f2(q[k], d))  # noqa: E731
        lines.append(f"{m} & {cell('cov')} & {cell('iou')} & {cell('dice')} & {cell('biou')} & {cell('r75')} & {cell('r90')} & "
                     f"{cell('missed', 0)} & {cell('wrong', 0)} & {cell('bkg', 0)} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_quality.tex", "\n".join(lines) + "\n")
    for m in IDS:
        k = W(m)
        for kk in ("cov", "iou", "biou", "r75", "r90", "missed", "wrong", "bkg"):
            mac(f"Q{ {'r75': 'rsev', 'r90': 'rnine'}.get(kk, kk)}{k}", f2(Q[m][kk], 0 if kk in ("missed", "wrong", "bkg") else 1))

    # ------------------------------------------------ H4 / H5 object-level companions
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
             r"\caption{Object-level companions of the architecture (H4) and occluder (H5) comparisons: "
             r"difference in mask recall at IoU 0.75 (all " + MACROS["NObjects"] + r" objects; missed = 0) and in "
             r"mask IoU of objects matched by both arms in every seed; cluster bootstrap 95\% CI and sign-flip "
             r"$p$ over " + MACROS["NImages"] + r" images, Holm within family.}\label{tab:objarch}",
             r"\begin{tabular}{@{}llccc@{}}\toprule",
             r"H & Comparison & $\Delta$ recall@.75 [95\% CI] & $\Delta$ IoU [95\% CI] & $n$ obj.\ (IoU) \\ \midrule"]
    for fam, keys in (("H4", [("02", "01"), ("03", "02"), ("04", "03"), ("08", "03")]), ("H5", [("07", "04"), ("06", "05")])):
        pr = holm({i: OBJ[k]["recall75"]["p"] for i, k in enumerate(keys)})
        pi = holm({i: OBJ[k]["iou"]["p"] for i, k in enumerate(keys)})
        for i, k in enumerate(keys):
            o = OBJ[k]
            lines.append(f"{fam} & {comp[k]['label']} & {f2(o['recall75']['mean'], 2, True)} {ci_txt(o['recall75']['ci95'])}, "
                         f"$p_H$={fp(pr[i])} & {f2(o['iou']['mean'], 2, True)} {ci_txt(o['iou']['ci95'])}, $p_H$={fp(pi[i])} & "
                         f"{o['iou']['n_obj']:,} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_objarch.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ H5 interaction + strata
    def interaction(a, b, bins, values, kind, hi, lo):
        d = ((np.nan_to_num(IOU[a]) >= 0.75).astype(float) - (np.nan_to_num(IOU[b]) >= 0.75)).mean(0)
        res = {}
        sels = {}
        for name, l, h in bins:
            if kind == "occlusion" and l < 0:
                sel = values == -0.5
            elif kind == "occlusion":
                sel = (values > l) & (values <= h)
            else:
                sel = (values >= l) & (values < h)
            sels[name] = sel
            res[name] = cluster_test(d, sel)
        def img_sums(sel):
            return (np.bincount(obj_img[sel], weights=d[sel], minlength=N),
                    np.bincount(obj_img[sel], minlength=N).astype(float))
        (sh, ch), (sl, cl) = img_sums(sels[hi]), img_sums(sels[lo])
        obs = sh.sum() / ch.sum() - sl.sum() / cl.sum()
        bt = (Wimg @ sh) / np.maximum(Wimg @ ch, 1e-12) - (Wimg @ sl) / np.maximum(Wimg @ cl, 1e-12)
        p = float(min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean())))
        return dict(bins=res, interaction=dict(delta=100 * float(obs), ci95=[100 * float(x) for x in np.percentile(bt, [2.5, 97.5])],
                                               p=p, contrast=f"{hi} minus {lo}"))
    H5I = {k: interaction(*k, OCC_BINS, cov, "occlusion", "heavy (>0.25)", "no occluder") for k in (("07", "04"), ("06", "05"))}
    EXI = {k: interaction(*k, FILL_BINS, fill, "thinness", "thin (<0.30)", "compact (>=0.60)") for k in (("09", "03"), ("09M", "03"))}
    EXO = {k: interaction(*k, OCC_BINS, cov, "occlusion", "heavy (>0.25)", "no occluder") for k in (("09", "03"), ("09M", "03"))}
    ph5 = holm({i: H5I[k]["interaction"]["p"] for i, k in enumerate(H5I)})
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.5pt}",
             r"\caption{Gain in mask recall at IoU 0.75 (points) by occlusion and thinness group; per object averaged "
             r"over three seeds; cluster bootstrap 95\% CI over images. Occlusion: fraction of the object covered by "
             r"its PIXray-A occluder (none / $\le$0.25 / $>$0.25); thinness: mask area over box area. The interaction "
             r"(last column) is gain(heavy) $-$ gain(none), or gain(thin) $-$ gain(compact), with bootstrap $p$; "
             r"H5 rows are Holm-adjusted ($m=2$). Rows marked \emph{expl.}\ were not planned in advance.}\label{tab:h5}",
             r"\begin{tabular}{@{}llcccc@{}}\toprule"]
    def block(title, groups, rowsd, padj=None):
        out = [f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"]
        names = list(next(iter(rowsd.values()))["bins"])
        ns = [next(iter(rowsd.values()))["bins"][nm]["n_obj"] for nm in names]
        out.append("& Comparison & " + " & ".join(f"{nm.split(' (')[0]} ($n$={n:,})" for nm, n in zip(names, ns)) +
                   " & Interaction [95\\% CI], $p$ \\\\ \\cmidrule(l){2-6}")
        for i, (k, lab) in enumerate(groups):
            r = rowsd[k]; it = r["interaction"]
            p = padj[i] if padj is not None else it["p"]
            cells = " & ".join(f"{f2(r['bins'][nm]['mean'], 2, True)} {ci_txt(r['bins'][nm]['ci95'], 1)}" for nm in names)
            out.append(f"& {lab} & {cells} & {f2(it['delta'], 2, True)} {ci_txt(it['ci95'], 1)}, {fp(p)} \\\\")
        return out
    lines += block("H5 --- occluder supervision vs occlusion", [(("07", "04"), "07$-$04 (Bilayer FCN)"),
                                                               (("06", "05"), "06$-$05 (Occ bilayer SAM)")], H5I, ph5)
    lines += [r"\addlinespace[3pt]"] + block("Expl.\\ --- refiners vs occlusion", [(("09", "03"), "09$-$03 transformer"),
                                                                                 (("09M", "03"), "09M$-$03 MLP")], EXO)
    lines += [r"\addlinespace[3pt]"] + block("Expl.\\ --- refiners vs thinness", [(("09", "03"), "09$-$03 transformer"),
                                                                                (("09M", "03"), "09M$-$03 MLP")], EXI)
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_h5.tex", "\n".join(lines) + "\n")
    for pre, dct in (("", H5I), ("Thin", EXI), ("Occ", EXO)):
      for k, r in dct.items():
        key = pre + W(*k)
        mac(f"INT{key}", f2(r["interaction"]["delta"], 2, True)); mac(f"INTCI{key}", ci_txt(r["interaction"]["ci95"]))
        mac(f"INTP{key}", fp(r["interaction"]["p"]))
        for nm, v in r["bins"].items():
            tag = {"no occluder": "none", "light (<=0.25)": "light", "heavy (>0.25)": "heavy", "thin (<0.30)": "thin",
                   "medium (0.30-0.60)": "medium", "compact (>=0.60)": "compact"}[nm]
            mac(f"G{key}{tag}", f2(v["mean"], 2, True)); mac(f"GCI{key}{tag}", ci_txt(v["ci95"], 2))
            mac(f"N{tag}", f"{v['n_obj']:,}")
    for i, k in enumerate(H5I):
        mac(f"INTPH{W(*k)}", fp(ph5[i]))

    # crowding (AP on image subsets, from statistics.json)
    cr = S["strata"]["crowding"]
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
             r"\caption{Mask AP by crowding (objects per image), mean over three seeds, and gain over the parent (03) "
             r"with 95\% bootstrap CI (images resampled within the subset).}\label{tab:crowd}",
             r"\begin{tabular}{@{}lccc@{}}\toprule",
             " & " + " & ".join(f"{b} ({v['images']} img., {v['objects']:,} obj.)".replace("objects (", "(") for b, v in cr.items()) + r" \\ \midrule"]
    for m in ("03", "07", "09", "09M"):
        cells = []
        for b, v in cr.items():
            txt = f2(v["methods"][m]["mean"])
            if m != "03":
                d = v["delta_vs_parent"][m]
                txt += f" ({f2(d['delta'], 2, True)} {ci_txt(d['ci95'], 1)})"
            cells.append(txt)
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_crowd.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ per-class table (09M-03, 09-03 with CI; Holm over 15)
    pcM, pcT = comp[("09M", "03")]["mask_per_class"], comp[("09", "03")]["mask_per_class"]
    pM = {c: p_from_ci(pcM[c]["delta"], pcM[c]["ci95"]) for c in cname}
    pT = {c: p_from_ci(pcT[c]["delta"], pcT[c]["ci95"]) for c in cname}
    hM, hT = holm(pM), holm(pT)
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
             r"\caption{Per-class mask AP and refiner gains over the Parent [95\% CI]. $n$: test objects; "
             r"$^{*}$Holm-adjusted $p<0.05$ over 15 classes.}\label{tab:perclass}",
             r"\begin{tabular}{@{}lrcccc@{}}\toprule",
             r"Class & $n$ & Parent & Bilayer FCN & Transformer $-$ Parent & MLP $-$ Parent \\ \midrule"]
    for c in sorted(cname, key=lambda c: S["methods"]["03"]["mask_per_class"][c]):
        st = lambda h: "$^{*}$" if h < 0.05 else ""  # noqa: E731
        lines.append(f"{c.replace('_', ' ')} & {per_class_n[c]} & {f2(S['methods']['03']['mask_per_class'][c], 1)} & "
                     f"{f2(S['methods']['07']['mask_per_class'][c], 1)} & {f2(pcT[c]['delta'], 1, True)}{st(hT[c])} "
                     f"{ci_txt(pcT[c]['ci95'], 1)} & {f2(pcM[c]['delta'], 1, True)}{st(hM[c])} {ci_txt(pcM[c]['ci95'], 1)} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_perclass.tex", "\n".join(lines) + "\n")
    mac("NSigClassM", str(sum(h < 0.05 for h in hM.values()))); mac("NSigClassT", str(sum(h < 0.05 for h in hT.values())))
    mac("NPosClassM", str(sum(pcM[c]["delta"] > 0 for c in cname))); mac("NPosClassT", str(sum(pcT[c]["delta"] > 0 for c in cname)))
    for c in ("Dart", "Saw_blade", "Pliers", "Screwdriver", "Knife", "Bat", "Pressure_vessel"):
        k = c.replace("_", "")
        mac(f"PC{k}M", f2(pcM[c]["delta"], 1, True)); mac(f"PC{k}T", f2(pcT[c]["delta"], 1, True))
        mac(f"PCCI{k}M", ci_txt(pcM[c]["ci95"], 1)); mac(f"PCCI{k}T", ci_txt(pcT[c]["ci95"], 1))

    # ------------------------------------------------ sample-size table
    lines = [r"\begin{table}[!htb]\centering\small\setlength{\tabcolsep}{3pt}",
             r"\caption{Units of analysis and sample sizes used by the tests.}\label{tab:samples}",
             r"\begin{tabular}{@{}lp{0.62\linewidth}@{}}\toprule",
             r"Unit & Size and role \\ \midrule",
             f"Training seeds & $S=3$ (42, 123, 2026) per method; 10 methods, 30 runs; paired by seed \\\\",
             f"Test images & $N={N:,}$ (PIXray test split, identical for every run); resampled by the bootstrap, "
             f"swapped by the permutation test \\\\",
             f"Test objects & {n_obj:,} instances in {len(cats)} classes ({min(per_class_n.values())}--{max(per_class_n.values())} "
             f"per class) \\\\",
             f"Matched pairs & {OBJ[('09M', '03')]['iou']['n_obj']:,} (09M/03) and {OBJ[('09', '03')]['iou']['n_obj']:,} (09/03) "
             f"objects matched by both arms in all seeds \\\\",
             f"Strata & crowding {', '.join(str(v['images']) for v in cr.values())} images; occlusion "
             f"{', '.join(MACROS['N' + t] for t in ('none', 'light', 'heavy'))} objects; thinness "
             f"{', '.join(MACROS['N' + t] for t in ('thin', 'medium', 'compact'))} objects \\\\",
             f"Resampling & $B={B:,}$ bootstrap replicates; $J={J:,}$ permutations ($p\\ge{1 / (J + 1):.4f}$) \\\\",
             r"Families & H1: 2 tests; H2: 1 (TOST); H3: 6 directional tests; H4: 4; H5: 2 (+2 interactions); per-class: 15 \\",
             r"\bottomrule\end{tabular}\end{table}"]
    write("tab_samples.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ failure statistics (from the qualitative suite)
    fs = json.loads((OUT / "figures/qualitative/failure_statistics.json").read_text())
    for a in ("09", "09M"):
        k = W(a); v = fs[f"{a}_vs_03"]
        mac(f"Worse{k}", f"{100 * np.mean(v['worse_ge_005']):.1f}"); mac(f"Better{k}", f"{100 * np.mean(v['better_ge_005']):.1f}")
    mac("BothWorse", str(fs["both_variants_below_base_in_all_seeds"]))
    mac("MissedAllTen", str(fs["missed_by_all_ten_in_all_seeds"]))
    mac("MissedThree", f2(fs["missed_per_method_mean_over_seeds"]["03"], 0))
    mac("MissedNine", f2(fs["missed_per_method_mean_over_seeds"]["09"], 0))
    mac("MissedNineM", f2(fs["missed_per_method_mean_over_seeds"]["09M"], 0))

    # ------------------------------------------------ detailed COCO metrics (box and mask), all methods
    KEYS = [("AP", "AP"), ("AP50", "AP$_{50}$"), ("AP75", "AP$_{75}$"), ("AP_small", "AP$_S$"), ("AP_medium", "AP$_M$"),
            ("AP_large", "AP$_L$"), ("AR1", "AR$_{1}$"), ("AR10", "AR$_{10}$"), ("AR100", "AR$_{100}$"),
            ("AR_small", "AR$_S$"), ("AR_medium", "AR$_M$"), ("AR_large", "AR$_L$")]
    for t, tname, tag in (("bbox", "Box", "box"), ("segm", "Mask", "mask")):
        vals = {m: {k: mean(lambda mm, s: 100 * full[(mm, s)][t]["stats"][k], m) for k, _ in KEYS} for m in IDS}
        sds = {m: {k: float(np.std([100 * full[(m, s)][t]["stats"][k] for s in SEEDS], ddof=1)) for k, _ in KEYS} for m in IDS}
        bestk = {k: max(vals[m][k] for m in IDS) for k, _ in KEYS}
        lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.3pt}",
                 f"\\caption{{All twelve COCO {tname.lower()} metrics (\\%), mean over three seeds (seed SD for AP in "
                 r"parentheses). S/M/L ranges use PIXray's box-area field. Best per column in bold.}"
                 f"\\label{{tab:coco{tag}}}",
                 r"\begin{tabular}{@{}l" + "c" * len(KEYS) + r"@{}}\toprule",
                 "Method & " + " & ".join(n for _, n in KEYS) + r" \\ \midrule"]
        for m in IDS:
            cells = []
            for k, _ in KEYS:
                v = f2(vals[m][k], 1)
                if abs(vals[m][k] - bestk[k]) < 1e-9:
                    v = f"\\textbf{{{v}}}"
                if k == "AP":
                    v += f" ({f2(sds[m][k], 2)})"
                cells.append(v)
            lines.append(f"{m} {SHORT[m]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule\end{tabular}\end{table*}")
        write(f"tab_coco_{tag}.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ change of every COCO metric vs the parent (09, 09M)
    lines = [r"\begin{table}[!htb]\centering\small\setlength{\tabcolsep}{2.8pt}",
             r"\caption{Change of every COCO metric relative to the parent (03), percentage points, mean $\pm$ SD of "
             r"the three per-seed differences; $k/3$: seeds with a positive difference. $^{\ddagger}$95\% two-level "
             r"bootstrap CI excludes 0 (available for AP, AP$_{50}$, AP$_{75}$).}\label{tab:delta}",
             r"\begin{tabular}{@{}llcccc@{}}\toprule",
             r" & Metric & 09 transformer & $k/3$ & 09M MLP & $k/3$ \\ \midrule"]
    cikey = {("segm", "AP"): "mask_AP", ("segm", "AP50"): "mask_AP50", ("segm", "AP75"): "mask_AP75",
             ("bbox", "AP"): "box_AP", ("bbox", "AP50"): "box_AP50", ("bbox", "AP75"): "box_AP75"}
    for t, tname in (("segm", "Mask"), ("bbox", "Box")):
        for i, (k, nm) in enumerate(KEYS):
            cells = []
            for a in ("09", "09M"):
                d = np.array([100 * (full[(a, s)][t]["stats"][k] - full[("03", s)][t]["stats"][k]) for s in SEEDS])
                sig = ""
                if (t, k) in cikey:
                    ci = comp[(a, "03")][cikey[(t, k)]]["ci95"]
                    sig = "$^{\\ddagger}$" if (ci[0] > 0 or ci[1] < 0) else ""
                cells += [f"{f2(d.mean(), 2, True)}{sig} $\\pm$ {f2(d.std(ddof=1), 2)}", f"{int((d > 0).sum())}/3"]
            lines.append(f"{tname if i == 0 else ''} & {nm} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if t == "segm" else "")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_delta.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ per-class mask AP, all methods
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.4pt}",
             r"\caption{Per-class mask AP (\%), mean over three seeds; classes from hardest to easiest by the mean over "
             r"the ten methods. $n$: test objects. Bold: best per class; last row counts classes won.}\label{tab:classall}",
             r"\begin{tabular}{@{}lr" + "c" * len(IDS) + r"@{}}\toprule",
             r"Class & $n$ & " + " & ".join(IDS) + r" \\ \midrule"]
    pcs = {m: S["methods"][m]["mask_per_class"] for m in IDS}
    wins = {m: 0 for m in IDS}
    for c in sorted(cname, key=lambda c: np.mean([pcs[m][c] for m in IDS])):
        bm = max(IDS, key=lambda m: pcs[m][c]); wins[bm] += 1
        cells = [f"\\textbf{{{f2(pcs[m][c], 1)}}}" if m == bm else f2(pcs[m][c], 1) for m in IDS]
        lines.append(f"{c.replace('_', ' ')} & {per_class_n[c]} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r"Classes won & & " + " & ".join(str(wins[m]) for m in IDS) + r" \\",
              r"\bottomrule\end{tabular}\end{table*}"]
    write("tab_classall.tex", "\n".join(lines) + "\n")
    mac("WinsNineM", str(wins["09M"])); mac("WinsNine", str(wins["09"]))
    mac("WinsOthers", str(sum(v for m, v in wins.items() if m not in ("09", "09M"))))

    # ------------------------------------------------ paired instance-level quality (all planned pairs)
    pq = json.loads((OUT / "analysis/paired_mask_quality.json").read_text())
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.6pt}",
             r"\caption{Instance-level mask quality on objects matched by \emph{both} methods (score $\ge0.5$, one-to-one, "
             r"same class, box IoU $\ge0.5$), percentage points, mean over seeds of the per-seed paired means. "
             r"$n$: common objects per seed (mean). Dice $=2|P\cap G|/(|P|+|G|)$; boundary IoU uses a band of 2\% of "
             r"the image diagonal. Impr.: share of common objects with higher IoU. The last two columns treat each "
             r"object's seed-averaged difference, matched in all seeds, with a cluster bootstrap CI and sign-flip "
             r"$p$ over images.}\label{tab:paired}",
             r"\begin{tabular}{@{}llcccccc@{}}\toprule",
             r"H & Pair & $n$ & $\Delta$IoU & $\Delta$Dice & $\Delta$B-IoU & Impr.\ (\%) & $\Delta$IoU [95\% CI], $p$ \\ \midrule"]
    for k in order:
        a, b = k
        ent = [x for x in pq if x["a"] == a and x["b"] == b]
        if not ent:
            continue
        mv = lambda f: float(np.mean([x[f] for x in ent]))  # noqa: E731
        o = OBJ[k]["iou"]
        lines.append(f"{fam_of[k]} & {a} vs {b} & {mv('common_gt'):,.0f} & {f2(100 * mv('d_iou'), 2, True)} & "
                     f"{f2(100 * mv('d_dice'), 2, True)} & {f2(100 * mv('d_boundary_iou'), 2, True)} & "
                     f"{100 * mv('share_improved'):.1f} & {f2(o['mean'], 2, True)} {ci_txt(o['ci95'])}, {fp(o['p'])} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_paired.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ precision / recall / F1 (mask, IoU 0.50 and 0.75)
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.4pt}",
             r"\caption{Mask precision, recall and F1 (\%) at the operating point score $\ge0.5$, for a match at mask "
             r"IoU $\ge0.50$ and $\ge0.75$ (" + MACROS["NObjects"] + r" objects; TP/FP/FN per run, mean over seeds), "
             r"and the best F1 over score thresholds with its threshold.}\label{tab:prf}",
             r"\begin{tabular}{@{}lcccccccccc@{}}\toprule",
             r" & \multicolumn{6}{c}{IoU $\ge$ 0.50} & \multicolumn{4}{c}{IoU $\ge$ 0.75} \\ \cmidrule(lr){2-7}\cmidrule(l){8-11}",
             r"Method & P & R & F1 & FP & FN & best F1 (thr.) & P & R & F1 & best F1 (thr.) \\ \midrule"]
    PRF = {}
    for m in IDS:
        g = lambda iou, sect, k: mean(lambda mm, s: full[(mm, s)]["segm"][f"prf_iou_{iou}"][sect][k], m)  # noqa: E731
        thr_ = lambda iou: np.mean([full[(m, s)]["segm"][f"prf_iou_{iou}"]["best"]["threshold"] for s in SEEDS])  # noqa: E731
        PRF[m] = dict(p5=100 * g("0.50", "at_0.50", "P"), r5=100 * g("0.50", "at_0.50", "R"), f5=100 * g("0.50", "at_0.50", "F1"),
                      fp=g("0.50", "at_0.50", "FP"), fn=g("0.50", "at_0.50", "FN"), b5=100 * g("0.50", "best", "F1"), t5=thr_("0.50"),
                      p7=100 * g("0.75", "at_0.50", "P"), r7=100 * g("0.75", "at_0.50", "R"), f7=100 * g("0.75", "at_0.50", "F1"),
                      b7=100 * g("0.75", "best", "F1"), t7=thr_("0.75"))
    bp = {k: max(PRF[m][k] for m in IDS) for k in ("p5", "r5", "f5", "b5", "p7", "r7", "f7", "b7")}
    for m in IDS:
        q = PRF[m]
        c = lambda k: f"\\textbf{{{f2(q[k], 1)}}}" if abs(q[k] - bp[k]) < 1e-9 else f2(q[k], 1)  # noqa: E731
        lines.append(f"{m} & {c('p5')} & {c('r5')} & {c('f5')} & {q['fp']:.0f} & {q['fn']:.0f} & {c('b5')} ({q['t5']:.2f}) & "
                     f"{c('p7')} & {c('r7')} & {c('f7')} & {c('b7')} ({q['t7']:.2f}) \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_prf.tex", "\n".join(lines) + "\n")
    for m in ("03", "07", "09", "09M"):
        for kk in ("f5", "f7", "p7", "r7"):
            mac(f"PRF{kk.replace('5', 'a').replace('7', 'b')}{W(m)}", f2(PRF[m][kk], 1))

    # ------------------------------------------------ mask AP at each IoU threshold
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.5pt}",
             r"\caption{Mask AP (\%) at each IoU threshold, mean over three seeds; the last two rows give the change of "
             r"each refiner over the parent (03). The strict thresholds carry the gain.}\label{tab:byiou}",
             r"\begin{tabular}{@{}l" + "c" * 10 + r"@{}}\toprule",
             r"Method & " + " & ".join(f"{t:.2f}" for t in thr) + r" \\ \midrule"]
    for m in IDS:
        lines.append(f"{m} & " + " & ".join(f2(v, 1) for v in by[m]) + r" \\")
    lines.append(r"\midrule")
    for a in ("09", "09M"):
        lines.append(f"{a}$-$03 & " + " & ".join(f2(x - y, 2, True) for x, y in zip(by[a], by["03"])) + r" \\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_byiou.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ error types (TIDE-style), all methods
    ek = [("TP", "TP"), ("missed", "missed"), ("wrong_class", "wrong class"), ("poor_localization", "poor loc."),
          ("duplicate", "duplicate"), ("background", "background"), ("other", "other")]
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{6pt}",
             r"\caption{Detection outcomes per run at score $\ge0.5$ (box IoU; mean over seeds; " + MACROS["NObjects"] +
             r" objects). FP = duplicate + wrong class + poor localisation + other + background.}\label{tab:errors}",
             r"\begin{tabular}{@{}l" + "c" * (len(ek) + 1) + r"@{}}\toprule",
             r"Method & " + " & ".join(n for _, n in ek) + r" & FP \\ \midrule"]
    for m in IDS:
        cnt = {k: mean(lambda mm, s: analysis[(mm, s)]["errors"]["counts"][k], m) for k, _ in ek}
        fpn = mean(lambda mm, s: analysis[(mm, s)]["errors"]["false_positives"], m)
        lines.append(f"{m} & " + " & ".join(f"{cnt[k]:.0f}" for k, _ in ek) + f" & {fpn:.0f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_errors.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ report: statistical tests used (three hypotheses)
    nb = OBJ[("09M", "03")]["biou"]["n_obj"]
    lines = [r"\begin{table}[!htb]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Statistical tests (paired, two-sided, $\alpha=0.05$).}\label{tab:design}",
             r"\begin{tabular}{@{}p{0.31\columnwidth}p{0.33\columnwidth}p{0.32\columnwidth}@{}}\toprule",
             r"Test & Used for & Sample \\ \midrule",
             f"Permutation test ($J={J:,}$) & H1; H2 difference & {N:,} images $\\times$ 3 seeds \\\\",
             f"Two-level bootstrap ($B={B:,}$) & 95\\% CI of AP differences; H3a, H3c & seeds and images resampled \\\\",
             r"TOST, margin $\pm1$ AP & H2 equivalence & 90\% CI \\",
             f"Cluster sign-flip test ($J={J:,}$) & H3b, H3c (object level) & {nb:,} matched / {n_obj:,} objects \\\\",
             r"One-sample $t$-test & sensitivity (seed as unit) & 3 seeds, $df=2$ \\",
             r"Holm--Bonferroni & multiplicity & H1: $m=2$; H3a--b: $m=4$; classes: $m=15$ \\",
             r"\bottomrule\end{tabular}\end{table}"]
    write("tab_design.tex", "\n".join(lines) + "\n")

    # ------------------------------------------------ report: the three hypothesis tests in one table
    def verdict_dir(delta, ci, p):
        excl = ci[0] > 0 or ci[1] < 0
        if p < 0.05 and excl:
            return "supported" if delta > 0 else "reversed"
        return "mixed" if p < 0.05 else "not supported"

    def tost_p(delta, ci95, margin=1.0):
        se = (ci95[1] - ci95[0]) / 3.92
        return float(max(sps.norm.sf((delta + margin) / se), sps.norm.sf((margin - delta) / se)))

    img_n = f"{N:,} img.$\\times$3"
    T = []  # (H, comparison, measure, test, n, delta, ci, p, p_adj, result)
    for k in (("09", "03"), ("09M", "03")):
        m = comp[k]["mask_AP"]
        T.append(("H1", PAIR[k], "mask AP", "permutation", img_n, m["delta"], m["ci95"], m["p_perm"], m["p_holm"],
                  verdict_dir(m["delta"], m["ci95"], m["p_holm"])))
    m = comp[("09M", "09")]["mask_AP"]
    pt = tost_p(m["delta"], m["ci95"])
    T.append(("H2", PAIR[("09M", "09")], "mask AP (90\\% CI)", "TOST $\\pm$1 AP$^{a}$", img_n, m["delta"], m["ci90"], pt, None,
              "equivalent" if pt < 0.05 else "not equivalent"))
    T.append(("H2", PAIR[("09M", "09")], "mask AP", "permutation", img_n, m["delta"], m["ci95"], m["p_perm"], None,
              "MLP higher" if m["p_perm"] < 0.05 and m["ci95"][0] > 0 else "no difference"))
    dirr = []
    for a in ("09", "09M"):
        g = comp[(a, "03")]["mask_gain_AP75_minus_AP50"]
        dirr.append(("H3a", PAIR[(a, "03")], "AP$_{75}$ gain $-$ AP$_{50}$ gain", "bootstrap$^{a}$", img_n, g["delta"],
                     g["ci95"], p_from_ci(g["delta"], g["ci95"])))
    for a in ("09", "09M"):
        o = OBJ[(a, "03")]["biou"]
        dirr.append(("H3b", PAIR[(a, "03")], "boundary IoU", "cluster sign-flip", f"{o['n_obj']:,} obj.", o["mean"],
                     o["ci95"], o["p"]))
    hh = holm({i: r[7] for i, r in enumerate(dirr)})
    for i, r in enumerate(dirr):
        T.append((*r, hh[i], verdict_dir(r[5], r[6], hh[i])))
    for a in ("09", "09M"):
        c = comp[(a, "03")]["box_AP"]
        pb = p_from_ci(c["delta"], c["ci95"])
        T.append(("H3c", PAIR[(a, "03")], "box AP", "bootstrap$^{a}$", img_n, c["delta"], c["ci95"], pb, None,
                  "no gain" if c["ci95"][0] <= 0 else "gain"))
    for a in ("09", "09M"):
        o = OBJ[(a, "03")]["coverage"]
        T.append(("H3c", PAIR[(a, "03")], "detection coverage", "cluster sign-flip", f"{o['n_obj']:,} obj.", o["mean"],
                  o["ci95"], o["p"], None, "no gain" if o["ci95"][0] <= 0 else "gain"))
    lines = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{3.4pt}",
             r"\caption{Hypothesis tests against the Parent. $\Delta$: mean of three seeds (points) with 95\% CI; "
             r"$p_\text{adj}$: Holm. H3c predicts no detection gain. $^{a}p$ from the bootstrap interval (normal "
             r"approximation).}\label{tab:tests}",
             r"\begin{tabular}{@{}lllllcccl@{}}\toprule",
             r"H & Comparison & Measure & Test & $n$ & $\Delta$ [CI] & $p$ & $p_\text{adj}$ & Result \\ \midrule"]
    prev = None
    for r in T:
        if prev is not None and r[0][:2] != prev[:2]:
            lines.append(r"\addlinespace[2pt]")
        prev = r[0]
        lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {f2(r[5], 2, True)} {ci_txt(r[6])} & {fp(r[7])} & "
                     f"{fp(r[8]) if r[8] is not None else '--'} & {r[9]} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_tests.tex", "\n".join(lines) + "\n")
    mac("PTOST", fp(pt)); mac("PHthreeGainNine", fp(hh[0])); mac("PHthreeGainNineM", fp(hh[1]))
    mac("PHthreeBiouNine", fp(hh[2])); mac("PHthreeBiouNineM", fp(hh[3]))
    for a in ("09", "09M"):
        c = comp[(a, "03")]["box_AP"]
        mac(f"PBox{W(a)}", fp(p_from_ci(c["delta"], c["ci95"])))
        mac(f"PCov{W(a)}", fp(OBJ[(a, "03")]["coverage"]["p"]))
    mac("NegBoxSeedsNine", str(sum(v < 0 for v in comp[("09", "03")]["box_AP"]["per_seed"].values())))

    # ------------------------------------------------ report figures: per-seed mask AP (H1, H2) and gain by IoU (H3)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 7})
    col = {"03": "#555555", "09": "#d95f02", "09M": "#1f6fb2"}
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    xs = {"03": 0, "09": 1, "09M": 2}
    for s_ in SEEDS:
        ys = [100 * full[(m, s_)]["segm"]["stats"]["AP"] for m in ("03", "09", "09M")]
        ax.plot([0, 1, 2], ys, color="#bbbbbb", lw=0.8, zorder=1)
        ax.scatter([0, 1, 2], ys, s=10, color="white", edgecolor="#888888", lw=0.7, zorder=2)
    for m in ("03", "09", "09M"):
        r = S["methods"][m]["mask_AP"]
        ax.errorbar(xs[m] + 0.12, r["mean"], yerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]], fmt="o",
                    ms=4, color=col[m], capsize=2, lw=1.2, zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Parent", "Transformer\nrefiner", "MLP\nrefiner"])
    ax.set_ylabel("mask AP (%)"); ax.set_xlim(-0.4, 2.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    fig.savefig(REP / "figures/fig_seeds.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    t10 = np.array(thr)
    for a, lab in (("09M", "MLP refiner $-$ Parent"), ("09", "Transformer refiner $-$ Parent")):
        d = np.array([[100 * (x - y) for x, y in zip(full[(a, s_)]["segm"]["ap_by_iou"], full[("03", s_)]["segm"]["ap_by_iou"])]
                      for s_ in SEEDS])
        ax.fill_between(t10, d.min(0), d.max(0), color=col[a], alpha=0.18, lw=0)
        ax.plot(t10, d.mean(0), "-o", ms=2.5, lw=1.2, color=col[a], label=lab)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("mask IoU threshold"); ax.set_ylabel("$\\Delta$ mask AP (points)")
    ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    fig.savefig(REP / "figures/fig_byiou.pdf", bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------ write macros + data
    import re
    for k, v in list(MACROS.items()):   # "...rel" variants for running text: $p<0.001$ or $p=0.005$
        if "P" in k and (v == "$<$0.001" or re.fullmatch(r"0\.\d{3}", v)):
            MACROS[k + "rel"] = "$p<0.001$" if v == "$<$0.001" else f"$p={v}$"
    tex = ["% generated by scripts/campaign_multiseed/results_section.py -- do not edit"]
    for k, v in sorted(MACROS.items()):
        assert k.isalpha(), k
        tex.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    write("numbers.tex", "\n".join(tex) + "\n")
    (DAT / "results_numbers.json").write_text(json.dumps(dict(
        macros=MACROS, main=rows, hypotheses=[{**r, "key": list(r["key"])} for r in hyp_rows],
        object_level={f"{a}-{b}": v for (a, b), v in OBJ.items()},
        h5_interaction={f"{a}-{b}": v for (a, b), v in H5I.items()},
        exploratory_thinness={f"{a}-{b}": v for (a, b), v in EXI.items()},
        exploratory_occlusion={f"{a}-{b}": v for (a, b), v in EXO.items()},
        per_class_p={"09M-03": pM, "09-03": pT}, per_class_p_holm={"09M-03": hM, "09-03": hT}), indent=1, default=float))
    print(f"{len(MACROS)} macros; tables: {sorted(p.name for p in TAB.glob('*.tex'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
