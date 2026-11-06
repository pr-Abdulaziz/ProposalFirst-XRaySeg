#!/usr/bin/env python
"""Aggregate the multi-seed campaign from its frozen manifest only.

Every number comes from a run listed in campaign_manifest.json, through that run's
write-once evaluation record, whose checkpoint, prediction and metrics hashes are
re-verified here. Nothing is discovered by globbing other output directories.

Writes summaries/ (per-seed results, mean and sample standard deviation over seeds,
paired per-seed differences, per-class AP, training and inference cost), figures/
and reports/campaign_results.md. Runs that have not finished are reported as
missing; no seed is dropped, replaced or selected.

    python scripts/campaign_multiseed/aggregate_results.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import BY_ID, MANIFEST, MAX_EPOCHS, OUT, eval_dir, run_dir, sha256  # noqa: E402

METRICS = [("bbox", "AP"), ("bbox", "AP50"), ("bbox", "AP75"), ("bbox", "AR100"),
           ("bbox", "AP_small"), ("bbox", "AP_medium"), ("bbox", "AP_large"),
           ("segm", "AP"), ("segm", "AP50"), ("segm", "AP75"), ("segm", "AR100"),
           ("segm", "AP_small"), ("segm", "AP_medium"), ("segm", "AP_large")]
COMPARISONS = [
    ("09", "03", "primary", "transformer point refinement vs parent (03)"),
    ("09M", "03", "primary", "MLP point refinement vs parent (03)"),
    ("09M", "09", "primary", "MLP vs transformer refinement (matched pair)"),
    ("02", "01", "architecture", "cascade vs Mask R-CNN; training proposals also 1000 -> 2000"),
    ("03", "02", "architecture", "AttFPN vs FPN"),
    ("04", "03", "architecture", "recall-enhanced RPN (anchor scales and both proposal budgets) vs 03"),
    ("08", "03", "architecture", "PIDray R0/R1 gate vs 03"),
    ("07", "04", "additional supervision", "bilayer FCN head with occluder supervision vs 04; not a target-only ablation"),
    ("06", "05", "additional supervision", "Occ bilayer decoder with occluder supervision vs RSPrompter-style SAM; "
                                           "not a target-only ablation"),
]


def load_run(man, entry) -> dict:
    mid, seed, key = entry["method"], entry["seed"], entry["run_key"]
    rec = dict(method=mid, seed=seed, run_key=key, name=BY_ID[mid].name, status="pending")
    st = run_dir(mid, seed) / "status.jsonl"
    events = [json.loads(x) for x in st.read_text().splitlines()] if st.is_file() else []
    rec["attempts"] = sum(e["event"] == "train_started" for e in events)
    completes = [e for e in events if e["event"] == "train_complete"]
    if not completes:
        rec["status"] = "failed" if any(e["event"] == "train_failed" for e in events) else "pending"
        return rec
    train = completes[-1]
    records = sorted(eval_dir(mid, seed).glob("eval_record_job*.json"))
    if not records:
        rec["status"] = "trained, not evaluated"
        return rec
    ev = [json.loads(p.read_text()) for p in records]
    ev = [e for e in ev if e["coverage"]["images_match"]]
    if not ev:
        rec["status"] = "evaluation coverage failed"
        return rec
    e = sorted(ev, key=lambda r: r["finished"])[-1]
    problems = []
    if e["checkpoint_sha256"] != train["checkpoint_sha256"]:
        problems.append("evaluated checkpoint differs from the completed training checkpoint")
    if e["resolved_config_sha256"] != man["runs"][key]["resolved_config_sha256"]:
        problems.append("evaluation config hash differs from the manifest")
    if sha256(ROOT / e["metrics_fullprec"]) != e["metrics_fullprec_sha256"]:
        problems.append("metrics file changed after evaluation")
    if sha256(Path(e["predictions"])) != e["predictions_sha256"]:
        problems.append("predictions changed after evaluation")
    full = json.loads((ROOT / e["metrics_fullprec"]).read_text())
    launches = [json.loads(p.read_text()) for p in sorted(run_dir(mid, seed).glob("launch_job*.json"))]
    acct = Path(man["runs"][key]["work_dir"]) / "update_accounting.jsonl"
    epochs = [json.loads(x) for x in acct.read_text().splitlines()] if acct.is_file() else []
    rec.update(status="complete" if not problems else "hash problem", problems=problems,
               evaluations=len(ev), eval_job=e["job"],
               metrics={f"{t}/{k}": full[t]["stats"][k] for t, k in METRICS},
               zero_id_safe={f"{t}/AP": full[t]["stats_zero_id_safe"]["AP"] for t in ("bbox", "segm")},
               per_class={t: {c: v["AP"] for c, v in full[t]["per_class"].items()} for t in ("bbox", "segm")},
               prf={t: {k: full[t]["prf_iou_0.50"]["at_0.50"][k] for k in ("P", "R", "F1")} for t in ("bbox", "segm")},
               training=dict(successful_updates=train["cumulative"].get("successful_updates"),
                             amp_skipped_updates=train["cumulative"].get("amp_skipped_updates"),
                             update_opportunities=train["cumulative"].get("update_opportunities"),
                             epoch_seconds_sum=sum(x.get("seconds", 0) for x in epochs[-MAX_EPOCHS:]),
                             peak_reserved_gb=max((x.get("peak_reserved_mb", 0) for x in epochs), default=0) / 1024,
                             nodes=sorted({l["node"] for l in launches}),
                             gpus=sorted({g["name"] for l in launches for g in l["gpus"]})),
               inference=dict(e["efficiency"], gpu=e["gpu"], node=e["node"]))
    return rec


def mean_sd(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None), len(vals)


def fmt(m, s, scale=100, nd=2):
    if m is None:
        return "pending"
    return f"{scale*m:.{nd}f} ± {scale*s:.{nd}f}" if s is not None else f"{scale*m:.{nd}f} (n=1)"


def main() -> int:
    man = json.loads(MANIFEST.read_text())
    seeds = man["seeds"]
    runs = [load_run(man, e) for e in man["run_order"]]
    by = {(r["method"], r["seed"]): r for r in runs}
    methods = list(dict.fromkeys(r["method"] for r in runs))
    summ = OUT / "summaries"
    summ.mkdir(parents=True, exist_ok=True)

    (summ / "individual_seed_results.json").write_text(json.dumps(runs, indent=1, default=float))
    with open(summ / "individual_seed_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "status"] + [f"{t}/{k}" for t, k in METRICS] + ["bbox/AP_zero_id_safe", "segm/AP_zero_id_safe",
                    "successful_updates", "amp_skipped_updates", "train_epoch_seconds", "eval_images_per_second"])
        for r in runs:
            ok = r["status"] == "complete"
            w.writerow([r["method"], r["seed"], r["status"]]
                       + [r["metrics"][f"{t}/{k}"] if ok else "" for t, k in METRICS]
                       + ([r["zero_id_safe"]["bbox/AP"], r["zero_id_safe"]["segm/AP"], r["training"]["successful_updates"],
                           r["training"]["amp_skipped_updates"], r["training"]["epoch_seconds_sum"],
                           r["inference"]["images_per_second"]] if ok else [""] * 6))

    summary = {}
    for mid in methods:
        vals = {f"{t}/{k}": mean_sd([by[(mid, s)]["metrics"][f"{t}/{k}"] if by[(mid, s)]["status"] == "complete" else None
                                     for s in seeds]) for t, k in METRICS}
        summary[mid] = dict(name=BY_ID[mid].name, seeds_complete=vals["segm/AP"][2],
                            **{k: dict(mean=v[0], sd=v[1], n=v[2]) for k, v in vals.items()})
    (summ / "method_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    paired = []
    for a, b, kind, label in COMPARISONS:
        row = dict(method=a, baseline=b, kind=kind, label=label, metrics={})
        for key in ("segm/AP", "bbox/AP", "segm/AP75", "segm/AR100", "bbox/AR100"):
            diffs = {s: by[(a, s)]["metrics"][key] - by[(b, s)]["metrics"][key] for s in seeds
                     if by[(a, s)]["status"] == "complete" and by[(b, s)]["status"] == "complete"}
            m, sd, n = mean_sd(list(diffs.values()))
            row["metrics"][key] = dict(per_seed=diffs, mean=m, sd=sd, n=n,
                                       positive=sum(d > 0 for d in diffs.values()),
                                       min=min(diffs.values()) if diffs else None,
                                       max=max(diffs.values()) if diffs else None)
        paired.append(row)
    (summ / "paired_differences.json").write_text(json.dumps(paired, indent=1, default=float))

    classes = sorted({c for r in runs if r["status"] == "complete" for c in r["per_class"]["segm"]})
    per_class = {mid: {t: {c: mean_sd([by[(mid, s)]["per_class"][t][c] if by[(mid, s)]["status"] == "complete" else None
                                       for s in seeds])[:2] for c in classes} for t in ("segm", "bbox")} for mid in methods}
    (summ / "per_class_summary.json").write_text(json.dumps(per_class, indent=1, default=float))

    _figures(methods, seeds, by, paired)
    _report(man, methods, seeds, runs, by, summary, paired, per_class, classes)
    done = sum(r["status"] == "complete" for r in runs)
    print(f"{done}/{len(runs)} runs complete; summaries in {summ}")
    return 0


def _figures(methods, seeds, by, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, key, title in ((axes[0], "segm/AP", "Mask AP"), (axes[1], "bbox/AP", "Box AP")):
        for i, mid in enumerate(methods):
            vals = [100 * by[(mid, s)]["metrics"][key] for s in seeds if by[(mid, s)]["status"] == "complete"]
            if not vals:
                continue
            ax.scatter([i] * len(vals), vals, s=14, color="0.4", zorder=3)
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            ax.errorbar(i, m, yerr=sd, fmt="o", color="C0", capsize=4, zorder=4)
        ax.set_xticks(range(len(methods)), methods)
        ax.set_title(f"{title}: seeds (grey), mean ± sample SD (blue)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "ap_by_method_seeds.pdf")
    fig.savefig(figs / "ap_by_method_seeds.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, row in enumerate(paired):
        d = row["metrics"]["segm/AP"]["per_seed"]
        if not d:
            continue
        vals = [100 * v for v in d.values()]
        ax.scatter(vals, [i] * len(vals), color="0.4", s=14, zorder=3)
        ax.plot([statistics.mean(vals)], [i], "D", color="C1", zorder=4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(paired)), [f"{r['method']} - {r['baseline']}" for r in paired])
    ax.set_xlabel("Mask AP difference (pp): per seed (grey), mean (orange)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "paired_mask_ap_differences.pdf")
    fig.savefig(figs / "paired_mask_ap_differences.png", dpi=160)
    plt.close(fig)


def _report(man, methods, seeds, runs, by, summary, paired, per_class, classes):
    rep = OUT / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    L = [f"# PIXray multi-seed campaign results ({man['campaign']})", "",
         f"Source: `campaign_manifest.json` (frozen {man['frozen_at']}). Seeds: {', '.join(map(str, seeds))}. "
         f"Runs complete: {sum(r['status'] == 'complete' for r in runs)}/{len(runs)}.", "",
         "Values are percentages from the official pycocotools evaluator on the 1,486 official test images. "
         "Mean ± sample standard deviation over seeds (n shown when fewer than three are complete). "
         "Size ranges bin instances by PIXray's `area` field, which is the bounding-box area. Three seeds give a "
         "limited variance estimate; no difference below is claimed to be statistically significant.", "",
         "## Mean ± SD", "",
         "| Method | Mask AP | Mask AP50 | Mask AP75 | Mask AR@100 | Box AP | Box AP50 | Box AP75 | Box AR@100 | Seeds |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for mid in methods:
        s = summary[mid]
        L.append(f"| {mid} {s['name']} | " + " | ".join(fmt(s[k]["mean"], s[k]["sd"]) for k in
                 ("segm/AP", "segm/AP50", "segm/AP75", "segm/AR100", "bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/AR100"))
                 + f" | {s['seeds_complete']}/3 |")
    L += ["", "## Individual seeds", "", "| Method | Seed | Status | Mask AP | Box AP | Mask AR@100 | Box AR@100 | "
          "Mask AP S/M/L | Successful / AMP-skipped updates | Train h (epoch sum) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if r["status"] != "complete":
            L.append(f"| {r['method']} | {r['seed']} | {r['status']} | | | | | | | |")
            continue
        m, t = r["metrics"], r["training"]
        L.append(f"| {r['method']} | {r['seed']} | complete | {100*m['segm/AP']:.3f} | {100*m['bbox/AP']:.3f} | "
                 f"{100*m['segm/AR100']:.3f} | {100*m['bbox/AR100']:.3f} | {100*m['segm/AP_small']:.2f}/"
                 f"{100*m['segm/AP_medium']:.2f}/{100*m['segm/AP_large']:.2f} | {t['successful_updates']} / "
                 f"{t['amp_skipped_updates']} | {t['epoch_seconds_sum']/3600:.2f} |")
    L += ["", "## Paired per-seed differences (percentage points)", "",
          "| Comparison | Kind | Metric | Per seed | Mean | SD | Seeds > 0 |", "|---|---|---|---|---|---|---|"]
    for row in paired:
        for key in ("segm/AP", "bbox/AP"):
            d = row["metrics"][key]
            per = ", ".join(f"{s}: {100*v:+.2f}" for s, v in d["per_seed"].items()) or "pending"
            L.append(f"| {row['method']} − {row['baseline']}: {row['label']} | {row['kind']} | {key} | {per} | "
                     f"{'' if d['mean'] is None else f'{100*d[chr(109)+chr(101)+chr(97)+chr(110)]:+.2f}'} | "
                     f"{'' if d['sd'] is None else f'{100*d[chr(115)+chr(100)]:.2f}'} | {d['positive']}/{d['n']} |")
    L += ["", "## Per-class mask AP (mean over seeds)", "", "| Class | " + " | ".join(methods) + " |",
          "|---|" + "---|" * len(methods)]
    for c in classes:
        L.append(f"| {c} | " + " | ".join("pending" if per_class[mid]["segm"][c][0] is None
                                          else f"{100*per_class[mid]['segm'][c][0]:.2f}" for mid in methods) + " |")
    L += ["", "## Disclosed differences", "",
          "- 06 uses PIXray-A occluder supervision; its comparison with 05 is an additional-supervision comparison.",
          "- 01 → 02 also raises training proposals from 1000 to 2000; 04 changes anchor scales and both proposal budgets.",
          "- 05 is an RSPrompter-style SAM ViT-L + LoRA adaptation, not an untouched SAM release.",
          "- 07 uses PIXray-A occluder supervision (user decision); its comparison with 04 is an additional-supervision comparison.",
          "- Inference throughput is end-to-end as logged by each evaluation job (data loading, COCO evaluation and dumping included), "
          "not a controlled latency benchmark."]
    (rep / "campaign_results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
