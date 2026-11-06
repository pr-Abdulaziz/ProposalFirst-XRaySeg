#!/usr/bin/env python
"""Collect the gradient-isolation diagnostic (Design A) in place and build a review ZIP.

Works inside output/pidray_diag_gradiso_2026-09-26/ (configs, verification, evaluations, runs and logs
are already there). Copies -- never moves -- the detached run's training accounting, logs and scalar
records from work_dirs into ``training/``, and the corrected (rescored) evaluation outputs and
accounting of the existing Parent and ordinary MLP seed-42 runs into ``reference/``. Writes
``comparison.json`` and ``comparison.md``: per-split and pooled mask / box AP (and AP75) for all three
models, all computed from the saved predictions with the same MMDetection label -> category mapping,
plus training accounting and window-averaged gradient-norm statistics. Finally writes
``pidray_diag_gradiso_2026-09-26_review.zip`` with paths relative to the folder, excluding checkpoints
(*.pth) and prediction dumps (*.pkl); raw predictions are referenced by path and SHA-256 only.
One seed: a diagnostic observation only.

    python scripts/campaign_pidray_bs1/collect_gradiso.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_pidray_bs1.coco_resample_mapped import label_to_category, to_coco_mapped  # noqa: E402
from scripts.campaign_pidray_bs1.evaluate_diag import OFF, grad_norm_stats  # noqa: E402

SRC = ROOT / "output/pidray_bs2_e10_2026-09-22"
DIAG = ROOT / "output/pidray_diag_gradiso_2026-09-26"
WSRC = ROOT / "work_dirs/pidray_bs2_e10_2026-09-22"
WDIAG = ROOT / "work_dirs/pidray_diag_gradiso_2026-09-26"
DET = "m09md_point_refine_mlp_detached"
NAMES = {"parent": "Parent (Cascade + AttFPN)", "mlp": "MLP refiner", "mlp_detached": "MLP refiner, detached"}


def copy_into(src: Path, dest: Path):
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def split_sources():
    """Per model and split: (predictions path, evaluator annotation file, existing corrected metrics file)."""
    rows = json.loads((SRC / "amendments/rescoring_category_mapping_2026-09-25.json").read_text())["rows"]
    out = {}
    for key, mid in (("parent", "03"), ("mlp", "09M")):
        for r in (x for x in rows if x["method"] == mid):
            metrics = next(SRC.glob(f"evaluations/*/seed_42/metrics_fullprec_rescored_job{r['job']}_{r['split']}.json"))
            out.setdefault(key, {})[r["split"]] = (Path(r["predictions"]), ROOT / r["annotations"], metrics)
    rec = json.loads(sorted((DIAG / "evaluations" / DET / "seed_42").glob("eval_record_job*.json"))[-1].read_text())
    for split, v in rec["splits"].items():
        metrics = next((DIAG / "evaluations" / DET / "seed_42").glob(f"metrics_fullprec_job{rec['job']}_{split}.json"))
        out.setdefault("mlp_detached", {})[split] = (Path(v["predictions"]), Path(v["evaluator"]), metrics)
    return out, rec


def evaluate(sources):
    """Per-split stats read from the corrected metrics files; pooled AP recomputed identically for all."""
    res = {}
    for key, by_split in sources.items():
        per, gt, dts = {}, dict(images=[], annotations=[], categories=None), {"bbox": [], "segm": []}
        for split in C.TEST_SPLITS:
            pred, ann, metrics = by_split[split]
            m = json.loads(metrics.read_text())
            per[split] = {f"{t}_{k}": 100 * m[t]["stats"][k] for t in ("segm", "bbox") for k in ("AP", "AP50", "AP75")}
            d = json.loads(ann.read_text()); o = OFF[split]
            with contextlib.redirect_stdout(io.StringIO()):
                g = COCO(str(ann))
            mapping = label_to_category(g, C.CLASSES)
            gt["categories"] = d["categories"]
            gt["images"] += [dict(im, id=im["id"] + o) for im in d["images"]]
            gt["annotations"] += [dict(a, id=a["id"] + 1 + o, image_id=a["image_id"] + o) for a in d["annotations"]]
            samples = pickle.load(open(pred, "rb"))
            for t in ("bbox", "segm"):
                for x in to_coco_mapped(samples, t, mapping):
                    x["image_id"] += o; dts[t].append(x)
            per[split]["predictions"] = str(pred); per[split]["predictions_sha256"] = C.sha256(pred)
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(); coco.dataset = gt; coco.createIndex()
            pooled = {}
            for t in ("segm", "bbox"):
                ev = COCOeval(coco, coco.loadRes(copy.deepcopy(dts[t])), t); ev.evaluate(); ev.accumulate(); ev.summarize()
                pooled.update({f"{t}_AP": 100 * ev.stats[0], f"{t}_AP50": 100 * ev.stats[1], f"{t}_AP75": 100 * ev.stats[2]})
        res[key] = dict(per_split=per, pooled=pooled)
    return res


ZIP_NAME = "pidray_diag_gradiso_2026-09-26_review.zip"
EXCLUDE_SUFFIXES = (".pth", ".pkl", ".zip")


def main() -> int:
    dest = DIAG
    sources, rec = split_sources()
    # ---- copies into the diagnostic folder (originals untouched)
    wd = WDIAG / DET / "seed_42"
    copy_into(wd / "update_accounting.jsonl", dest / "training/update_accounting.jsonl")
    for f in sorted(wd.glob("*/*.log")) + sorted(wd.glob("*/vis_data/*.json")):
        copy_into(f, dest / "training" / f.relative_to(wd))
    for key, mid_slug in (("parent", "m03_cascade_r101_attfpn"), ("mlp", "m09m_point_refine_mlp")):
        for split, (_, _, metrics) in sources[key].items():
            copy_into(metrics, dest / f"reference/{mid_slug}/{metrics.name}")
        copy_into(WSRC / mid_slug / "seed_42/update_accounting.jsonl", dest / f"reference/{mid_slug}/update_accounting.jsonl")
    copy_into(SRC / "amendments/rescoring_category_mapping_2026-09-25.json", dest / "reference/rescoring_category_mapping_2026-09-25.json")
    copy_into(SRC / "audit_point_refine_2026-09-26", dest / "reference/audit_point_refine_2026-09-26")
    # ---- comparison
    res = evaluate(sources)
    acc = {"parent": WSRC / "m03_cascade_r101_attfpn/seed_42", "mlp": WSRC / "m09m_point_refine_mlp/seed_42", "mlp_detached": wd}
    for key, d in acc.items():
        last = json.loads((d / "update_accounting.jsonl").read_text().splitlines()[-1])
        res[key]["training"] = dict(cumulative=last["cumulative"], grad_norm=grad_norm_stats(d))
    (dest / "comparison.json").write_text(json.dumps(dict(
        note="One seed (42), batch 2, 10 epochs, fp32; diagnostic observation, not evidence of generalisation or causality.",
        models=NAMES, results=res), indent=1, default=float))
    lines = ["# PIDray gradient-isolation diagnostic (Design A, seed 42)", "",
             "One seed; batch 2; 10 epochs; fp32. A diagnostic observation, not evidence of generalisation or causality.", "",
             "| Model | " + " | ".join(f"{s} mask / box" for s in C.TEST_SPLITS) + " | pooled mask / box | pooled mask AP75 |",
             "|---|" + "---|" * (len(C.TEST_SPLITS) + 2)]
    for key, name in NAMES.items():
        r = res[key]
        cells = [f"{r['per_split'][s]['segm_AP']:.2f} / {r['per_split'][s]['bbox_AP']:.2f}" for s in C.TEST_SPLITS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['pooled']['segm_AP']:.2f} / {r['pooled']['bbox_AP']:.2f} | "
                     f"{r['pooled']['segm_AP75']:.2f} |")
    lines += ["", "| Model | successful updates | AMP-skipped | non-finite losses | mean window grad_norm | windows with mean > 1.0 |",
              "|---|---|---|---|---|---|"]
    for key, name in NAMES.items():
        t = res[key]["training"]; c = t["cumulative"]; g = t["grad_norm"]
        lines.append(f"| {name} | {c['successful_updates']:,} | {c['amp_skipped_updates']} | {c['nonfinite_loss_iterations']} | "
                     f"{g['mean']:.2f} | {100 * g['share_windows_mean_above_max_norm']:.1f}% of {g['logged_windows']:,} |")
    lines += ["", "Gradient norms are pre-clipping 50-iteration window means (clip at 1.0 on the joint norm over all "
              "parameters). A window mean above 1.0 implies at least one clipped iteration in that window; the "
              "per-iteration clipping rate is not logged."]
    (dest / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # ---- review ZIP (relative paths; no checkpoints, prediction dumps or nested zips)
    zpath = dest / ZIP_NAME
    files = sorted(f for f in dest.rglob("*") if f.is_file() and not f.name.endswith(EXCLUDE_SUFFIXES))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(dest).as_posix())
    print(f"ZIP: {zpath} ({zpath.stat().st_size / 2**20:.1f} MiB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
