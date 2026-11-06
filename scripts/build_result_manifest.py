"""Build the explicit result manifest the report consumes.

The report must not choose a result by glob order or by preferring whichever
corrected file happens to exist. This writes one selected entry per active
method, each naming its checkpoint, training config, evaluation config,
prediction file and their hashes, the evaluator mode, and full-precision AP
read from COCOeval.stats rather than from a rounded metric file.

A method with no completed evaluation is recorded as pending, never as zero.
"""
from __future__ import annotations

import contextlib, copy, hashlib, io, json, pickle, sys
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from projects.proposal_first_pidray_occ.registry_seven import METHODS  # noqa: E402
from projects.proposal_first_pidray_occ.evaluation.zero_id_compat import (  # noqa: E402
    ZeroIdSafeCOCOeval)
from scripts.audit_zero_id_effect import to_coco  # noqa: E402

ANN = ROOT / "data/PIXray/annotations/test.json"
# Source files whose content determines how predictions were produced. Hashed
# into every manifest so a later report can tell whether a result predates a
# code change.
# Which inference sources actually determine each method's predictions. A file
# that the method never executes is not part of its provenance.
RELEVANT_SOURCES = {
    "01": ["pidray_r0_detector"], "02": ["pidray_r0_detector"],
    "03": ["pidray_r0_detector"], "04": ["pidray_r0_detector"],
    "05": ["sam_mask_geometry", "occ_bilayer_head"],
    "06": ["sam_mask_geometry", "occ_bilayer_head"],
    "07": ["bilayer_fcn_head"],
    "08": ["pidray_r0_detector", "pidray_r0_head"],
}
INFERENCE_SOURCES = {
    "sam_mask_geometry": "mmdet/occlu/models.py",
    "occ_bilayer_head": "mmdet/occlu/occlu.py",
    "pidray_r0_detector": "projects/proposal_first_pidray_occ/models/pidray_cascade_rcnn.py",
    "pidray_r0_head": "projects/proposal_first_pidray_occ/models/r0_head.py",
    "bilayer_fcn_head": "projects/proposal_first_pidray_occ/models/bilayer_fcn_mask_head.py",
}
EVALUATOR_SOURCES = {
    "xray_coco_metric": "projects/proposal_first_pidray_occ/evaluation/xray_coco_metric.py",
    "zero_id_compat": "projects/proposal_first_pidray_occ/evaluation/zero_id_compat.py",
}
# Methods whose stored predictions came from the legacy (released) SAM geometry
# rather than the corrected path. None currently: 05/06 report the corrected set.
LEGACY_GEOMETRY_PREDICTIONS = {
    "05": "output/audit/2026-09-11/predictions/legacy/pixray_b5.pkl",
    "06": "output/audit/2026-09-11/predictions/legacy/pixray_s2.pkl",
}
METRIC_UNITS = {
    "AP": "COCO average precision, fraction in [0,1] (multiply by 100 for pp)",
    "AP50": "same, IoU 0.50 only", "AP75": "same, IoU 0.75 only",
    "AP_s": "same, COCO small area bin from the released 'area' field (bbox area)",
    "AP_m": "same, medium bin", "AP_l": "same, large bin",
    "AR@100": "COCO average recall at maxDets 100, fraction in [0,1]",
}
STATS = {"AP": 0, "AP50": 1, "AP75": 2, "AP_s": 3, "AP_m": 4, "AP_l": 5, "AR@100": 8}
# Methods whose predictions were regenerated with the corrected SAM geometry.
# The live work_dirs copy is NOT authoritative for these: the legacy-geometry
# measurement run (job 11029) overwrote it afterwards. Both prediction sets were
# preserved with hashes when they were produced, and the corrected set is the
# one the main table reports.
GEOMETRY = {"05": "corrected", "06": "corrected"}
PRESERVED = {
    "05": Path("output/audit/2026-09-11/predictions/corrected/pixray_b5.pkl"),
    "06": Path("output/audit/2026-09-11/predictions/corrected/pixray_s2.pkl"),
}


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(coco, dets, iou_type, eval_cls):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = eval_cls(coco, coco.loadRes(copy.deepcopy(dets)), iou_type)
        ev.params.maxDets = [100, 300, 1000]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: float(ev.stats[i]) for k, i in STATS.items()}


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ANN))
    import datetime, pycocotools
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "manifest_version": stamp,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "annotations": str(ANN.relative_to(ROOT)),
        "annotation_sha256": sha(ANN),
        "evaluator": "pycocotools COCOeval",
        "pycocotools_version": getattr(pycocotools, "__version__", "unversioned"),
        "pycocotools_path": pycocotools.__file__,
        "compat_evaluator": "ZeroIdSafeCOCOeval (appendix only, never replaces official)",
        "evaluator_source_sha256": {
            k: sha(ROOT / v) for k, v in EVALUATOR_SOURCES.items()},
        "inference_source_sha256": {
            k: sha(ROOT / v) for k, v in INFERENCE_SOURCES.items()},
        "metric_units": METRIC_UNITS,
        "methods": [],
    }
    for nid, method in METHODS.items():
        work = ROOT / method.work_dir
        summary = work / "training_summary.json"
        preds = work / "test/test/predictions.pkl"
        entry = {
            "id": nid,
            "label": method.name,
            "role": method.role,
            "historical_id": method.historical_id or None,
            "training_config": method.config,
            "training_config_sha256": sha(ROOT / method.config),
            "geometry": GEOMETRY.get(nid, "unchanged"),
            "legacy_geometry": False,
            "legacy_geometry_predictions": LEGACY_GEOMETRY_PREDICTIONS.get(nid),
        }
        # Config actually resolved at training time, dumped by MMEngine into the
        # work directory. Distinct from the current config file, which may have
        # been edited since the run.
        dumped = sorted(work.glob("*/vis_data/config.py")) + sorted(work.glob("*.py"))
        if dumped:
            entry["resolved_training_config"] = str(dumped[0].relative_to(ROOT))
            entry["resolved_training_config_sha256"] = sha(dumped[0])
        eval_cfg = sorted((work / "test/test").glob("*.py")) if (work / "test/test").is_dir() else []
        if eval_cfg:
            entry["resolved_evaluation_config"] = str(eval_cfg[0].relative_to(ROOT))
            entry["resolved_evaluation_config_sha256"] = sha(eval_cfg[0])
        if not summary.is_file():
            entry["status"] = "pending"
            entry["reason"] = "training has not written training_summary.json"
            manifest["methods"].append(entry)
            print(f"  {nid}  PENDING (no training summary)")
            continue
        info = json.loads(summary.read_text())
        ckpt = work / (work / "last_checkpoint").read_text().strip().split("/")[-1]
        entry.update(
            status="evaluated" if preds.is_file() else "trained",
            checkpoint=str(ckpt.relative_to(ROOT)),
            checkpoint_sha256=sha(ckpt),
            epochs=info["protocol"]["epochs"],
            seed=info["protocol"].get("seed", 42),
            gpu_count=info["gpu_count"],
            resumed=info.get("resumed"),
            supervision=info.get("supervision"),
        )
        if not preds.is_file():
            entry["reason"] = "no predictions.pkl; evaluation has not run"
            manifest["methods"].append(entry)
            print(f"  {nid}  TRAINED, not evaluated")
            continue
        if nid in PRESERVED and (ROOT / PRESERVED[nid]).is_file():
            preds = ROOT / PRESERVED[nid]
            entry["predictions_note"] = (
                "preserved corrected-geometry set; work_dirs copy was later "
                "overwritten by the legacy-geometry measurement run")
        entry["predictions"] = str(preds.relative_to(ROOT))
        entry["predictions_sha256"] = sha(preds)

        # Provenance of the predictions themselves. No snapshot of the inference
        # source was taken at generation time, so the best available evidence is
        # whether the relevant source files have been modified since the
        # predictions were written. Where a source changed afterwards, the
        # current hash does NOT describe how these predictions were produced and
        # is recorded as such rather than attached retrospectively.
        made_at = preds.stat().st_mtime
        prov, incomplete = {}, []
        for key in RELEVANT_SOURCES.get(nid, []):
            src = ROOT / INFERENCE_SOURCES[key]
            changed_at = src.stat().st_mtime if src.is_file() else None
            unchanged = changed_at is not None and changed_at <= made_at
            prov[key] = {
                "path": INFERENCE_SOURCES[key],
                "unmodified_since_predictions_written": unchanged,
                "current_sha256": sha(src) if unchanged else None,
                "note": None if unchanged else (
                    "source modified after these predictions were written; the "
                    "generation-time content was not snapshotted, so no hash is "
                    "claimed"),
            }
            if not unchanged:
                incomplete.append(key)
        entry["inference_provenance"] = prov
        entry["inference_provenance_complete"] = not incomplete
        if incomplete:
            entry["inference_provenance_gaps"] = incomplete
        # Metrics below are computed now, by the current evaluator, so the
        # global evaluator hashes do describe them.
        entry["metrics_computed_by"] = "current evaluator (see evaluator_source_sha256)"
        loaded = pickle.load(open(preds, "rb"))
        results = {}
        for iou_type in ("bbox", "segm"):
            dets = to_coco(loaded, iou_type)
            if not dets:
                results[iou_type] = None
                continue
            results[iou_type] = {
                "official": evaluate(coco, dets, iou_type, COCOeval),
                "compat_zeroid": evaluate(coco, dets, iou_type, ZeroIdSafeCOCOeval),
            }
        entry["results"] = results
        manifest["methods"].append(entry)
        ap_b = results["bbox"]["official"]["AP"] * 100 if results["bbox"] else float("nan")
        ap_m = results["segm"]["official"]["AP"] * 100 if results["segm"] else float("nan")
        print(f"  {nid}  box {ap_b:8.5f}  mask {ap_m:8.5f}  geometry={entry['geometry']}")

    # Every manifest version is kept; the stable name is a pointer to the newest.
    versions = ROOT / "output/audit/2026-09-11/manifests"
    versions.mkdir(parents=True, exist_ok=True)
    archived = versions / f"result_manifest_{stamp}.json"
    archived.write_text(json.dumps(manifest, indent=2))
    out = ROOT / "output/audit/2026-09-11/result_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    kept = sorted(versions.glob("result_manifest_*.json"))
    print(f"\nwrote {archived.relative_to(ROOT)}")
    print(f"wrote {out.relative_to(ROOT)} (pointer to the newest)")
    print(f"manifest versions retained: {len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
