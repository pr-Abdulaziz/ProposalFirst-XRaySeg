"""Four-combination matrix for the SAM methods, from preserved prediction sets.

Each combination is evaluated from its own deep copy of the detections, because
``COCO.loadRes`` mutates its input and its bbox branch is tested before its segm
branch, so a reused list silently changes every detection's COCO area range.

    legacy    predictions produced with XRAY_LEGACY_SAM_GEOMETRY=1 (job 11029)
    corrected predictions produced with the default corrected path (job 11028)

Identical checkpoints, unchanged annotations, read-only.
"""
from __future__ import annotations
import contextlib, copy, hashlib, io, json, pickle, sys
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projects.proposal_first_pidray_occ.evaluation.zero_id_compat import ZeroIdSafeCOCOeval
from scripts.audit_zero_id_effect import to_coco

ROOT = Path("output/audit/2026-09-11/predictions")
ARMS = [("05", "B5", "pixray_b5"), ("06", "S2", "pixray_s2")]
STATS = {"AP": 0, "AP50": 1, "AP75": 2, "AP_s": 3, "AP_m": 4, "AP_l": 5, "AR@100": 8}


def score(coco, dets, iou_type, eval_cls):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = eval_cls(coco, coco.loadRes(copy.deepcopy(dets)), iou_type)
        ev.params.maxDets = [100, 300, 1000]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: float(ev.stats[i]) for k, i in STATS.items()}


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO("data/PIXray/annotations/test.json")
    out = {"annotations": "data/PIXray/annotations/test.json", "arms": []}
    for nid, hist, wd in ARMS:
        entry = {"id": nid, "historical_id": hist, "predictions": {}, "results": {}}
        preds = {}
        for geom in ("legacy", "corrected"):
            p = ROOT / geom / f"{wd}.pkl"
            entry["predictions"][geom] = {
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            preds[geom] = pickle.load(open(p, "rb"))
        for iou_type in ("bbox", "segm"):
            rec = {}
            for geom in ("legacy", "corrected"):
                dets = to_coco(preds[geom], iou_type)
                rec[f"{geom}+official"] = score(coco, dets, iou_type, COCOeval)
                rec[f"{geom}+compat"] = score(coco, dets, iou_type, ZeroIdSafeCOCOeval)
            entry["results"][iou_type] = rec
        out["arms"].append(entry)

    print("=== four-combination matrix, independent prediction copies ===")
    print("  identical checkpoints; annotations unchanged; AP in percentage points\n")
    for a in out["arms"]:
        print(f"--- method {a['id']} ({a['historical_id']}) ---")
        for g, info in a["predictions"].items():
            print(f"  {g:<10} predictions sha256 {info['sha256'][:16]}")
        print(f"  {'combination':<38}{'bbox AP':>12}{'segm AP':>12}")
        for combo in ("legacy+official", "legacy+compat",
                      "corrected+official", "corrected+compat"):
            b = a["results"]["bbox"][combo]["AP"] * 100
            s = a["results"]["segm"][combo]["AP"] * 100
            print(f"  {combo:<38}{b:>12.5f}{s:>12.5f}")
        for label, x, y in (("GEOMETRY-ONLY", "corrected+official", "legacy+official"),
                            ("ID-ZERO on legacy geometry", "legacy+compat", "legacy+official"),
                            ("ID-ZERO on corrected geometry", "corrected+compat", "corrected+official"),
                            ("COMBINED", "corrected+compat", "legacy+official")):
            db = (a["results"]["bbox"][x]["AP"] - a["results"]["bbox"][y]["AP"]) * 100
            ds = (a["results"]["segm"][x]["AP"] - a["results"]["segm"][y]["AP"]) * 100
            print(f"  {label:<38}{db:>+12.5f}{ds:>+12.5f}")
        g = (a["results"]["segm"]["corrected+official"]["AP"]
             - a["results"]["segm"]["legacy+official"]["AP"]) * 100
        z = (a["results"]["segm"]["corrected+compat"]["AP"]
             - a["results"]["segm"]["corrected+official"]["AP"]) * 100
        c = (a["results"]["segm"]["corrected+compat"]["AP"]
             - a["results"]["segm"]["legacy+official"]["AP"]) * 100
        print(f"  {'additivity check (segm): g + z vs combined':<38}"
              f"{g + z:>+12.5f}{c:>12.5f}  residual {c - (g + z):+.5f}\n")

    Path("output/audit/2026-09-11/four_combination_matrix.json").write_text(
        json.dumps(out, indent=2))
    print("wrote output/audit/2026-09-11/four_combination_matrix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
