#!/usr/bin/env python
"""Pre-results evaluation amendment: drop one undecodable test_easy image.

The data audit of 2026-09-24 found xray_easy22241.png corrupt in the released
archive (its first ~11.8 KB are not PNG; a replacement copy from the user's source has
the same size and damaged header). It carries no annotations. mmcv cannot load it, so the frozen
evaluation would crash on test_easy. Decided by the user on 2026-09-24, before
any evaluation had run.

This writes derived copies of test_easy.json and test_easy_occ.json without
that one image entry (every annotation is kept byte-for-byte), plus a
write-once amendment record. The released files are only read.

    python scripts/campaign_pidray_bs1/build_eval_exclusion.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

EXCLUDED = ("xray_easy22241.png",)
AMEND = C.OUT / "amendments"
RECORD = AMEND / "eval_exclusion_2026-09-24.json"
DERIVED = dict(
    test_easy=AMEND / "annotations" / "test_easy_eval.json",
    occ_test_easy=AMEND / "annotations" / "test_easy_occ_eval.json",
)


def derive(src: Path, dst: Path) -> dict:
    payload = json.loads(src.read_text())
    drop = {i["id"] for i in payload["images"] if i["file_name"] in EXCLUDED}
    if len(drop) != len(EXCLUDED):
        raise SystemExit(f"{src}: expected {len(EXCLUDED)} excluded images, found {len(drop)}")
    touching = [a["id"] for a in payload["annotations"] if a["image_id"] in drop]
    if touching:
        raise SystemExit(f"{src}: excluded image has annotations {touching[:5]}; refusing")
    out = dict(payload)
    out["images"] = [i for i in payload["images"] if i["id"] not in drop]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"{dst} exists; derived files are write-once")
    dst.write_text(json.dumps(out))
    dst.chmod(0o444)
    back = json.loads(dst.read_text())
    assert back["annotations"] == payload["annotations"]
    assert back["categories"] == payload["categories"]
    assert len(back["images"]) == len(payload["images"]) - len(drop)
    assert {i["file_name"] for i in payload["images"]} - {i["file_name"] for i in back["images"]} == set(EXCLUDED)
    return dict(source=str(src.relative_to(ROOT)), source_sha256=C.sha256(src),
                path=str(dst.relative_to(ROOT)), sha256=C.sha256(dst),
                images=len(back["images"]), source_images=len(payload["images"]),
                annotations=len(back["annotations"]), removed_image_ids=sorted(drop))


def main() -> int:
    if RECORD.exists():
        raise SystemExit(f"{RECORD} exists; the amendment is write-once")
    done = sorted(str(p.relative_to(ROOT)) for p in C.OUT.rglob("eval_record_job*.json"))
    if done:
        raise SystemExit(f"evaluation results already exist ({done[:3]}); this amendment must precede them")
    manifest = json.loads(C.MANIFEST.read_text())
    if C.sha256(ROOT / manifest["annotations"]["test_easy"]["path"]) != manifest["annotations"]["test_easy"]["sha256"]:
        raise SystemExit("released test_easy.json no longer matches the frozen manifest")
    files = dict(test_easy=derive(C.ANN["test_easy"], DERIVED["test_easy"]),
                 occ_test_easy=derive(C.ANN["occ_test_easy"], DERIVED["occ_test_easy"]))
    bad = ROOT / "data/PIDray/test" / EXCLUDED[0]
    record = dict(
        amendment="exclude undecodable test_easy image from evaluation",
        decided="2026-09-24 by the user", written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        before_any_evaluation=True,
        excluded_images=list(EXCLUDED),
        excluded_file=dict(path=str(bad.relative_to(ROOT)), bytes=bad.stat().st_size, sha256=C.sha256(bad),
                           state="first ~11.8 KB are not PNG (no signature, no IHDR); cv2 and PIL cannot decode; "
                                 "a replacement copy from the user's source has the same size and the same damaged header"),
        excluded_annotations=0,
        scope="test_easy only, every method; hard and hidden unchanged; training unaffected",
        restored_same_day=["xray_easy10732.png", "xray_easy15075.png", "xray_easy14802.png", "xray_easy16776.png"],
        evidence="output/pidray_bs2_e10_2026-09-22/data_audit_2026-09-24/",
        derived_files=files,
        released_files="unchanged (read only)",
        evaluation_entry_point="scripts/campaign_pidray_bs1/evaluate_run_amended.py (wraps the frozen evaluate_run.py)")
    C.write_once(RECORD, record)
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
