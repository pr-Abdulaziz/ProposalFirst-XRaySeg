#!/usr/bin/env python
"""Write de-leaked PIDray-30 training annotations.

User decision (2026-09-21): drop the 84 training images whose bytes also appear
in a test split. The released files under data/ are read only and never modified;
the derived files are written read-only under the campaign output directory, and
both the exclusion list and the resulting digests are recorded there.

Test splits are untouched, so evaluation stays exactly the released benchmark.

    python scripts/campaign_pidray_bs1/build_deleaked_annotations.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

AUDIT = C.OUT / "audit"
EXPECTED_EXCLUSIONS = 84


def derive(src: Path, dst: Path, drop_names: set[str]) -> dict:
    data = json.loads(src.read_text())
    keep_images, dropped_ids = [], set()
    for image in data["images"]:
        if Path(image["file_name"]).name in drop_names:
            dropped_ids.add(image["id"])
        else:
            keep_images.append(image)
    keep_ann = [a for a in data["annotations"] if a["image_id"] not in dropped_ids]
    if len(dropped_ids) != len(drop_names):
        raise SystemExit(f"{src}: matched {len(dropped_ids)} of {len(drop_names)} names")
    out = dict(data)
    out["images"], out["annotations"] = keep_images, keep_ann
    out["deleak_provenance"] = dict(
        derived_from=str(src.relative_to(ROOT)),
        source_sha256=C.sha256(src),
        removed_images=sorted(drop_names),
        removed_image_count=len(drop_names),
        removed_annotation_count=len(data["annotations"]) - len(keep_ann),
        reason=("byte-identical to an image in a PIDray test split; "
                "scripts/audit_pidray.py --check-content-hashes, 2026-09-21"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"{dst} already exists; refusing to overwrite a derived annotation")
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as stream:
        json.dump(out, stream)
    return dict(path=str(dst.relative_to(ROOT)), sha256=C.sha256(dst),
                images=len(keep_images), annotations=len(keep_ann),
                images_removed=len(drop_names),
                annotations_removed=len(data["annotations"]) - len(keep_ann),
                source=str(src.relative_to(ROOT)), source_sha256=out["deleak_provenance"]["source_sha256"],
                source_images=len(data["images"]), source_annotations=len(data["annotations"]))


def main() -> int:
    exclusions = json.loads((AUDIT / "train_exclusions.json").read_text())
    names = set(exclusions)
    if len(names) != EXPECTED_EXCLUSIONS:
        raise SystemExit(f"expected {EXPECTED_EXCLUSIONS} exclusions, found {len(names)}")
    record = dict(
        policy="exclude_train_duplicates",
        decided="2026-09-21",
        excluded_images=len(names),
        excluded_annotated_images=sum(v["instances"] > 0 for v in exclusions.values()),
        excluded_instances=sum(v["instances"] for v in exclusions.values()),
        standard=derive(C.ANN["train"], C.DELEAKED_TRAIN_ANN, names),
        occluder=derive(C.ANN["occ_train"], C.DELEAKED_OCC_TRAIN_ANN, names),
        test_splits="untouched; evaluation uses the released easy/hard/hidden files")
    if record["standard"]["images"] != 22990:
        raise SystemExit(f"unexpected train image count {record['standard']['images']}")
    if record["standard"]["images"] != record["occluder"]["images"]:
        raise SystemExit("standard and occluder training sets disagree after de-leaking")
    (AUDIT / "deleak_record.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
