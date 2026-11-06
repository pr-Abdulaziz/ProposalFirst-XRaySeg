#!/usr/bin/env python
"""Re-check the leakage gate against the de-leaked training annotation.

leakage_report.py hashed every image whose file size occurs in more than one
split, so leakage_groups.json is the complete set of cross-split byte-identical
images: any duplicate not in it would have had a size unique to its own split,
which is impossible. The gate therefore closes if and only if no image remaining
in the de-leaked training file appears in a group that also contains a test image.

    python scripts/campaign_pidray_bs1/verify_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

AUDIT = C.AUDIT


def names(path: Path) -> set[str]:
    return {Path(i["file_name"]).name for i in json.loads(path.read_text())["images"]}


def main() -> int:
    groups = json.loads((AUDIT / "leakage_groups.json").read_text())
    train = names(C.DELEAKED_TRAIN_ANN)
    occ_train = names(C.DELEAKED_OCC_TRAIN_ANN)
    released = names(C.ANN["train"])
    test = {s: names(C.ANN[f"test_{s}"]) for s in C.TEST_SPLITS}

    checks, failures = {}, []
    checks["train_images"] = len(train)
    checks["occluder_matches_standard"] = train == occ_train
    checks["subset_of_released_train"] = train <= released
    checks["removed"] = len(released - train)

    remaining = []
    for g in groups:
        members = g["splits"]
        if "train" not in members:
            continue
        if any(s in members for s in C.TEST_SPLITS):
            still = [m["name"] for m in members["train"] if m["name"] in train]
            if still:
                remaining.append((g["sha256"], still))
    checks["train_test_duplicate_groups_remaining"] = len(remaining)

    for split, tnames in test.items():
        overlap = train & tnames
        checks[f"filename_overlap_train_{split}"] = len(overlap)
        if overlap:
            failures.append(f"filename overlap train/{split}: {sorted(overlap)[:5]}")
    if remaining:
        failures.append(f"byte duplicates still shared with a test split: {remaining[:5]}")
    if not checks["occluder_matches_standard"]:
        failures.append("de-leaked occluder and standard training sets differ")
    if not checks["subset_of_released_train"]:
        failures.append("de-leaked training set is not a subset of the released one")
    if len(train) != 22990:
        failures.append(f"unexpected training image count {len(train)}")

    # test/test duplicates are recorded, not removed: they are inside the
    # evaluation sets, not between training and evaluation.
    checks["test_internal_duplicate_groups"] = sum(
        1 for g in groups if "train" not in g["splits"] and len(g["splits"]) > 1)

    result = dict(gate="PIDray train/test byte isolation", policy="exclude_train_duplicates",
                  checks=checks, failures=failures,
                  status="PASS" if not failures else "FAIL")
    (AUDIT / "gate_verification.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
