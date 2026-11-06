#!/usr/bin/env python
"""Run the official PIDray annotation audit against the campaign's own files.

verify_gate.py only proves the de-leaked training set no longer shares bytes with
a test split. This runs the checks in scripts/audit_pidray.py -- imported, not
re-implemented -- on the derived files themselves: canonical categories, unique
image and annotation ids, required fields (including bg_object_segmentation on
the occluder track), valid boxes, every referenced image present on disk, and
target parity between the standard and occluder files.

    python scripts/campaign_pidray_bs1/verify_annotations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_pidray import (assert_pairing, image_index, load,  # noqa: E402
                                  validate)
from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402


def main() -> int:
    C.require_leakage_policy()
    train_files = image_index(C.DATA / "train", train=True)
    test_files = image_index(C.DATA / "test", train=False)
    checks, failures = {}, []

    def run(name, fn):
        try:
            fn()
            checks[name] = "PASS"
        except Exception as exc:  # noqa: BLE001
            checks[name] = f"FAIL: {type(exc).__name__}: {exc}"
            failures.append(f"{name}: {exc}")

    standard = load(C.train_annotation(occ=False))
    occluder = load(C.train_annotation(occ=True))
    n = len(standard["images"])
    checks["deleaked_train_images"] = n
    checks["deleaked_train_instances"] = len(standard["annotations"])
    run("train_standard_validate",
        lambda: validate(standard, C.train_annotation(False), train_files, n, overlap=False))
    run("train_occluder_validate",
        lambda: validate(occluder, C.train_annotation(True), train_files, n, overlap=True))
    run("train_standard_vs_occluder_pairing",
        lambda: assert_pairing(standard, occluder, "train(deleaked)"))

    for split in C.TEST_SPLITS:
        std = load(C.ANN[f"test_{split}"])
        occ = load(C.ANN[f"occ_test_{split}"])
        count = C.TEST_IMAGES[split]
        run(f"test_{split}_standard_validate",
            lambda std=std, split=split, count=count: validate(
                std, C.ANN[f"test_{split}"], test_files, count, overlap=False))
        run(f"test_{split}_occluder_validate",
            lambda occ=occ, split=split, count=count: validate(
                occ, C.ANN[f"occ_test_{split}"], test_files, count, overlap=True))
        run(f"test_{split}_pairing",
            lambda std=std, occ=occ, split=split: assert_pairing(std, occ, split))

    # occluder supervision must survive de-leaking
    released_occ = load(C.ANN["occ_train"])
    def counted(payload):
        return sum(bool(a.get("bg_object_segmentation")) for a in payload["annotations"])
    checks["overlap_instances_released"] = counted(released_occ)
    checks["overlap_instances_deleaked"] = counted(occluder)
    checks["categories_in_campaign_order"] = tuple(
        c["name"] for c in sorted(standard["categories"], key=lambda c: c["id"])) == C.CLASSES
    if not checks["categories_in_campaign_order"]:
        failures.append("category order does not match campaign_common.CLASSES")
    if checks["overlap_instances_deleaked"] > checks["overlap_instances_released"]:
        failures.append("de-leaking increased the occluder instance count")

    # the derived files must be a strict subset of the released ones
    rel_std = load(C.ANN["train"])
    rel_ids = {i["id"] for i in rel_std["images"]}
    checks["image_ids_subset_of_released"] = {i["id"] for i in standard["images"]} <= rel_ids
    rel_ann = {a["id"] for a in rel_std["annotations"]}
    checks["annotation_ids_subset_of_released"] = {a["id"] for a in standard["annotations"]} <= rel_ann
    for key in ("image_ids_subset_of_released", "annotation_ids_subset_of_released"):
        if not checks[key]:
            failures.append(key)
    # annotations kept must be byte-identical to the released ones
    rel_by_id = {a["id"]: a for a in rel_std["annotations"]}
    changed = [a["id"] for a in standard["annotations"]
               if json.dumps(rel_by_id[a["id"]], sort_keys=True) != json.dumps(a, sort_keys=True)]
    checks["kept_annotations_unmodified"] = not changed
    if changed:
        failures.append(f"{len(changed)} kept annotations differ from the released ones")

    result = dict(checks=checks, failures=failures, status="PASS" if not failures else "FAIL")
    (C.AUDIT / "annotation_verification.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
