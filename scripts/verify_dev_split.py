#!/usr/bin/env python
"""Re-verify the development split before a run uses it.

Cheap enough to run at the head of every training job. It confirms that the
source annotations, the manifest and the two derived files are the exact bytes
the split was certified on, and that the partition still holds. It does not
re-hash the images: that is what ``build_dev_split.py`` does once, at creation.

Exit status is 1 on any mismatch, so a launcher can refuse to start.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(split_dir: str = "splits/pixray_dev_v1") -> int:
    root = ROOT / split_dir
    manifest = json.loads((root / "manifest.json").read_text())
    failures = []

    def check(condition, message):
        print(("PASS " if condition else "FAIL ") + message)
        if not condition:
            failures.append(message)

    check(manifest.get("certified") is True, "manifest is certified")
    for key in ("train_json", "test_json"):
        path = ROOT / manifest["source"][key]
        check(sha256_file(path) == manifest["source"][f"{key}_sha256"],
              f"source {manifest['source'][key]} is unmodified")
    for name, expected in manifest["derived"].items():
        check(sha256_file(root / name) == expected, f"derived {name} matches manifest")

    train = json.loads((root / "dev_train.json").read_text())
    val = json.loads((root / "dev_val.json").read_text())
    train_ids = {i["id"] for i in train["images"]}
    val_ids = {i["id"] for i in val["images"]}
    check(not (train_ids & val_ids), "dev-train and dev-val image ids still disjoint")
    check(train_ids == set(manifest["dev_train_image_ids"]), "dev-train ids match manifest")
    check(val_ids == set(manifest["dev_val_image_ids"]), "dev-val ids match manifest")
    check(len(train_ids) == manifest["sizes"]["dev_train_images"]
          and len(val_ids) == manifest["sizes"]["dev_val_images"],
          f"sizes still {len(train_ids)} / {len(val_ids)}")

    print("SPLIT VERIFIED" if not failures else f"SPLIT VERIFICATION FAILED ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
