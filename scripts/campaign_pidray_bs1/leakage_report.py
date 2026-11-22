#!/usr/bin/env python
"""Enumerate every byte-identical image shared between PIDray splits.

Read-only. scripts/audit_pidray.py reports the first five examples and stops;
this lists all of them, says which side carries annotations, and writes the
train-side exclusion list a de-leaked training annotation would need.

Original annotation files are never touched.

    python scripts/campaign_pidray_bs1/leakage_report.py
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pidray_bs1_e50_2026-09-21/audit"
DATA = ROOT / "data/PIDray"
SPLITS = ("train", "easy", "hard", "hidden")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_index(train: bool) -> dict[str, Path]:
    dirs = ((DATA / "train/train0", DATA / "train/train1") if train
            else (DATA / "test",))
    out = {}
    for d in dirs:
        with os.scandir(d) as entries:
            for e in entries:
                if e.is_file():
                    out[e.name] = Path(e.path)
    return out


def main() -> int:
    train_files, test_files = image_index(True), image_index(False)
    meta, paths = {}, {}
    for split in SPLITS:
        name = "train.json" if split == "train" else f"test_{split}.json"
        data = json.loads((DATA / "annotations" / name).read_text())
        annotated = defaultdict(int)
        for a in data["annotations"]:
            annotated[a["image_id"]] += 1
        available = train_files if split == "train" else test_files
        meta[split] = {
            Path(i["file_name"]).name: dict(id=i["id"], instances=annotated.get(i["id"], 0))
            for i in data["images"]}
        paths[split] = {n: available[n] for n in meta[split]}
        print(f"{split}: {len(meta[split])} images, "
              f"{sum(v['instances'] > 0 for v in meta[split].values())} annotated", flush=True)

    sizes = defaultdict(set)
    for split, ps in paths.items():
        for p in ps.values():
            sizes[p.stat().st_size].add(split)
    shared = {s for s, owners in sizes.items() if len(owners) > 1}
    digests: dict[str, dict[str, str]] = {}
    for split, ps in paths.items():
        digests[split] = {n: sha256(p) for n, p in ps.items()
                          if p.stat().st_size in shared}
        print(f"hashed {len(digests[split])} candidates in {split}", flush=True)

    by_hash: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split, d in digests.items():
        for name, h in d.items():
            by_hash[h][split].append(name)

    pairs, train_exclude = [], {}
    for h, owners in by_hash.items():
        if len(owners) < 2:
            continue
        rec = {s: [dict(name=n, instances=meta[s][n]["instances"], image_id=meta[s][n]["id"])
                   for n in sorted(names)] for s, names in owners.items()}
        pairs.append(dict(sha256=h, splits=rec))
        if "train" in owners and len(owners) > 1:
            for n in owners["train"]:
                train_exclude[n] = dict(
                    image_id=meta["train"][n]["id"],
                    instances=meta["train"][n]["instances"],
                    duplicate_of={s: sorted(v) for s, v in owners.items() if s != "train"})

    summary = defaultdict(lambda: dict(groups=0, train_annotated=0, test_annotated=0))
    for p in pairs:
        present = sorted(p["splits"])
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                k = f"{a}/{b}"
                summary[k]["groups"] += 1
                summary[k]["train_annotated"] += sum(
                    x["instances"] > 0 for x in p["splits"][a])
                summary[k]["test_annotated"] += sum(
                    x["instances"] > 0 for x in p["splits"][b])

    train_test = {k: v for k, v in summary.items() if k.startswith("train/")}
    report = dict(
        candidates_hashed={s: len(d) for s, d in digests.items()},
        duplicate_groups=len(pairs),
        by_split_pair={k: dict(v) for k, v in sorted(summary.items())},
        train_test_images_to_exclude=len(train_exclude),
        train_test_annotated_images_to_exclude=sum(
            v["instances"] > 0 for v in train_exclude.values()),
        train_test_instances_to_exclude=sum(v["instances"] for v in train_exclude.values()),
        train_images_after_exclusion=len(meta["train"]) - len(train_exclude),
        note=("train-side removal keeps all three test splits exactly as released; "
              "test/test duplicates are inside the evaluation sets and are not leakage "
              "between training and evaluation"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_groups.json").write_text(json.dumps(pairs, indent=1))
    (OUT / "train_exclusions.json").write_text(json.dumps(train_exclude, indent=1))
    (OUT / "leakage_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("train/test pairs:", json.dumps({k: dict(v) for k, v in train_test.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
