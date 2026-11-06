#!/usr/bin/env python
"""Build and leakage-check a training-only development split for PIXray.

Why this exists
---------------
Every historical PIXray checkpoint in this project was trained on the whole
official training split and evaluated on the official test split. Any design
decision taken by looking at those numbers -- which of three refinement arms to
keep, for instance -- is a decision taken on the test set, and a test set used
for selection no longer measures generalisation. A development split that is
carved out of the *training* images only, and never touches ``test.json``, is
what makes a three-arm comparison honest.

What it does and does not touch
-------------------------------
``data/PIXray/annotations/train.json`` and ``test.json`` are read and never
written. The split is an external manifest under ``splits/`` listing image ids.
Derived annotation files are written next to the manifest, are reproducible
from it, and preserve every field of every retained record verbatim: image ids,
annotation ids, category ids and their order, ``bbox``, ``segmentation``,
``area``, ``iscrowd``. Nothing is reindexed, renumbered, recomputed or dropped
other than by whole-image membership, and the two derived files partition the
original exactly.

Stratification
--------------
Images carry multiple categories, so marginal class balance is not enough.
Images are grouped by their exact set of present categories and each group is
divided at the target fraction, which balances joint label combinations and, as
a consequence, the marginals too. Groups too small to divide are tracked with a
running deficit so they do not all drift into training.

Leakage checks
--------------
  1. dev-train and dev-val image ids are disjoint and reconstruct the original
  2. file names are disjoint
  3. image *content* hashes are disjoint -- two different file names holding
     identical pixels would leak an image across the boundary
  4. no dev-val file name or content hash occurs in the official test split
  5. every category occurs in both parts
  6. annotation counts and ids are conserved exactly
  7. the source annotation files' own hashes are recorded, so the split can be
     shown to derive from unmodified annotations
Any failure is fatal: a split that cannot be certified must not be usable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(values, seed: int):
    """Deterministic order that does not depend on Python's hash randomisation
    or on dict insertion order, so the split is reproducible anywhere."""
    return sorted(values, key=lambda v: hashlib.sha256(
        f"{seed}:{v}".encode()).hexdigest())


def build_split(train: dict, val_fraction: float, seed: int):
    per_image = defaultdict(set)
    for ann in train["annotations"]:
        per_image[ann["image_id"]].add(ann["category_id"])

    groups = defaultdict(list)
    for image in train["images"]:
        signature = tuple(sorted(per_image.get(image["id"], ())))
        groups[signature].append(image["id"])

    val_ids, deficit = set(), 0.0
    for signature in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        members = stable_order(groups[signature], seed)
        want = len(members) * val_fraction + deficit
        take = int(round(want))
        take = max(0, min(take, len(members)))
        deficit = want - take
        val_ids.update(members[:take])

    train_ids = {img["id"] for img in train["images"]} - val_ids
    return train_ids, val_ids, groups


def subset(source: dict, keep_ids: set) -> dict:
    """Whole-image subset. Every retained record is the original object; no
    field is rewritten, so ids and geometry cannot drift."""
    out = {k: v for k, v in source.items() if k not in ("images", "annotations")}
    out["images"] = [i for i in source["images"] if i["id"] in keep_ids]
    out["annotations"] = [a for a in source["annotations"]
                          if a["image_id"] in keep_ids]
    return out


def hash_images(directory: Path, names) -> dict:
    return {name: sha256_file(directory / name) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data/PIXray"))
    parser.add_argument("--out", default=str(ROOT / "splits/pixray_dev_v1"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--skip-image-hashes", action="store_true",
                        help="skip checks 3 and 4; the manifest is then marked "
                             "uncertified and must not be used for a campaign")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ann_dir = data_root / "annotations"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path, test_path = ann_dir / "train.json", ann_dir / "test.json"
    train = json.loads(train_path.read_text())
    test = json.loads(test_path.read_text())

    train_ids, val_ids, groups = build_split(train, args.val_fraction, args.seed)
    dev_train = subset(train, train_ids)
    dev_val = subset(train, val_ids)

    failures, notes = [], []

    def check(condition, message):
        (notes if condition else failures).append(
            ("PASS " if condition else "FAIL ") + message)

    # 1 / 6 -- partition and conservation
    check(not (train_ids & val_ids), "1 image ids disjoint")
    check(train_ids | val_ids == {i["id"] for i in train["images"]},
          "1 image ids reconstruct the official training split")
    check(len(dev_train["images"]) + len(dev_val["images"]) == len(train["images"]),
          f"6 image count conserved ({len(train['images'])})")
    check(len(dev_train["annotations"]) + len(dev_val["annotations"])
          == len(train["annotations"]),
          f"6 annotation count conserved ({len(train['annotations'])})")
    original_ann_ids = [a["id"] for a in train["annotations"]]
    derived_ann_ids = ([a["id"] for a in dev_train["annotations"]]
                       + [a["id"] for a in dev_val["annotations"]])
    check(sorted(original_ann_ids) == sorted(derived_ann_ids),
          "6 annotation ids preserved exactly, none reindexed")
    check(dev_train["categories"] == train["categories"]
          == dev_val["categories"], "6 categories identical and in order")

    # 2 -- file names
    train_names = {i["file_name"] for i in dev_train["images"]}
    val_names = {i["file_name"] for i in dev_val["images"]}
    check(not (train_names & val_names), "2 file names disjoint")

    # 5 -- category coverage
    def present(part):
        return {a["category_id"] for a in part["annotations"]}
    all_cats = {c["id"] for c in train["categories"]}
    check(present(dev_train) == all_cats,
          f"5 all {len(all_cats)} categories present in dev-train")
    missing_val = all_cats - present(dev_val)
    check(not missing_val, f"5 all categories present in dev-val "
                           f"(missing: {sorted(missing_val)})")

    # 3 / 4 -- content hashes
    image_hashes = {}
    if args.skip_image_hashes:
        failures.append("FAIL 3/4 image content hashes were skipped; "
                        "this manifest is UNCERTIFIED")
    else:
        print("hashing training images ...", flush=True)
        image_hashes = hash_images(data_root / "train", sorted(train_names | val_names))
        print("hashing test images ...", flush=True)
        test_names = [i["file_name"] for i in test["images"]]
        test_hashes = hash_images(data_root / "test", sorted(test_names))

        train_digests = {image_hashes[n] for n in train_names}
        val_digests = {image_hashes[n] for n in val_names}
        overlap = train_digests & val_digests
        check(not overlap, f"3 no identical image content across dev-train/dev-val "
                           f"({len(overlap)} colliding digests)")

        test_digest_set = set(test_hashes.values())
        val_in_test = val_digests & test_digest_set
        train_in_test = train_digests & test_digest_set
        check(not val_in_test,
              f"4 no dev-val image content occurs in the official test split "
              f"({len(val_in_test)} colliding digests)")
        check(not (val_names & set(test_names)),
              "4 no dev-val file name occurs in the official test split")
        if train_in_test:
            notes.append(
                f"NOTE  {len(train_in_test)} digests are shared between the "
                "official train and test splits. That is a property of the "
                "released dataset, not of this split, and it affects every "
                "method equally; it is recorded, not repaired.")
        else:
            notes.append("PASS  official train and test splits share no image content")

        duplicates = Counter(image_hashes.values())
        repeated = {d: c for d, c in duplicates.items() if c > 1}
        # Reported for completeness. Check 3 already guarantees no duplicate
        # straddles the dev boundary; this says whether any exist at all.
        notes.append(f"NOTE  {len(repeated)} image digests occur more than once "
                     f"inside the official training split")

    # marginal balance, reported so the split can be judged, not just trusted
    def marginal(part):
        counter = Counter(a["category_id"] for a in part["annotations"])
        total = sum(counter.values())
        return {c: counter.get(c, 0) / total for c in sorted(all_cats)}, counter
    train_frac, train_count = marginal(dev_train)
    val_frac, val_count = marginal(dev_val)
    names = {c["id"]: c["name"] for c in train["categories"]}
    balance = [dict(category_id=c, name=names[c], dev_train=train_count.get(c, 0),
                    dev_val=val_count.get(c, 0),
                    dev_train_share=round(train_frac[c], 5),
                    dev_val_share=round(val_frac[c], 5),
                    abs_share_gap=round(abs(train_frac[c] - val_frac[c]), 5))
               for c in sorted(all_cats)]
    worst = max(b["abs_share_gap"] for b in balance)

    manifest = dict(
        name=out_dir.name,
        created="2026-09-12",
        purpose="Training-only development split for PIXray. dev-val is held out "
                "of training for all development arms. The official test split is "
                "not read by any development run and is not used for selection.",
        dataset="PIXray",
        source=dict(
            train_json=str(train_path.relative_to(ROOT)),
            train_json_sha256=sha256_file(train_path),
            test_json=str(test_path.relative_to(ROOT)),
            test_json_sha256=sha256_file(test_path),
        ),
        parameters=dict(val_fraction=args.val_fraction, seed=args.seed,
                        stratification="exact category-set signature groups"),
        sizes=dict(
            official_train_images=len(train["images"]),
            official_train_annotations=len(train["annotations"]),
            dev_train_images=len(dev_train["images"]),
            dev_train_annotations=len(dev_train["annotations"]),
            dev_val_images=len(dev_val["images"]),
            dev_val_annotations=len(dev_val["annotations"]),
            achieved_val_fraction=round(len(dev_val["images"]) / len(train["images"]), 5),
            signature_groups=len(groups),
        ),
        class_balance=balance,
        worst_class_share_gap=worst,
        dev_train_image_ids=sorted(train_ids),
        dev_val_image_ids=sorted(val_ids),
        leakage_checks=notes + failures,
        certified=not failures,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not failures:
        for name, part in (("dev_train.json", dev_train), ("dev_val.json", dev_val)):
            (out_dir / name).write_text(json.dumps(part))
        manifest["derived"] = {
            name: sha256_file(out_dir / name)
            for name in ("dev_train.json", "dev_val.json")}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if image_hashes:
            (out_dir / "image_digests.json").write_text(
                json.dumps(image_hashes, indent=0))

    print("\n".join(notes + failures))
    print(f"\ndev-train {len(dev_train['images'])} images / "
          f"{len(dev_train['annotations'])} anns")
    print(f"dev-val   {len(dev_val['images'])} images / "
          f"{len(dev_val['annotations'])} anns "
          f"({manifest['sizes']['achieved_val_fraction']:.3%})")
    print(f"worst per-class share gap: {worst:.5f}")
    print(f"certified: {manifest['certified']}  ->  {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
