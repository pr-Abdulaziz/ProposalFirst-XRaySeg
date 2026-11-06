#!/usr/bin/env python
"""Do 03, 09 and 09M actually see the same augmented data?

Matching seeds and worker counts are a reason to *expect* the same stream, not
evidence that it happens. This builds each arm's training dataloader exactly the
way its Runner does -- same seed, same sampler, same worker count, same
worker_init_fn -- iterates a fixed number of batches with no model involved, and
hashes what comes out: image pixels after augmentation and normalisation, plus
boxes, labels and mask areas.

Identical digests over the tested prefix are direct evidence. They do not prove
the streams stay identical for all 106,800 iterations; the claim is scoped to
what was measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.runner import set_random_seed  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402

register_all_modules(init_default_scope=True)

ARMS = {
    "03": "work_dirs/pixray_p2/pixray_p2_cascade_attfpn.py",
    "09": "work_dirs/pixray_09_point_refine/pixray_09_point_refine.py",
    "09M": "output/full_split_campaign/2026-09-12/resolved_09M.py",
}


def digest_batch(batch) -> str:
    h = hashlib.sha256()
    for tensor in batch["inputs"]:
        h.update(np.ascontiguousarray(tensor.numpy()).tobytes())
    for sample in batch["data_samples"]:
        h.update(str(sample.img_id).encode())
        h.update(str(tuple(sample.img_shape)).encode())
        gt = sample.gt_instances
        h.update(np.ascontiguousarray(
            gt.bboxes.tensor.numpy().round(4)).tobytes())
        h.update(np.ascontiguousarray(gt.labels.numpy()).tobytes())
        if "masks" in gt:
            h.update(np.ascontiguousarray(
                np.asarray(gt.masks.areas, dtype=np.int64)).tobytes())
    return h.hexdigest()


def stream(config_path, batches, seed):
    cfg = Config.fromfile(str(ROOT / config_path))
    set_random_seed(seed, deterministic=cfg.randomness.get("deterministic", False))
    loader = Runner.build_dataloader(
        cfg.train_dataloader, seed=seed,
        diff_rank_seed=cfg.randomness.get("diff_rank_seed", False))
    out, ids = [], []
    iterator = iter(loader)
    for _ in range(batches):
        batch = next(iterator)
        out.append(digest_batch(batch))
        ids.append([int(s.img_id) for s in batch["data_samples"]])
    del iterator, loader
    return out, ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = {}
    for arm, path in ARMS.items():
        print(f"streaming {arm} ...", flush=True)
        results[arm] = stream(path, args.batches, args.seed)

    base_d, base_i = results["03"]
    report = {"batches": args.batches, "seed": args.seed, "arms": {}}
    print()
    for arm, (digests, ids) in results.items():
        same_order = ids == base_i
        same_pixels = digests == base_d
        first_diff = next((i for i, (a, b) in enumerate(zip(digests, base_d))
                           if a != b), None)
        report["arms"][arm] = dict(
            sample_order_matches_03=same_order,
            augmented_batches_match_03=same_pixels,
            first_differing_batch=first_diff,
            first_batch_digest=digests[0],
            first_batch_image_ids=ids[0],
        )
        print(f"  {arm:<4} order=={'03' if same_order else 'DIFFERS'}  "
              f"pixels=={'03' if same_pixels else f'DIFFERS at batch {first_diff}'}  "
              f"first digest {digests[0][:16]}")

    out = ROOT / "output/audit/2026-09-12-fairness/augmentation_stream.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    all_match = all(v["augmented_batches_match_03"] for v in report["arms"].values())
    print("VERDICT: identical augmented stream over the tested prefix"
          if all_match else "VERDICT: streams DIFFER -- see first_differing_batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
