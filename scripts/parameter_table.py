#!/usr/bin/env python
"""Total / trainable / frozen parameter counts for methods 01-09.

Counts are taken from the built model, not from a checkpoint, so an arm that has
never been trained is reported on the same footing as one that has. No weights
are downloaded and ``init_weights`` is never called: ``MODELS.build`` alone is
enough to fix every parameter's shape and its ``requires_grad`` flag, because
every freezing rule in this project (ResNet ``frozen_stages``, the SAM image
encoder, LoRA) is applied inside ``__init__``.

Frozen is defined as ``requires_grad is False`` at build time. Buffers -- BN
running statistics above all -- are counted separately: they are state, they are
not parameters, and folding them into either column would misreport both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402

from mmdet.registry import MODELS  # noqa: E402
from projects.proposal_first_pidray_occ import registry_seven  # noqa: E402

# 09 and 09M are development candidates, deliberately outside the active
# registry. They are listed here because a parameter table that stopped at 08
# could not be used to judge the cost of the thing being proposed.
EXTRA = {
    "09": ("Method 03 + transformer point refinement",
           "projects/proposal_first_pidray_occ/configs/pixray_09_point_refine.py"),
    "09M": ("Method 03 + capacity-matched MLP point refinement (control)",
            "projects/proposal_first_pidray_occ/configs/pixray_09m_point_refine_mlp.py"),
}


def counts(model: torch.nn.Module) -> dict:
    total = trainable = frozen = 0
    for parameter in model.parameters():
        n = parameter.numel()
        total += n
        if parameter.requires_grad:
            trainable += n
        else:
            frozen += n
    buffers = sum(b.numel() for b in model.buffers())
    assert trainable + frozen == total
    return dict(total=total, trainable=trainable, frozen=frozen,
                buffers=buffers,
                frozen_fraction=(frozen / total if total else 0.0))


def submodules(model: torch.nn.Module) -> dict:
    out = {}
    for name in ("backbone", "neck", "rpn_head", "roi_head", "r0_head",
                 "point_refine_head"):
        module = getattr(model, name, None)
        if module is None and name == "point_refine_head":
            module = getattr(getattr(model, "roi_head", None), name, None)
        if isinstance(module, torch.nn.Module):
            out[name] = sum(p.numel() for p in module.parameters())
    return out


def build(config_path: str):
    cfg = Config.fromfile(str(ROOT / config_path))
    init_default_scope(cfg.get("default_scope", "mmdet"))
    return MODELS.build(cfg.model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/dev_campaign/2026-09-12/parameter_table.json")
    args = parser.parse_args()

    entries = [(m.id, m.name, m.config) for m in registry_seven.METHODS.values()]
    entries += [(k, name, path) for k, (name, path) in EXTRA.items()]

    rows, failures = [], []
    for method_id, name, config_path in entries:
        try:
            model = build(config_path)
            row = dict(id=method_id, name=name, config=config_path,
                       **counts(model), parts=submodules(model))
            del model
        except Exception as error:                     # reported, never hidden
            row = dict(id=method_id, name=name, config=config_path,
                       error=f"{type(error).__name__}: {error}")
            failures.append(method_id)
        rows.append(row)
        print(f"built {method_id}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))

    head = f"{'ID':<4} {'total':>12} {'trainable':>12} {'frozen':>12} {'frozen %':>9}  name"
    print("\n" + head)
    print("-" * len(head))
    for row in rows:
        if "error" in row:
            print(f"{row['id']:<4} {'BUILD FAILED':>50}  {row['name']}")
            print(f"     {row['error']}")
            continue
        print(f"{row['id']:<4} {row['total']:>12,} {row['trainable']:>12,} "
              f"{row['frozen']:>12,} {100*row['frozen_fraction']:>8.2f}%  {row['name']}")
    print(f"\nwritten to {out_path}")
    if failures:
        print(f"BUILD FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
