#!/usr/bin/env python
"""Total and trainable parameters for the ten campaign experiments (Table 6).

Builds each model from its frozen resolved config (seed 42) on CPU, without loading weights, and counts
parameters with requires_grad after model.train(), which applies MMDetection's frozen_stages and
norm_eval rules. Writes output/<campaign>/revision_2026-11/a5_parameters.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.campaign_common import METHODS, OUT, run_key  # noqa: E402
from scripts.campaign_multiseed.run_guard_amended import verified_manifest  # noqa: E402


def main() -> int:
    from mmdet.registry import MODELS
    init_default_scope("mmdet")
    man = json.loads(verified_manifest().read_text())["runs"]
    rows = {}
    for m in METHODS:
        cfg = Config.fromfile(str(ROOT / man[run_key(m.id, 42)]["resolved_config"]), import_custom_modules=True)
        if cfg.get("custom_imports"):
            import_modules_from_strings(**cfg.custom_imports)
        model = MODELS.build(cfg.model)
        model.train()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        by_top = Counter()
        for n, p in model.named_parameters():
            if p.requires_grad:
                by_top[n.split(".")[0]] += p.numel()
        rows[m.id] = dict(name=m.name, total=total, trainable=trainable, frozen=total - trainable,
                          trainable_by_module=dict(by_top))
        print(f"{m.id:4s} {m.name:45s} total {total:>12,} trainable {trainable:>12,}", flush=True)
        del model
    dest = OUT / "revision_2026-11" / "a5_parameters.json"
    dest.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
