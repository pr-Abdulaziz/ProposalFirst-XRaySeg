#!/usr/bin/env python
"""Build the resolved configs of the PIDray gradient-isolation diagnostic (no training is launched).

Arms, all under the frozen PIDray protocol (batch 2, 10 epochs, fp32, same data, schedule and losses):
  m03_cascade_r101_attfpn            Parent                               (seed 42 exists: reuse)
  m09m_point_refine_mlp              MLP refiner, original                (seed 42 exists: reuse)
  m09md_point_refine_mlp_detached    MLP refiner, P2 samples and coarse-logit samples detached
  m09d_point_refine_transformer_detached   transformer refiner, same detachment
Each config is derived from the frozen PIDray resolved config of its source method. The only keys that
may differ are the seed-dependent ones (randomness seed, point-selection seed, RNG-hook run key,
work_dir) and, for detached arms, the point-head type and one added custom import. The builder
asserts this key by key. Point-selection seeds use the original campaign's formula, so arms with the
same seed share data order and point sampling.

    python scripts/campaign_pidray_bs1/build_gradiso_configs.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmengine.config import Config  # noqa: E402

SRC_CAMPAIGN = "pidray_bs2_e10_2026-09-22"
DIAG = "pidray_diag_gradiso_2026-09-26"
SRC = ROOT / "output" / SRC_CAMPAIGN / "protocol/resolved_configs"
OUT = ROOT / "output" / DIAG
WORK = ROOT / "work_dirs" / DIAG
SEEDS = (42, 123, 2026)
DETACH_IMPORT = "projects.proposal_first_pidray_occ.models.point_refine.detached"
ARMS = {  # slug: (source slug, detached head type or None, existing seed-42 run to reuse or None)
    "m03_cascade_r101_attfpn": ("m03_cascade_r101_attfpn", None, "m03_cascade_r101_attfpn"),
    "m09m_point_refine_mlp": ("m09m_point_refine_mlp", None, "m09m_point_refine_mlp"),
    "m09md_point_refine_mlp_detached": ("m09m_point_refine_mlp", "PointRefineMLPHeadDetached", None),
    "m09d_point_refine_transformer_detached": ("m09_point_refine_transformer", "PointRefineHeadDetached", None),
}


def point_seed(seed: int) -> int:
    return int(hashlib.sha256(f"{SRC_CAMPAIGN}/point_refine/{seed}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def main() -> int:
    plan = dict(diagnostic=DIAG, source_campaign=SRC_CAMPAIGN, seeds=list(SEEDS), runs=[])
    for slug, (src_slug, head_type, reuse) in ARMS.items():
        src_cfg = Config.fromfile(str(SRC / src_slug / "seed_42.py"))
        base = flatten(src_cfg.to_dict())
        for seed in SEEDS:
            cfg = Config.fromfile(str(SRC / src_slug / "seed_42.py"))
            cfg.randomness = dict(seed=seed, deterministic=False, diff_rank_seed=False)
            if "seed" in cfg.model["roi_head"]:
                cfg.model["roi_head"]["seed"] = point_seed(seed)
            for hook in cfg.custom_hooks:
                if "run_key" in hook:
                    hook["run_key"] = f"{slug}/seed_{seed}"
            cfg.work_dir = str(WORK / slug / f"seed_{seed}")
            if head_type:
                cfg.model["roi_head"]["point_refine_head"]["type"] = head_type
                cfg.custom_imports["imports"] = list(cfg.custom_imports["imports"]) + [DETACH_IMPORT]
            new = flatten(cfg.to_dict())
            changed = sorted(k for k in set(base) | set(new) if base.get(k, "<absent>") != new.get(k, "<absent>"))
            allowed = {"randomness.seed", "model.roi_head.seed", "work_dir", "model.roi_head.point_refine_head.type"}
            unexpected = [k for k in changed if k not in allowed and not k.startswith("custom_hooks[")
                          and not k.startswith("custom_imports.imports[")]
            if unexpected:
                raise SystemExit(f"{slug}/seed_{seed}: unexpected config changes {unexpected}")
            dest = OUT / "protocol/resolved_configs" / slug / f"seed_{seed}.py"
            dest.parent.mkdir(parents=True, exist_ok=True)
            cfg.dump(str(dest))
            reuse_run = (reuse is not None and seed == 42)
            plan["runs"].append(dict(arm=slug, seed=seed, source=f"{src_slug}/seed_42.py", config=str(dest.relative_to(ROOT)),
                                     changed_keys=changed, point_seed=cfg.model["roi_head"].get("seed"),
                                     status="reuse existing PIDray run" if reuse_run else "to train",
                                     reuse=f"output/{SRC_CAMPAIGN}/runs/{reuse}/seed_42" if reuse_run else None))
            print(f"{slug:40s} seed {seed:4d}  {'reuse' if reuse_run else 'train'}  changed: {changed}")
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    (OUT / "protocol/plan.json").write_text(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
