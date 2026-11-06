#!/usr/bin/env python
"""Run the frozen evaluate_run.py with the 2026-09-24 test_easy amendment applied.

evaluate_run.py and the resolved configs stay byte-identical, so every frozen
hash check inside it still runs and must still pass. The only change is what
Config.fromfile hands back: the test_easy loader/evaluator paths in
``campaign_test_splits`` point to the derived copies without the one
undecodable image (see build_eval_exclusion.py). The derived files' paths and
hashes land in the eval record through evaluate_run's own per-split fields.

    python scripts/campaign_pidray_bs1/evaluate_run_amended.py <method id> [<seed>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_pidray_bs1 import evaluate_run as E  # noqa: E402
from scripts.campaign_pidray_bs1.build_eval_exclusion import RECORD  # noqa: E402

AMENDED_SPLIT = "easy"
_FROZEN_CONFIG = E.Config


def amended_paths() -> dict[str, str]:
    """Released path -> derived path, after re-checking every recorded hash."""
    record = json.loads(RECORD.read_text())
    mapping = {}
    for rec in record["derived_files"].values():
        src, dst = ROOT / rec["source"], ROOT / rec["path"]
        if C.sha256(src) != rec["source_sha256"]:
            raise SystemExit(f"ABORT: released {src} changed since the amendment")
        if C.sha256(dst) != rec["sha256"]:
            raise SystemExit(f"ABORT: derived {dst} changed since the amendment")
        mapping[str(src)] = str(dst)
    return mapping


class _AmendedConfig(_FROZEN_CONFIG):
    MAPPING: dict[str, str] = {}

    @staticmethod
    def fromfile(*args, **kwargs):
        cfg = _FROZEN_CONFIG.fromfile(*args, **kwargs)
        split = cfg.campaign_test_splits[AMENDED_SPLIT]
        for role in ("loader", "evaluator"):
            if split[role] not in _AmendedConfig.MAPPING:
                raise SystemExit(f"ABORT: {role} {split[role]} has no amended copy")
            split[role] = _AmendedConfig.MAPPING[split[role]]
        return cfg


def main() -> int:
    _AmendedConfig.MAPPING = amended_paths()
    E.Config = _AmendedConfig  # evaluate_run resolves Config from its module namespace
    return E.main()


if __name__ == "__main__":
    sys.exit(main())
