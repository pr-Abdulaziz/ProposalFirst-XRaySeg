#!/usr/bin/env python
"""evaluate_run.py against the amended manifest (06 activation checkpointing, 2026-09-17).

evaluate_run.py is frozen and unchanged. For every method except 06 the amended manifest is identical
to the frozen one; for 06 it names the checkpointed config. All evaluate_run checks still apply,
including that all 30 approved runs have completed training.

    python scripts/campaign_multiseed/evaluate_amended.py <method id> <seed>
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.campaign_multiseed.evaluate_run as evaluate_run  # noqa: E402
from scripts.campaign_multiseed.run_guard_amended import verified_manifest  # noqa: E402

if __name__ == "__main__":
    evaluate_run.MANIFEST = verified_manifest()
    sys.exit(evaluate_run.main())
