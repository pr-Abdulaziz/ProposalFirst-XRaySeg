#!/usr/bin/env python
"""run_guard.py against the amended manifest (06 activation checkpointing, 2026-09-17).

run_guard.py itself is frozen and unchanged; this wrapper verifies the amended manifest against its
amendment record and points run_guard's MANIFEST at it, so every guard check still applies.

    python scripts/campaign_multiseed/run_guard_amended.py {gpu-check|train-start|train-end} ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.campaign_multiseed.run_guard as run_guard  # noqa: E402
from scripts.campaign_multiseed.amend_06_checkpointing import AMENDED_MANIFEST, RECORD  # noqa: E402
from scripts.campaign_multiseed.campaign_common import MANIFEST, sha256  # noqa: E402


def verified_manifest() -> Path:
    rec = json.loads(RECORD.read_text())
    if sha256(MANIFEST) != rec["frozen_manifest_sha256"] or sha256(AMENDED_MANIFEST) != rec["amended_manifest_sha256"]:
        raise SystemExit("ABORT: manifest hashes do not match the amendment record")
    return AMENDED_MANIFEST


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gpu-check":
        state = run_guard.gpu_check()
        print(json.dumps(state), flush=True)
        sys.exit(0 if state["clean"] else 4)
    if cmd == "config-path":
        man = json.loads(verified_manifest().read_text())
        print(ROOT / man["runs"][f"{run_guard.BY_ID[sys.argv[2]].slug}/seed_{int(sys.argv[3])}"]["resolved_config"])
        sys.exit(0)
    run_guard.MANIFEST = verified_manifest()
    mid, seed = sys.argv[2], int(sys.argv[3])
    if cmd == "train-start":
        run_guard.train_start(mid, seed, "--resume" in sys.argv[4:])
    elif cmd == "train-end":
        run_guard.train_end(mid, seed, int(sys.argv[4]))
    else:
        raise SystemExit(f"unknown command {cmd}")
