#!/usr/bin/env python
"""Evaluate one run under the seed-42 scope amendment (user instruction, 2026-09-15).

The frozen campaign manifest is never edited. scope_amendment_seed42.py writes a
write-once scoped copy in which only seed-42 runs remain launchable. This wrapper
verifies both hashes against the amendment record, then runs evaluate_run.py
unchanged against the scoped copy, so all of its checks still apply:
evaluator sources, config hash, final-epoch checkpoint, and every in-scope
training run being complete.

    python scripts/campaign_multiseed/evaluate_scoped.py <method id> 42
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import MANIFEST, OUT, sha256  # noqa: E402
import scripts.campaign_multiseed.evaluate_run as evaluate_run  # noqa: E402

AMENDMENT = OUT / "summaries" / "scope_amendment_seed42.json"


def main() -> int:
    amend = json.loads(AMENDMENT.read_text())
    if sha256(MANIFEST) != amend["frozen_manifest_sha256"]:
        raise SystemExit("ABORT: frozen manifest does not match the amendment record")
    scoped = ROOT / amend["scoped_manifest"]
    if sha256(scoped) != amend["scoped_manifest_sha256"]:
        raise SystemExit("ABORT: scoped manifest changed after the amendment")
    if int(sys.argv[2]) not in amend["seeds_in_scope"]:
        raise SystemExit(f"ABORT: seed {sys.argv[2]} is outside the amended scope {amend['seeds_in_scope']}")
    evaluate_run.MANIFEST = scoped
    return evaluate_run.main()


if __name__ == "__main__":
    sys.exit(main())
