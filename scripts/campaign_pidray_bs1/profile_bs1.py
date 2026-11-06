#!/usr/bin/env python
"""PIDray-30 time and memory profile at physical batch 1, no accumulation, 50 epochs.

Measurement only: no checkpoint is written and no metric is computed, so nothing
here is a PIDray result. It reuses scripts/pidray_profile/profile_pidray.py
unchanged and overrides only the protocol constants, so the batch-2 profile and
this one are produced by exactly the same code path.

    python scripts/campaign_pidray_bs1/profile_bs1.py <method id>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Must be set before profile_pidray imports torch/mmengine: it reads these at import.
os.environ.update(XRAY_BATCH_SIZE_PER_GPU="1", XRAY_EFFECTIVE_BATCH_SIZE="1",
                  XRAY_NUM_WORKERS="2")

from scripts.pidray_profile import profile_pidray as P  # noqa: E402

CAMPAIGN = "pidray_bs1_e50_2026-09-21"
P.PHYSICAL_BATCH = 1            # user instruction (2026-09-21)
P.ACCUMULATION = 1              # no gradient accumulation, as in the PIXray campaign
P.EPOCHS = 50                   # user instruction (2026-09-21)
P.OUT = ROOT / "output" / CAMPAIGN / "profile"
# 400 timing images is 400 iterations at batch 1 (it was 200 at batch 2).

if __name__ == "__main__":
    sys.exit(P.main())
