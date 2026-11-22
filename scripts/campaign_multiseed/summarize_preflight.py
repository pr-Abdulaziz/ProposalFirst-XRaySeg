#!/usr/bin/env python
"""Write preflight/PREFLIGHT_SUMMARY.md from the per-method preflight and geometry results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import METHODS, OUT  # noqa: E402


def main() -> int:
    L = ["# Preflight summary (final source)", "",
         "Each method ran its frozen seed-42 resolved config on the eight most populated training images × 20 "
         "(160 images in batches of 4, 40 updates, no accumulation), plus a memory stress pass at resize ratio 1.5, an "
         "empty-instance batch, checkpoint save/strict reload, RNG restore, and inference on difficult test images. "
         "Timing uses the densest images and includes host overhead, so it is an upper-side estimate.", "",
         "| Method | Checks | Geometry | Params total (M) | Trainable (M) | Frozen (M) | Peak train GB | Peak stress GB | "
         "Peak infer GB | Headroom GB | s/iteration | data wait s | Projected h/run | AMP skipped (of 40) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    notes = []
    for m in METHODS:
        p = OUT / "preflight" / m.slug / "preflight.json"
        if not m.launchable:
            L.append(f"| {m.id} | held | | | | | | | | | | | | |")
            continue
        r = json.loads(p.read_text())
        g = OUT / "preflight" / m.slug / "inference_geometry.json"
        geo = json.loads(g.read_text()) if g.is_file() else {}
        acct = r["update_accounting"][-1]["cumulative"]
        pk = max(i["peak_reserved_mb"] for i in r["iterations"]) / 1024
        L.append(f"| {m.id} | {r['summary']['passed']}/{r['summary']['passed'] + r['summary']['failed']} | "
                 f"{geo.get('status', 'pending')} ({geo.get('nonempty_masks_checked', '-')} masks) | "
                 f"{r['parameters']['total']/1e6:.2f} | {r['parameters']['trainable']/1e6:.2f} | {r['parameters']['frozen']/1e6:.2f} | "
                 f"{pk:.2f} | {r['stress_train']['peak_reserved_mb']/1024:.2f} | {r['inference']['peak_reserved_mb']/1024:.2f} | "
                 f"{r['headroom_mb']/1024:.1f} | {r['timing']['median_s_per_iteration']:.3f} | "
                 f"{r['timing']['median_data_wait_s'] or 0:.3f} | "
                 f"{r['timing']['projected_train_hours']:.1f} | {acct['amp_skipped_updates']} |")
        silent = sorted({".".join(n.split(".")[:3]) for n in r["trainable_tensors_without_gradient_signal"]})
        safe = [n for n, b in r["batchnorm"].items() if b["type"] == "BatchOneSafeBatchNorm2d"]
        notes.append(f"- **{m.id}** — GPU {r['gpu']} ({r['node']}). Trainable tensors with no gradient signal: "
                     f"{', '.join(silent) if silent else 'none'}. AttFPN attention BN modules: {len(safe)}. "
                     f"Frozen tensors held by the optimizer (never receive gradients): {r['optimizer']['frozen_in_optimizer']}.")
    L += ["", "## Per-method notes", ""] + notes + ["",
          "## Findings carried into the fairness audit", "",
          "- AttFPN methods (03, 04, 08, 09, 09M): the channel-attention BatchNorm after global pooling normalises "
          "over the 4 images of each physical batch and updates its running statistics, which inference uses. The "
          "historical runs normalised over 2 images. The batch-one fallback in BatchOneSafeBatchNorm2d never triggers.",
          "- ResNet BatchNorm is in eval mode throughout (norm_eval) for every detector, so it is unaffected by batch size.",
          "- The SAM methods' prompt-head BatchNorm normalises over RoIs within the image (≥49 values per channel).",
          "- AttFPN level-4 (P6) attention receives no gradient: at ≤512 px crops every stride-64 anchor crosses the "
          "border and is excluded by allowed_border=0. SAM's IoU-prediction head is unused (multimask_output=False). "
          "Both are properties of the historical architectures, identical in every seed.",
          "- Box decoding clamps to the padded 512×512 canvas (MMCV Pad sets img_shape), so rescaled boxes can extend "
          "past the original lower/right edge; masks are pasted at original resolution. Shared by every method."]
    (OUT / "preflight" / "PREFLIGHT_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
