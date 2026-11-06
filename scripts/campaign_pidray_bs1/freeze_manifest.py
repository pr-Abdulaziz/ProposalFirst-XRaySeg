#!/usr/bin/env python
"""Freeze the PIDray-30 batch-1 campaign: one write-once manifest, then no edits.

Records the protocol, the leakage decision and its verification, every resolved
configuration and its hash, the annotation and pretrained-weight digests, and a
full source snapshot. run_guard.py refuses to launch a run whose entry, config,
annotations, weights or guarded source no longer match this file.

    python scripts/campaign_pidray_bs1/freeze_manifest.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

PRETRAINED = dict(
    resnet101_imagenet=ROOT / "pretrained/torch/hub/checkpoints/resnet101-63fe2227.pth",
    sam_vit_large_converted=ROOT / "pretrained/sam_vit_large/pytorch_model_new.bin",
    sam_vit_large_hf=ROOT / "pretrained/sam_vit_large/pytorch_model.bin",
    sam_vit_large_config=ROOT / "pretrained/sam_vit_large/config.json",
)


def main() -> int:
    C.require_leakage_policy()
    if C.MANIFEST.exists():
        raise SystemExit(f"{C.MANIFEST} already exists; a frozen manifest is never rewritten")
    gate = json.loads((C.AUDIT / "gate_verification.json").read_text())
    deleak = json.loads((C.AUDIT / "deleak_record.json").read_text())
    build = json.loads((C.OUT / "protocol" / "build_report.json").read_text())
    if gate["status"] != "PASS":
        raise SystemExit("ABORT: leakage gate is not PASS")
    if build["problems"]:
        raise SystemExit(f"ABORT: build report lists problems: {build['problems']}")

    annotations = {
        "train": dict(path=str(C.train_annotation(False).relative_to(ROOT)),
                      sha256=C.sha256(C.train_annotation(False)), role="target-only training"),
        "occ_train": dict(path=str(C.train_annotation(True).relative_to(ROOT)),
                          sha256=C.sha256(C.train_annotation(True)),
                          role="occluder-supervised training (methods 06, 07)"),
    }
    for split in C.TEST_SPLITS:
        annotations[f"test_{split}"] = dict(
            path=str(C.ANN[f"test_{split}"].relative_to(ROOT)),
            sha256=C.sha256(C.ANN[f"test_{split}"]), role=f"evaluation, {split} split, as released")
    released = dict(train=dict(path=str(C.ANN["train"].relative_to(ROOT)), sha256=C.sha256(C.ANN["train"])),
                    occ_train=dict(path=str(C.ANN["occ_train"].relative_to(ROOT)),
                                   sha256=C.sha256(C.ANN["occ_train"])))

    snap = C.source_snapshot()
    snap_path = C.OUT / "protocol" / "source_snapshot.json"
    snap_path.write_text(json.dumps(snap, indent=1))

    runs = {}
    for mid, seed in C.run_order():
        m = C.BY_ID[mid]
        cfg = C.resolved_config(mid, seed)
        runs[C.run_key(mid, seed)] = dict(
            method=mid, seed=seed, slug=m.slug, name=m.name, supervision=m.supervision,
            family=m.family, role=m.role, launchable=m.launchable,
            source_config=m.historical_config,
            resolved_config=str(cfg.relative_to(ROOT)), resolved_config_sha256=C.sha256(cfg),
            work_dir=str(C.work_dir(mid, seed)), run_dir=str(C.run_dir(mid, seed)),
            eval_dir=str(C.eval_dir(mid, seed)))

    manifest = dict(
        campaign=C.CAMPAIGN,
        frozen=dt.datetime.now().isoformat(timespec="seconds"),
        instructions=dict(
            dated="2026-09-21 to 2026-09-22",
            dataset="PIDray-30, a separate campaign from the PIXray one, comparing the same architectures",
            physical_batch=C.PHYSICAL_BATCH, gradient_accumulation="none", epochs=C.MAX_EPOCHS,
            precision="fp32 for every method", warmup_iterations=C.WARMUP_ITERS,
            methods=[m for m, _ in C.run_order()], seeds=list(C.SEEDS),
            history=("batch 1 / 50 epochs diverged (job 12317); batch 2 fp16 diverged on 08; "
                     "fp32 + warmup 500 adopted; 15-epoch run superseded by 10 epochs")),
        protocol=build["protocol"],
        dataset=dict(
            name="PIDray-30",
            train_images=C.train_images(), released_train_images=C.RELEASED_TRAIN_IMAGES,
            test_images=C.TEST_IMAGES, classes=list(C.CLASSES),
            empty_images="kept (filter_empty_gt=False); PIDray normal images are part of the benchmark"),
        leakage=dict(
            audit="scripts/audit_pidray.py --check-content-hashes",
            result_as_released="FAIL: 84 training images byte-identical to test images "
                               "(50 easy, 19 hard, 15 hidden; 11 annotated)",
            decision=C.LEAKAGE_POLICY, decided="2026-09-21 by the user",
            derived_training_files=deleak,
            verification=gate,
            released_files_unchanged=released,
            test_splits="released easy/hard/hidden files, untouched"),
        methods=[C.method_record(m) for m in C.METHODS],
        scope=dict(
            decided="2026-09-21 by the user",
            included=[m for m, _ in C.run_order()],
            excluded=C.EXCLUDED_METHODS,
            order_rationale="run order given by the user",
            note="the three excluded architectures are reported from the PIXray campaign only"),
        runs=runs,
        annotations=annotations,
        pretrained={k: dict(path=str(v.relative_to(ROOT)), sha256=C.sha256(v))
                    for k, v in PRETRAINED.items()},
        source_snapshot=dict(aggregate_sha256=snap["aggregate_sha256"], file_count=snap["file_count"],
                             per_file=str(snap_path.relative_to(ROOT)),
                             guarded_runtime_files=list(C.RUNTIME_FILES)),
        evaluation_policy=("each run is scored on all three released test splits "
                           "(easy, hard, hidden) with the campaign scorer; no evaluation "
                           "starts until every approved training run has completed"),
        schedule_basis="output/%s/profile/*.json (batch-1 measurement, 2026-09-21)" % C.CAMPAIGN)
    C.write_once(C.MANIFEST, manifest)
    print(json.dumps(dict(manifest=str(C.MANIFEST.relative_to(ROOT)), runs=len(runs),
                          train_images=C.train_images(),
                          source_files=snap["file_count"],
                          aggregate=snap["aggregate_sha256"][:16]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
