#!/usr/bin/env python
"""Freeze the full-split campaign's architecture, hyperparameters and
checkpoint-selection rule, before any training starts.

A pre-registration is only worth something if it is written down before the
numbers exist and can be checked afterwards against what actually ran. This
records, with hashes: the configs and their fully resolved forms, every source
file that defines the two new architectures, the evaluator sources, the
annotation files, the reused Method 03 artefacts, and the decision rules.

Re-running it after training would overwrite the record, so it refuses to run if
a frozen work_dir already contains a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/full_split_campaign/2026-09-12"
C = "projects/proposal_first_pidray_occ/configs"
PR = "projects/proposal_first_pidray_occ/models/point_refine"

ARMS = {
    "03": dict(role="reference baseline, REUSED not retrained",
               config=f"{C}/pixray_p2_cascade_attfpn.py",
               work_dir="work_dirs/pixray_p2"),
    "09": dict(role="candidate: transformer point refinement",
               config=f"{C}/pixray_09_point_refine.py",
               work_dir="work_dirs/pixray_09_point_refine"),
    "09M": dict(role="control: capacity-matched MLP point refinement",
                config=f"{C}/pixray_09m_point_refine_mlp.py",
                work_dir="work_dirs/pixray_09m_point_refine_mlp"),
}

SOURCES = [f"{PR}/{n}.py" for n in
           ("__init__", "embeddings", "transformer", "sampling", "head",
            "cascade_adapter", "mlp_control")] + [
    "projects/proposal_first_pidray_occ/evaluation/xray_coco_metric.py",
    "projects/proposal_first_pidray_occ/evaluation/zero_id_compat.py",
    f"{C}/pixray_detector_base.py",
    f"{C}/pidray_r101_attfpn_r0_legacy.py",
]

ANNOTATIONS = ["data/PIXray/annotations/train.json",
               "data/PIXray/annotations/test.json"]


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    # Refuse to re-freeze a campaign that has already started.
    for arm, spec in ARMS.items():
        if arm == "03":
            continue
        started = list((ROOT / spec["work_dir"]).glob("*.pth")) if (
            ROOT / spec["work_dir"]).exists() else []
        if started:
            print(f"REFUSING: {spec['work_dir']} already contains {len(started)} "
                  "checkpoint(s). A freeze must precede training.")
            return 1

    # Forced, not defaulted: common.sh exports 4, and Method 03 trained with 2.
    # Worker count seeds the per-worker augmentation RNG, so matching it is part
    # of matching the protocol.
    os.environ["XRAY_NUM_WORKERS"] = "2"
    from mmengine.config import Config

    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for arm, spec in ARMS.items():
        cfg = Config.fromfile(str(ROOT / spec["config"]))
        resolved = OUT / f"resolved_{arm}.py"
        resolved.write_text(cfg.pretty_text)
        train, test = cfg.train_dataloader, cfg.test_dataloader
        frozen[arm] = dict(
            role=spec["role"],
            config=spec["config"],
            config_sha256=sha256(ROOT / spec["config"]),
            resolved_config=str(resolved.relative_to(ROOT)),
            resolved_config_sha256=sha256(resolved),
            work_dir=spec["work_dir"],
            architecture=dict(
                detector=cfg.model["type"],
                roi_head=str(cfg.model["roi_head"]["type"]),
                point_refine_head=(cfg.model["roi_head"].get("point_refine_head") or {}).get("type"),
                point_refine_config=cfg.model["roi_head"].get("point_refine_head"),
            ),
            data=dict(
                train_ann=train["dataset"]["ann_file"],
                test_ann=test["dataset"]["ann_file"],
                num_workers=train["num_workers"],
                batch_size=train["batch_size"],
            ),
            hyperparameters=dict(
                max_epochs=cfg.train_cfg["max_epochs"],
                seed=cfg.randomness["seed"],
                deterministic=cfg.randomness["deterministic"],
                optimizer=cfg.optim_wrapper["optimizer"],
                accumulative_counts=cfg.optim_wrapper["accumulative_counts"],
                clip_grad=cfg.optim_wrapper["clip_grad"],
                precision=f"{cfg.optim_wrapper['type']}/{cfg.optim_wrapper['dtype']}",
                param_scheduler=cfg.param_scheduler,
                load_from=cfg.load_from,
            ),
            checkpoint_hook=cfg.default_hooks["checkpoint"],
            validation=dict(val_cfg=cfg.val_cfg, val_dataloader=cfg.val_dataloader,
                            val_evaluator=cfg.val_evaluator),
            test_evaluator=cfg.test_evaluator,
        )

    manifest = json.loads(
        (ROOT / "output/audit/2026-09-11/result_manifest.json").read_text())
    baseline = [m for m in manifest["methods"] if str(m.get("id")) == "03"][0]

    payload = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        campaign="PIXray full official split; Method 09 and its matched MLP "
                 "control trained under Method 03's protocol",
        split=dict(
            train="data/PIXray/annotations/train.json (all 3560 images)",
            test="data/PIXray/annotations/test.json (all 1486 images)",
            note="The official released split, unmodified. The 85/15 development "
                 "split under splits/pixray_dev_v1 is INACTIVE and is used by "
                 "nothing in this campaign.",
        ),
        decision_rules=dict(
            checkpoint_selection="Final epoch (epoch_60.pth). No validation loop "
                                 "exists, no save_best rule exists, and no "
                                 "checkpoint is chosen among alternatives. This "
                                 "is the rule Method 03 was trained under and it "
                                 "is independent of test performance.",
            checkpoint_selection_verified_from=(
                "work_dirs/pixray_p2/pixray_p2_cascade_attfpn.py: "
                "val_cfg=None, val_dataloader=None, val_evaluator=None, "
                "CheckpointHook(interval=1, max_keep_ckpts=3, save_last=True) "
                "with no save_best; and work_dirs/pixray_p2 contains exactly one "
                "checkpoint, epoch_60.pth, with no best_*.pth."),
            architecture="Frozen as recorded above. No architectural change may "
                         "be made after training begins without discarding the "
                         "run and re-freezing.",
            hyperparameters="Frozen as recorded above; identical to Method 03 "
                            "except the roi_head.",
            evaluation="Single evaluation of each final checkpoint on the "
                       "official test split, with the same evaluator Method 03's "
                       "reported metrics were computed by.",
            comparison="03 vs 09 asks whether point refinement helps. "
                       "09M vs 09 asks whether attention between points helps "
                       "beyond the same capacity applied per point.",
        ),
        arms=frozen,
        reused_baseline=dict(
            id="03",
            retrained=False,
            checkpoint=baseline["checkpoint"],
            checkpoint_sha256=baseline["checkpoint_sha256"],
            checkpoint_sha256_reverified=sha256(ROOT / baseline["checkpoint"]),
            predictions=baseline["predictions"],
            predictions_sha256=baseline["predictions_sha256"],
            predictions_sha256_reverified=sha256(ROOT / baseline["predictions"]),
            training_config_sha256=baseline["training_config_sha256"],
            training_config_sha256_reverified=sha256(ROOT / baseline["training_config"]),
            reported_segm_AP=baseline["results"]["segm"]["official"]["AP"],
            reported_bbox_AP=baseline["results"]["bbox"]["official"]["AP"],
            metrics_computed_by=baseline["metrics_computed_by"],
            training_seconds=json.loads(
                (ROOT / "work_dirs/pixray_p2/training_summary.json").read_text()
            )["training_seconds"],
        ),
        source_sha256={p: sha256(ROOT / p) for p in SOURCES},
        annotation_sha256={p: sha256(ROOT / p) for p in ANNOTATIONS},
        evaluator_source_sha256=manifest["evaluator_source_sha256"],
        environment=dict(
            python=sys.version.split()[0],
            torch=__import__("torch").__version__,
            cuda=__import__("torch").version.cuda,
            xray_num_workers=os.environ["XRAY_NUM_WORKERS"],
            nvidia_smi=subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip() or "not queried on this host",
        ),
        known_protocol_differences_from_method_03=[
            dict(item="max_keep_ckpts", method_03=3, campaign=1,
                 effect="Disk retention only. save_last=True keeps the final "
                        "epoch in both cases, and the final epoch is the "
                        "selected checkpoint, so this cannot affect any result."),
            dict(item="test_evaluator",
                 method_03="CocoMetric at training time",
                 campaign="XRayCocoMetric + ProposalFirstRuntimeMetric",
                 effect="Method 03's REPORTED metrics were recomputed with the "
                        "current evaluator, not taken from its 2026-09-07 test "
                        "run. The campaign therefore matches the evaluator that "
                        "produced 03's reported numbers, which is what the "
                        "comparison requires."),
            dict(item="train_num_workers", method_03=2, campaign=2,
                 effect="Matched deliberately. The current default is 4; the "
                        "launcher forces 2 and a guard enforces it. Worker count "
                        "seeds the augmentation RNG per worker, so matching it "
                        "with the same seed and sampler gives the same "
                        "augmentation stream Method 03 saw."),
        ],
    )

    out = OUT / "freeze.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"frozen -> {out}")
    for arm, spec in frozen.items():
        print(f"  {arm:<4} {spec['architecture']['roi_head']:<28} "
              f"workers={spec['data']['num_workers']} "
              f"epochs={spec['hyperparameters']['max_epochs']} "
              f"seed={spec['hyperparameters']['seed']}")
    r = payload["reused_baseline"]
    print(f"  03 checkpoint reverified: "
          f"{r['checkpoint_sha256'] == r['checkpoint_sha256_reverified']}")
    print(f"  03 predictions reverified: "
          f"{r['predictions_sha256'] == r['predictions_sha256_reverified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
