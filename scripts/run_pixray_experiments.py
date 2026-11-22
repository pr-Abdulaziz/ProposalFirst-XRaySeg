#!/usr/bin/env python3
"""Train and evaluate the controlled PIXray experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "projects/proposal_first_pidray_occ/configs"
PIXRAY_ROOT = ROOT / "data/PIXray"
PIXRAYA_ROOT = ROOT / "data/PIXray-A/annotations"
# The eleven controlled arms, followed by the unified baselines added so the
# published comparison table can be reproduced under this repository's protocol
# rather than quoted. See docs/unified-baseline-plan.md.
# The seven active methods are defined once, in
# projects/proposal_first_pidray_occ/registry_seven.py. Canonical ids are the
# strings "01".."07"; historical aliases resolve to them without creating a
# second entry or scheduling a second run.
try:
    from projects.proposal_first_pidray_occ.registry_seven import (
        METHODS as SEVEN_METHODS, resolve_all as resolve_seven,
    )
    ACTIVE_IDS = tuple(SEVEN_METHODS)
except Exception:                                   # registry optional at import
    SEVEN_METHODS, resolve_seven, ACTIVE_IDS = {}, None, ()

PRIMARY_IDS = (
    "P1", "P2", "P3", "P4", "S1", "S2", "S3-P", "S3-MBoost", "E1",
    "E2", "E3",
)
BASELINE_IDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "08")
EXPECTED_IDS = PRIMARY_IDS + BASELINE_IDS

os.environ.setdefault("XRAY_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("PIXRAY_DATA_ROOT", str(PIXRAY_ROOT))
os.environ.setdefault("PIXRAYA_ANN_ROOT", str(PIXRAYA_ROOT))
os.environ.setdefault("SAM_VIT_LARGE_DIR", str(ROOT / "pretrained/sam_vit_large"))
os.environ.setdefault(
    "SAM_VIT_LARGE_CKPT",
    str(Path(os.environ["SAM_VIT_LARGE_DIR"]) / "pytorch_model_new.bin"),
)

PHYSICAL_BATCH_SIZE = int(os.getenv("XRAY_BATCH_SIZE_PER_GPU", "2"))
EFFECTIVE_BATCH_SIZE = int(os.getenv("XRAY_EFFECTIVE_BATCH_SIZE", "16"))
if PHYSICAL_BATCH_SIZE <= 0 or EFFECTIVE_BATCH_SIZE % PHYSICAL_BATCH_SIZE:
    raise ValueError("Physical batch size must evenly divide effective batch size")
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE

PROTOCOL = {
    "dataset": "PIXray",
    "train_images": 3560,
    "test_images": 1486,
    "seed": 42,
    "epochs": 60,
    "physical_batch_per_gpu": PHYSICAL_BATCH_SIZE,
    "gradient_accumulation": ACCUMULATION_STEPS,
    "effective_batch_single_gpu": EFFECTIVE_BATCH_SIZE,
    "optimizer": "AdamW",
    "base_lr": 1e-4,
    "weight_decay": 0.05,
    "precision": "AMP_float16",
    "gradient_clip_norm": 1.0,
    "input_size": [512, 512],
    "validation": "disabled",
    "metric_annotations": "PIXray/annotations/test.json",
}


@dataclass(frozen=True)
class Experiment:
    config: Path
    label: str
    work_dir_name: str
    supervision: str
    initialization: str


EXPERIMENTS = {
    "P1": Experiment(CONFIG_ROOT / "pixray_p1_cascade_fpn.py", "Cascade-FPN", "pixray_p1", "target_masks_only", "ImageNet_ResNet101"),
    "P2": Experiment(CONFIG_ROOT / "pixray_p2_cascade_attfpn.py", "Cascade-AttFPN", "pixray_p2", "target_masks_only", "ImageNet_ResNet101"),
    "P3": Experiment(CONFIG_ROOT / "pixray_p3_cascade_attfpn_recall_rpn.py", "Cascade-AttFPN-RecallRPN", "pixray_p3", "target_masks_only", "ImageNet_ResNet101"),
    "P4": Experiment(CONFIG_ROOT / "pixray_p4_enhanced_proposal_learning.py", "Cascade-AttFPN-CascadeRPN", "pixray_p4", "target_masks_only", "ImageNet_ResNet101"),
    "S1": Experiment(CONFIG_ROOT / "pixray_s1_single_decoder.py", "ProposalFirst-SingleDecoder", "pixray_s1", "target_masks_only", "ImageNet_ResNet101_plus_SAM_ViT_L"),
    "S2": Experiment(ROOT / "configs/occlusion/pixray_s2_occ_bisam.py", "Occ-BiSAM", "pixray_s2", "target_plus_overlap_masks", "SAM_ViT_L"),
    "S3-P": Experiment(CONFIG_ROOT / "pixray_s3_proposal_first_bisam.py", "RecallRPN-GuidedBiSAM", "pixray_s3_p", "target_plus_overlap_masks", "ImageNet_ResNet101_plus_SAM_ViT_L"),
    "S3-MBoost": Experiment(CONFIG_ROOT / "pixray_s3_mboost.py", "GuidedBiSAM-MaskBooster", "pixray_s3_mboost", "target_plus_overlap_masks", "ImageNet_ResNet101_plus_SAM_ViT_L"),
    "E1": Experiment(CONFIG_ROOT / "pixray_e1_adapter.py", "RecallRPN-SingleDecoder-Adapter", "pixray_e1", "target_masks_only", "ImageNet_ResNet101_plus_SAM_ViT_L"),
    "E2": Experiment(CONFIG_ROOT / "pixray_e2_hard_aware.py", "E1-HardAwareLoss", "pixray_e2", "target_plus_overlap_masks", "ImageNet_ResNet101_plus_SAM_ViT_L"),
    "E3": Experiment(CONFIG_ROOT / "pixray_e3_conditional_bisam.py", "Conditional-GuidedBiSAM-MaskBooster", "pixray_e3", "target_plus_overlap_masks", "ImageNet_ResNet101_plus_SAM_ViT_L"),
    # Unified baselines. B1-B4 are the published comparison rows retrained under
    # this protocol; B5 is RSPrompter; B6 is the occlusion-aware architecture
    # with its occluder supervision switched off.
    "B1": Experiment(CONFIG_ROOT / "pixray_b1_mask_rcnn_r50.py", "MaskRCNN-R50", "pixray_b1", "target_masks_only", "ImageNet_ResNet50"),
    "B2": Experiment(CONFIG_ROOT / "pixray_b2_mask_rcnn_r101.py", "MaskRCNN-R101", "pixray_b2", "target_masks_only", "ImageNet_ResNet101"),
    "B3": Experiment(CONFIG_ROOT / "pixray_b3_cascade_rcnn_r101.py", "CascadeRCNN-R101-BoxOnly", "pixray_b3", "boxes_only", "ImageNet_ResNet101"),
    "B4": Experiment(CONFIG_ROOT / "pixray_b4_cascade_mask_rcnn_r50.py", "CascadeMaskRCNN-R50", "pixray_b4", "target_masks_only", "ImageNet_ResNet50"),
    "B5": Experiment(ROOT / "configs/occlusion/pixray_b5_rsprompter.py", "RSPrompter", "pixray_b5", "target_masks_only", "SAM_ViT_L"),
    # B6-B8 apply occluder supervision to an ordinary FCN mask head, so the
    # occlusion-aware idea is tested without the SAM decoder it is normally
    # bundled with. Each mirrors one detector arm exactly.
    "B6": Experiment(CONFIG_ROOT / "pixray_b6_bilayer_fcn_fpn.py", "Cascade-FPN-BilayerFCN", "pixray_b6", "target_plus_overlap_masks", "ImageNet_ResNet101"),
    "B7": Experiment(CONFIG_ROOT / "pixray_b7_bilayer_fcn_attfpn.py", "Cascade-AttFPN-BilayerFCN", "pixray_b7", "target_plus_overlap_masks", "ImageNet_ResNet101"),
    "B8": Experiment(CONFIG_ROOT / "pixray_b8_bilayer_fcn_recall_rpn.py", "RecallRPN-BilayerFCN", "pixray_b8", "target_plus_overlap_masks", "ImageNet_ResNet101"),
    # Method 08 has never been run before, so it carries no historical alias and
    # is keyed by its canonical id. Adapted to PIXray; not a PIDray benchmark run.
    "08": Experiment(CONFIG_ROOT / "pixray_08_pidray_r0r1.py", "PIDray-R0R1-adapted", "pixray_08_pidray_r0r1", "target_masks_only", "ImageNet_ResNet101"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "all"))
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help='methods to run: canonical ids "01".."07", historical aliases '
             '(B2, P1, S2, ...), or "all" for the seven active methods. '
             "Archived arms are retained on disk but are not selectable here.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-error-analysis", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="evaluate the newest checkpoint even if training never finished",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from its last checkpoint instead of "
             "refusing to touch a work directory that already has one",
    )
    parser.add_argument("--keep-predictions", action="store_true")
    parser.add_argument("--error-samples", type=int, default=20)
    parser.add_argument("--report", type=Path, default=Path("output/reports/pixray/pixray_primary60.xlsx"))
    return parser.parse_args()


def directory_for(name: str) -> Path:
    return ROOT / "work_dirs" / EXPERIMENTS[name].work_dir_name


def validate_registry() -> None:
    """Check the registry is internally consistent.

    The eleven controlled arms must all still be present -- dropping one would
    silently change what the primary comparison means. Beyond that the registry
    is open, so baselines can be added, but ids, config paths and work
    directories stay unique: two entries sharing a work directory would have one
    run overwrite the other's checkpoints.
    """
    registered = tuple(EXPERIMENTS)
    missing_primary = [name for name in PRIMARY_IDS if name not in registered]
    if missing_primary:
        raise RuntimeError(
            f"PIXray registry is missing controlled experiments: {missing_primary}"
        )
    if len(registered) != len(set(registered)):
        raise RuntimeError("PIXray experiment ids must be unique")
    unknown = [name for name in registered if name not in EXPECTED_IDS]
    if unknown:
        raise RuntimeError(
            f"PIXray registry has ids absent from EXPECTED_IDS: {unknown}. Add "
            "them there so --experiments accepts them."
        )
    configs = [item.config.resolve() for item in EXPERIMENTS.values()]
    work_dirs = [item.work_dir_name for item in EXPERIMENTS.values()]
    if len(configs) != len(set(configs)) or len(work_dirs) != len(set(work_dirs)):
        raise RuntimeError("PIXray config and work-directory paths must be unique")
    missing = [str(path) for path in configs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PIXray configs: {missing}")


def execute(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def assert_training_complete(name: str) -> dict:
    """Refuse to evaluate a run that never finished.

    ``training_summary.json`` is written only after ``tools/train.py`` returns,
    so its absence means the run is still going, crashed, or was cancelled.
    Without this check the newest ``epoch_*.pth`` is evaluated silently and the
    partial result is reported as though it were a completed 60-epoch run.
    """
    directory = directory_for(name)
    summary_path = directory / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{name} has no training_summary.json in {directory}; the run has "
            "not finished. Wait for it, or pass --allow-incomplete to evaluate "
            "a partial checkpoint deliberately."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def checkpoint_for(name: str) -> Path:
    directory = directory_for(name)
    marker = directory / "last_checkpoint"
    if marker.is_file():
        recorded = Path(marker.read_text(encoding="utf-8").strip())
        for candidate in (recorded, ROOT / recorded, directory / recorded.name):
            if candidate.is_file():
                return candidate.resolve()
    checkpoints = sorted(directory.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found for {name} below {directory}")
    return checkpoints[-1].resolve()


def preflight(experiment: Experiment, dry_run: bool) -> None:
    execute([sys.executable, "scripts/preflight_pixray.py", "--config", str(experiment.config), "--build-model"], dry_run)


def train(name: str, experiment: Experiment, dry_run: bool, resume: bool = False) -> None:
    directory = directory_for(name)
    existing = list(directory.glob("*.pth")) if directory.is_dir() else []
    has_checkpoints = (directory / "last_checkpoint").exists() or existing
    if has_checkpoints and not resume:
        raise FileExistsError(
            f"{name} already contains checkpoints. Archive the directory, or "
            "pass --resume to continue a run that was interrupted."
        )
    if resume and not has_checkpoints:
        raise FileNotFoundError(
            f"{name} has no checkpoint in {directory} to resume from."
        )
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc)
    started = time.perf_counter()
    command = [sys.executable, "tools/train.py", str(experiment.config), "--work-dir", str(directory)]
    if resume:
        # "auto" continues from last_checkpoint. The protocol is unchanged: this
        # finishes the same run, it does not start a differently-configured one.
        command.extend(["--resume", "auto"])
    execute(command, dry_run)
    if dry_run:
        return
    elapsed = time.perf_counter() - started
    summary = {
        "experiment": name,
        "label": experiment.label,
        # A resumed run only times the segment after the interruption, so
        # training_seconds and the GPU-hours derived from it understate the
        # true cost. Flagged rather than silently reported.
        "resumed": bool(resume),
        "config": str(experiment.config),
        "started_utc": started_utc.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": elapsed,
        "gpu_count": int(os.getenv("WORLD_SIZE", "1")),
        "gpu_hours": elapsed * int(os.getenv("WORLD_SIZE", "1")) / 3600.0,
        "initialization": experiment.initialization,
        "supervision": experiment.supervision,
        "protocol": PROTOCOL,
    }
    (directory / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def evaluate(name: str, experiment: Experiment, report: Path, args: argparse.Namespace) -> None:
    # Stage-level timings are a test-time measurement only. Training never sets
    # this, so completed checkpoints stay valid.
    os.environ["XRAY_PROFILE_INFERENCE"] = "1"
    if not args.dry_run and not args.allow_incomplete:
        summary = assert_training_complete(name)
        expected = summary.get("protocol", {}).get("epochs")
        checkpoint = checkpoint_for(name)
        reached = re.search(r"epoch_(\d+)\.pth$", checkpoint.name)
        if expected and reached and int(reached.group(1)) != int(expected):
            raise FileNotFoundError(
                f"{name} resolved {checkpoint.name} but the protocol declares "
                f"{expected} epochs. Refusing to report a partial run; pass "
                "--allow-incomplete to override."
            )
    else:
        checkpoint = Path(f"<checkpoint:{name}>") if args.dry_run else checkpoint_for(name)
    output = directory_for(name) / "test" / "test"
    predictions = output / "predictions.pkl"
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoint_source.txt").write_text(
            f"experiment={name}\ncheckpoint={checkpoint}\nsplit=test\n", encoding="utf-8"
        )
    notes = ";".join((
        f"experiment={name}", f"label={experiment.label}", "seed=42", "epochs=60",
        f"supervision={experiment.supervision}", f"physical_batch_per_gpu={PHYSICAL_BATCH_SIZE}",
        f"gradient_accumulation={ACCUMULATION_STEPS}", f"effective_batch_single_gpu={EFFECTIVE_BATCH_SIZE}",
        "optimizer=AdamW", "base_lr=1e-4", "precision=AMP_float16",
        "metric_annotations=PIXray/annotations/test.json",
        "profile_inference=1",
    ))
    command = [
        sys.executable, "tools/test.py", str(experiment.config), str(checkpoint),
        "--work-dir", str(output), "--report", str(report),
        "--report-dataset", "PIXray", "--report-notes", notes,
    ]
    if not args.skip_error_analysis:
        command.extend(["--out", str(predictions)])
    execute(command, args.dry_run)
    if args.skip_error_analysis:
        return
    if experiment.supervision == "boxes_only":
        # The mask error analysis matches predicted to target instances by mask
        # IoU. An arm with no mask branch has nothing to match, and the analyser
        # would write a summary of zeros that reads as a collapsed model rather
        # than as an arm the analysis does not apply to.
        print(f"{name} has no mask branch; skipping mask error analysis")
        if not args.dry_run and not args.keep_predictions:
            predictions.unlink(missing_ok=True)
        return
    execute([
        sys.executable, "scripts/analyze_segmentation_errors.py",
        "--predictions", str(predictions),
        "--annotations", str(PIXRAY_ROOT / "annotations/test.json"),
        "--occlusion-annotations", str(PIXRAYA_ROOT / "test_occ.json"),
        "--image-root", str(PIXRAY_ROOT / "test"),
        "--output-dir", str(ROOT / "output/error_analysis/pixray" / name.lower()),
        "--samples", str(args.error_samples),
    ], args.dry_run)
    if not args.dry_run and not args.keep_predictions:
        predictions.unlink(missing_ok=True)


def _select(requested) -> list[str]:
    """Resolve the requested methods to historical registry keys.

    Accepts canonical numeric ids, historical aliases, or "all". Duplicates
    that arise from mixing the two naming schemes are rejected rather than
    silently running the same checkpoint twice.
    """
    if resolve_seven is None:
        return list(requested or EXPECTED_IDS)
    if not requested or [str(x).lower() for x in requested] == ["all"]:
        canonical = list(ACTIVE_IDS)
    else:
        canonical = resolve_seven(requested)
    # 08 has no historical alias; it is keyed in EXPERIMENTS by its canonical id.
    return [SEVEN_METHODS[c].historical_id or c for c in canonical]


def main() -> int:
    args = parse_args()
    validate_registry()
    args.experiments = _select(args.experiments)
    if not args.report.is_absolute():
        args.report = ROOT / args.report
    if not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    preflighted: set[Path] = set()
    stages = ("train", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        for name in dict.fromkeys(args.experiments):
            experiment = EXPERIMENTS[name]
            try:
                if not args.skip_preflight and experiment.config not in preflighted:
                    preflight(experiment, args.dry_run)
                    preflighted.add(experiment.config)
                if stage == "train":
                    train(name, experiment, args.dry_run, resume=args.resume)
                else:
                    evaluate(name, experiment, args.report, args)
            except (OSError, subprocess.CalledProcessError) as exc:
                failures.append(f"{name}/{stage}: {exc}")
                print(f"ERROR: {failures[-1]}", file=sys.stderr)
                if not args.continue_on_error:
                    return 1
    if failures:
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
