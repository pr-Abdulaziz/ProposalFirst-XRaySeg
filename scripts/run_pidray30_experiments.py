"""Train and evaluate the PIDray-30 experiment matrix with isolated outputs.

This entrypoint is PIDray-30-only. Use run_pixray_experiments.py for PIXray.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "projects/proposal_first_pidray_occ/configs"
PIDRAY_ROOT = ROOT / "data/PIDray"
PIDRAYA_ANN_ROOT = ROOT / "data/PIDray-A/annotations"
TEST_IMAGE_ROOT = PIDRAY_ROOT / "test"
TEST_SPLITS = ("easy", "hard", "hidden")
EXPECTED_TEST_IMAGES = {"easy": 24758, "hard": 9746, "hidden": 13069}
EXPECTED_EXPERIMENT_IDS = (
    "P1", "P2", "P3", "P4", "S1", "S2", "S3-P", "S3-MBoost", "E1",
    "E2", "E3",
)

os.environ.setdefault("XRAY_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("PIDRAY_DATA_ROOT", str(PIDRAY_ROOT))
os.environ.setdefault("OCC_PIDRAY_DATA_ROOT", str(PIDRAY_ROOT))
os.environ.setdefault("PIDRAYA_ANN_ROOT", str(PIDRAYA_ANN_ROOT))
os.environ.setdefault("SAM_VIT_LARGE_DIR", str(ROOT / "pretrained/sam_vit_large"))
os.environ.setdefault(
    "SAM_VIT_LARGE_CKPT",
    str(Path(os.environ["SAM_VIT_LARGE_DIR"]) / "pytorch_model_new.bin"),
)

PHYSICAL_BATCH_SIZE = int(os.getenv("XRAY_BATCH_SIZE_PER_GPU", "2"))
EFFECTIVE_BATCH_SIZE = int(os.getenv("XRAY_EFFECTIVE_BATCH_SIZE", "16"))
if PHYSICAL_BATCH_SIZE <= 0 or EFFECTIVE_BATCH_SIZE % PHYSICAL_BATCH_SIZE:
    raise ValueError(
        "XRAY_BATCH_SIZE_PER_GPU must be positive and evenly divide "
        "XRAY_EFFECTIVE_BATCH_SIZE"
    )
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE

SHARED_PROTOCOL = {
    "dataset": "PIDray-30",
    "train_images": 23074,
    "test_images": EXPECTED_TEST_IMAGES,
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
}


@dataclass(frozen=True)
class Experiment:
    config: Path
    label: str
    work_dir_name: str
    dataset: str
    initialization: str
    trainable_scope: str = "full_model"
    epochs: int = 60


EXPERIMENTS = {
    "P1": Experiment(
        CONFIG_ROOT / "pidray30_p1_cascade_fpn.py",
        "Cascade-FPN", "pidray30_p1", "PIDray-30", "ImageNet_ResNet101",
    ),
    "P2": Experiment(
        CONFIG_ROOT / "pidray30_p2_cascade_attfpn.py",
        "Cascade-AttFPN", "pidray30_p2", "PIDray-30", "ImageNet_ResNet101",
    ),
    "P3": Experiment(
        CONFIG_ROOT / "pidray30_p3_recall_rpn.py",
        "Cascade-AttFPN-RecallRPN", "pidray30_p3", "PIDray-30",
        "ImageNet_ResNet101",
    ),
    "P4": Experiment(
        CONFIG_ROOT / "pidray30_p4_cascade_rpn.py",
        "Cascade-AttFPN-CascadeRPN", "pidray30_p4", "PIDray-30",
        "ImageNet_ResNet101",
    ),
    "S1": Experiment(
        CONFIG_ROOT / "pidray30_s1_single_decoder.py",
        "ProposalFirst-SingleDecoder", "pidray30_s1", "PIDray-30+PIDray-A",
        "ImageNet_ResNet101_plus_SAM_ViT_L",
    ),
    "S2": Experiment(
        ROOT / "configs/occlusion/pidray30_s2_occ_bisam.py",
        "Occ-BiSAM", "pidray30_s2", "PIDray-30+PIDray-A", "SAM_ViT_L",
    ),
    "S3-P": Experiment(
        CONFIG_ROOT / "pidray30_s3_proposal_first_bisam.py",
        "RecallRPN-GuidedBiSAM", "pidray30_s3_p", "PIDray-30+PIDray-A",
        "ImageNet_ResNet101_plus_SAM_ViT_L",
    ),
    "S3-MBoost": Experiment(
        CONFIG_ROOT / "pidray30_s3_mboost.py",
        "RecallRPN-GuidedBiSAM-MaskBooster", "pidray30_s3_mboost",
        "PIDray-30+PIDray-A", "ImageNet_ResNet101_plus_SAM_ViT_L",
    ),
    "E1": Experiment(
        CONFIG_ROOT / "pidray30_e1_adapter.py",
        "RecallRPN-SingleDecoder-Adapter", "pidray30_e1",
        "PIDray-30+PIDray-A", "ImageNet_ResNet101_plus_SAM_ViT_L",
    ),
    "E2": Experiment(
        CONFIG_ROOT / "pidray30_e2_hard_aware.py",
        "E1-HardAwareLoss", "pidray30_e2", "PIDray-30+PIDray-A",
        "ImageNet_ResNet101_plus_SAM_ViT_L",
    ),
    "E3": Experiment(
        CONFIG_ROOT / "pidray30_e3_conditional_bisam.py",
        "Conditional-GuidedBiSAM-MaskBooster", "pidray30_e3",
        "PIDray-30+PIDray-A", "ImageNet_ResNet101_plus_SAM_ViT_L",
    ),
}

SUPERVISION = {
    "P1": "target_masks_only",
    "P2": "target_masks_only",
    "P3": "target_masks_only",
    "P4": "target_masks_only",
    "S1": "target_masks_only",
    "S2": "target_plus_overlap_masks",
    "S3-P": "target_plus_overlap_masks",
    "S3-MBoost": "target_plus_overlap_masks",
    "E1": "target_masks_only",
    "E2": "target_plus_overlap_masks",
    "E3": "target_plus_overlap_masks",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "all"))
    parser.add_argument(
        "--experiments", nargs="+", default=list(EXPERIMENTS),
        choices=tuple(EXPERIMENTS),
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(TEST_SPLITS), choices=TEST_SPLITS,
        help="PIDray test splits used during evaluation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--report", type=Path,
        default=Path("output/reports/pidray30/pidray30_experiment_results.xlsx"),
    )
    parser.add_argument("--skip-error-analysis", action="store_true")
    parser.add_argument("--keep-predictions", action="store_true")
    parser.add_argument("--error-samples", type=int, default=20)
    return parser.parse_args()


def work_dir(name: str) -> Path:
    return ROOT / "work_dirs" / EXPERIMENTS[name].work_dir_name


def validate_registry() -> None:
    if tuple(EXPERIMENTS) != EXPECTED_EXPERIMENT_IDS:
        raise RuntimeError(
            "The active registry must contain the controlled eleven-experiment "
            "matrix in protocol order"
        )
    work_dirs = [item.work_dir_name for item in EXPERIMENTS.values()]
    if len(work_dirs) != len(set(work_dirs)):
        raise RuntimeError("Experiment work directories must be unique")
    config_paths = [item.config.resolve() for item in EXPERIMENTS.values()]
    if len(config_paths) != len(set(config_paths)):
        raise RuntimeError("Experiment config paths must be unique")
    missing = [str(path) for path in config_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment configs are missing: {missing}")
    if any(item.epochs != SHARED_PROTOCOL["epochs"] for item in EXPERIMENTS.values()):
        raise RuntimeError("Every primary experiment must train for 60 epochs")
    if set(SUPERVISION) != set(EXPERIMENTS):
        raise RuntimeError("Every experiment must declare its supervision")


def checkpoint_for(name: str) -> Path:
    directory = work_dir(name)
    marker = directory / "last_checkpoint"
    if marker.is_file():
        value = Path(marker.read_text(encoding="utf-8").strip())
        for candidate in (
            value if value.is_absolute() else ROOT / value,
            directory / value.name,
        ):
            if candidate.is_file():
                return candidate.resolve()
    checkpoints = sorted(
        directory.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime
    )
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No checkpoint found for {name} below {directory}")


def execute(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def gpu_inventory() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def preflight(experiment: Experiment, dry_run: bool) -> None:
    execute(
        [
            sys.executable, "scripts/preflight.py", "--config",
            str(experiment.config), "--build-model",
        ],
        dry_run,
    )


def train(name: str, experiment: Experiment, dry_run: bool) -> None:
    directory = work_dir(name)
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
    if (directory / "last_checkpoint").exists() and not dry_run:
        raise FileExistsError(
            f"{name} already has a checkpoint. Archive {directory}; fresh runs never resume."
        )
    command = [
        sys.executable, "tools/train.py", str(experiment.config),
        "--work-dir", str(directory),
    ]
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    execute(command, dry_run)
    if dry_run:
        return
    elapsed_seconds = time.perf_counter() - started
    gpu_count = int(os.getenv("WORLD_SIZE", "1"))
    summary = {
        "experiment": name,
        "label": experiment.label,
        "config": str(experiment.config),
        "started_utc": started_at.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": elapsed_seconds,
        "gpu_count": gpu_count,
        "gpu_hours": elapsed_seconds * gpu_count / 3600.0,
        "resumed": False,
        "epochs": experiment.epochs,
        "initialization": experiment.initialization,
        "trainable_scope": experiment.trainable_scope,
        "supervision": SUPERVISION[name],
        "protocol": SHARED_PROTOCOL,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
        "system_gpu_inventory": gpu_inventory(),
    }
    (directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def evaluate_split(
    name: str,
    experiment: Experiment,
    split: str,
    report: Path,
    dry_run: bool,
    error_analysis: bool,
    keep_predictions: bool,
    error_samples: int,
) -> None:
    checkpoint = Path(f"<checkpoint:{name}>") if dry_run else checkpoint_for(name)
    directory = work_dir(name) / "test" / split
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
    predictions = directory / "predictions.pkl"
    annotation = PIDRAY_ROOT / "annotations" / f"test_{split}.json"
    report_notes = ";".join(
        (
            f"experiment={name}", f"label={experiment.label}",
            f"split={split}", f"expected_images={EXPECTED_TEST_IMAGES[split]}",
            f"checkpoint_source={name}", "seed=42", "epochs=60",
            f"initialization={experiment.initialization}",
            f"trainable_scope={experiment.trainable_scope}",
            f"supervision={SUPERVISION[name]}",
            f"physical_batch_per_gpu={PHYSICAL_BATCH_SIZE}",
            f"gradient_accumulation={ACCUMULATION_STEPS}",
            f"effective_batch_single_gpu={EFFECTIVE_BATCH_SIZE}",
            "optimizer=AdamW", "base_lr=1e-4",
            "schedule=LinearWarmup50_then_Cosine60", "precision=AMP_float16",
            "metric_annotations=main_PIDray",
        )
    )
    if not dry_run:
        (directory / "checkpoint_source.txt").write_text(
            f"experiment={name}\ncheckpoint={checkpoint}\nsplit={split}\n",
            encoding="utf-8",
        )
    command = [
        sys.executable, "tools/test.py", str(experiment.config), str(checkpoint),
        "--work-dir", str(directory), "--test-split", split,
        "--report", str(report), "--report-dataset",
        f"PIDray-{split.capitalize()}", "--report-notes", report_notes,
    ]
    if error_analysis:
        command.extend(["--out", str(predictions)])
    execute(command, dry_run)
    if not error_analysis:
        return
    execute(
        [
            sys.executable, "scripts/analyze_segmentation_errors.py",
            "--predictions", str(predictions), "--annotations", str(annotation),
            "--image-root", str(TEST_IMAGE_ROOT), "--output-dir",
            str(ROOT / "output/error_analysis/pidray30" / name.lower() / split),
            "--samples", str(error_samples),
        ],
        dry_run,
    )
    if not dry_run and not keep_predictions:
        predictions.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    validate_registry()
    if not args.report.is_absolute():
        args.report = ROOT / args.report
    if not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    selected = list(dict.fromkeys(args.experiments))
    splits = list(dict.fromkeys(args.splits))
    stages = ("train", "evaluate") if args.stage == "all" else (args.stage,)
    failures: list[str] = []
    preflighted: set[Path] = set()
    for stage in stages:
        for name in selected:
            experiment = EXPERIMENTS[name]
            try:
                if not args.skip_preflight and experiment.config not in preflighted:
                    preflight(experiment, args.dry_run)
                    preflighted.add(experiment.config)
                if stage == "train":
                    train(name, experiment, args.dry_run)
                else:
                    for split in splits:
                        evaluate_split(
                            name, experiment, split, args.report, args.dry_run,
                            not args.skip_error_analysis, args.keep_predictions,
                            args.error_samples,
                        )
            except (OSError, subprocess.CalledProcessError) as exc:
                failures.append(f"{name}/{stage}: {exc}")
                print(f"ERROR: {failures[-1]}", file=sys.stderr)
                if not args.continue_on_error:
                    return 1
    if failures:
        print("Failed experiments:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
