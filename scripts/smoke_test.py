"""Run one real GPU train step and the available inference paths."""

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings


def positive_count(batch: dict) -> int:
    return sum(len(sample.gt_instances) for sample in batch["data_samples"])


def hard_overlap_count(batch: dict, threshold: float) -> int:
    count = 0
    for sample in batch["data_samples"]:
        instances = sample.gt_instances
        if "bo_masks" not in instances or len(instances) == 0:
            continue
        target_area = torch.as_tensor(
            instances.masks.areas.astype("float32", copy=False)
        ).clamp_min(1.0)
        overlap_area = torch.as_tensor(
            instances.bo_masks.areas.astype("float32", copy=False)
        )
        count += int(((overlap_area / target_area) >= threshold).sum())
    return count


def expects_overlap_loss(value: object) -> bool:
    """Return whether a resolved model config supervises an occluder mask."""
    if isinstance(value, dict):
        if "loss_mask_bo" in value:
            loss = value["loss_mask_bo"]
            return not isinstance(loss, dict) or loss.get("loss_weight", 1.0) > 0
        if float(value.get("overlap_loss_weight", 0.0)) > 0:
            return True
        return any(expects_overlap_loss(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(expects_overlap_loss(child) for child in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--require-retained", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The smoke test requires a CUDA GPU.")

    temporary_work_dir = tempfile.TemporaryDirectory(prefix="xray_seg_smoke_")
    cfg = Config.fromfile(str(args.config.resolve()), import_custom_modules=True)
    custom_imports = cfg.get("custom_imports")
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    cfg.work_dir = temporary_work_dir.name
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    detector = runner.model.module if hasattr(runner.model, "module") else runner.model
    roi_cfg = cfg.model.get("roi_head", {})
    if roi_cfg.get("use_mask_booster", False) and not hasattr(
        detector.roi_head, "mask_booster"
    ):
        raise RuntimeError("Mask-booster config did not build the booster module.")

    train_batch = None
    needs_hard_route = bool(roi_cfg.get("conditional_bisam", False))
    hard_threshold = float(roi_cfg.get("difficulty_threshold", 0.05))
    for index, candidate in enumerate(runner.train_dataloader):
        has_positive = positive_count(candidate) > 0
        has_hard = hard_overlap_count(candidate, hard_threshold) > 0
        if has_positive and (not needs_hard_route or has_hard):
            train_batch = candidate
            break
        if index + 1 >= args.max_train_batches:
            break
    if train_batch is None:
        raise RuntimeError(
            "No suitable positive training batch was found; the mask path was "
            "not tested. Conditional BiSAM additionally requires an overlap-hard "
            "instance."
        )

    if not hasattr(runner.optim_wrapper, "optim_context"):
        runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    runner.model.train()
    losses = runner.model.train_step(train_batch, runner.optim_wrapper)
    if not any("loss_mask" in name for name in losses):
        raise RuntimeError("Training step did not report the target-mask loss.")
    if roi_cfg.get("conditional_bisam", False) and not any(
        "loss_difficulty_gate" in name for name in losses
    ):
        raise RuntimeError("Conditional BiSAM did not report its gate loss.")
    overlap_expected = expects_overlap_loss(cfg.model)
    overlap_losses = [value for name, value in losses.items() if "loss_mask_bo" in name]
    if overlap_expected and not overlap_losses:
        raise RuntimeError("Training step did not report the occluder-mask loss.")
    if not overlap_expected and any(
        float(value.detach().abs().max()) > 1e-8 for value in overlap_losses
    ):
        raise RuntimeError("A non-zero overlap loss was reported when disabled.")
    print("Positive train step: PASS")
    print("Loss keys:", sorted(losses))

    checkpoint_path = Path(cfg.work_dir) / "smoke_test.pth"
    torch.save({"state_dict": runner.model.state_dict()}, checkpoint_path)
    load_checkpoint(runner.model, str(checkpoint_path), map_location="cpu")
    print("Checkpoint save/reload: PASS")

    test_batch = next(iter(runner.test_dataloader))
    if not hasattr(detector, "r0_head"):
        runner.model.eval()
        with torch.inference_mode():
            predictions = runner.model.test_step(test_batch)
        if len(predictions) != len(test_batch["data_samples"]):
            raise RuntimeError("Inference output count does not match the batch.")
        print("Standard PIDray inference: PASS")
        print("GPU smoke test: PASS")
        temporary_work_dir.cleanup()
        return

    original = (
        detector.use_r0_screening,
        detector.use_proposal_screening,
        detector.r0_head.threshold,
        detector.det_score_threshold,
        detector.combined_score_threshold,
        detector.max_sam_proposals,
        detector.r0_rescue_enabled,
        detector.rescue_det_threshold,
        detector.rescue_topk,
    )
    runner.model.eval()
    with torch.inference_mode():
        detector.use_r0_screening = True
        detector.use_proposal_screening = True
        detector.r0_head.threshold = 2.0
        detector.r0_rescue_enabled = False
        skipped = runner.model.test_step(test_batch)
        if any(sample.metainfo.get("sam_executed") for sample in skipped):
            raise RuntimeError(
                "Forced R0-negative inference unexpectedly executed SAM."
            )
        print("Zero-retained-proposal inference: PASS")

        detector.r0_head.threshold = -1.0
        detector.det_score_threshold = -1.0
        detector.combined_score_threshold = -1.0
        detector.max_sam_proposals = 1
        retained = runner.model.test_step(test_batch)
        sam_executed = any(sample.metainfo.get("sam_executed") for sample in retained)
        if args.require_retained and not sam_executed:
            raise RuntimeError(
                "No final detector proposal survived, so the SAM inference path "
                "was not exercised."
            )
        print(
            "Retained-proposal SAM inference:",
            "PASS" if sam_executed else "NOT EXERCISED",
        )

        detector.r0_head.threshold = 2.0
        detector.r0_rescue_enabled = True
        detector.rescue_det_threshold = -1.0
        detector.rescue_topk = 1
        rescued = runner.model.test_step(test_batch)
        rescue_executed = any(
            sample.metainfo.get("screening_route") == "rescue"
            and sample.metainfo.get("sam_executed")
            for sample in rescued
        )
        if args.require_retained and not rescue_executed:
            raise RuntimeError(
                "The forced R0-negative rescue route did not execute SAM."
            )
        print(
            "R0-rescue SAM inference:",
            "PASS" if rescue_executed else "NOT EXERCISED",
        )

    (
        detector.use_r0_screening,
        detector.use_proposal_screening,
        detector.r0_head.threshold,
        detector.det_score_threshold,
        detector.combined_score_threshold,
        detector.max_sam_proposals,
        detector.r0_rescue_enabled,
        detector.rescue_det_threshold,
        detector.rescue_topk,
    ) = original
    print("GPU smoke test: PASS")
    temporary_work_dir.cleanup()


if __name__ == "__main__":
    main()
