"""Hooks for the PIXray multi-seed campaign: RNG state and optimizer-update accounting.

Neither hook changes what the model computes. ``CampaignRNGStateHook`` stores the
Python / NumPy / PyTorch CPU and CUDA generator states, plus the point-refinement
generator where one exists, in every checkpoint and restores them when a run is
resumed. ``UpdateAccountingHook`` counts optimizer-update opportunities, AMP
skipped updates and non-finite losses, and writes one JSON line per epoch.

Data-loader worker processes are re-created on resume and reseeded by MMEngine's
``worker_init_fn``; their augmentation stream is therefore not continued
bit-for-bit after an interruption. The sampler order is (it is a function of the
seed and the epoch).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class CampaignRNGStateHook(Hook):
    """Save and restore every RNG state a training run depends on."""

    priority = "VERY_LOW"

    def __init__(self, run_key: str) -> None:
        self.run_key = run_key

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        state = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(),
            torch_cuda=(torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None),
        )
        roi_head = getattr(_unwrap(runner.model), "roi_head", None)
        if roi_head is not None and hasattr(roi_head, "point_generator_state"):
            state["point_refine_generator"] = roi_head.point_generator_state()
        checkpoint["campaign_rng"] = state
        checkpoint.setdefault("meta", {})["campaign_run"] = self.run_key

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        state = checkpoint.get("campaign_rng")
        if state is None:
            return
        saved_run = checkpoint.get("meta", {}).get("campaign_run")
        if saved_run != self.run_key:
            raise RuntimeError(
                f"checkpoint belongs to run {saved_run!r}, not {self.run_key!r}; "
                "a campaign run must never start from another run's checkpoint")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        roi_head = getattr(_unwrap(runner.model), "roi_head", None)
        if "point_refine_generator" in state:
            roi_head.load_point_generator_state(state["point_refine_generator"])
        runner.logger.info(f"restored RNG states for {self.run_key}")


@HOOKS.register_module()
class UpdateAccountingHook(Hook):
    """Count update opportunities, AMP-skipped updates and non-finite losses.

    Runs before ``ParamSchedulerHook`` (priority LOW) so the learning rate it
    records is the one the optimizer step actually used.
    """

    priority = "HIGH"

    def __init__(self, filename: str = "update_accounting.jsonl") -> None:
        self.filename = filename
        self.totals = dict(iterations=0, update_opportunities=0, successful_updates=0,
                           amp_skipped_updates=0, nonfinite_loss_iterations=0,
                           nonfinite_grad_norm_updates=0)
        self._epoch = None
        self._scale_before = None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _is_update(optim_wrapper, count: int) -> bool:
        acc = optim_wrapper._accumulative_counts
        return count % acc == 0 or count == optim_wrapper._max_counts

    @staticmethod
    def _scale(optim_wrapper):
        scaler = getattr(optim_wrapper, "loss_scaler", None)
        return float(scaler.get_scale()) if scaler is not None else None

    def _new_epoch(self, runner) -> None:
        self._epoch = dict(epoch=runner.epoch + 1, iterations=0, update_opportunities=0,
                           successful_updates=0, amp_skipped_updates=0,
                           nonfinite_loss_iterations=0, nonfinite_grad_norm_updates=0,
                           lr_first_update=None, lr_last_update=None,
                           loss_scale_last=None, started=time.time())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # -- hook points -------------------------------------------------------
    def before_train_epoch(self, runner) -> None:
        self._new_epoch(runner)

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:
        ow = runner.optim_wrapper
        # _inner_count is incremented inside backward(); the coming iteration is an
        # update opportunity when the incremented count qualifies.
        self._scale_before = (self._scale(ow)
                              if self._is_update(ow, ow._inner_count + 1) else None)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        ow = runner.optim_wrapper
        e = self._epoch
        e["iterations"] += 1
        self.totals["iterations"] += 1
        loss = (outputs or {}).get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            e["nonfinite_loss_iterations"] += 1
            self.totals["nonfinite_loss_iterations"] += 1
        if not self._is_update(ow, ow._inner_count):
            return
        e["update_opportunities"] += 1
        self.totals["update_opportunities"] += 1
        lr = float(ow.get_lr()["lr"][0])
        e["lr_first_update"] = lr if e["lr_first_update"] is None else e["lr_first_update"]
        e["lr_last_update"] = lr
        scale_after = self._scale(ow)
        skipped = (scale_after is not None and self._scale_before is not None
                   and scale_after < self._scale_before)
        key = "amp_skipped_updates" if skipped else "successful_updates"
        e[key] += 1
        self.totals[key] += 1
        e["loss_scale_last"] = scale_after
        grad_norm = runner.message_hub.get_scalar("train/grad_norm").current() \
            if "train/grad_norm" in runner.message_hub.log_scalars else None
        if grad_norm is not None and not skipped and not math.isfinite(float(grad_norm)):
            e["nonfinite_grad_norm_updates"] += 1
            self.totals["nonfinite_grad_norm_updates"] += 1

    def after_train_epoch(self, runner) -> None:
        e = dict(self._epoch)
        e["seconds"] = time.time() - e.pop("started")
        if torch.cuda.is_available():
            e["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            e["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        e["cumulative"] = dict(self.totals)
        with (Path(runner.work_dir) / self.filename).open("a") as f:
            f.write(json.dumps(e) + "\n")

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        checkpoint.setdefault("meta", {})["update_accounting"] = dict(self.totals)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        saved = checkpoint.get("meta", {}).get("update_accounting")
        if saved:
            self.totals.update(saved)
