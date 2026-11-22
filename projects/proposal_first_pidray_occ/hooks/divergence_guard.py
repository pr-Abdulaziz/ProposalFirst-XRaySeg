"""Abort a training run that has provably diverged, instead of burning the job.

The PIDray batch-1 attempt (job 12317, method 08) reached a loss of 438 at
iteration ~511, went 79% non-finite for the rest of epoch 1 and 100% in epoch 2,
and the AMP loss scale collapsed to 0.0. Nothing was recoverable after the first
hour, but the job would have held a GPU for another 51 hours and then been
rejected by run_guard for exactly the condition this hook detects.

Two signals, both unambiguous:

* a long unbroken run of non-finite losses -- a healthy run has none at all
  (the batch-1/batch-2 profiles measured 400 consecutive finite iterations for
  every method), so a streak of ``patience`` is not a transient;
* an AMP loss scale of exactly zero, which cannot recover by construction:
  the scaler only grows by multiplication, and zero times anything is zero.

A new module, registered through ``custom_imports``, so no existing file changes.
"""
from __future__ import annotations

import math

from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class DivergenceGuardHook(Hook):
    """Raise as soon as the run is known to be dead.

    Args:
        patience: consecutive non-finite-loss iterations tolerated before abort.
        warn_every: log a warning every this many consecutive non-finite losses.
        check_loss_scale_after: skip the loss-scale test for this many iterations,
            so the scaler's initial calibration cannot trip it.
    """

    priority = "LOW"

    def __init__(self, patience: int = 500, warn_every: int = 100,
                 check_loss_scale_after: int = 200) -> None:
        self.patience = patience
        self.warn_every = warn_every
        self.check_loss_scale_after = check_loss_scale_after
        self.streak = 0
        self.seen = 0
        self.worst_loss = 0.0

    @staticmethod
    def _scale(optim_wrapper):
        scaler = getattr(optim_wrapper, "loss_scaler", None)
        return None if scaler is None else float(scaler.get_scale())

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.seen += 1
        loss = float(outputs["loss"]) if outputs and "loss" in outputs else float("nan")
        finite = math.isfinite(loss)
        if finite:
            self.streak = 0
            self.worst_loss = max(self.worst_loss, abs(loss))
        else:
            self.streak += 1
            if self.streak % self.warn_every == 0:
                runner.logger.warning(
                    f"DivergenceGuardHook: {self.streak} consecutive non-finite losses "
                    f"at iteration {runner.iter + 1} (abort at {self.patience})")
            if self.streak >= self.patience:
                raise RuntimeError(
                    f"training diverged: {self.streak} consecutive non-finite losses "
                    f"ending at iteration {runner.iter + 1}; worst finite loss seen was "
                    f"{self.worst_loss:.4g}. Aborting instead of holding the GPU.")
        scale = self._scale(runner.optim_wrapper)
        if (scale is not None and scale == 0.0
                and self.seen > self.check_loss_scale_after):
            raise RuntimeError(
                f"training diverged: AMP loss scale collapsed to 0 at iteration "
                f"{runner.iter + 1}; it cannot recover, since the scaler only grows "
                "by multiplication. Aborting instead of holding the GPU.")
