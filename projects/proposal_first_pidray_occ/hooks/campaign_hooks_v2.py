"""Resumable RNG-state hook for the PIDray batch-1 campaign.

``CampaignRNGStateHook`` saves the generator states correctly but cannot restore
them when the checkpoint was loaded onto the GPU: MMEngine maps checkpoint
tensors to the current device, and ``torch.set_rng_state`` requires a CPU
``ByteTensor``, so a resume fails with

    TypeError: RNG state must be a torch.ByteTensor

This subclass changes only the restore path -- the saved contents and the run-key
check are inherited unchanged -- by moving each state back to the CPU as a uint8
tensor first. That includes the point-refinement generator state, which fails
later than the rest: cascade_adapter keeps it pending until the generator is
built, so a CUDA tensor there raises at the first iteration after the resume
rather than during loading. It is a new module so
that the PIXray campaign's frozen source snapshot stays byte-identical; configs
register it through ``custom_imports``.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from mmdet.registry import HOOKS

from .campaign_hooks import CampaignRNGStateHook, _unwrap


def _cpu_byte(state):
    """A CPU uint8 tensor, whatever device or dtype the checkpoint loader produced."""
    if torch.is_tensor(state):
        return state.detach().to(device="cpu", dtype=torch.uint8, copy=True)
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")


@HOOKS.register_module()
class CampaignRNGStateHookV2(CampaignRNGStateHook):
    """``CampaignRNGStateHook`` with a device-safe restore."""

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
        torch.set_rng_state(_cpu_byte(state["torch_cpu"]))
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_cpu_byte(s) for s in state["torch_cuda"]])
        roi_head = getattr(_unwrap(runner.model), "roi_head", None)
        if "point_refine_generator" in state:
            # The point-selection generator state needs the same treatment, and it
            # fails later than the tensors above: cascade_adapter stashes it in
            # _pending_generator_state and only calls set_state() when the generator
            # is first built, i.e. at the first training iteration after the resume.
            # Hand it over already on the CPU so both branches there are safe.
            record = dict(state["point_refine_generator"])
            if record.get("state") is not None:
                record["state"] = _cpu_byte(record["state"])
            roi_head.load_point_generator_state(record)
        runner.logger.info(f"restored RNG states for {self.run_key} (v2, device-safe)")
