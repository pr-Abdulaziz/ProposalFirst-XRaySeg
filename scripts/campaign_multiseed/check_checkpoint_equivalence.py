#!/usr/bin/env python
"""GPU check that Method 06 with activation checkpointing computes the same losses and gradients.

Builds the frozen 06 model and the amended (checkpointed) 06 model, copies identical weights into
both (including the randomly initialised LoRA and heads), draws one real training batch of two
images once, and runs forward + backward under the campaign's float16 autocast in train mode with
identical RNG seeds and deterministic cuDNN. Two images keep the uncheckpointed model within memory;
checkpointing does not depend on the batch size. Compared: every loss term and every trainable
gradient. Also records peak memory of each pass. Writes
protocol/amendments/06_activation_checkpointing/equivalence_check.json.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.amend_06_checkpointing import EQUIV, amended_config  # noqa: E402
from scripts.campaign_multiseed.campaign_common import resolved_config, sha256  # noqa: E402

LOSS_REL_TOL = 1e-4
GRAD_COS_MIN = 0.9999
GRAD_NORM_REL_TOL = 1e-3


def seed_all(s: int = 1234) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_pass(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    seed_all()
    torch.cuda.reset_peak_memory_stats()
    data = model.data_preprocessor(copy.deepcopy(batch), True)
    with torch.autocast("cuda", dtype=torch.float16):
        losses = model(**data, mode="loss")
    total, log_vars = model.parse_losses(losses)
    total.backward()
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    return ({k: float(v) for k, v in log_vars.items()}, grads, torch.cuda.max_memory_allocated() / 2**20)


def main() -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    frozen_cfg = Config.fromfile(str(resolved_config("06", 42)), import_custom_modules=True)
    amended_cfg = Config.fromfile(str(amended_config(42)), import_custom_modules=True)
    import_modules_from_strings(**amended_cfg.custom_imports)
    init_default_scope("mmdet")
    from mmdet.registry import MODELS

    loader_cfg = copy.deepcopy(frozen_cfg.train_dataloader)
    loader_cfg.update(batch_size=2, num_workers=0, persistent_workers=False,
                      sampler=dict(type="DefaultSampler", shuffle=False))
    loader = Runner.build_dataloader(loader_cfg, seed=0)
    seed_all()
    batch = next(iter(loader))

    plain = MODELS.build(copy.deepcopy(frozen_cfg.model))
    plain.init_weights()
    plain = plain.cuda()
    ckpt = MODELS.build(copy.deepcopy(amended_cfg.model))
    ckpt.load_state_dict(plain.state_dict(), strict=True)
    ckpt = ckpt.cuda()
    n_ckpt_layers = ckpt.backbone.checkpointed_layers

    loss_p, grad_p, mem_p = run_pass(plain, batch)
    # Run-to-run baseline: the same uncheckpointed model twice (some CUDA kernels, e.g. RoI Align
    # backward, are not bitwise deterministic). Checkpointing must not differ by more than that.
    loss_p2, grad_p2, _ = run_pass(plain, batch)
    del plain
    gc.collect()
    torch.cuda.empty_cache()
    loss_c, grad_c, mem_c = run_pass(ckpt, batch)

    def compare(ga, gb):
        worst_cos, worst_norm = 1.0, 0.0
        for n in ga:
            a, b = ga[n].flatten(), gb[n].flatten()
            na, nb = float(a.norm()), float(b.norm())
            if na == 0 and nb == 0:
                continue
            worst_cos = min(worst_cos, float(torch.dot(a, b) / (na * nb + 1e-30)))
            worst_norm = max(worst_norm, abs(na - nb) / max(na, 1e-30))
        return worst_cos, worst_norm

    loss_rows, ok = {}, True
    for k in loss_p:
        rel = abs(loss_p[k] - loss_c[k]) / max(abs(loss_p[k]), 1e-8)
        base = abs(loss_p[k] - loss_p2[k]) / max(abs(loss_p[k]), 1e-8)
        loss_rows[k] = dict(plain=loss_p[k], checkpointed=loss_c[k], rel_diff=rel, run_to_run_rel_diff=base)
        ok &= rel <= max(LOSS_REL_TOL, 10 * base)
    ok &= set(grad_p) == set(grad_c)
    worst_cos, worst_norm = compare(grad_p, grad_c)
    base_cos, base_norm = compare(grad_p, grad_p2)
    ok &= (worst_cos >= min(GRAD_COS_MIN, 1 - 10 * (1 - base_cos))
           and worst_norm <= max(GRAD_NORM_REL_TOL, 10 * base_norm) and n_ckpt_layers > 0)
    res = dict(status="PASS" if ok else "FAIL", job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
               gpu=torch.cuda.get_device_name(0), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
               frozen_config_sha256=sha256(resolved_config("06", 42)), amended_config_sha256=sha256(amended_config(42)),
               batch="first two training images of the frozen train pipeline, batch 2, float16 autocast, train mode",
               checkpointed_layers=n_ckpt_layers, losses=loss_rows, trainable_tensors_with_grad=len(grad_p),
               worst_grad_cosine=worst_cos, worst_grad_norm_rel_diff=worst_norm,
               run_to_run_baseline=dict(worst_grad_cosine=base_cos, worst_grad_norm_rel_diff=base_norm),
               tolerances=dict(loss_rel=LOSS_REL_TOL, grad_cos_min=GRAD_COS_MIN, grad_norm_rel=GRAD_NORM_REL_TOL),
               peak_allocated_mb=dict(plain=mem_p, checkpointed=mem_c))
    EQUIV.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("status", "checkpointed_layers", "worst_grad_cosine",
                                          "worst_grad_norm_rel_diff", "peak_allocated_mb")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
