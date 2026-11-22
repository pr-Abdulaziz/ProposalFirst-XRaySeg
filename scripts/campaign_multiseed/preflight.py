#!/usr/bin/env python
"""GPU preflight for one campaign method, run against its frozen seed-42 resolved config.

Preflight-only changes, recorded in the output: the training set is the eight most
populated PIXray training images repeated 20 times (160 images in physical batches,
one optimizer update per batch without accumulation), one epoch, and a scratch work directory. Model,
pipeline, optimizer, AMP, accumulation, schedule and campaign hooks are the frozen
config's.

Checks: annotation fields; model construction and parameter counts; optimizer
coverage; finite losses and gradients; loss bookkeeping; updates to trainable
modules and untouched frozen ones; batch-one normalisation behaviour; AMP and
accumulation counts; learning rate at update opportunities; checkpoint save and
strict reload; RNG-state restore; an empty-instance batch; a memory stress pass at
the largest resize; inference geometry and memory on difficult test images.

    python scripts/campaign_multiseed/preflight.py <method id>
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.hooks import Hook  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import Runner, load_checkpoint  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    ACCUMULATION, BY_ID, MICROBATCHES, OUT, PHYSICAL_BATCH, point_seed, resolved_config, run_key, sha256)

HARD_TRAIN = [239, 500, 509, 519, 553, 233, 217, 2881]   # 7,7,7,7,7,6,6,5 instances; largest box areas
HARD_TEST = [1070, 1135, 62, 331, 348]                   # 8 instances; widest image; largest box areas
REPEAT = 20


class Probe(Hook):
    priority = "VERY_HIGH"

    def __init__(self):
        self.iters, self.grad_steps, self.bn = [], [], {}
        self.never_nonzero = None      # trainable tensors whose gradient was None or all zero at every step
        self.frozen_with_grad = set()

    def before_train(self, runner):
        model = runner.model
        self.params = dict(model.named_parameters())
        self.before = {n: p.detach().float().cpu().clone() for n, p in self.params.items() if p.requires_grad}
        self.frozen = {n: (float(p.detach().double().sum()), float(p.detach().double().norm()))
                       for n, p in self.params.items() if not p.requires_grad}
        opt = runner.optim_wrapper.optimizer
        ids = {id(p) for g in opt.param_groups for p in g["params"]}
        self.optimizer = dict(
            optimizer_tensors=len(ids),
            trainable_tensors=sum(p.requires_grad for p in self.params.values()),
            trainable_in_optimizer=sum(1 for p in self.params.values() if p.requires_grad and id(p) in ids),
            frozen_in_optimizer=sum(1 for p in self.params.values() if not p.requires_grad and id(p) in ids),
            frozen_in_optimizer_names=[n for n, p in self.params.items() if not p.requires_grad and id(p) in ids],
            param_groups=len(opt.param_groups),
            group_hparams=[{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups])
        original = opt.step

        def probed(*args, **kwargs):
            grads = [(n, p.grad) for n, p in self.params.items() if p.requires_grad]
            present = [g for _, g in grads if g is not None]
            silent = {n for n, g in grads if g is None or not bool(torch.any(g))}
            self.never_nonzero = silent if self.never_nonzero is None else self.never_nonzero & silent
            self.frozen_with_grad |= {n for n, p in self.params.items() if not p.requires_grad and p.grad is not None}
            self.grad_steps.append(dict(
                iteration=runner.iter + 1,
                tensors_without_grad=sum(1 for _, g in grads if g is None),
                all_finite=bool(all(torch.isfinite(g).all() for g in present)),
                clipped_grad_norm=float(torch.norm(torch.stack([g.detach().float().norm() for g in present])))))
            return original(*args, **kwargs)
        opt.step = probed
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.register_forward_pre_hook(self._bn_hook(name, mod))
        torch.cuda.reset_peak_memory_stats()

    def _bn_hook(self, name, mod):
        rec = self.bn.setdefault(name, dict(type=type(mod).__name__, calls=0, training_calls=0,
                                            min_values_per_channel=None, single_value_training_calls=0))

        def hook(module, inputs):
            x = inputs[0]
            vpc = x.numel() // x.shape[1]
            rec["calls"] += 1
            rec["training_calls"] += int(module.training)
            rec["min_values_per_channel"] = vpc if rec["min_values_per_channel"] is None else min(rec["min_values_per_channel"], vpc)
            rec["single_value_training_calls"] += int(module.training and vpc == 1)
        return hook

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        torch.cuda.synchronize()
        now = time.perf_counter()
        # Time since the previous iteration ended: data loading plus other hooks.
        self.data_wait = now - self.t_end if getattr(self, "t_end", None) else None
        self.t0 = now

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        torch.cuda.synchronize()
        sample = data_batch["data_samples"][0]
        gi = sample.gt_instances
        loss = float(outputs["loss"])
        components = sum(float(v) for k, v in outputs.items() if k != "loss" and "loss" in k)
        self.iters.append(dict(
            iteration=runner.iter + 1, seconds=time.perf_counter() - self.t0, loss=loss,
            loss_minus_components=loss - components, finite=math.isfinite(loss),
            gt_instances=len(gi), gt_fields=sorted(gi.keys()),
            input_shape=list(data_batch["inputs"][0].shape),
            peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20, data_wait_seconds=self.data_wait))
        if len(self.iters) == 1:
            self.loss_keys = sorted(outputs.keys())
        self.t_end = time.perf_counter()


def dataset_positions(cfg_ds, img_ids):
    from mmdet.registry import DATASETS
    ds = DATASETS.build(copy.deepcopy(cfg_ds))
    pos = {ds.get_data_info(i)["img_id"]: i for i in range(len(ds))}
    info = [ds.get_data_info(pos[i]) for i in img_ids]
    return [pos[i] for i in img_ids], info


def expected_lrs(cfg, iterations, acc):
    from mmengine.registry import PARAM_SCHEDULERS
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4, weight_decay=0.05)
    scheds = [PARAM_SCHEDULERS.build(dict(copy.deepcopy(dict(s)), optimizer=opt)) for s in cfg.param_scheduler]
    lrs = []
    for count in range(1, iterations + 1):
        if count % acc == 0:
            lrs.append(opt.param_groups[0]["lr"])
        for s in scheds:
            if not s.by_epoch:
                s.step()
    return lrs


def main() -> int:
    mid = sys.argv[1]
    m = BY_ID[mid]
    # PREFLIGHT_CONFIG / PREFLIGHT_OUT_SUFFIX: preflight of an amended config (06 activation checkpointing).
    cfg_path = Path(os.environ["PREFLIGHT_CONFIG"]) if os.environ.get("PREFLIGHT_CONFIG") else resolved_config(mid, 42)
    out_dir = OUT / "preflight" / (m.slug + os.environ.get("PREFLIGHT_OUT_SUFFIX", ""))
    work = out_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = dict(method=mid, slug=m.slug, name=m.name, run_key=run_key(mid, 42), config=str(cfg_path),
               config_sha256=sha256(cfg_path), node=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
               gpu=torch.cuda.get_device_name(0), gpu_total_mb=torch.cuda.get_device_properties(0).total_memory / 2**20,
               preflight_changes=[f"train set = {len(HARD_TRAIN)} most populated train images x {REPEAT}",
                                  "max_epochs = 1", f"work_dir = {work}"],
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), checks=[])

    gc.collect()
    torch.cuda.empty_cache()
    free0, total0 = torch.cuda.mem_get_info()
    res["gpu_memory_at_start"] = dict(free_mib=free0 / 2**20, total_mib=total0 / 2**20, used_elsewhere_mib=(total0 - free0) / 2**20,
                                      pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    def check(name, ok, detail=None):
        res["checks"].append(dict(name=name, status="PASS" if ok else "FAIL", detail=detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail is not None and not ok else ""), flush=True)

    def guarded(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(name, False, f"{type(exc).__name__}: {exc}")
            res.setdefault("tracebacks", {})[name] = traceback.format_exc()

    cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet")
    acc = cfg.optim_wrapper.accumulative_counts
    check(f"frozen config: physical batch {PHYSICAL_BATCH}, accumulation {ACCUMULATION}, 2 workers",
          cfg.train_dataloader.batch_size == PHYSICAL_BATCH and acc == ACCUMULATION
          and cfg.train_dataloader.num_workers == 2)
    n_iters = len(HARD_TRAIN) * REPEAT // PHYSICAL_BATCH
    n_updates = n_iters // acc

    idx, infos = dataset_positions(cfg.train_dataloader.dataset, HARD_TRAIN)
    inst_keys = sorted(infos[0]["instances"][0].keys())
    check("annotation fields: occluder masks loaded only by occluder-supervised methods (06, 07)",
          ("bo_mask" in inst_keys) == (m.supervision == "occ"), inst_keys)
    res["train_images"] = [dict(img_id=i["img_id"], instances=len(i["instances"]),
                                width=i["width"], height=i["height"]) for i in infos]

    ds_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    ds_cfg["indices"] = idx
    cfg.train_dataloader.dataset = dict(type="RepeatDataset", times=REPEAT, dataset=ds_cfg)
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1)
    cfg.work_dir = str(work)
    cfg.default_hooks["logger"]["interval"] = 20
    runner = Runner.from_cfg(cfg)
    probe = Probe()
    runner.register_hook(probe)
    model = runner.model

    counts = dict(total=0, trainable=0, frozen=0)
    for p in model.parameters():
        counts["total"] += p.numel()
        counts["trainable" if p.requires_grad else "frozen"] += p.numel()
    counts["buffers"] = sum(b.numel() for b in model.buffers())
    counts["by_module"] = {n: dict(total=sum(p.numel() for p in c.parameters()),
                                   trainable=sum(p.numel() for p in c.parameters() if p.requires_grad))
                           for n, c in model.named_children()}
    res["parameters"] = counts

    t_train = time.time()
    try:
        runner.train()
        check(f"training loop completes ({n_iters} batches of {PHYSICAL_BATCH})", len(probe.iters) == n_iters, len(probe.iters))
    except Exception as exc:  # noqa: BLE001
        check(f"training loop completes ({n_iters} batches of {PHYSICAL_BATCH})", False, f"{type(exc).__name__}: {exc}")
        res["tracebacks"] = {"train": traceback.format_exc()}
        (out_dir / "preflight.json").write_text(json.dumps(res, indent=1, default=str))
        return 1
    res["train_wall_seconds"] = time.time() - t_train
    res["optimizer"] = probe.optimizer
    res["loss_keys"] = probe.loss_keys
    res["iterations"] = probe.iters
    res["grad_steps"] = probe.grad_steps

    o = probe.optimizer
    check("optimizer holds every trainable tensor; frozen tensors it holds never receive a gradient",
          o["trainable_in_optimizer"] == o["trainable_tensors"] and not probe.frozen_with_grad,
          dict(trainable=o["trainable_tensors"], trainable_in_optimizer=o["trainable_in_optimizer"],
               frozen_in_optimizer=o["frozen_in_optimizer"], frozen_with_grad=sorted(probe.frozen_with_grad)))
    check("all losses finite", all(i["finite"] for i in probe.iters))
    check("total loss equals the sum of loss terms (each term counted once)",
          all(abs(i["loss_minus_components"]) <= 1e-3 * max(1.0, abs(i["loss"])) for i in probe.iters))
    non_loss_diag = [k for k in probe.loss_keys if "point" in k and "loss" not in k]
    res["diagnostic_keys_not_summed"] = non_loss_diag
    if mid in ("09", "09M"):
        check("point loss present once per cascade stage",
              all(f"s{s}.loss_point" in probe.loss_keys for s in range(3)), probe.loss_keys)
        check("point generator seed derived from run seed",
              model.roi_head._seed == point_seed(42), model.roi_head._seed)
    fields = {tuple(i["gt_fields"]) for i in probe.iters}
    has_bo = any("bo_masks" in f for f in fields)
    check("model receives occluder targets only for occluder-supervised methods (06, 07)", has_bo == (m.supervision == "occ"), sorted(fields))

    # accounting, AMP, learning rate
    lines = [json.loads(x) for x in (work / "update_accounting.jsonl").read_text().splitlines()]
    acct = lines[-1]["cumulative"]
    res["update_accounting"] = lines
    check(f"update opportunities = {n_iters} / {acc}", acct["update_opportunities"] == n_updates, acct)
    check("at least one successful AMP update; skipped updates counted",
          acct["successful_updates"] >= 1 and acct["successful_updates"] + acct["amp_skipped_updates"] == n_updates, acct)
    check("gradient probe saw one optimizer.step per successful update",
          len(probe.grad_steps) == acct["successful_updates"], len(probe.grad_steps))
    check("gradients finite at every optimizer step", all(g["all_finite"] for g in probe.grad_steps))
    lrs = expected_lrs(cfg, n_iters, acc)
    check("learning rate at update opportunities matches the schedule",
          math.isclose(lines[-1]["lr_first_update"], lrs[0], rel_tol=1e-9)
          and math.isclose(lines[-1]["lr_last_update"], lrs[-1], rel_tol=1e-9),
          dict(logged=(lines[-1]["lr_first_update"], lines[-1]["lr_last_update"]), expected=(lrs[0], lrs[-1])))

    # parameter updates
    changed, unchanged = {}, {}
    for n, p in probe.params.items():
        if not p.requires_grad:
            continue
        group = ".".join(n.split(".")[:2])
        same = torch.equal(probe.before[n], p.detach().float().cpu())
        (unchanged if same else changed)[group] = (unchanged if same else changed).get(group, 0) + p.numel()
    res["trainable_numel_changed_by_module"] = changed
    res["trainable_numel_unchanged_by_module"] = unchanged
    unchanged_names = [n for n, p in probe.params.items() if p.requires_grad
                       and torch.equal(probe.before[n], p.detach().float().cpu())]
    res["trainable_tensors_unchanged"] = unchanged_names
    total_trainable = sum(changed.values()) + sum(unchanged.values())
    check("trainable parameters updated (>99% of trainable elements)",
          sum(changed.values()) >= 0.99 * total_trainable, dict(unchanged=unchanged))
    silent = sorted(probe.never_nonzero or [])
    res["trainable_tensors_without_gradient_signal"] = silent
    unexplained = [n for n in unchanged_names if n not in (probe.never_nonzero or set())]
    check("trainable tensors stay unchanged only where no gradient ever arrived", not unexplained, unexplained[:20])
    new_modules = {"09": "roi_head.point_refine_head", "09M": "roi_head.point_refine_head", "08": "r0_head"}.get(mid)
    if new_modules:
        missed = [n for n in silent if n.startswith(new_modules)]
        check(f"every tensor of {new_modules} receives a gradient", not missed, missed[:20])
    if m.family == "sam":
        lora = [n for n, q in probe.params.items() if "lora" in n and q.requires_grad]
        missed = [n for n in silent if "lora" in n]
        check("every LoRA tensor receives a gradient", lora and not missed, missed[:20])
    frozen_changed = [n for n, (s, nr) in probe.frozen.items()
                      if (float(probe.params[n].detach().double().sum()), float(probe.params[n].detach().double().norm())) != (s, nr)]
    check("frozen parameters unchanged", not frozen_changed, frozen_changed[:10])
    if m.family == "sam":
        bad = [n for n, p in probe.params.items() if n.startswith("backbone") and p.requires_grad and "lora" not in n]
        check("SAM image encoder frozen except LoRA", not bad, bad[:10])

    # normalisation at batch one
    res["batchnorm"] = probe.bn
    safe = {n: r for n, r in probe.bn.items() if r["type"] == "BatchOneSafeBatchNorm2d"}
    backbone_bn = {n: r for n, r in probe.bn.items() if n.startswith("backbone") and r["type"] != "BatchOneSafeBatchNorm2d"}
    if backbone_bn:
        check("backbone BatchNorm in eval mode during training (norm_eval)",
              all(r["training_calls"] == 0 for r in backbone_bn.values()))
    if safe:
        mods = dict(model.named_modules())
        tracked = {n: int(mods[n].num_batches_tracked) for n in safe}
        check(f"AttFPN channel-attention BN normalises over {PHYSICAL_BATCH} images (batch-one fallback never "
              "triggers); running statistics update",
              all(r["single_value_training_calls"] == 0 and r["training_calls"] > 0
                  and r["min_values_per_channel"] == PHYSICAL_BATCH for r in safe.values())
              and all(v > 0 for v in tracked.values()),
              dict(bn=safe, num_batches_tracked=tracked))
    others = {n: r for n, r in probe.bn.items() if n not in safe and n not in backbone_bn and r["training_calls"]}
    check("no other BatchNorm receives a single value per channel in training",
          all((r["min_values_per_channel"] or 2) > 1 for r in others.values()), others)

    # checkpoint save / strict reload / RNG restore
    def ckpt_checks():
        path = work / "epoch_1.pth"
        ck = torch.load(path, map_location="cpu")
        sd = model.state_dict()
        same = all(torch.equal(ck["state_dict"][k], v.detach().cpu()) for k, v in sd.items()) and set(ck["state_dict"]) == set(sd)
        check("checkpoint state_dict equals the trained model", same)
        check("checkpoint carries campaign RNG state and update accounting",
              "campaign_rng" in ck and ck["meta"].get("update_accounting", {}).get("update_opportunities") == n_updates
              and ck["meta"].get("campaign_run") == run_key(mid, 42))
        from mmdet.registry import MODELS
        fresh = MODELS.build(copy.deepcopy(cfg.model))
        load_checkpoint(fresh, str(path), map_location="cpu", strict=True, logger="silent")
        fsd = fresh.state_dict()
        check("fresh model loads the checkpoint strictly and identically",
              all(torch.equal(fsd[k].cpu(), ck["state_dict"][k]) for k in fsd))
        del fresh
        from projects.proposal_first_pidray_occ.hooks.campaign_hooks import CampaignRNGStateHook
        saved = ck["campaign_rng"]
        torch.manual_seed(987); np.random.seed(987); random.seed(987); torch.cuda.manual_seed_all(987)
        if "point_refine_generator" in saved:
            model.roi_head._generator.manual_seed(987)
        CampaignRNGStateHook(run_key(mid, 42)).after_load_checkpoint(runner, ck)
        ok = (torch.equal(torch.get_rng_state(), saved["torch_cpu"])
              and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved["torch_cuda"]))
              and random.getstate() == saved["python"]
              and np.array_equal(np.random.get_state()[1], saved["numpy"][1]))
        if "point_refine_generator" in saved:
            ok = ok and torch.equal(model.roi_head._generator.get_state(), saved["point_refine_generator"]["state"])
        check("resume restores Python/NumPy/Torch CPU+CUDA" + (" and point-generator" if "point_refine_generator" in saved else "") + " RNG state", ok)
        try:
            CampaignRNGStateHook(run_key(mid, 123)).after_load_checkpoint(runner, ck)
            refused = False
        except RuntimeError:
            refused = True
        check("a checkpoint from another run is refused", refused)
    guarded("checkpoint and RNG checks", ckpt_checks)

    # empty-instance batch
    def empty_instances():
        batch = next(iter(runner.train_dataloader))
        for s in batch["data_samples"]:
            s.gt_instances = s.gt_instances[0:0]
        model.train()
        with runner.optim_wrapper.optim_context(model):
            data = model.data_preprocessor(batch, True)
            losses = model._run_forward(data, mode="loss")
        loss, _ = model.parse_losses(losses)
        loss.backward()
        finite_grads = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        runner.optim_wrapper.zero_grad()
        check("empty-instance image: finite loss and gradients", math.isfinite(float(loss)) and finite_grads, float(loss))
    guarded("empty-instance image: finite loss and gradients", empty_instances)

    # memory stress at the largest resize
    def stress():
        sds = copy.deepcopy(ds_cfg)
        for step in sds["pipeline"]:
            if step["type"] == "RandomResize":
                step["ratio_range"] = (1.5, 1.5)
        loader = Runner.build_dataloader(dict(batch_size=PHYSICAL_BATCH, num_workers=2, persistent_workers=False,
                                              sampler=dict(type="DefaultSampler", shuffle=False),
                                              dataset=dict(type="RepeatDataset", times=2, dataset=sds)), seed=0)
        runner.optim_wrapper.initialize_count_status(model, 0, acc * 1000)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for batch in loader:
            out = model.train_step(batch, optim_wrapper=runner.optim_wrapper)
            losses.append(float(out["loss"]))
        res["stress_train"] = dict(iterations=len(losses), peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                                   peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
                                   all_finite=all(math.isfinite(x) for x in losses))
        check(f"memory stress (resize ratio 1.5, densest images, 2 batches of {PHYSICAL_BATCH} incl. 1 update) finite",
              res["stress_train"]["all_finite"], res["stress_train"])
    guarded(f"memory stress (resize ratio 1.5, densest images, 2 batches of {PHYSICAL_BATCH} incl. 1 update) finite", stress)

    # inference
    def inference():
        tcfg = copy.deepcopy(cfg.test_dataloader)
        tidx, tinfo = dataset_positions(tcfg.dataset, HARD_TEST)
        tds = copy.deepcopy(tcfg.dataset)
        tds["indices"] = tidx
        loader = Runner.build_dataloader(dict(batch_size=1, num_workers=2, persistent_workers=False,
                                              sampler=dict(type="DefaultSampler", shuffle=False), dataset=tds), seed=0)
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rows, geometry_ok = [], True
        for batch in loader:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                outs = model.test_step(batch)
            torch.cuda.synchronize()
            pred = outs[0].pred_instances
            h, w = outs[0].metainfo["ori_shape"]
            mshape = list(pred.masks.shape) if "masks" in pred else None
            boxes = pred.bboxes
            overflow = 0.0 if len(boxes) == 0 else float(max(-boxes[:, 0].min(), -boxes[:, 1].min(),
                                                              boxes[:, 2].max() - w, boxes[:, 3].max() - h, 0))
            extents = None if len(boxes) == 0 else [float(boxes[:, 0].min()), float(boxes[:, 1].min()),
                                                     float(boxes[:, 2].max()), float(boxes[:, 3].max())]
            inside = overflow <= 2.0
            # Box decoding clamps to img_shape, which MMCV Pad sets to the padded 512x512 canvas, so
            # rescaled boxes may extend past the original lower/right edge. Shared by every method and
            # by the historical evaluation; recorded, not a pass criterion.
            ok = mshape is not None and mshape[1:] == [h, w] and len(pred) <= 100
            geometry_ok &= ok
            rows.append(dict(img_id=outs[0].metainfo["img_id"], ori_shape=[h, w], masks=mshape, detections=len(pred),
                             box_extents=extents, box_overflow_px=overflow, boxes_inside=inside,
                             scale_factor=outs[0].metainfo.get("scale_factor"), img_shape=outs[0].metainfo.get("img_shape"),
                             seconds=time.perf_counter() - t0))
        res["inference"] = dict(images=rows, peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                                peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20)
        check("inference masks at original resolution, <=100 detections, box overflow recorded", geometry_ok, rows)
    guarded("inference masks at original resolution, <=100 detections, box overflow recorded", inference)

    # memory headroom and timing
    peaks = [max(i["peak_reserved_mb"] for i in probe.iters),
             res.get("stress_train", {}).get("peak_reserved_mb", 0),
             res.get("inference", {}).get("peak_reserved_mb", 0)]
    allocated = [max(i["peak_allocated_mb"] for i in probe.iters),
                 res.get("stress_train", {}).get("peak_allocated_mb", 0),
                 res.get("inference", {}).get("peak_allocated_mb", 0)]
    res["peak_reserved_mb_max"] = max(peaks)
    res["peak_allocated_mb_max"] = max(allocated)
    res["headroom_mb"] = res["gpu_total_mb"] - max(peaks)
    res["headroom_allocated_mb"] = res["gpu_total_mb"] - max(allocated)
    check("peak allocated memory below 90% of GPU memory (reserved recorded)",
          max(allocated) < 0.9 * res["gpu_total_mb"],
          dict(allocated_train_stress_infer=allocated, reserved_train_stress_infer=peaks, total=res["gpu_total_mb"]))
    secs = [i["seconds"] for i in probe.iters[4:]]
    waits = [i["data_wait_seconds"] for i in probe.iters[4:] if i["data_wait_seconds"] is not None]
    res["timing"] = dict(median_s_per_iteration=statistics.median(secs), mean_s_per_iteration=statistics.mean(secs),
                         median_data_wait_s=statistics.median(waits) if waits else None,
                         projected_train_hours=(statistics.median(secs) + (statistics.median(waits) if waits else 0))
                         * MICROBATCHES / 3600,
                         note=f"batches of {PHYSICAL_BATCH} from the densest training images; compute plus data wait")
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    fails = [c for c in res["checks"] if c["status"] == "FAIL"]
    res["summary"] = dict(passed=len(res["checks"]) - len(fails), failed=len(fails))
    (out_dir / "preflight.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(dict(method=mid, **res["summary"], params=counts["total"], trainable=counts["trainable"],
                          peak_reserved_mb=res["peak_reserved_mb_max"], timing=res["timing"]), default=str))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
