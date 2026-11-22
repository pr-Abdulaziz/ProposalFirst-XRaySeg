#!/usr/bin/env python
"""Measure training time and GPU memory for one comparison method on PIDray at batch 2.

Measurement only: no checkpoint is saved and no metric is computed, so nothing here is a
PIDray result (PIDray training remains gated by the leakage audit noted in README.md).

The method's model, 512-px pipeline, optimizer, AMP and clipping come from its PIXray
comparison config; only the data layer is replaced with the PIDray-30 track:
annotations/train.json (23,074 images, empty images kept) or PIDray-A train_occ.json
for the occluder-supervised methods (06, 07), 12 classes, test images from the easy,
hard and hidden splits. Physical batch 2, accumulation 8, 2 workers.

Phases: (1) 200 iterations on 400 random training images, timed per iteration including
data loading; (2) a memory stress pass on the most populated training images at resize
ratio 1.5; (3) batch-1 inference on 60 random test images (20 per split).

    python scripts/pidray_profile/profile_pidray.py <method id> [--dry]
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
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PIDRAY = ROOT / "data/PIDray"
PIDRAYA = ROOT / "data/PIDray-A/annotations"
os.environ.update(
    XRAY_BATCH_SIZE_PER_GPU="2", XRAY_EFFECTIVE_BATCH_SIZE="16", XRAY_NUM_WORKERS="2",
    XRAY_PROJECT_ROOT=str(ROOT), PIXRAY_DATA_ROOT=str(ROOT / "data/PIXray"),
    PIXRAYA_ANN_ROOT=str(ROOT / "data/PIXray-A/annotations"), PIDRAY_DATA_ROOT=str(PIDRAY),
    PIDRAYA_ANN_ROOT=str(PIDRAYA), SAM_VIT_LARGE_DIR=str(ROOT / "pretrained/sam_vit_large"),
    SAM_VIT_LARGE_CKPT=str(ROOT / "pretrained/sam_vit_large/pytorch_model_new.bin"),
    TORCH_HOME=str(ROOT / "pretrained/torch"))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.hooks import Hook  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.campaign_common import BY_ID  # noqa: E402

OUT = ROOT / "output/pidray_profile_2026-09-14"
CLASSES = ("Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench",
           "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter")
PHYSICAL_BATCH, ACCUMULATION, WORKERS, EPOCHS = 2, 8, 2, 60
TRAIN_IMAGES = 23074
TEST_IMAGES = {"easy": 24758, "hard": 9746, "hidden": 13069}
N_TIMING_IMAGES, N_TEST_PER_SPLIT, WARMUP_ITERS = 400, 20, 20


def replace_num_classes(obj, old=15, new=12):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "num_classes" and v == old:
                obj[k] = new
            else:
                replace_num_classes(v, old, new)
    elif isinstance(obj, list):
        for v in obj:
            replace_num_classes(v, old, new)


def build_config(mid: str):
    m = BY_ID[mid]
    cfg = Config.fromfile(str(ROOT / m.historical_config), import_custom_modules=True)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    occ = m.supervision in ("occ", "conflict")
    dtype = "PIDrayOcclusionDataset" if occ else "PIDrayDataset"
    train_ann = PIDRAYA / "train_occ.json" if occ else PIDRAY / "annotations/train.json"
    train_ds = dict(type=dtype, data_root=str(PIDRAY), ann_file=str(train_ann),
                    data_prefix=dict(img="train/"), metainfo=dict(classes=CLASSES),
                    filter_cfg=dict(filter_empty_gt=False, min_size=0),
                    pipeline=copy.deepcopy(cfg.train_dataloader.dataset.pipeline))
    test_ds = {split: dict(type=dtype, data_root=str(PIDRAY),
                           ann_file=str(PIDRAYA / f"test_{split}_occ.json" if occ
                                        else PIDRAY / f"annotations/test_{split}.json"),
                           data_prefix=dict(img="test/"), metainfo=dict(classes=CLASSES), test_mode=True,
                           pipeline=copy.deepcopy(cfg.test_dataloader.dataset.pipeline))
               for split in TEST_IMAGES}
    cfg.train_dataloader.update(batch_size=PHYSICAL_BATCH, num_workers=WORKERS, persistent_workers=True,
                                sampler=dict(type="DefaultSampler", shuffle=True), batch_sampler=None,
                                dataset=train_ds)
    cfg.optim_wrapper["accumulative_counts"] = ACCUMULATION
    replace_num_classes(cfg.model)
    cfg.test_dataloader = None
    cfg.test_cfg = None
    cfg.test_evaluator = None
    cfg.default_hooks["checkpoint"] = dict(type="CheckpointHook", interval=10**9, save_last=False)
    cfg.default_hooks["logger"]["interval"] = 50
    cfg.custom_hooks = []
    cfg.load_from = None
    cfg.resume = False
    cfg.randomness = dict(seed=42, deterministic=False, diff_rank_seed=False)
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1)
    cfg.work_dir = str(OUT / "scratch" / m.slug)
    return cfg, train_ds, test_ds, occ


class Timer(Hook):
    priority = "VERY_HIGH"

    def __init__(self):
        self.rows, self.t_end = [], None

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.wait = None if self.t_end is None else now - self.t_end
        self.t0 = now

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        torch.cuda.synchronize()
        now = time.perf_counter()
        n_inst = sum(len(s.gt_instances) for s in data_batch["data_samples"])
        self.rows.append(dict(compute_s=now - self.t0, data_wait_s=self.wait,
                              cycle_s=(now - self.t0) + (self.wait or 0.0), instances=n_inst,
                              loss=float(outputs["loss"]),
                              peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                              peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20))
        self.t_end = now


def gpu_state() -> dict:
    """GPU memory/utilisation and compute processes other than this one (shared-GPU record)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    apps = [line.strip() for line in subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.splitlines() if line.strip()]
    foreign = [a for a in apps if a.split(",")[0].strip() != str(os.getpid())]
    return dict(memory_used_mib_and_utilization=q, foreign_compute_processes=foreign,
                time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def summary(vals):
    return dict(median=statistics.median(vals), mean=statistics.mean(vals),
                p95=sorted(vals)[int(0.95 * (len(vals) - 1))], n=len(vals))


def main() -> int:
    mid = sys.argv[1]
    dry = "--dry" in sys.argv[2:]
    m = BY_ID[mid]
    cfg, train_ds, test_ds, occ = build_config(mid)
    init_default_scope("mmdet")
    from mmdet.registry import DATASETS
    full = DATASETS.build(copy.deepcopy(train_ds))
    n_train = len(full)
    counts = [len(full.get_data_info(i)["instances"]) for i in range(n_train)]
    rng = random.Random(0)
    timing_idx = sorted(rng.sample(range(n_train), N_TIMING_IMAGES))
    dense_idx = sorted(range(n_train), key=lambda i: -counts[i])[:8]
    res = dict(method=mid, name=m.name, slug=m.slug, source_config=m.historical_config,
               dataset="PIDray-30 train (23,074 images, empty images kept)" + (" + PIDray-A occluder masks" if occ else ""),
               train_images=n_train, empty_train_images=sum(c == 0 for c in counts),
               timing_sample=dict(images=N_TIMING_IMAGES, empty=sum(counts[i] == 0 for i in timing_idx),
                                  instances=sum(counts[i] for i in timing_idx)),
               dense_images_instances=[counts[i] for i in dense_idx],
               physical_batch=PHYSICAL_BATCH, accumulation=ACCUMULATION, workers=WORKERS,
               num_classes=12, node=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"))
    del full
    print(json.dumps(dict(dry=dry, **{k: res[k] for k in ("method", "train_images", "empty_train_images",
                                                           "timing_sample", "dense_images_instances")})), flush=True)
    if dry:
        return 0

    gc.collect()
    torch.cuda.empty_cache()
    free0, total0 = torch.cuda.mem_get_info()
    res.update(gpu=torch.cuda.get_device_name(0), gpu_total_mb=total0 / 2**20,
               gpu_free_at_start_mb=free0 / 2**20,
               pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    res["shared_gpu_allowed"] = os.environ.get("PROFILE_ALLOW_SHARED_GPU") == "1"
    res["gpu_state"] = dict(start=gpu_state())
    ds = copy.deepcopy(train_ds)
    ds["indices"] = timing_idx
    cfg.train_dataloader.dataset = ds
    runner = Runner.from_cfg(cfg)
    timer = Timer()
    runner.register_hook(timer)
    params = sum(p.numel() for p in runner.model.parameters())
    trainable = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    runner.train()
    res["timing_phase_wall_s"] = time.time() - t0
    res["gpu_state"]["after_training"] = gpu_state()
    rows = timer.rows[WARMUP_ITERS:]
    iters_per_epoch = math.ceil(TRAIN_IMAGES / PHYSICAL_BATCH)
    cyc = summary([r["cycle_s"] for r in rows])
    res.update(
        parameters=dict(total=params, trainable=trainable),
        timing=dict(iterations_measured=len(rows), warmup_excluded=WARMUP_ITERS,
                    cycle_s=cyc, compute_s=summary([r["compute_s"] for r in rows]),
                    data_wait_s=summary([r["data_wait_s"] for r in rows if r["data_wait_s"] is not None])),
        projection=dict(iterations_per_epoch=iters_per_epoch, epochs=EPOCHS,
                        total_iterations=iters_per_epoch * EPOCHS,
                        hours_per_epoch=cyc["mean"] * iters_per_epoch / 3600,
                        train_hours_60_epochs=cyc["mean"] * iters_per_epoch * EPOCHS / 3600,
                        basis="mean wall time per iteration (compute + data loading) on a random training sample"),
        memory_train=dict(peak_allocated_mb=max(r["peak_allocated_mb"] for r in timer.rows),
                          peak_reserved_mb=max(r["peak_reserved_mb"] for r in timer.rows)),
        all_losses_finite=all(math.isfinite(r["loss"]) for r in timer.rows))

    # memory stress on the most populated images at the largest resize
    model = runner.model
    sds = copy.deepcopy(train_ds)
    sds["indices"] = dense_idx
    for step in sds["pipeline"]:
        if step["type"] == "RandomResize":
            step["ratio_range"] = (1.5, 1.5)
    loader = Runner.build_dataloader(dict(batch_size=PHYSICAL_BATCH, num_workers=WORKERS, persistent_workers=False,
                                          sampler=dict(type="DefaultSampler", shuffle=False),
                                          dataset=dict(type="RepeatDataset", times=2, dataset=sds)), seed=0)
    runner.optim_wrapper.initialize_count_status(model, 0, ACCUMULATION * 1000)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    finite = True
    for batch in loader:
        out = model.train_step(batch, optim_wrapper=runner.optim_wrapper)
        finite &= math.isfinite(float(out["loss"]))
    res["memory_stress"] = dict(peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                                peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20, all_finite=finite,
                                note="8 most populated training images x 2, resize ratio 1.5, batch 2")

    # inference
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    per_split = {}
    for split, tcfg in test_ds.items():
        tfull = DATASETS.build(copy.deepcopy(tcfg))
        idx = sorted(random.Random(1).sample(range(len(tfull)), N_TEST_PER_SPLIT))
        del tfull
        t = copy.deepcopy(tcfg)
        t["indices"] = idx
        tl = Runner.build_dataloader(dict(batch_size=1, num_workers=WORKERS, persistent_workers=False,
                                          sampler=dict(type="DefaultSampler", shuffle=False), dataset=t), seed=0)
        times, walls, t_prev = [], [], time.perf_counter()
        for batch in tl:
            torch.cuda.synchronize()
            ts = time.perf_counter()
            with torch.no_grad():
                model.test_step(batch)
            torch.cuda.synchronize()
            te = time.perf_counter()
            times.append(te - ts)
            walls.append(te - t_prev)
            t_prev = te
        per_split[split] = dict(forward_s=summary(times[3:]), wall_s=summary(walls[3:]))
    mean_wall = statistics.mean(v["wall_s"]["mean"] for v in per_split.values())
    res["inference"] = dict(per_split=per_split,
                            peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                            peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
                            projected_hours_all_test_images=mean_wall * sum(TEST_IMAGES.values()) / 3600,
                            note="batch 1; forward plus data loading; COCO scoring and result encoding not included")
    res["gpu_state"]["after_inference"] = gpu_state()
    peak = max(res["memory_train"]["peak_allocated_mb"], res["memory_stress"]["peak_allocated_mb"],
               res["inference"]["peak_allocated_mb"])
    res["peak_allocated_mb"] = peak
    res["peak_allocated_fraction"] = peak / res["gpu_total_mb"]
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{m.slug}.json").write_text(json.dumps(res, indent=1, default=float))
    shutil.rmtree(cfg.work_dir, ignore_errors=True)
    print(json.dumps(dict(method=mid, s_per_iter=round(cyc["mean"], 3), data_wait=round(res["timing"]["data_wait_s"]["mean"], 3),
                          train_h_60ep=round(res["projection"]["train_hours_60_epochs"], 1),
                          peak_alloc_gb=round(peak / 1024, 2), peak_frac=round(res["peak_allocated_fraction"], 3),
                          eval_h=round(res["inference"]["projected_hours_all_test_images"], 2))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
