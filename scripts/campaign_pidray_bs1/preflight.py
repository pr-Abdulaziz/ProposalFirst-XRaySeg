#!/usr/bin/env python
"""Pre-launch check of one resolved PIDray batch-1 config: build, iterate, measure.

Runs a short number of real training iterations from the frozen config (not a
copy), confirms the training loader length, the class count, the annotation
files, finite losses and peak memory, and builds each of the three test loaders.
No checkpoint is written and no metric is computed.

    python scripts/campaign_pidray_bs1/preflight.py <method id> [<seed>] [--iters N]
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.hooks import Hook  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

OUT = C.OUT / "preflight"


class Probe(Hook):
    priority = "VERY_HIGH"

    def __init__(self):
        self.rows = []

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.rows.append(dict(
            loss=float(outputs["loss"]),
            instances=sum(len(s.gt_instances) for s in data_batch["data_samples"]),
            images=len(data_batch["data_samples"]),
            peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20))


def main() -> int:
    C.require_leakage_policy()
    mid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else C.SEEDS[0]
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 60
    fp32 = "--fp32" in sys.argv
    warmup = int(sys.argv[sys.argv.index("--warmup") + 1]) if "--warmup" in sys.argv else None
    m = C.BY_ID[mid]
    cfg_path = C.resolved_config(mid, seed)
    cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet")
    from mmdet.registry import DATASETS

    facts = dict(method=mid, seed=seed, name=m.name, slug=m.slug,
                 resolved_config=str(cfg_path.relative_to(ROOT)),
                 resolved_config_sha256=C.sha256(cfg_path),
                 node=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
                 train_ann=cfg.train_dataloader.dataset["ann_file"],
                 train_dataset_type=cfg.train_dataloader.dataset["type"],
                 batch_size=cfg.train_dataloader["batch_size"],
                 accumulative_counts=cfg.optim_wrapper["accumulative_counts"],
                 max_epochs=cfg.train_cfg["max_epochs"],
                 classes=len(cfg.train_dataloader.dataset["metainfo"]["classes"]))
    problems = []
    if facts["batch_size"] != C.PHYSICAL_BATCH or facts["accumulative_counts"] != C.ACCUMULATION:
        problems.append("batch/accumulation do not match the campaign protocol")
    if facts["max_epochs"] != C.MAX_EPOCHS:
        problems.append("max_epochs does not match the campaign protocol")
    if Path(facts["train_ann"]) != C.train_annotation(m.supervision in ("occ", "conflict")):
        problems.append("training annotation is not the campaign file")

    full = DATASETS.build(copy.deepcopy(dict(cfg.train_dataloader.dataset)))
    facts["train_images"] = len(full)
    facts["empty_train_images"] = sum(
        len(full.get_data_info(i)["instances"]) == 0 for i in range(len(full)))
    if facts["train_images"] != C.train_images():
        problems.append(f"train loader has {facts['train_images']} images, expected {C.train_images()}")
    del full

    facts["test_loaders"] = {}
    for split, paths in cfg.campaign_test_splits.items():
        ds = copy.deepcopy(dict(cfg.test_dataloader.dataset))
        ds["ann_file"] = paths["loader"]
        built = DATASETS.build(ds)
        facts["test_loaders"][split] = dict(images=len(built), loader=paths["loader"],
                                            evaluator=paths["evaluator"])
        if len(built) != C.TEST_IMAGES[split]:
            problems.append(f"test split {split} has {len(built)} images, expected {C.TEST_IMAGES[split]}")
        del built

    # short real training run on the first images, into a scratch work dir
    probe_cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
    ds = copy.deepcopy(dict(probe_cfg.train_dataloader.dataset))
    # A seeded random sample, so the empty/annotated mix matches real training
    # (61.6% of PIDray training images carry no annotation); the first N images
    # in file order are all annotated and would not exercise that path.
    import random as _random
    ds["indices"] = sorted(_random.Random(42).sample(range(facts["train_images"]),
                                                     min(iters * C.PHYSICAL_BATCH,
                                                         facts["train_images"])))
    probe_cfg.train_dataloader.dataset = ds
    probe_cfg.train_dataloader["persistent_workers"] = False
    probe_cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1)
    probe_cfg.default_hooks["checkpoint"] = dict(type="CheckpointHook", interval=10**9, save_last=False)
    probe_cfg.custom_hooks = [h for h in probe_cfg.custom_hooks
                              if h["type"] not in ("CampaignRNGStateHookV2", "UpdateAccountingHook")]
    probe_cfg.work_dir = str(OUT / "scratch" / m.slug)
    # Diagnostic overrides. They never touch the frozen config; the variant used
    # is recorded in the result so a probe run can never be mistaken for the
    # campaign protocol.
    variant = []
    if fp32:
        ow = probe_cfg.optim_wrapper
        probe_cfg.optim_wrapper = dict(
            type="OptimWrapper", optimizer=dict(ow["optimizer"]),
            clip_grad=dict(ow["clip_grad"]), accumulative_counts=ow["accumulative_counts"])
        variant.append("fp32 (AMP disabled)")
    if warmup is not None:
        probe_cfg.param_scheduler[0]["end"] = warmup
        variant.append(f"warmup {warmup} iterations")
    facts["variant"] = variant or ["campaign protocol"]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner = Runner.from_cfg(probe_cfg)
    probe = Probe()
    runner.register_hook(probe)
    facts["parameters"] = dict(
        total=sum(p.numel() for p in runner.model.parameters()),
        trainable=sum(p.numel() for p in runner.model.parameters() if p.requires_grad))
    t0 = time.time()
    runner.train()
    facts["probe"] = dict(
        iterations=len(probe.rows), seconds=time.time() - t0,
        s_per_iter=(time.time() - t0) / max(len(probe.rows), 1),
        all_losses_finite=all(math.isfinite(r["loss"]) for r in probe.rows),
        nonfinite_losses=sum(not math.isfinite(r["loss"]) for r in probe.rows),
        max_finite_loss=max((abs(r["loss"]) for r in probe.rows if math.isfinite(r["loss"])),
                            default=float("nan")),
        instances_seen=sum(r["instances"] for r in probe.rows),
        iterations_without_instances=sum(r["instances"] == 0 for r in probe.rows),
        images_per_iteration=sorted({r["images"] for r in probe.rows}),
        peak_allocated_mb=max(r["peak_allocated_mb"] for r in probe.rows),
        gpu=torch.cuda.get_device_name(0),
        gpu_total_mb=torch.cuda.mem_get_info()[1] / 2**20)
    if not facts["probe"]["all_losses_finite"]:
        problems.append("non-finite loss during the probe")
    if facts["probe"]["images_per_iteration"] != [C.PHYSICAL_BATCH]:
        problems.append(f"iteration saw {facts['probe']['images_per_iteration']} images, expected [1]")
    facts["problems"] = problems
    facts["status"] = "PASS" if not problems else "FAIL"
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if not variant else "__" + "_".join(
        v.split()[0].replace("(", "") for v in variant)
    (OUT / f"{m.slug}{tag}.json").write_text(json.dumps(facts, indent=1, default=float))
    import shutil
    shutil.rmtree(probe_cfg.work_dir, ignore_errors=True)
    print(json.dumps(dict(method=mid, variant=facts["variant"], status=facts["status"], problems=problems,
                          train_images=facts["train_images"],
                          test={k: v["images"] for k, v in facts["test_loaders"].items()},
                          params=facts["parameters"]["total"],
                          iters=facts["probe"]["iterations"],
                          nonfinite=facts["probe"]["nonfinite_losses"],
                          max_loss=round(facts["probe"]["max_finite_loss"], 2),
                          s_per_iter=round(facts["probe"]["s_per_iter"], 3),
                          peak_gb=round(facts["probe"]["peak_allocated_mb"] / 1024, 2))), flush=True)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
