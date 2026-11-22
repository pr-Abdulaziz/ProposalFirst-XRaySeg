#!/usr/bin/env python
"""Measure, per development arm, the quantities a cost projection needs.

Three different things get called "runtime" and they are not interchangeable:

  * **microbatch step time** -- one forward, one backward, and the amortised
    share of an optimizer step, over ``physical_batch_size`` images. This is
    what is measured here.
  * **optimizer updates** -- one per ``accumulative_counts`` microbatches. A
    60-epoch schedule is defined in epochs, so the update count is derived, not
    chosen, and quoting updates as if they were steps understates the work by
    the accumulation factor.
  * **projected full-run cost** -- microbatch time times the microbatch count,
    plus validation. It is arithmetic on a measured rate, not a measurement:
    it excludes dataloader startup, checkpoint writes, epoch boundaries, any
    contention with other jobs on the node, and any drift over hours of running.

Nothing here is a full run and nothing here should be reported as one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402

register_all_modules(init_default_scope=True)

C = "projects/proposal_first_pidray_occ/configs"
ARMS = {
    "A": ("Method 03, no refinement", f"{C}/pixray_dev_a_cascade_attfpn.py"),
    "B": ("matched MLP control", f"{C}/pixray_dev_b_point_refine_mlp.py"),
    "C": ("transformer (Method 09)", f"{C}/pixray_dev_c_point_refine_transformer.py"),
}


def total_loss(losses):
    out = 0
    for key, value in losses.items():
        if "loss" not in key:
            continue
        out = out + (sum(v.sum() for v in value)
                     if isinstance(value, (list, tuple)) else value.sum())
    return out


def measure(path, warmup, timed, val_warmup, val_timed):
    cfg = Config.fromfile(str(ROOT / path))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.work_dir = "/tmp/m09runtime"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    accumulation = cfg.optim_wrapper["accumulative_counts"]
    batch_size = cfg.train_dataloader["batch_size"]

    optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)
    optim_wrapper.initialize_count_status(model, 0, (warmup + timed))
    model.train()

    loader = iter(runner.train_dataloader)

    def one_step():
        nonlocal loader
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(runner.train_dataloader)
            batch = next(loader)
        data = model.data_preprocessor(batch, True)
        with optim_wrapper.optim_context(model):
            loss = total_loss(model.loss(data["inputs"], data["data_samples"]))
        optim_wrapper.update_params(loss)

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(timed):
        one_step()
    torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - start
    train_peak = torch.cuda.max_memory_allocated() / 2 ** 20

    model.eval()
    val_loader = runner.val_dataloader
    iterator = iter(val_loader)
    seen = 0

    def one_image():
        nonlocal iterator, seen
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            batch = next(iterator)
        with torch.no_grad():
            data = model.data_preprocessor(batch, False)
            out = model.predict(data["inputs"], data["data_samples"])
        seen += len(out)

    for _ in range(val_warmup):
        one_image()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    seen = 0
    start = time.perf_counter()
    for _ in range(val_timed):
        one_image()
    torch.cuda.synchronize()
    val_elapsed = time.perf_counter() - start
    val_peak = torch.cuda.max_memory_allocated() / 2 ** 20

    result = dict(
        config=path,
        physical_batch_size=batch_size,
        accumulative_counts=accumulation,
        effective_batch_size=batch_size * accumulation,
        train_microbatches_timed=timed,
        train_warmup_microbatches=warmup,
        train_seconds=round(train_elapsed, 3),
        seconds_per_microbatch=round(train_elapsed / timed, 5),
        seconds_per_optimizer_update=round(train_elapsed / timed * accumulation, 5),
        images_per_second_train=round(timed * batch_size / train_elapsed, 3),
        train_peak_mib=round(train_peak, 1),
        val_images_timed=seen,
        val_seconds=round(val_elapsed, 3),
        seconds_per_val_image=round(val_elapsed / max(seen, 1), 5),
        val_peak_mib=round(val_peak, 1),
    )
    del model, runner
    torch.cuda.empty_cache()
    return result, cfg


def project(result, train_images, epochs, val_images, val_interval):
    batch = result["physical_batch_size"]
    microbatches_per_epoch = -(-train_images // batch)       # drop_last is False
    microbatches = microbatches_per_epoch * epochs
    updates = microbatches // result["accumulative_counts"]
    train_seconds = microbatches * result["seconds_per_microbatch"]
    validations = epochs // val_interval
    val_seconds = validations * val_images * result["seconds_per_val_image"]
    return dict(
        train_images=train_images, epochs=epochs,
        microbatches_per_epoch=microbatches_per_epoch,
        total_microbatches=microbatches,
        total_optimizer_updates=updates,
        validations=validations,
        projected_train_hours=round(train_seconds / 3600, 3),
        projected_val_hours=round(val_seconds / 3600, 3),
        projected_total_hours=round((train_seconds + val_seconds) / 3600, 3),
        basis="seconds_per_microbatch and seconds_per_val_image measured on this "
              "node; excludes dataloader startup, checkpoint writes, epoch "
              "boundaries, COCO evaluation, logging and node contention",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--val-warmup", type=int, default=10)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--out",
                        default="output/dev_campaign/2026-09-12/runtime_measurements.json")
    args = parser.parse_args()

    import socket
    payload = dict(
        node=socket.gethostname(),
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        note="Single process, world_size=1. Multi-rank DDP is UNTESTED in this "
             "project; no multi-GPU figure is measured or projected here.",
        arms={},
    )
    for arm, (label, path) in ARMS.items():
        print(f"\n=== arm {arm}: {label} ===", flush=True)
        result, cfg = measure(path, args.warmup, args.steps,
                              args.val_warmup, args.val_steps)
        result["label"] = label
        result["projection"] = project(
            result, train_images=3026, epochs=cfg.train_cfg["max_epochs"],
            val_images=534, val_interval=cfg.train_cfg["val_interval"])
        payload["arms"][arm] = result
        print(f"  measured  {result['seconds_per_microbatch']:.4f} s/microbatch "
              f"({result['physical_batch_size']} images), "
              f"{result['seconds_per_optimizer_update']:.4f} s/optimizer update")
        print(f"  measured  {result['seconds_per_val_image']:.4f} s/val image, "
              f"train peak {result['train_peak_mib']:.0f} MiB, "
              f"val peak {result['val_peak_mib']:.0f} MiB")
        p = result["projection"]
        print(f"  projected {p['total_microbatches']:,} microbatches -> "
              f"{p['total_optimizer_updates']:,} updates, "
              f"{p['projected_train_hours']:.2f} h train + "
              f"{p['projected_val_hours']:.2f} h val = "
              f"{p['projected_total_hours']:.2f} h")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    total = sum(a["projection"]["projected_total_hours"] for a in payload["arms"].values())
    print(f"\nthree arms, projected sum: {total:.2f} GPU-h (projection, not a measurement)")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
