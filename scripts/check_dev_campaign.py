#!/usr/bin/env python
"""Focused integration checks for the three PIXray development arms.

Run against the current source, after the point-sampling memory fix and after
the ``_build_mixer`` hook. It re-verifies the properties that the ten Method 09
checkpoints established, on the configs that will actually be launched, and
adds the checks the development campaign introduces: the split wiring, the
matched control, and the structural impossibility of reading the official test
split from a development run.

Nothing here trains to convergence and nothing is written to a work_dir that a
campaign would later use.
"""
from __future__ import annotations

import copy
import json
import sys
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
    "A": f"{C}/pixray_dev_a_cascade_attfpn.py",
    "B": f"{C}/pixray_dev_b_point_refine_mlp.py",
    "C": f"{C}/pixray_dev_c_point_refine_transformer.py",
}
EXPECTED = {"B": ("PointRefineMLPHead", 458_127), "C": ("PointRefineHead", 458_383)}

results = []


def check(label, condition, extra=""):
    results.append((bool(condition), label, extra))
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + extra if extra else ''}")


def build(path, **override):
    cfg = Config.fromfile(str(ROOT / path))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.work_dir = "/tmp/m09devcheck"
    cfg.train_dataloader["num_workers"] = 0
    cfg.train_dataloader["persistent_workers"] = False
    cfg.val_dataloader["num_workers"] = 0
    cfg.val_dataloader["persistent_workers"] = False
    for key, value in override.items():
        cfg.model["roi_head"][key] = value
    return cfg, Runner.from_cfg(cfg)


def total_loss(losses):
    out = 0
    for key, value in losses.items():
        if "loss" not in key:
            continue
        out = out + (sum(v.sum() for v in value)
                     if isinstance(value, (list, tuple)) else value.sum())
    return out


print("=== split wiring (shared by all three arms) ===")
manifest = json.loads((ROOT / "splits/pixray_dev_v1/manifest.json").read_text())
cfgA, runnerA = build(ARMS["A"])
train_ds, val_ds = runnerA.train_dataloader.dataset, runnerA.val_dataloader.dataset
check("dev-train dataset has 3026 images", len(train_ds) == 3026, f"{len(train_ds)}")
check("dev-val dataset has 534 images", len(val_ds) == 534, f"{len(val_ds)}")
train_ids = {train_ds.get_data_info(i)["img_id"] for i in range(len(train_ds))}
val_ids = {val_ds.get_data_info(i)["img_id"] for i in range(len(val_ds))}
check("no dev-val image id is reachable from the training loader",
      not (train_ids & val_ids), f"{len(train_ids & val_ids)} shared")
check("loader ids match the manifest",
      train_ids == set(manifest["dev_train_image_ids"])
      and val_ids == set(manifest["dev_val_image_ids"]))
check("manifest certified", manifest["certified"] is True)
train_paths = {Path(train_ds.get_data_info(i)["img_path"]).parts[-2]
               for i in range(len(train_ds))}
val_paths = {Path(val_ds.get_data_info(i)["img_path"]).parts[-2]
             for i in range(len(val_ds))}
check("every dev image is read from data/PIXray/train/",
      train_paths == val_paths == {"train"}, f"{train_paths | val_paths}")
check("no test loader, evaluator or loop exists in a dev config",
      cfgA.test_dataloader is None and cfgA.test_evaluator is None
      and cfgA.test_cfg is None)
try:
    runnerA.test_dataloader
    built_test = True
except Exception as error:
    built_test = False
    reason = f"{type(error).__name__}"
check("runner cannot build a test dataloader", not built_test,
      reason if not built_test else "it built one")
check("all arms start from ImageNet, not a full-training checkpoint",
      all(Config.fromfile(str(ROOT / p)).load_from is None for p in ARMS.values()))

batchA = next(iter(runnerA.train_dataloader))
modelA = runnerA.model.cuda().eval()
dataA = modelA.data_preprocessor(copy.deepcopy(batchA), True)
valbatch = next(iter(runnerA.val_dataloader))

for arm in ("B", "C"):
    expected_type, expected_params = EXPECTED[arm]
    print(f"\n=== arm {arm}: {expected_type} ===")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg, runner = build(ARMS[arm])
    model = runner.model.cuda().train()

    heads = [m for m in model.modules()
             if type(m).__name__ in ("PointRefineHead", "PointRefineMLPHead")]
    check("exactly one shared refinement head", len(heads) == 1, f"{len(heads)}")
    head = heads[0]
    check(f"head class is {expected_type}", type(head).__name__ == expected_type)
    count = sum(p.numel() for p in head.parameters())
    check(f"refinement parameters = {expected_params:,}", count == expected_params,
          f"{count:,}")

    torch.manual_seed(42)
    data = model.data_preprocessor(copy.deepcopy(batchA), True)
    losses = model.loss(data["inputs"], data["data_samples"])
    loss_keys = sorted(k for k in losses if "point" in k and "loss" in k)
    diag_keys = sorted(k for k in losses if "point" in k and "loss" not in k)
    check("one summed point loss per cascade stage", len(loss_keys) == 3, str(loss_keys))
    check("diagnostics carry no 'loss' in their key",
          diag_keys and all("loss" not in k for k in diag_keys), f"{len(diag_keys)} keys")
    check("point losses finite", all(torch.isfinite(losses[k]).all() for k in loss_keys))

    raw = {}
    original = head.loss_and_refine

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        raw[kwargs.get("stage", 0)] = float(out["loss_point"].detach())
        return out

    head.loss_and_refine = spy
    torch.manual_seed(42)
    weighted = model.loss(data["inputs"], data["data_samples"])
    head.loss_and_refine = original
    weights = cfg.model["roi_head"]["stage_loss_weights"]
    for stage in range(3):
        if stage in raw and f"s{stage}.loss_point" in weighted:
            got, want = float(weighted[f"s{stage}.loss_point"]), raw[stage] * weights[stage]
            check(f"stage {stage} weight applied exactly once (w={weights[stage]})",
                  abs(got - want) < 1e-4,
                  f"raw {raw[stage]:.5f} -> {got:.5f}, expected {want:.5f}")

    model.zero_grad()
    total_loss(losses).backward()
    level = head.embedding.level_embedding.grad
    check("both refinement-level embeddings receive gradient",
          level is not None and level[0].abs().sum() > 0 and level[1].abs().sum() > 0,
          f"l0={level[0].abs().sum():.3e} l1={level[1].abs().sum():.3e}"
          if level is not None else "no grad")
    mixer = sum(p.grad.abs().sum() for p in head.transformer.parameters()
                if p.grad is not None)
    check("mixer receives gradient", mixer > 0, f"{mixer:.3e}")
    backbone = sum(p.grad.abs().sum() for p in model.backbone.parameters()
                   if p.grad is not None)
    check("backbone receives gradient", backbone > 0, f"{backbone:.3e}")
    check("frozen stem and layer1 stay frozen",
          all(not p.requires_grad for n, p in model.backbone.named_parameters()
              if n.startswith(("conv1", "bn1", "layer1"))))

    # Inference at the real detection budget: this is the path that ran out of
    # memory before the per-image feature sampler.
    model.eval()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        vd = model.data_preprocessor(copy.deepcopy(valbatch), False)
        out = model.predict(vd["inputs"], vd["data_samples"])
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    detections = sum(len(s.pred_instances) for s in out)
    shapes_ok = all(tuple(s.pred_instances.masks.shape[-2:]) == tuple(s.ori_shape)
                    for s in out if len(s.pred_instances))
    check("inference completes at the full point budget", True,
          f"{detections} detections, peak {peak:.0f} MiB")
    check("masks are pasted at original resolution", shapes_ok)

    del model, runner
    torch.cuda.empty_cache()

print("\n=== arm C with refinement disabled must equal arm A ===")
_, runner_off = build(ARMS["C"], refine_enabled=False)
model_off = runner_off.model.cuda().eval()
missing, unexpected = model_off.load_state_dict(modelA.state_dict(), strict=False)
check("only refinement keys are missing when loading arm A weights",
      missing and all("point_refine_head" in k for k in missing) and not unexpected,
      f"{len(missing)} missing, {len(unexpected)} unexpected")
with torch.no_grad():
    d1 = model_off.data_preprocessor(copy.deepcopy(valbatch), False)
    d2 = modelA.data_preprocessor(copy.deepcopy(valbatch), False)
    a = model_off.predict(d1["inputs"], d1["data_samples"])
    b = modelA.predict(d2["inputs"], d2["data_samples"])
check("refine_enabled=False is bit-identical to arm A",
      all(torch.equal(x.pred_instances.masks, y.pred_instances.masks)
          for x, y in zip(a, b)))

failed = [label for ok, label, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:")
    for label in failed:
        print("  -", label)
print("ALL CHECKS PASSED" if not failed else "SOME CHECKS FAILED")
raise SystemExit(1 if failed else 0)
