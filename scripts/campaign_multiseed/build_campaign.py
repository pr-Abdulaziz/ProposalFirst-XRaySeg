#!/usr/bin/env python
"""Resolve, audit and hash every configuration of the PIXray multi-seed campaign.

Nothing trains. For every method and seed the historical configuration is loaded,
the shared campaign protocol is checked (fields that must already match are
asserted, never silently overwritten) and the recorded campaign changes are
applied: physical batch, accumulation, epochs and cosine length, 2 workers, warmup measured in images,
seed, work directory and the two campaign hooks. The result is dumped as a
standalone resolved config, reloaded and compared, and hashed.

Then: annotation-policy checks, a side-by-side difference audit of the resolved
configs, the learning-rate sequence at every optimizer-update opportunity, loader
lengths, and hashes of annotations, images, pretrained weights, source and
environment. Output goes to output/<campaign>/protocol/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.update(
    XRAY_BATCH_SIZE_PER_GPU="8", XRAY_EFFECTIVE_BATCH_SIZE="16", XRAY_NUM_WORKERS="2",
    XRAY_PROJECT_ROOT=str(ROOT), PIXRAY_DATA_ROOT=str(ROOT / "data/PIXray"),
    PIXRAYA_ANN_ROOT=str(ROOT / "data/PIXray-A/annotations"),
    SAM_VIT_LARGE_DIR=str(ROOT / "pretrained/sam_vit_large"),
    SAM_VIT_LARGE_CKPT=str(ROOT / "pretrained/sam_vit_large/pytorch_model_new.bin"),
    TORCH_HOME=str(ROOT / "pretrained/torch"))

from mmengine.config import Config  # noqa: E402

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    ACCUMULATION, ANN, BY_ID, CAMPAIGN, EFFECTIVE_BATCH, IMAGE_DIRS, ITERS_PER_EPOCH, MAX_EPOCHS, METHODS,
    MICROBATCHES, OUT, PHYSICAL_BATCH, PRETRAINED, ROOT as CROOT, SEEDS, UPDATES, WARMUP_ITERS,
    method_record, point_seed, resolved_config, run_key, run_order, sha256,
    source_files, source_snapshot, work_dir)

assert CROOT == ROOT
PROT = OUT / "protocol"
RUNNER_KEYS = {
    "model", "work_dir", "train_dataloader", "val_dataloader", "test_dataloader",
    "train_cfg", "val_cfg", "test_cfg", "auto_scale_lr", "optim_wrapper",
    "param_scheduler", "val_evaluator", "test_evaluator", "default_hooks",
    "custom_hooks", "load_from", "resume", "launcher", "env_cfg", "log_processor",
    "log_level", "visualizer", "default_scope", "randomness", "experiment_name",
    "custom_imports"}
PROTOCOL = dict(
    dataset="official PIXray split: 3,560 train / 1,486 test images",
    physical_batch_per_gpu=PHYSICAL_BATCH, world_size=1, accumulative_counts=ACCUMULATION,
    effective_batch=EFFECTIVE_BATCH,
    optimizer="AdamW lr=1e-4 weight_decay=0.05, betas/eps torch defaults (0.9, 0.999) / 1e-8, one parameter group; "
              "learning rate kept from the batch-16 protocol, not rescaled for the effective batch of 4",
    amp="AmpOptimWrapper float16, dynamic GradScaler (defaults)",
    clip_grad="global L2 max_norm=1.0 (after unscaling)",
    warmup=f"LinearLR start_factor=0.001, iteration-based, {WARMUP_ITERS} iterations x {PHYSICAL_BATCH} = "
           f"{WARMUP_ITERS * PHYSICAL_BATCH} images (historical: 50 iterations x 2 = 100 images)",
    cosine=f"CosineAnnealingLR epoch-based, begin=1 end={MAX_EPOCHS} T_max={MAX_EPOCHS} eta_min=1e-7 "
           "(historical end=60 T_max=60; shortened with the schedule at the user's instruction, same shape)",
    max_epochs=MAX_EPOCHS, validation="none",
    checkpoint_rule=f"final epoch_{MAX_EPOCHS}.pth; no best-checkpoint selection",
    train_loader="DefaultSampler shuffle=True, no batch sampler, num_workers=2, persistent workers",
    test_loader="batch 1, num_workers=2, DefaultSampler shuffle=False",
    randomness="randomness.seed=<run seed>, deterministic=False, diff_rank_seed=False, cudnn_benchmark=False",
    point_refine_seed="sha256('<campaign>/point_refine/<seed>')[:4] & 0x7fffffff",
    cuda_launch_blocking="0 for every campaign run (historical tools/train.py forced 1; no effect on computation)",
)


# ---------------------------------------------------------------------------
def tname(t):
    return t if isinstance(t, str) else getattr(t, "__name__", str(t))


def normalize(obj):
    """JSON-comparable form: classes become their names, tuples become lists."""
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, range):
        return list(obj)
    if isinstance(obj, type) or callable(obj):
        return tname(obj)
    if isinstance(obj, float):
        return float(f"{obj:.15g}")
    return obj


def ann_path(ds) -> Path:
    p = Path(ds["ann_file"])
    return p if p.is_absolute() else Path(ds["data_root"]) / p


def all_types(obj, acc=None):
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "type":
                acc.add(tname(v))
            all_types(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            all_types(v, acc)
    return acc


def all_keys(obj, acc=None):
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(k)
            all_keys(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            all_keys(v, acc)
    return acc


def apply_protocol(cfg, m, seed):
    """Assert the inherited protocol, then apply the recorded campaign changes."""
    problems, changes = [], []

    def expect(cond, msg):
        if not cond:
            problems.append(msg)

    ow = cfg.optim_wrapper
    expect(tname(ow["type"]) == "AmpOptimWrapper" and ow.get("dtype") == "float16", "AMP wrapper/dtype")
    expect(normalize(ow["optimizer"]) == dict(type="AdamW", lr=1e-4, weight_decay=0.05),
           f"optimizer {normalize(ow['optimizer'])}")
    expect(normalize(ow.get("clip_grad")) == dict(max_norm=1.0, norm_type=2), "clip_grad")
    expect("loss_scale" not in ow and "paramwise_cfg" not in ow, "loss_scale/paramwise_cfg present")
    ps = cfg.param_scheduler
    expect(len(ps) == 2, "two schedulers")
    w, c = ps[0], ps[1]
    expect(tname(w["type"]) == "LinearLR" and w["start_factor"] == 0.001 and w["by_epoch"] is False
           and w["begin"] == 0 and w["end"] == 50, f"warmup {normalize(w)}")
    expect(tname(c["type"]) == "CosineAnnealingLR" and c["by_epoch"] is True and c["begin"] == 1
           and c["end"] == 60 and c["T_max"] == 60 and math.isclose(c["eta_min"], 1e-7), f"cosine {normalize(c)}")
    expect(normalize(cfg.train_cfg) == dict(type="EpochBasedTrainLoop", max_epochs=60), "train_cfg")
    expect(cfg.get("val_cfg") is None and cfg.get("val_dataloader") is None and cfg.get("val_evaluator") is None,
           "validation loop present")
    expect(cfg.env_cfg.get("cudnn_benchmark") is False, "cudnn_benchmark")
    expect(cfg.get("load_from") is None and cfg.get("resume") is False, "load_from/resume")
    expect("save_best" not in cfg.default_hooks["checkpoint"], "save_best")
    expect(not normalize(cfg.get("auto_scale_lr", {})).get("enable", False), "auto_scale_lr enabled")
    ted = cfg.test_dataloader
    expect(ted["batch_size"] == 1 and ted["num_workers"] == 2
           and normalize(ted["sampler"]) == dict(type="DefaultSampler", shuffle=False), "test loader")

    td = cfg.train_dataloader
    before = dict(batch_size=td.get("batch_size"), num_workers=td.get("num_workers"),
                  sampler=normalize(td.get("sampler")), batch_sampler=normalize(td.get("batch_sampler", "ABSENT")),
                  accumulative_counts=ow.get("accumulative_counts"), warmup_end=w["end"],
                  cosine_end_T_max=(c["end"], c["T_max"]), max_epochs=cfg.train_cfg["max_epochs"])
    td["batch_size"] = PHYSICAL_BATCH
    td["num_workers"] = 2
    td["persistent_workers"] = True
    td["sampler"] = dict(type="DefaultSampler", shuffle=True)
    td["batch_sampler"] = None
    ow["accumulative_counts"] = ACCUMULATION
    w["end"] = WARMUP_ITERS
    c["end"] = MAX_EPOCHS
    c["T_max"] = MAX_EPOCHS
    cfg.train_cfg["max_epochs"] = MAX_EPOCHS
    cfg.randomness = dict(seed=seed, deterministic=False, diff_rank_seed=False)
    cfg.default_hooks["checkpoint"] = dict(type="CheckpointHook", interval=1, max_keep_ckpts=1, save_last=True)
    cfg.work_dir = str(work_dir(m.id, seed))
    hooks = [h for h in (cfg.get("custom_hooks") or [])]
    hooks += [dict(type="CampaignRNGStateHook", run_key=run_key(m.id, seed)),
              dict(type="UpdateAccountingHook")]
    cfg.custom_hooks = hooks
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=EFFECTIVE_BATCH)
    for e in (cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]):
        if "ann_file" in e:
            expect(Path(e["ann_file"]).resolve() == ANN["test"], f"evaluator ann {e['ann_file']}")
            e["ann_file"] = str(ANN["test"])
    if "seed" in cfg.model["roi_head"]:
        cfg.model["roi_head"]["seed"] = point_seed(seed)
        changes.append("model.roi_head.seed = point_seed(seed)")
    changes.append(dict(before=before, after=dict(batch_size=PHYSICAL_BATCH, num_workers=2, sampler="DefaultSampler shuffle=True",
                                                  batch_sampler=None, accumulative_counts=ACCUMULATION,
                                                  warmup_end=WARMUP_ITERS, cosine_end_T_max=(MAX_EPOCHS, MAX_EPOCHS),
                                                  max_epochs=MAX_EPOCHS)))
    return problems, changes


def policy_check(cfg, m):
    """Annotation policy: only 06 (Occ) and 07 (bilayer FCN, user decision) may read occluder annotations."""
    problems, facts = [], {}
    tr, te = ann_path(cfg.train_dataloader["dataset"]), ann_path(cfg.test_dataloader["dataset"])
    ev = [Path(e["ann_file"]) for e in (cfg.test_evaluator if isinstance(cfg.test_evaluator, list)
                                        else [cfg.test_evaluator]) if "ann_file" in e]
    types = all_types(cfg.model) | all_types(cfg.train_dataloader) | all_types(cfg.test_dataloader)
    occ_types = sorted(t for t in types if t.startswith("Oc") or t.startswith("WeightedOc") or "Bilayer" in t
                       or t == "PIXrayOcclusionDataset")
    occ_keys = sorted(k for k in all_keys(cfg.model) if "bo_" in k or k == "loss_mask_bo")
    facts.update(train_annotations=str(tr), test_loader_annotations=str(te),
                 evaluator_annotations=[str(p) for p in ev], occlusion_components=occ_types,
                 occluder_config_keys=occ_keys,
                 train_dataset_type=tname(cfg.train_dataloader["dataset"]["type"]))
    if any(p != ANN["test"] for p in ev) or not ev:
        problems.append("evaluator must score against data/PIXray/annotations/test.json")
    if m.supervision == "original":
        if tr != ANN["train"] or te != ANN["test"]:
            problems.append(f"target-only method reads {tr.name} / {te.name}")
        if m.id == "05" and "OcFeatureAggregator" in occ_types:
            # mmdet/occlu/occlu.py: no layers, forward returns inputs[0]. It adapts the
            # single-output MMPretrain SAM encoder used with LoRA; it reads no occluder
            # data. Its parameter count (0) is verified again in preflight.
            occ_types = [t for t in occ_types if t != "OcFeatureAggregator"]
            facts["inherited_passthrough"] = "OcFeatureAggregator (parameter-free identity)"
        if occ_types or occ_keys:
            problems.append(f"target-only method carries occlusion components {occ_types} {occ_keys}")
    elif m.supervision == "occ":
        if tr != ANN["occ_train"] or te != ANN["occ_test"]:
            problems.append(f"Occ method must read the PIXray-A track, got {tr} / {te}")
        # 06: Occ bilayer SAM decoder; 07: bilayer FCN head (occluder supervision by user decision 2026-09-14).
        expected = {"06": "WeightedOcPrompterAnchorMaskHead", "07": "BilayerFCNMaskHead"}.get(m.id)
        mh = cfg.model["roi_head"]["mask_head"]
        heads = mh if isinstance(mh, (list, tuple)) else [mh]
        if expected is None or any(tname(h["type"]) != expected or "bo_loss_weight" not in h for h in heads):
            problems.append("occluder-supervised method lost its occluder decoder/supervision")
    elif m.supervision == "conflict":
        facts["conflict"] = (f"reads {tr.name} and builds {occ_types}; occluder supervision "
                             "outside the Occ method violates the campaign annotation rule")
    # initialisation
    if m.family == "detector":
        init = normalize(cfg.model["backbone"].get("init_cfg"))
        facts["backbone_init"] = init
        if init != dict(type="Pretrained", checkpoint="torchvision://resnet101"):
            problems.append(f"detector backbone init {init}")
    else:
        paths = set()
        for key in ("backbone", "shared_image_embedding"):
            paths.add(cfg.model[key]["init_cfg"]["checkpoint"])
        paths.add(cfg.model["roi_head"]["mask_head"]["mask_decoder"]["init_cfg"]["checkpoint"])
        facts["sam_init_checkpoints"] = sorted(paths)
        if paths != {str(PRETRAINED["sam_vit_large_converted"])}:
            problems.append(f"SAM init checkpoints {paths}")
        facts["sam_adaptation"] = normalize(cfg.model["backbone"].get("peft_config"))
    return problems, facts


_NAMES = None


def registered_names() -> dict:
    """id(class or function) -> names it is registered under, across every registry scope."""
    global _NAMES
    if _NAMES is None:
        import mmengine.registry as R
        from mmdet.utils import register_all_modules
        import mmdet.occlu  # noqa: F401
        import projects.proposal_first_pidray_occ  # noqa: F401
        register_all_modules(init_default_scope=False)
        _NAMES = {}

        def visit(reg):
            for k, v in reg._module_dict.items():
                _NAMES.setdefault(id(v), []).append(k)
            for child in reg.children.values():
                visit(child)
        for attr in dir(R):
            obj = getattr(R, attr)
            if isinstance(obj, R.Registry) and obj.parent is None:
                visit(obj)
    return _NAMES


def resolves_to(name, target) -> bool:
    import mmdet.registry as DR
    for attr in dir(DR):
        reg = getattr(DR, attr)
        if hasattr(reg, "get") and hasattr(reg, "_module_dict"):
            try:
                if reg.get(name) is target:
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


def to_registered(obj):
    """Replace class/function ``type`` values with names the registries resolve.

    A dumped pure-Python config otherwise stores import paths such as
    'mmdet.models.losses.smooth_l1_loss.L1Loss', which the registry cannot build.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "type" and (not isinstance(v, str) or "." in v):
                target = v
                if isinstance(v, str):
                    import importlib
                    module, _, attr = v.rpartition(".")
                    target = getattr(importlib.import_module(module), attr)
                cands = registered_names().get(id(target), [])
                name = getattr(target, "__name__", None)
                if cands:
                    chosen = name if name in cands else sorted(cands)[0]
                    if not resolves_to(chosen, target):
                        raise ValueError(f"{v!r} -> {chosen!r} resolves to a different object in scope mmdet")
                elif getattr(target, "__module__", "").startswith("mmcv.ops") and name:
                    # Not registry-built: SingleRoIExtractor does getattr(mmcv.ops, name) and
                    # batched_nms looks the NMS op up by name in mmcv.ops.
                    import mmcv.ops
                    if getattr(mmcv.ops, name, None) is not target:
                        raise ValueError(f"{v!r} is not mmcv.ops.{name}")
                    chosen = name
                else:
                    raise ValueError(f"cannot name type {v!r}")
                out[k] = chosen
            else:
                out[k] = to_registered(v)
        return out
    if isinstance(obj, list):
        return [to_registered(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_registered(v) for v in obj)
    return obj


def resolve_one(m, seed):
    cfg = Config.fromfile(str(ROOT / m.historical_config))
    problems, changes = apply_protocol(cfg, m, seed)
    pol_problems, facts = policy_check(cfg, m)
    full = cfg.to_dict()
    removed = sorted(k for k in full if k not in RUNNER_KEYS)
    pruned = to_registered({k: v for k, v in full.items() if k in RUNNER_KEYS})
    dest = resolved_config(m.id, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Resolved config for {run_key(m.id, seed)} ({CAMPAIGN}).\n"
              f"# Source: {m.historical_config}. Generated by scripts/campaign_multiseed/build_campaign.py.\n"
              f"# Do not edit: the campaign manifest records this file's SHA-256.\n")
    text = Config(pruned).pretty_text
    dest.write_text(header + text)
    reloaded = Config.fromfile(str(dest)).to_dict()
    fidelity = normalize(reloaded) == normalize(pruned)
    if not fidelity:
        problems.append("dumped config does not reload identically")
    return dict(config=pruned, problems=problems + pol_problems, changes=changes, facts=facts,
                removed_namespace_keys=removed, path=str(dest.relative_to(ROOT)), sha256=sha256(dest),
                reload_identical=fidelity)


# ---------------------------------------------------------------------------
def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from flatten(obj[k], f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list) and obj and all(isinstance(v, dict) for v in obj):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, json.dumps(obj, sort_keys=True)


IGNORED_IN_DIFF = [r"^work_dir$", r"^randomness\.seed$", r"^custom_hooks\[\d+\]\.run_key$",
                   r"^model\.roi_head\.seed$"]
MUST_MATCH = [r"^optim_wrapper", r"^param_scheduler", r"^train_cfg", r"^val_", r"^env_cfg", r"^randomness",
              r"^load_from$", r"^resume$", r"^auto_scale_lr", r"^train_dataloader\.(batch_size|num_workers|"
              r"persistent_workers|sampler|batch_sampler)", r"^test_dataloader\.(batch_size|num_workers|"
              r"persistent_workers|drop_last|sampler)", r"^default_hooks\.(checkpoint|param_scheduler|logger|timer|"
              r"sampler_seed)", r"^log_processor", r"^custom_hooks", r"^test_evaluator"]


def diff_table(resolved: dict[str, dict]) -> dict:
    flat = {}
    for mid, r in resolved.items():
        cfg = normalize(r["config"])
        for side in ("train_dataloader", "test_dataloader"):
            ds = cfg[side]["dataset"]
            ds["ann_file"] = str(ann_path(ds))
        flat[mid] = dict(flatten(cfg))
    keys = sorted(set().union(*[set(f) for f in flat.values()]))
    rows = {}
    for k in keys:
        if any(re.search(p, k) for p in IGNORED_IN_DIFF):
            continue
        vals = {mid: flat[mid].get(k, "<absent>") for mid in flat}
        if len(set(vals.values())) > 1:
            rows[k] = vals
    return rows


def classify(rows: dict, rules: list[dict]) -> tuple[dict, list]:
    out, unexplained = {}, []
    for k, vals in rows.items():
        if any(re.search(p, k) for p in MUST_MATCH):
            unexplained.append(dict(key=k, reason="protocol field differs", values=vals))
            continue
        hit = next((r for r in rules if re.search(r["pattern"], k)), None)
        if hit is None:
            unexplained.append(dict(key=k, reason="no rationale", values=vals))
            continue
        out[k] = dict(category=hit["category"], rationale=hit["rationale"], values=vals)
    return out, unexplained


def simulate_schedule(cfg, iters_per_epoch: int) -> dict:
    import torch
    from mmengine.registry import PARAM_SCHEDULERS
    import mmengine.optim  # noqa: F401  registers schedulers
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4, weight_decay=0.05)
    scheds = [PARAM_SCHEDULERS.build(dict(dict(s), optimizer=opt)) for s in cfg["param_scheduler"]]
    acc = cfg["optim_wrapper"]["accumulative_counts"]
    epochs = cfg["train_cfg"]["max_epochs"]
    max_iters = epochs * iters_per_epoch
    count, updates, per_epoch, warm_updates = 0, [], [], 0
    for epoch in range(epochs):
        first = last = None
        for _ in range(iters_per_epoch):
            count += 1
            if count % acc == 0 or count == max_iters:
                lr = opt.param_groups[0]["lr"]
                updates.append((count, lr))
                first = lr if first is None else first
                last = lr
                warm_updates += int(lr < 1e-4 * (1 - 1e-9) and epoch == 0)
            for s in scheds:
                if not s.by_epoch:
                    s.step()
        per_epoch.append(dict(epoch=epoch + 1, lr_first_update=first, lr_last_update=last))
        for s in scheds:
            if s.by_epoch:
                s.step()
    return dict(iterations_per_epoch=iters_per_epoch, microbatch_iterations=max_iters,
                accumulative_counts=acc, update_opportunities=len(updates),
                remainder_microbatches=max_iters % acc,
                updates_crossing_epoch_boundary=sum(1 for c, _ in updates if (c - acc) // iters_per_epoch
                                                    != (c - 1) // iters_per_epoch),
                warmup_iterations=cfg["param_scheduler"][0]["end"],
                warmup_images=cfg["param_scheduler"][0]["end"] * cfg["train_dataloader"]["batch_size"],
                updates_below_base_lr_in_warmup=warm_updates,
                first_12_updates=[dict(microbatch=c, lr=lr) for c, lr in updates[:12]],
                last_update=dict(microbatch=updates[-1][0], lr=updates[-1][1]), per_epoch=per_epoch)


def annotation_audit() -> dict:
    def load(p):
        return json.loads(Path(p).read_text())
    out = {}
    for split in ("train", "test"):
        a, b = load(ANN[split]), load(ANN[f"occ_{split}"])
        exact = ("image_id", "category_id", "segmentation", "iscrowd")
        same_order = len(a["annotations"]) == len(b["annotations"])
        field_mismatch = {f: 0 for f in exact}
        id_offsets, max_bbox, max_area, repr_only = {}, 0.0, 0.0, dict(bbox=0, area=0)
        for x, y in zip(a["annotations"], b["annotations"]):
            for f in exact:
                field_mismatch[f] += int(x.get(f) != y.get(f))
            db = max(abs(p - q) for p, q in zip(x["bbox"], y["bbox"]))
            da = abs(x["area"] - y["area"])
            max_bbox, max_area = max(max_bbox, db), max(max_area, da)
            repr_only["bbox"] += int(x["bbox"] != y["bbox"])
            repr_only["area"] += int(x["area"] != y["area"])
            off = y["id"] - x["id"]
            id_offsets[off] = id_offsets.get(off, 0) + 1
        field_mismatch["bbox_beyond_1e-6"] = int(max_bbox > 1e-6)
        field_mismatch["area_beyond_0.05"] = int(max_area > 0.05)
        occ_nonempty = sum(1 for x in b["annotations"] if x.get("bg_object_segmentation"))
        out[split] = dict(images_identical=a["images"] == b["images"],
                          categories_identical=a["categories"] == b["categories"],
                          target_instances_identical=same_order and not any(field_mismatch.values()),
                          target_field_mismatches_in_order=field_mismatch,
                          float_representation_differences=dict(count=repr_only, max_bbox_abs=max_bbox,
                                                                max_area_abs=max_area),
                          annotation_id_offsets_occ_minus_original=id_offsets,
                          annotation_ids_renumbered=set(id_offsets) != {0},
                          instances=len(a["annotations"]), images=len(a["images"]),
                          occluder_masks_nonempty=occ_nonempty,
                          occ_file_info=b.get("info"))
    return out


def image_digests() -> dict:
    digests = {}
    for split, d in IMAGE_DIRS.items():
        for p in sorted(d.iterdir()):
            if p.is_file():
                digests[f"{split}/{p.name}"] = sha256(p)
    agg = hashlib.sha256("\n".join(f"{k} {v}" for k, v in digests.items()).encode()).hexdigest()
    return dict(aggregate_sha256=agg, count=len(digests), files=digests)


def environment() -> dict:
    import importlib
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform())
    for mod in ("torch", "torchvision", "mmengine", "mmcv", "mmdet", "mmpretrain", "numpy", "pycocotools",
                "transformers", "peft", "cv2"):
        try:
            m = importlib.import_module(mod)
            env[mod] = dict(version=getattr(m, "__version__", "?"), path=getattr(m, "__file__", "?"))
        except Exception as exc:  # noqa: BLE001
            env[mod] = dict(error=str(exc))
    import torch
    env["torch_cuda"] = torch.version.cuda
    env["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (PROT / "pip_freeze.txt").write_text(freeze)
    env["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
    return env


def loader_lengths(resolved) -> dict:
    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS
    import mmdet.occlu  # noqa: F401
    import projects.proposal_first_pidray_occ  # noqa: F401
    init_default_scope("mmdet")
    out, seen = {}, {}
    for mid, r in resolved.items():
        ds_cfg = r["config"]["train_dataloader"]["dataset"]
        key = (tname(ds_cfg["type"]), str(ann_path(ds_cfg)))
        if key not in seen:
            seen[key] = len(DATASETS.build(ds_cfg))
        out[mid] = dict(dataset=key[0], annotations=Path(key[1]).name, images=seen[key])
    return out


def main() -> int:
    PROT.mkdir(parents=True, exist_ok=True)
    report = dict(campaign=CAMPAIGN, created=dt.datetime.now().isoformat(timespec="seconds"),
                  protocol=PROTOCOL, seeds=list(SEEDS), run_order=[dict(method=m, seed=s) for m, s in run_order()],
                  methods=[method_record(m) for m in METHODS])

    # 1. registry mapping
    from projects.proposal_first_pidray_occ import registry_seven as reg
    mapping = []
    for m in METHODS:
        r = reg.METHODS.get(m.id)
        ok = (r is None and m.id in ("09", "09M")) or (r is not None and r.config == m.historical_config
                                                         and r.historical_id == m.historical_id)
        mapping.append(dict(id=m.id, name=m.name, config=m.historical_config, registry_config=getattr(r, "config", None),
                            registry_name=getattr(r, "name", "not in registry (09/09M are outside registry_seven)"),
                            historical_id=m.historical_id, config_exists=(ROOT / m.historical_config).is_file(),
                            matches_registry=ok))
    report["registry_mapping"] = mapping

    # 2. resolve every run
    runs, problems, seed42 = {}, [], {}
    for m in METHODS:
        for seed in SEEDS:
            r = resolve_one(m, seed)
            key = run_key(m.id, seed)
            runs[key] = dict(method=m.id, seed=seed, launchable=m.launchable, resolved_config=r["path"],
                             resolved_config_sha256=r["sha256"], work_dir=str(work_dir(m.id, seed)),
                             reload_identical=r["reload_identical"], point_refine_seed=(
                                 point_seed(seed) if "seed" in r["config"]["model"]["roi_head"] else None),
                             removed_namespace_keys=r["removed_namespace_keys"], changes=r["changes"],
                             policy=r["facts"], problems=r["problems"])
            if r["problems"] and m.launchable:
                problems += [f"{key}: {p}" for p in r["problems"]]
            if seed == 42:
                seed42[m.id] = r
        print(f"resolved {m.id} ({len(SEEDS)} seeds)", flush=True)
    report["runs"] = runs

    # 3. differences between methods (seed 42; seeds differ only in ignored keys)
    for m in METHODS:
        per_seed = {s: dict(flatten(normalize(Config.fromfile(str(resolved_config(m.id, s))).to_dict())))
                    for s in SEEDS}
        seed_keys = sorted(k for k in per_seed[42] if len({per_seed[s].get(k) for s in SEEDS}) > 1)
        bad = [k for k in seed_keys if not any(re.search(p, k) for p in IGNORED_IN_DIFF)]
        if bad:
            problems.append(f"{m.id}: seeds differ beyond seed/work_dir/run_key: {bad}")
    approved = {mid: r for mid, r in seed42.items() if BY_ID[mid].launchable}
    rules = json.loads((ROOT / "scripts/campaign_multiseed/difference_rationale.json").read_text())
    rows = diff_table(approved)
    classified, unexplained = classify(rows, rules)
    pair = diff_table({k: approved[k] for k in ("09", "09M")})
    pair_ok = sorted(pair) == ["model.roi_head.point_refine_head.type"]
    if not pair_ok:
        problems.append(f"09 vs 09M differ beyond the head type: {sorted(pair)}")
    held = diff_table({"04": seed42["04"], "07": seed42["07"]})
    (PROT / "fairness_differences.json").write_text(json.dumps(dict(
        classified=classified, unexplained=unexplained, pair_09_09M=pair, historical_07_vs_04=held), indent=1))
    report["fairness"] = dict(differing_keys=len(rows), unexplained=len(unexplained),
                              categories={c: sum(1 for v in classified.values() if v["category"] == c)
                                          for c in sorted({v["category"] for v in classified.values()})},
                              pair_09_09M_only_head_type=pair_ok)
    if unexplained:
        problems.append(f"{len(unexplained)} unexplained configuration differences")

    # 4. schedule and loader
    lengths = loader_lengths(approved | {"07": seed42["07"]})
    report["loader_lengths"] = lengths
    if any(v["images"] != 3560 for v in lengths.values()):
        problems.append(f"loader lengths {lengths}")
    sched = simulate_schedule(normalize(approved["03"]["config"]), ITERS_PER_EPOCH)
    report["schedule"] = sched
    if sched["update_opportunities"] != UPDATES or sched["microbatch_iterations"] != MICROBATCHES:
        problems.append("schedule arithmetic")

    # 5. data, weights, source, environment
    report["annotations"] = {k: dict(path=str(v.relative_to(ROOT)), sha256=sha256(v)) for k, v in ANN.items()}
    report["annotation_identity"] = annotation_audit()
    for split, a in report["annotation_identity"].items():
        if not (a["images_identical"] and a["categories_identical"] and a["target_instances_identical"]):
            problems.append(f"{split}: occlusion track does not preserve the target instances")
    imgs = image_digests()
    (PROT / "image_digests.json").write_text(json.dumps(imgs, indent=0))
    report["images"] = dict(aggregate_sha256=imgs["aggregate_sha256"], count=imgs["count"])
    report["pretrained"] = {k: dict(path=str(v.relative_to(ROOT)), sha256=sha256(v), bytes=v.stat().st_size)
                            for k, v in PRETRAINED.items()}
    dev = json.loads((ROOT / "splits/pixray_dev_v1/manifest.json").read_text())
    report["dev_split"] = dict(status=dev.get("status"), referenced_by_campaign_configs=any(
        "pixray_dev" in Path(r["resolved_config"]).read_text() if False else
        "pixray_dev" in (ROOT / r["resolved_config"]).read_text() for r in runs.values()))
    if report["dev_split"]["status"] != "inactive" or report["dev_split"]["referenced_by_campaign_configs"]:
        problems.append("development split is not inactive")
    snap = source_snapshot()
    (PROT / "source_snapshot.json").write_text(json.dumps(snap, indent=0))
    with tarfile.open(PROT / "source_snapshot.tar.gz", "w:gz") as tar:
        for p in source_files():
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    report["source_snapshot"] = dict(aggregate_sha256=snap["aggregate_sha256"], file_count=snap["file_count"],
                                     tarball_sha256=sha256(PROT / "source_snapshot.tar.gz"))
    report["environment"] = environment()

    report["problems"] = problems
    (PROT / "build_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(dict(problems=problems, fairness=report["fairness"], loader=lengths,
                          schedule={k: sched[k] for k in ("microbatch_iterations", "update_opportunities",
                                                          "remainder_microbatches", "updates_crossing_epoch_boundary",
                                                          "updates_below_base_lr_in_warmup")}), indent=1))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
