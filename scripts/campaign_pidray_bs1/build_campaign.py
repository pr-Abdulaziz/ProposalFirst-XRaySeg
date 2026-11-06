#!/usr/bin/env python
"""Resolve, audit and hash every configuration of the PIDray-30 batch-1 campaign.

Nothing trains. For each method the PIXray comparison configuration is loaded
(same architecture object, so the two datasets compare the same models), the
protocol it already carries is asserted rather than silently overwritten, the
data layer is replaced with the PIDray-30 track, and the campaign protocol is
applied: physical batch 1, no accumulation, 50 epochs, cosine length 50, warmup
measured in images, 12 classes, seed 42. Each resolved config is dumped,
reloaded, compared and hashed into output/<campaign>/protocol/.

    python scripts/campaign_pidray_bs1/build_campaign.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.update(
    XRAY_BATCH_SIZE_PER_GPU="8", XRAY_EFFECTIVE_BATCH_SIZE="16", XRAY_NUM_WORKERS="2",
    XRAY_PROJECT_ROOT=str(ROOT), PIXRAY_DATA_ROOT=str(ROOT / "data/PIXray"),
    PIXRAYA_ANN_ROOT=str(ROOT / "data/PIXray-A/annotations"),
    PIDRAY_DATA_ROOT=str(ROOT / "data/PIDray"),
    PIDRAYA_ANN_ROOT=str(ROOT / "data/PIDray-A/annotations"),
    SAM_VIT_LARGE_DIR=str(ROOT / "pretrained/sam_vit_large"),
    SAM_VIT_LARGE_CKPT=str(ROOT / "pretrained/sam_vit_large/pytorch_model_new.bin"),
    TORCH_HOME=str(ROOT / "pretrained/torch"))

from mmengine.config import Config  # noqa: E402

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

PROT = C.OUT / "protocol"
SOURCE_CLASSES = 15
# Precision: fp32 for every method (user instruction, 2026-09-21), so all ten run
# under identical numerics and the AMP loss scaler is removed as a failure mode.
#
# It was method 08 that forced the question. In fp16 its scaler collapsed to 0 at
# iteration 291-558 and the run was permanently dead. In fp32 the same run rides
# out a violent early transient -- loss 2.0 at iteration 50, 4,043 at 250, 13,657
# at 500 -- and then recovers and converges: 1.08 at 1,000 and 0.82 at 3,000,
# with gradient clipping at max_norm 1.0 holding the updates bounded. fp16 cannot
# recover from that transient because the scaler only grows by multiplication, so
# once it reaches zero it stays there.
FP32_ALL = True
HOOK_MODULE = "projects.proposal_first_pidray_occ.hooks.campaign_hooks_v2"
GUARD_MODULE = "projects.proposal_first_pidray_occ.hooks.divergence_guard"


# --------------------------------------------------------------------- helpers
def tname(t):
    return t if isinstance(t, str) else getattr(t, "__name__", str(t))


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return tname(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj


def count_num_classes(obj, value, found=None):
    found = [] if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "num_classes" and v == value:
                found.append(k)
            else:
                count_num_classes(v, value, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            count_num_classes(v, value, found)
    return found


def set_num_classes(obj, old, new):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "num_classes" and v == old:
                obj[k] = new
            else:
                set_num_classes(v, old, new)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            set_num_classes(v, old, new)


def all_types(obj, acc=None):
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "type":
                acc.add(tname(v))
            else:
                all_types(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            all_types(v, acc)
    return acc



_NAMES = None
RUNNER_KEYS = {
    "model", "work_dir", "train_dataloader", "val_dataloader", "test_dataloader",
    "train_cfg", "val_cfg", "test_cfg", "auto_scale_lr", "optim_wrapper",
    "param_scheduler", "val_evaluator", "test_evaluator", "default_hooks",
    "custom_hooks", "load_from", "resume", "launcher", "env_cfg", "log_processor",
    "log_level", "visualizer", "default_scope", "randomness", "experiment_name",
    "custom_imports", "campaign_test_splits"}


def registered_names() -> dict:
    """id(class or function) -> names it is registered under, across every registry scope."""
    global _NAMES
    if _NAMES is None:
        import mmengine.registry as R
        from mmdet.utils import register_all_modules
        import mmdet.occlu  # noqa: F401
        import projects.proposal_first_pidray_occ  # noqa: F401
        import projects.proposal_first_pidray_occ.hooks.campaign_hooks_v2  # noqa: F401
        import projects.proposal_first_pidray_occ.hooks.divergence_guard  # noqa: F401
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
                        raise ValueError(f"{v!r} -> {chosen!r} resolves to a different object")
                elif getattr(target, "__module__", "").startswith("mmcv.ops") and name:
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


# ------------------------------------------------------------- source protocol
def assert_source(cfg, m):
    """The PIXray comparison config must be exactly the one the campaign expects."""
    bad = []

    def expect(cond, msg):
        if not cond:
            bad.append(msg)

    ow = cfg.optim_wrapper
    expect(tname(ow["type"]) == "AmpOptimWrapper" and ow.get("dtype") == "float16", "AMP wrapper/dtype")
    expect(normalize(ow["optimizer"]) == dict(type="AdamW", lr=1e-4, weight_decay=0.05),
           f"optimizer {normalize(ow['optimizer'])}")
    expect(normalize(ow.get("clip_grad")) == dict(max_norm=1.0, norm_type=2), "clip_grad")
    w, c = cfg.param_scheduler
    expect(tname(w["type"]) == "LinearLR" and w["start_factor"] == 0.001 and w["by_epoch"] is False
           and w["begin"] == 0, f"warmup {normalize(w)}")
    expect(tname(c["type"]) == "CosineAnnealingLR" and c["by_epoch"] is True and c["begin"] == 1
           and math.isclose(c["eta_min"], 1e-7), f"cosine {normalize(c)}")
    expect(cfg.get("val_cfg") is None and cfg.get("val_dataloader") is None, "validation loop present")
    expect(cfg.env_cfg.get("cudnn_benchmark") is False, "cudnn_benchmark")
    expect(cfg.get("load_from") is None and cfg.get("resume") is False, "load_from/resume")
    expect(cfg.test_dataloader["batch_size"] == 1, "test batch size")
    expect(len(count_num_classes(cfg.model, SOURCE_CLASSES)) > 0, "no num_classes=15 sites in the model")
    return bad


# ------------------------------------------------------------------ data layer
def test_ann(split: str, occ: bool) -> Path:
    """Annotation the test *loader* reads: the occluder track for 06 and 07."""
    return C.ANN[f"occ_test_{split}"] if occ else C.ANN[f"test_{split}"]


def swap_dataset(cfg, m):
    """Replace the PIXray data layer with the PIDray-30 track."""
    occ = m.supervision in ("occ", "conflict")
    dtype = "PIDrayOcclusionDataset" if occ else "PIDrayDataset"
    before = dict(train_type=tname(cfg.train_dataloader.dataset["type"]),
                  train_ann=str(cfg.train_dataloader.dataset.get("ann_file")),
                  test_ann=str(cfg.test_dataloader.dataset.get("ann_file")),
                  num_classes_sites=len(count_num_classes(cfg.model, SOURCE_CLASSES)))
    train_ann = C.train_annotation(occ)
    cfg.train_dataloader.dataset = dict(
        type=dtype, data_root=str(C.DATA), ann_file=str(train_ann),
        data_prefix=dict(img="train/"), metainfo=dict(classes=list(C.CLASSES)),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=copy.deepcopy(cfg.train_dataloader.dataset["pipeline"]))
    # The resolved config carries the easy split; evaluate_run.py swaps ann_file to
    # score all three released splits with the same loader and pipeline.
    #
    # Occluder-supervised methods must *load* the PIDray-A track even at test time:
    # OcCocoDataset.parse_data_info reads ann["bg_object_segmentation"] directly, so
    # the standard file raises KeyError. Scoring always uses the released standard
    # annotations; assert_pairing guarantees identical image records, so the image
    # ids in the predictions match either file. This mirrors the PIXray campaign,
    # where method 06 loads test_occ.json and is scored against test.json.
    test_pipeline = copy.deepcopy(cfg.test_dataloader.dataset["pipeline"])
    cfg.test_dataloader.dataset = dict(
        type=dtype, data_root=str(C.DATA), ann_file=str(test_ann("easy", occ)),
        data_prefix=dict(img="test/"), metainfo=dict(classes=list(C.CLASSES)),
        test_mode=True, pipeline=test_pipeline)
    evaluators = cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]
    for e in evaluators:
        if "ann_file" in e:
            e["ann_file"] = str(C.ANN["test_easy"])
    set_num_classes(cfg.model, SOURCE_CLASSES, C.NUM_CLASSES)
    cfg.campaign_test_splits = {
        s: dict(loader=str(test_ann(s, occ)), evaluator=str(C.ANN[f"test_{s}"]))
        for s in C.TEST_SPLITS}
    after = dict(train_type=dtype, train_ann=str(train_ann),
                 test_loader_ann=str(test_ann("easy", occ)),
                 test_evaluator_ann=str(C.ANN["test_easy"]),
                 remaining_num_classes_15=len(count_num_classes(cfg.model, SOURCE_CLASSES)),
                 num_classes_12_sites=len(count_num_classes(cfg.model, C.NUM_CLASSES)))
    return dict(before=before, after=after), occ


# -------------------------------------------------------------------- protocol
def apply_protocol(cfg, m, seed):
    td, ow = cfg.train_dataloader, cfg.optim_wrapper
    w, c = cfg.param_scheduler
    before = dict(batch_size=td.get("batch_size"), num_workers=td.get("num_workers"),
                  accumulative_counts=ow.get("accumulative_counts"), warmup_end=w["end"],
                  cosine_end_T_max=(c["end"], c["T_max"]), max_epochs=cfg.train_cfg["max_epochs"])
    td["batch_size"] = C.PHYSICAL_BATCH
    td["num_workers"] = 2
    td["persistent_workers"] = True
    td["sampler"] = dict(type="DefaultSampler", shuffle=True)
    td["batch_sampler"] = None
    ow["accumulative_counts"] = C.ACCUMULATION
    w["end"] = C.WARMUP_ITERS
    c["end"] = C.MAX_EPOCHS
    c["T_max"] = C.MAX_EPOCHS
    cfg.train_cfg["max_epochs"] = C.MAX_EPOCHS
    cfg.randomness = dict(seed=seed, deterministic=False, diff_rank_seed=False)
    cfg.default_hooks["checkpoint"] = dict(type="CheckpointHook", interval=1, max_keep_ckpts=1, save_last=True)
    cfg.work_dir = str(C.work_dir(m.id, seed))
    cfg.custom_hooks = list(cfg.get("custom_hooks") or []) + [
        dict(type="CampaignRNGStateHookV2", run_key=C.run_key(m.id, seed)),
        dict(type="UpdateAccountingHook"),
        # Added after the batch-1 attempt diverged silently for 50 hours.
        dict(type="DivergenceGuardHook", patience=500)]
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=C.EFFECTIVE_BATCH)
    if FP32_ALL:
        ow_amp = cfg.optim_wrapper
        cfg.optim_wrapper = dict(
            type="OptimWrapper", optimizer=dict(ow_amp["optimizer"]),
            clip_grad=dict(ow_amp["clip_grad"]),
            accumulative_counts=ow_amp["accumulative_counts"])
    # Register the device-safe RNG hook without touching the PIXray campaign's
    # frozen projects/ tree: a new module, imported by the config itself.
    imports = list(cfg.custom_imports["imports"])
    for module in (HOOK_MODULE, GUARD_MODULE):
        if module not in imports:
            imports.append(module)
    cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)
    if "seed" in cfg.model["roi_head"]:
        cfg.model["roi_head"]["seed"] = C.point_seed(seed)
    after = dict(batch_size=C.PHYSICAL_BATCH, num_workers=2, accumulative_counts=C.ACCUMULATION,
                 warmup_end=C.WARMUP_ITERS, cosine_end_T_max=(C.MAX_EPOCHS, C.MAX_EPOCHS),
                 max_epochs=C.MAX_EPOCHS)
    return dict(before=before, after=after)


def policy_check(cfg, m, occ):
    problems, facts = [], {}
    tr = Path(cfg.train_dataloader.dataset["ann_file"])
    facts["train_annotations"] = str(tr)
    facts["train_dataset_type"] = tname(cfg.train_dataloader.dataset["type"])
    facts["test_annotations"] = cfg.campaign_test_splits
    types = all_types(cfg.model)
    occ_types = sorted(t for t in types if t.startswith("Oc") or t.startswith("WeightedOc")
                       or "Bilayer" in t)
    facts["occlusion_components"] = occ_types
    if tr != C.train_annotation(occ):
        problems.append(f"train annotation {tr} is not the campaign file for this method")
    if m.supervision == "original":
        if occ:
            problems.append("target-only method marked as occluder-supervised")
        if "OcFeatureAggregator" in occ_types and m.id == "05":
            occ_types = [t for t in occ_types if t != "OcFeatureAggregator"]
            facts["inherited_passthrough"] = "OcFeatureAggregator (parameter-free identity)"
        if occ_types:
            problems.append(f"target-only method carries occlusion components {occ_types}")
    elif m.supervision == "occ":
        expected = {"06": "WeightedOcPrompterAnchorMaskHead", "07": "BilayerFCNMaskHead"}.get(m.id)
        mh = cfg.model["roi_head"]["mask_head"]
        heads = mh if isinstance(mh, (list, tuple)) else [mh]
        if expected is None or any(tname(h["type"]) != expected or "bo_loss_weight" not in h for h in heads):
            problems.append("occluder-supervised method lost its occluder decoder/supervision")
    for split, paths in cfg.campaign_test_splits.items():
        if Path(paths["evaluator"]) != C.ANN[f"test_{split}"]:
            problems.append(f"test split {split} is scored against {paths['evaluator']}, "
                            "not the released standard annotations")
        if Path(paths["loader"]) != test_ann(split, occ):
            problems.append(f"test split {split} loader reads {paths['loader']}")
    loader = Path(cfg.test_dataloader.dataset["ann_file"])
    if loader != test_ann("easy", occ):
        problems.append(f"test loader annotation {loader} does not match the supervision track")
    return problems, facts


def simulate_schedule(cfg, iters_per_epoch: int) -> dict:
    """Learning rate at the first and last update of every epoch, from the resolved schedulers."""
    w, c = cfg.param_scheduler
    base, lrs = 1e-4, []
    for epoch in range(C.MAX_EPOCHS):
        cos = base if epoch < c["begin"] else (
            c["eta_min"] + (base - c["eta_min"]) * (1 + math.cos(math.pi * epoch / c["T_max"])) / 2)
        first_it, last_it = epoch * iters_per_epoch, (epoch + 1) * iters_per_epoch - 1
        def with_warmup(it, lr):
            if it >= w["end"]:
                return lr
            f = w["start_factor"] + (1 - w["start_factor"]) * it / w["end"]
            return lr * f
        lrs.append(dict(epoch=epoch + 1, lr_first=with_warmup(first_it, cos),
                        lr_last=with_warmup(last_it, cos)))
    return dict(updates_total=iters_per_epoch * C.MAX_EPOCHS, iters_per_epoch=iters_per_epoch,
                warmup_iterations=C.WARMUP_ITERS, per_epoch=lrs)


def environment() -> dict:
    import torch
    return dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, node=os.uname().nodename,
                git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip() or None)


def main() -> int:
    C.require_leakage_policy()
    gate = json.loads((C.AUDIT / "gate_verification.json").read_text())
    if gate["status"] != "PASS":
        raise SystemExit("ABORT: leakage gate is not PASS")
    PROT.mkdir(parents=True, exist_ok=True)
    iters = C.iters_per_epoch()
    report, problems = {}, []
    for mid, seed in C.run_order():
        m = C.BY_ID[mid]
        cfg = Config.fromfile(str(ROOT / m.historical_config), import_custom_modules=True)
        src_problems = assert_source(cfg, m)
        data_change, occ = swap_dataset(cfg, m)
        proto_change = apply_protocol(cfg, m, seed)
        pol_problems, facts = policy_check(cfg, m, occ)
        full = cfg.to_dict()
        removed = sorted(k for k in full if k not in RUNNER_KEYS)
        pruned = to_registered({k: v for k, v in full.items() if k in RUNNER_KEYS})
        out = C.resolved_config(mid, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# Resolved config for {C.run_key(mid, seed)} ({C.CAMPAIGN}).\n"
                  f"# Source: {m.historical_config}. Generated by "
                  f"scripts/campaign_pidray_bs1/build_campaign.py.\n"
                  f"# Do not edit: the campaign manifest records this file's SHA-256.\n")
        out.write_text(header + Config(pruned).pretty_text)
        reloaded = Config.fromfile(str(out), import_custom_modules=True)
        identical = normalize(reloaded.to_dict()) == normalize(pruned)
        # The dump must also build: a type dumped as an import path would not.
        import copy as _copy
        from mmengine.registry import init_default_scope
        from mmdet.registry import MODELS
        init_default_scope("mmdet")
        try:
            MODELS.build(_copy.deepcopy(reloaded.model))
            builds = True
        except Exception as exc:  # noqa: BLE001
            builds = False
            src_problems.append(f"reloaded model does not build: {exc}")
        entry = dict(method=mid, seed=seed, source_config=m.historical_config,
                     source_config_sha256=C.sha256(ROOT / m.historical_config),
                     resolved_config=str(out.relative_to(ROOT)),
                     resolved_config_sha256=C.sha256(out),
                     reload_identical=identical, model_builds_from_dump=builds,
                     precision="fp32" if FP32_ALL else "AMP float16",
                     removed_namespace_keys=removed, supervision=m.supervision,
                     data_layer=data_change, protocol=proto_change, policy=facts,
                     problems=src_problems + pol_problems + ([] if identical else ["reload differs"]))
        report[C.run_key(mid, seed)] = entry
        problems += [f"{mid}/{seed}: {p}" for p in entry["problems"]]
        print(f"{mid:>3} seed {seed}: {out.name} "
              f"{'OK' if not entry['problems'] else 'PROBLEMS ' + str(entry['problems'])}", flush=True)

    schedule = simulate_schedule(Config.fromfile(str(C.resolved_config("01", 42))), iters)
    summary = dict(
        campaign=C.CAMPAIGN, built=__import__("datetime").datetime.now().isoformat(timespec="seconds"),
        dataset=dict(name="PIDray-30", train_annotation=str(C.train_annotation(False)),
                     train_images=C.train_images(), released_train_images=C.RELEASED_TRAIN_IMAGES,
                     leakage_policy=C.LEAKAGE_POLICY, test_images=C.TEST_IMAGES,
                     classes=len(C.CLASSES)),
        protocol=dict(physical_batch=C.PHYSICAL_BATCH, accumulation=C.ACCUMULATION,
                      effective_batch=C.EFFECTIVE_BATCH, epochs=C.MAX_EPOCHS,
                      iterations_per_epoch=iters, total_iterations=iters * C.MAX_EPOCHS,
                      warmup_iterations=C.WARMUP_ITERS, seeds=list(C.SEEDS),
                      optimizer="AdamW lr=1e-4 wd=0.05",
                      precision="fp32 (AMP disabled for every method)",
                      clip_grad="max_norm 1.0", validation="disabled",
                      checkpoint="final epoch (interval 1, max_keep_ckpts 1)"),
        environment=environment(), runs=report, schedule=schedule, problems=problems)
    (PROT / "build_report.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(dict(runs=len(report), problems=problems,
                          iterations_per_epoch=iters, total_iterations=iters * C.MAX_EPOCHS), indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
