"""Validate the exact eleven-experiment PIDray-30 comparison contract."""

from __future__ import annotations

import ast
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config
# This checker validates the pidray30_* configs, so it reads the PIDray-30
# runtime. scripts/run_experiments.py is now only a guard that refuses to run
# without an explicit dataset runtime, and importing it aborted this script.
from scripts.run_pidray30_experiments import (
    EXPECTED_EXPERIMENT_IDS,
    EXPERIMENTS,
    SUPERVISION,
)

CONFIGS = {
    "P1": "projects/proposal_first_pidray_occ/configs/pidray30_p1_cascade_fpn.py",
    "P2": "projects/proposal_first_pidray_occ/configs/pidray30_p2_cascade_attfpn.py",
    "P3": "projects/proposal_first_pidray_occ/configs/pidray30_p3_recall_rpn.py",
    "P4": "projects/proposal_first_pidray_occ/configs/pidray30_p4_cascade_rpn.py",
    "S1": "projects/proposal_first_pidray_occ/configs/pidray30_s1_single_decoder.py",
    "S2": "configs/occlusion/pidray30_s2_occ_bisam.py",
    "S3-P": "projects/proposal_first_pidray_occ/configs/pidray30_s3_proposal_first_bisam.py",
    "S3-MBoost": "projects/proposal_first_pidray_occ/configs/pidray30_s3_mboost.py",
    "E1": "projects/proposal_first_pidray_occ/configs/pidray30_e1_adapter.py",
    "E2": "projects/proposal_first_pidray_occ/configs/pidray30_e2_hard_aware.py",
    "E3": "projects/proposal_first_pidray_occ/configs/pidray30_e3_conditional_bisam.py",
}
ENHANCEMENT_RUNS = {"E1", "E2", "E3"}
EXPECTED_PHYSICAL_BATCH = int(os.getenv("XRAY_BATCH_SIZE_PER_GPU", "2"))
EXPECTED_EFFECTIVE_BATCH = int(os.getenv("XRAY_EFFECTIVE_BATCH_SIZE", "16"))
EXPECTED_WORKERS = int(os.getenv("XRAY_NUM_WORKERS", "4"))
assert EXPECTED_PHYSICAL_BATCH > 0
assert EXPECTED_EFFECTIVE_BATCH % EXPECTED_PHYSICAL_BATCH == 0
EXPECTED_ACCUMULATION = EXPECTED_EFFECTIVE_BATCH // EXPECTED_PHYSICAL_BATCH


def _load(relative_path: str) -> Config:
    return Config.fromfile(str(ROOT / relative_path), import_custom_modules=False)


def _assert_shared_training(name: str, cfg: Config) -> None:
    assert cfg.randomness.seed == 42, name
    assert cfg.randomness.deterministic is False, name
    assert cfg.randomness.diff_rank_seed is False, name
    assert cfg.train_cfg.max_epochs == 60, name
    assert cfg.train_dataloader.batch_size == EXPECTED_PHYSICAL_BATCH, name
    assert cfg.train_dataloader.num_workers == EXPECTED_WORKERS, name
    assert cfg.train_dataloader.persistent_workers is True, name
    assert cfg.train_dataloader.sampler.type == "DefaultSampler", name
    assert cfg.train_dataloader.sampler.shuffle is True, name
    assert cfg.train_dataloader.get("batch_sampler") is None, name
    assert cfg.optim_wrapper.type == "AmpOptimWrapper", name
    assert cfg.optim_wrapper.dtype == "float16", name
    assert cfg.optim_wrapper.accumulative_counts == EXPECTED_ACCUMULATION, name
    assert cfg.optim_wrapper.optimizer.type == "AdamW", name
    assert cfg.optim_wrapper.optimizer.lr == 1e-4, name
    assert cfg.optim_wrapper.optimizer.weight_decay == 0.05, name
    assert cfg.optim_wrapper.clip_grad.max_norm == 1.0, name
    assert cfg.optim_wrapper.clip_grad.norm_type == 2, name
    assert cfg.param_scheduler[0].type == "LinearLR", name
    assert cfg.param_scheduler[0].start_factor == 0.001, name
    assert cfg.param_scheduler[0].begin == 0, name
    assert cfg.param_scheduler[0].end == 50, name
    assert cfg.param_scheduler[1].type == "CosineAnnealingLR", name
    assert cfg.param_scheduler[1].begin == 1, name
    assert cfg.param_scheduler[1].eta_min == 1e-7, name
    assert cfg.param_scheduler[1].end == cfg.train_cfg.max_epochs, name
    assert cfg.param_scheduler[1].T_max == cfg.train_cfg.max_epochs, name
    assert cfg.val_dataloader is None, name
    assert cfg.val_evaluator is None, name
    assert cfg.val_cfg is None, name
    assert cfg.test_dataloader.batch_size == 1, name
    assert cfg.test_dataloader.num_workers == 2, name
    assert cfg.test_dataloader.persistent_workers is True, name
    assert cfg.test_dataloader.drop_last is False, name
    assert cfg.test_dataloader.sampler.type == "DefaultSampler", name
    assert cfg.test_dataloader.sampler.shuffle is False, name
    assert cfg.test_dataloader.dataset.test_mode is True, name
    assert cfg.auto_scale_lr.enable is False, name
    assert cfg.auto_scale_lr.base_batch_size == EXPECTED_EFFECTIVE_BATCH, name
    checkpoint_hook = cfg.default_hooks.checkpoint
    assert checkpoint_hook.interval == 1, name
    assert checkpoint_hook.max_keep_ckpts == 3, name
    assert checkpoint_hook.save_last is True, name
    assert "save_best" not in checkpoint_hook, name
    assert cfg.default_hooks.logger.interval == 50, name
    assert cfg.resume is False, name
    assert cfg.load_from is None, name


def _pipeline_step(cfg: Config, suffix: str, test: bool = False):
    dataloader = cfg.test_dataloader if test else cfg.train_dataloader
    matches = [
        step
        for step in dataloader.dataset.pipeline
        if str(step.type).endswith(suffix)
    ]
    assert len(matches) == 1, (suffix, matches)
    return matches[0]


def _assert_shared_data_protocol(name: str, cfg: Config) -> None:
    flip = _pipeline_step(cfg, "RandomFlip")
    resize = _pipeline_step(cfg, "RandomResize")
    crop = _pipeline_step(cfg, "RandomCrop")
    filtered = _pipeline_step(cfg, "FilterAnnotations")
    test_resize = _pipeline_step(cfg, "Resize", test=True)
    test_pad = _pipeline_step(cfg, "Pad", test=True)
    assert flip.prob == 0.5, name
    assert tuple(resize.scale) == (512, 512), name
    assert tuple(resize.ratio_range) == (0.5, 1.5), name
    assert resize.keep_ratio is True, name
    assert tuple(crop.crop_size) == (512, 512), name
    assert crop.crop_type == "absolute", name
    assert crop.recompute_bbox is True, name
    assert crop.allow_negative_crop is True, name
    assert tuple(filtered.min_gt_bbox_wh) == (1e-5, 1e-5), name
    assert filtered.by_mask is True, name
    assert filtered.keep_empty is True, name
    assert tuple(test_resize.scale) == (512, 512), name
    assert test_resize.keep_ratio is True, name
    assert tuple(test_pad.size) == (512, 512), name

    evaluator = cfg.test_evaluator
    evaluators = evaluator if isinstance(evaluator, list) else [evaluator]
    coco = [item for item in evaluators if item.type == "CocoMetric"]
    assert len(coco) == 1, name
    assert list(coco[0].metric) == ["bbox", "segm", "proposal"], name
    assert coco[0].classwise is True, name


def _has_recall_rpn(cfg: Config) -> bool:
    return (
        list(cfg.model.rpn_head.anchor_generator.scales) == [4, 8]
        and cfg.model.train_cfg.rpn_proposal.nms_pre == 3000
        and cfg.model.train_cfg.rpn_proposal.max_per_img == 3000
        and cfg.model.test_cfg.rpn.nms_pre == 2000
        and cfg.model.test_cfg.rpn.max_per_img == 2000
    )


def _bash_array(source: str, name: str) -> list[str]:
    match = re.search(rf"{re.escape(name)}=\(\s*(.*?)\s*\)", source, re.DOTALL)
    assert match is not None, f"Missing shell array: {name}"
    return shlex.split(match.group(1), comments=True, posix=True)


def main() -> None:
    configs = {}
    for name, path in CONFIGS.items():
        print(f"Loading {name}: {path}", flush=True)
        configs[name] = _load(path)
    for name, cfg in configs.items():
        _assert_shared_training(name, cfg)
        _assert_shared_data_protocol(name, cfg)

    assert set(CONFIGS) == set(EXPERIMENTS)
    assert set(SUPERVISION) == set(EXPERIMENTS)
    assert tuple(CONFIGS) == EXPECTED_EXPERIMENT_IDS
    assert tuple(EXPERIMENTS) == EXPECTED_EXPERIMENT_IDS
    assert len({item.config.resolve() for item in EXPERIMENTS.values()}) == 11
    assert len({item.work_dir_name for item in EXPERIMENTS.values()}) == 11
    assert all(item.config.is_file() for item in EXPERIMENTS.values())
    assert all(item.epochs == 60 for item in EXPERIMENTS.values())
    data_roots = {
        Path(cfg.train_dataloader.dataset.data_root).resolve()
        for cfg in configs.values()
    }
    assert data_roots == {(ROOT / "data/PIDray").resolve()}, data_roots
    for name, cfg in configs.items():
        experiment = EXPERIMENTS[name]
        assert Path(cfg.work_dir).resolve() == (
            ROOT / "work_dirs" / experiment.work_dir_name
        ).resolve(), name
        assert experiment.config.resolve() == (ROOT / CONFIGS[name]).resolve(), name

    assert configs["P1"].model.neck.type == "FPN"
    assert configs["P2"].model.neck.type == "AttFPN"
    assert configs["P3"].model.neck.type == "AttFPN"
    assert configs["P4"].model.neck.type == "AttFPN"
    assert not _has_recall_rpn(configs["P2"])
    for name in ("P3", "S3-P", "S3-MBoost", "E1", "E2", "E3"):
        assert _has_recall_rpn(configs[name]), name

    # P4 changes proposal learning rather than proposal quantity. It inherits
    # P2's final candidate limits and downstream three-stage Cascade RoI head.
    p2 = configs["P2"].model
    p4 = configs["P4"].model
    assert p4.rpn_head.type == "CascadeRPNHead"
    assert p4.rpn_head.num_stages == 2
    assert len(p4.rpn_head.stages) == 2
    assert p4.rpn_head.stages[0].type == "StageCascadeRPNHead"
    assert p4.rpn_head.stages[0].bridged_feature is True
    assert p4.rpn_head.stages[0].with_cls is False
    assert p4.rpn_head.stages[1].type == "StageCascadeRPNHead"
    assert p4.rpn_head.stages[1].adapt_cfg.type == "offset"
    assert p4.rpn_head.stages[1].with_cls is True
    assert p4.train_cfg.rpn_proposal == p2.train_cfg.rpn_proposal
    assert p4.test_cfg.rpn == p2.test_cfg.rpn
    assert len(p4.roi_head.bbox_head) == len(p2.roi_head.bbox_head) == 3

    s1 = configs["S1"].model.roi_head
    assert not s1.use_occluder_decoder
    assert not s1.use_residual_guidance
    assert s1.mask_head.overlap_loss_weight == 0.0

    s2 = configs["S2"].model.roi_head.mask_head
    assert s2.loss_mask.loss_weight == 1.0
    assert s2.loss_mask_bo.loss_weight == 1.0

    s3_p = configs["S3-P"].model.roi_head
    assert s3_p.use_occluder_decoder and s3_p.use_residual_guidance
    assert not s3_p.use_mask_booster
    assert s3_p.mask_head.overlap_loss_weight == 1.0
    s3_mboost = configs["S3-MBoost"].model.roi_head
    assert s3_mboost.use_mask_booster
    assert not s3_mboost.conditional_bisam
    assert s3_mboost.mask_head.overlap_loss_weight == 1.0

    for name in ENHANCEMENT_RUNS:
        model = configs[name].model
        assert "detector_frozen" not in model
        assert not model.sam_encoder_frozen
        assert not model.sam_decoder_frozen
        assert model.roi_head.use_prompt_adapter
    e1 = configs["E1"].model.roi_head
    assert not e1.use_occluder_decoder
    assert not e1.hard_aware_loss
    assert not e1.conditional_bisam
    assert not e1.use_mask_booster
    assert e1.mask_head.overlap_loss_weight == 0.0
    e2 = configs["E2"].model.roi_head
    assert not e2.use_occluder_decoder
    assert e2.hard_aware_loss
    assert not e2.conditional_bisam
    assert e2.mask_head.overlap_loss_weight == 0.0
    e3 = configs["E3"].model.roi_head
    assert e3.use_occluder_decoder and e3.use_residual_guidance
    assert e3.hard_aware_loss and e3.conditional_bisam and e3.use_mask_booster
    assert e3.mask_head.overlap_loss_weight == 1.0

    standard_annotations = {"P1", "P2", "P3", "P4"}
    for name, cfg in configs.items():
        expected = "train.json" if name in standard_annotations else "train_occ.json"
        assert Path(cfg.train_dataloader.dataset.ann_file).name == expected, name
        assert cfg.train_dataloader.dataset.data_prefix.img == "train/", name
        assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False, name
        assert cfg.test_dataloader.dataset.data_prefix.img == "test/", name
        classes = tuple(cfg.train_dataloader.dataset.metainfo.classes)
        assert len(classes) == 12, name
        assert classes == tuple(cfg.test_dataloader.dataset.metainfo.classes), name

        assert set(cfg.test_split_ann_files) == {"easy", "hard", "hidden"}, name
        assert set(cfg.metric_split_ann_files) == {"easy", "hard", "hidden"}, name
        for split in ("easy", "hard", "hidden"):
            dataset_suffix = (
                f"test_{split}.json"
                if name in standard_annotations
                else f"test_{split}_occ.json"
            )
            assert str(cfg.test_split_ann_files[split]).endswith(dataset_suffix), name
            assert str(cfg.metric_split_ann_files[split]).endswith(
                f"test_{split}.json"
            ), name
            assert str(ROOT / "data/PIDray") in str(
                Path(cfg.test_dataloader.dataset.data_root).resolve()
            ), name

    for name in ("S1", "S3-P", "S3-MBoost", "E1", "E2", "E3"):
        model = configs[name].model
        assert Path(model.sam_image_encoder.hf_pretrain_name).is_absolute(), name
        assert Path(
            model.sam_image_encoder.init_cfg.checkpoint
        ).is_absolute(), name

    sources = (
        ROOT / "projects/proposal_first_pidray_occ/models/proposal_first_detector.py",
        ROOT / "projects/proposal_first_pidray_occ/models/proposal_first_roi_head.py",
        ROOT / "scripts/run_pidray30_experiments.py",
    )
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    roi_source = sources[1].read_text(encoding="utf-8")
    for obsolete in ("use_mask_ensemble", "ensemble_weights", "mask_fusion"):
        assert obsolete not in roi_source, obsolete
    assert "self.mask_booster" in roi_source
    assert "self.difficulty_gate" in roi_source
    assert "_hard_aware_mask_loss" in roi_source

    launcher = (ROOT / "scripts/run_pidray30_primary60.sh").read_text(
        encoding="utf-8"
    )
    assert _bash_array(launcher, "experiments") == list(EXPECTED_EXPERIMENT_IDS)
    assert _bash_array(launcher, "configs") == [CONFIGS[name] for name in CONFIGS]
    assert _bash_array(launcher, "work_dirs") == [
        EXPERIMENTS[name].work_dir_name for name in EXPERIMENTS
    ]
    assert "--splits easy hard hidden" in launcher
    assert "check_static_contracts.py" in launcher
    assert "check_hybrid_contracts.py" in launcher
    assert "scripts/smoke_test.py" in launcher
    print("PIDray-30 eleven-experiment contracts: PASS")


if __name__ == "__main__":
    main()
