"""Static parity checks for the PIDray MMDetection 3.2 config."""

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmengine.config import Config

DEFAULT_EXPERIMENT_SEED = 42


def main() -> None:
    new_config = PROJECT_ROOT / (
        "projects/proposal_first_pidray_occ/configs/"
        "pidray_r101_attfpn_r0_legacy.py"
    )
    original_config = PROJECT_ROOT / (
        "references/legacy_pidray/configs/"
        "cascade_mask_rcnn_r101_with_R0R1.py"
    )
    paper_config = PROJECT_ROOT / (
        "projects/proposal_first_pidray_occ/configs/"
        "pidray_r101_attfpn_r0_paper_full.py"
    )
    new = Config.fromfile(str(new_config))
    original = runpy.run_path(str(original_config))
    paper = Config.fromfile(str(paper_config))
    model = new.model
    original_model = original["model"]

    assert model.backbone.depth == original_model["backbone"]["depth"] == 101
    assert model.neck.type == "AttFPN"
    assert model.r0_head == {
        "type": "PIDrayR0Head",
        "hidden_dim": 256,
        "threshold": 0.5,
        "loss_weight": 0.1,
        "loss_type": "bce",
    }
    assert model.training_gate == "legacy"
    assert [head.num_classes for head in model.roi_head.bbox_head] == [12, 12, 12]
    assert [head.bbox_coder.target_stds for head in model.roi_head.bbox_head] == [
        head["bbox_coder"]["target_stds"]
        for head in original_model["roi_head"]["bbox_head"]
    ]
    assert new.train_dataloader.dataset.filter_cfg == {
        "filter_empty_gt": False,
        "min_size": 0,
    }
    assert new.train_dataloader.dataset.ann_file == "annotations/train.json"
    assert new.val_dataloader is None
    assert new.val_evaluator is None
    assert new.val_cfg is None
    assert new.test_dataloader.dataset.ann_file == "annotations/test.json"
    assert new.test_dataloader.dataset.data_prefix.img == "test/"
    assert new.test_evaluator.type == "PIDrayPaperMetric"
    assert set(new.test_evaluator.subset_ann_files) == {"easy", "hard", "hidden"}
    assert new.train_dataloader.batch_size == original["data"]["samples_per_gpu"] == 4
    assert new.randomness.seed == DEFAULT_EXPERIMENT_SEED
    assert new.randomness.diff_rank_seed is False
    assert new.default_hooks.progress.type == "TqdmProgressHook"
    assert new.train_pipeline[2].scale == (500, 500)
    assert new.param_scheduler[1].milestones == original["lr_config"]["step"]
    assert new.train_cfg.max_epochs == original["runner"]["max_epochs"]
    assert len(new.metainfo.classes) == 12
    assert new.custom_hooks[0].training_scope == "official_full_train"
    assert paper.train_dataloader.dataset.ann_file == "annotations/train.json"
    assert paper.val_dataloader is None
    assert paper.val_evaluator is None
    assert paper.val_cfg is None
    assert paper.custom_hooks[0].training_scope == "official_full_train"
    assert paper.test_evaluator.type == "PIDrayPaperMetric"
    print("Static PIDray parity contracts: PASS")


if __name__ == "__main__":
    main()
