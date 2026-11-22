#!/usr/bin/env python
"""Pre-launch checks for the detached-MLP diagnostic run (CPU, nothing trained or written back).

1. Initialisation: the original and the detached MLP model, built under the run seed (42) and
   initialised by the same protocol (ImageNet ResNet-101 backbone + random heads), must have identical
   initial weights, and those must differ from the trained final checkpoint.
2. Data order: the first training batches under both configs (same seed) must be identical
   (image files, flips, shapes, instance counts, input tensors).
3. Category mapping: the analysis scripts' label -> category mapping must equal the dataset objects'
   actual ``cat2label`` for every annotation file used in training and evaluation (PIDray, PIXray).

    python scripts/campaign_pidray_bs1/prelaunch_gradiso.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "pretrained/torch"))
from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import Runner, set_random_seed  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_pidray_bs1.coco_resample_mapped import label_to_category  # noqa: E402

ORIG = ROOT / "output/pidray_bs2_e10_2026-09-22/protocol/resolved_configs/m09m_point_refine_mlp/seed_42.py"
DET = ROOT / "output/pidray_diag_gradiso_2026-09-26/protocol/resolved_configs/m09md_point_refine_mlp_detached/seed_42.py"
CKPT = ROOT / "work_dirs/pidray_bs2_e10_2026-09-22/m09m_point_refine_mlp/seed_42/epoch_10.pth"
OUT = ROOT / "output/pidray_diag_gradiso_2026-09-26/verification/prelaunch_gradiso.json"
torch.set_num_threads(4)


def load(path):
    cfg = Config.fromfile(str(path))
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet")
    return cfg


def init_state(cfg):
    from mmdet.registry import MODELS
    set_random_seed(cfg.randomness["seed"], deterministic=cfg.randomness["deterministic"])
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model.state_dict()


def main() -> int:
    rep = {}
    co, cd = load(ORIG), load(DET)
    rep["config"] = dict(load_from=[co.load_from, cd.load_from], resume=[co.resume, cd.resume],
                         seed=[co.randomness["seed"], cd.randomness["seed"]],
                         point_seed=[co.model.roi_head.seed, cd.model.roi_head.seed],
                         head=[co.model.roi_head.point_refine_head.type, cd.model.roi_head.point_refine_head.type],
                         backbone_init=[co.model.backbone.init_cfg, cd.model.backbone.init_cfg],
                         optim_wrapper=[co.optim_wrapper.to_dict(), cd.optim_wrapper.to_dict()])

    # 1. initial weights
    so, sd = init_state(co), init_state(cd)
    ck = torch.load(CKPT, map_location="cpu")["state_dict"]
    head = [k for k in so if "point_refine_head" in k]
    rep["initialisation"] = dict(
        keys_equal=list(so) == list(sd),
        all_tensors_identical=all(torch.equal(so[k], sd[k]) for k in so),
        point_head_tensors=len(head),
        differs_from_trained_checkpoint=sum(not torch.equal(so[k].float(), ck[k].float()) for k in head),
        backbone_from_imagenet=bool(torch.equal(so["backbone.conv1.weight"],
                                               torch.load(ROOT / "pretrained/torch/hub/checkpoints/resnet101-63fe2227.pth",
                                                          map_location="cpu")["conv1.weight"])))
    print("init:", rep["initialisation"], flush=True)

    # 2. data order of the first training batches
    batches = {}
    for name, cfg in (("orig", co), ("det", cd)):
        dl_cfg = cfg.train_dataloader.to_dict(); dl_cfg["num_workers"] = 2; dl_cfg["persistent_workers"] = False
        dl = Runner.build_dataloader(dl_cfg, seed=cfg.randomness["seed"])
        seq = []
        for i, b in enumerate(dl):
            seq.append(dict(paths=[s.img_path.split("/")[-1] for s in b["data_samples"]],
                            flip=[bool(s.flip) for s in b["data_samples"]],
                            shapes=[list(s.img_shape) for s in b["data_samples"]],
                            instances=[len(s.gt_instances) for s in b["data_samples"]],
                            input_sum=[float(x.double().sum()) for x in b["inputs"]]))
            if i == 5:
                break
        batches[name] = seq
    rep["data_order_first_6_batches"] = dict(identical=batches["orig"] == batches["det"], orig=batches["orig"][:2])
    print("data:", rep["data_order_first_6_batches"]["identical"], batches["orig"][0]["paths"], flush=True)

    # 3. category mapping vs the dataset objects' own cat2label
    from mmdet.registry import DATASETS
    files = {"train (de-leaked)": C.train_annotation(False),
             "test easy (amended)": ROOT / "output/pidray_bs2_e10_2026-09-22/amendments/annotations/test_easy_eval.json",
             "test hard": C.ANN["test_hard"], "test hidden": C.ANN["test_hidden"]}
    rows = {}
    for name, ann in files.items():
        dcfg = co.test_dataloader.dataset.to_dict(); dcfg["ann_file"] = str(ann); dcfg["pipeline"] = []
        ds = DATASETS.build(dcfg); ds.full_init()
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(ann))
        script = label_to_category(gt, C.CLASSES)
        dataset = {v: k for k, v in ds.cat2label.items()}          # label -> category id
        rows[name] = dict(equal=script == dataset, dataset_label_to_category=dataset,
                          metainfo_classes_equal=tuple(ds.metainfo["classes"]) == tuple(C.CLASSES))
    from projects.proposal_first_pidray_occ.datasets.pidray_dataset import PIXRAY_CLASSES
    with contextlib.redirect_stdout(io.StringIO()):
        gpx = COCO(str(ROOT / "data/PIXray/annotations/test.json"))
    rows["PIXray test"] = dict(equal=label_to_category(gpx, PIXRAY_CLASSES) == {i: i for i in range(15)})
    rep["category_mapping"] = rows
    print("mapping:", {k: v["equal"] for k, v in rows.items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, default=str))
    ok = (rep["initialisation"]["all_tensors_identical"] and rep["data_order_first_6_batches"]["identical"]
          and all(v["equal"] for v in rows.values()))
    print("ALL CHECKS PASS" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
