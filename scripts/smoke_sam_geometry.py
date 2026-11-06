"""Bounded GPU smoke for the corrected SAM mask geometry.

Loads each SAM checkpoint, runs inference on a handful of real test images
including the 940x448 shape the geometry defect affects, and checks that mask
and box frames agree. Does not write metrics.
"""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, "/nfs/ahalqahtani/ProposalFirst-XRaySeg")
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules

register_all_modules(init_default_scope=True)

ARMS = [("05", "configs/occlusion/pixray_b5_rsprompter.py", "work_dirs/pixray_b5/epoch_60.pth"),
        ("06", "configs/occlusion/pixray_s2_occ_bisam.py",  "work_dirs/pixray_s2/epoch_60.pth")]
gt = json.load(open("data/PIXray/annotations/test.json"))
affected = [i["id"] for i in gt["images"] if (i["width"], i["height"]) == (940, 448)][:3]
other    = [i["id"] for i in gt["images"] if (i["width"], i["height"]) != (940, 448)][:3]
want = set(affected + other)
print(f"probe images: {sorted(want)}  (940x448 cases: {affected})")

ok = True
for nid, cfg_path, ckpt in ARMS:
    print(f"\n=== arm {nid}: {cfg_path} ===")
    cfg = Config.fromfile(cfg_path)
    # tools/train.py and tools/test.py do this explicitly; Runner does not.
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.work_dir = f"/tmp/smoke_{nid}"
    cfg.load_from = ckpt
    cfg.test_evaluator = dict(type="XRayCocoMetric",
                              ann_file="data/PIXray/annotations/test.json",
                              metric=["bbox", "segm"], classwise=False)
    cfg.test_dataloader["num_workers"] = 0
    cfg.test_dataloader["persistent_workers"] = False
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(ckpt, map_location="cpu")
    model = runner.model.eval().cuda()
    n = 0
    for batch in runner.test_dataloader:
        ids = [d.img_id for d in batch["data_samples"]]
        if not (set(ids) & want):
            continue
        with torch.no_grad():
            out = model.test_step(batch)
        for s in out:
            h, w = s.ori_shape
            m = s.pred_instances.masks
            b = s.pred_instances.bboxes
            shape_ok = (m.shape[-2:] == (h, w)) if len(m) else True
            box_ok = True if not len(b) else bool(
                (b[:, 0] >= -2).all() and (b[:, 2] <= w + 2).all()
                and (b[:, 1] >= -2).all() and (b[:, 3] <= h + 2).all())
            ok &= shape_ok and box_ok
            print(f"  img {s.img_id:<5} ori {w}x{h}  masks {tuple(m.shape)} "
                  f"frame_ok={shape_ok} boxes_in_frame={box_ok} n={len(b)}")
        n += len(out)
        if n >= 6:
            break
    del runner, model
    torch.cuda.empty_cache()
print(f"\nSMOKE {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
