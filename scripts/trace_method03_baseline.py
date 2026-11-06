"""Checkpoint 1: capture real Method 03 tensor shapes on GPU.

Records the actual input, backbone, AttFPN, ROI and mask shapes, the stage
positive-ROI counts, and the frozen parameter names, so Method 09 is built
against observed dimensions rather than assumed ones.
"""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, "/nfs/ahalqahtani/ProposalFirst-XRaySeg")
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

CFG = "projects/proposal_first_pidray_occ/configs/pixray_p2_cascade_attfpn.py"
cfg = Config.fromfile(CFG)
if cfg.get("custom_imports"):
    import_modules_from_strings(**cfg["custom_imports"])
cfg.work_dir = "/tmp/trace03"
cfg.train_dataloader["num_workers"] = 0
cfg.train_dataloader["persistent_workers"] = False
runner = Runner.from_cfg(cfg)
model = runner.model.cuda().train()
torch.manual_seed(42)

trace = {"config": CFG, "steps": []}
def rec(step, shape, note):
    trace["steps"].append({"step": step, "shape": list(shape) if shape else None, "note": note})
    print(f"  {step:<34}{str(list(shape) if shape else '-'):<34}{note}")

print(f"  {'step':<34}{'observed shape':<34}note")
print("  " + "-"*96)
batch = next(iter(runner.train_dataloader))
data = model.data_preprocessor(batch, True)
inputs, samples = data["inputs"], data["data_samples"]
rec("preprocessed input", inputs.shape, "NCHW after synchronized transforms")
for i, s in enumerate(samples):
    print(f"      image {i}: img_shape={s.img_shape} batch_input_shape={s.batch_input_shape} "
          f"gt_instances={len(s.gt_instances)}")

with torch.no_grad():
    c = model.backbone(inputs)
for i, t in enumerate(c):
    rec(f"backbone C{i+2}", t.shape, f"stride {inputs.shape[-1]//t.shape[-1]}")
with torch.no_grad():
    p = model.neck(c)
for i, t in enumerate(p):
    rec(f"AttFPN P{i+2}", t.shape, f"stride {inputs.shape[-1]//t.shape[-1]}")
trace["p2_stride"] = inputs.shape[-1] // p[0].shape[-1]
trace["num_levels"] = len(p)

# Stage shapes are captured with hooks on the real cascade loss path rather
# than by reimplementing assignment and box refinement, so the observed N_s
# values are the ones training actually produces.
roi = model.roi_head
captured = []
def mask_hook(idx):
    def hook(module, args, output):
        feats = args[0]
        captured.append({"stage": idx, "feats": list(feats.shape),
                         "logits": list(output.shape)})
    return hook
handles = [roi.mask_head[i].register_forward_hook(mask_hook(i))
           for i in range(roi.num_stages)]
extract_handles = []
roi_shapes = {}
def extractor_hook(idx):
    def hook(module, args, output):
        roi_shapes[idx] = list(args[1].shape)   # rois tensor
    return hook
extract_handles = [roi.mask_roi_extractor[i].register_forward_hook(extractor_hook(i))
                   for i in range(roi.num_stages)]

losses = model.loss(inputs, samples)
for h in handles + extract_handles:
    h.remove()
for c in captured:
    i = c["stage"]
    rec(f"stage {i} positive ROIs", roi_shapes.get(i, []), "[img_idx,x1,y1,x2,y2]")
    rec(f"stage {i} mask ROI feats", c["feats"],
        f"extractor output_size={roi.mask_roi_extractor[i].roi_layers[0].output_size}")
    rec(f"stage {i} mask logits", c["logits"], "foreground channels, no background")
    trace[f"stage{i}_rois"] = roi_shapes.get(i)
    trace[f"stage{i}_mask_logits"] = c["logits"]
trace["loss_keys"] = sorted(losses.keys())
print(f"\n  loss keys: {sorted(losses.keys())}")

frozen = [n for n, q in model.named_parameters() if not q.requires_grad]
trace["frozen_parameters"] = frozen
trace["frozen_count"] = len(frozen)
print(f"\n  frozen parameters: {len(frozen)} tensors (first: {frozen[0] if frozen else '-'})")
trace["num_classes"] = model.roi_head.mask_head[0].num_classes
print(f"  mask head num_classes: {trace['num_classes']}")
out = Path("output/audit/method09/baseline_trace.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(trace, indent=2))
print(f"\nwrote {out}")
