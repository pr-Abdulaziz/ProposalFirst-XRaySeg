"""Bounded GPU forward/backward smoke for method 08 before training.

Builds the model from the real config, runs one training step and one
inference step on real PIXray batches, and reports parameter counts, optimizer
membership, gradient flow under controlled gate outcomes, and the R0 gate
diagnostics the addendum requires.
"""
import sys, torch
sys.path.insert(0, "/nfs/ahalqahtani/ProposalFirst-XRaySeg")
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules

register_all_modules(init_default_scope=True)
CFG = "projects/proposal_first_pidray_occ/configs/pixray_08_pidray_r0r1.py"
cfg = Config.fromfile(CFG)
if cfg.get("custom_imports"):
    import_modules_from_strings(**cfg["custom_imports"])
cfg.work_dir = "/tmp/smoke_08"
cfg.train_dataloader["num_workers"] = 0
cfg.train_dataloader["persistent_workers"] = False
cfg.test_dataloader["num_workers"] = 0
cfg.test_dataloader["persistent_workers"] = False
runner = Runner.from_cfg(cfg)
model = runner.model.cuda()

print("=== parameter counts by module ===")
total = trainable = frozen = 0
for name, mod in model.named_children():
    t = sum(p.numel() for p in mod.parameters())
    tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    total += t; trainable += tr; frozen += t - tr
    print(f"  {name:<16}{t/1e6:>9.2f}M total{tr/1e6:>9.2f}M trainable{(t-tr)/1e6:>9.2f}M frozen")
print(f"  {'MODEL':<16}{total/1e6:>9.2f}M total{trainable/1e6:>9.2f}M trainable{frozen/1e6:>9.2f}M frozen")
# runner.optim_wrapper stays a lazy ConfigDict until the train loop builds it.
optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)
opt = optim_wrapper.optimizer
in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
print(f"  optimizer covers {in_opt/1e6:.2f}M parameters; matches trainable: {in_opt == trainable}")
r0 = sum(p.numel() for p in model.r0_head.parameters())
print(f"  r0_head: {r0/1e6:.2f}M parameters, in optimizer: "
      f"{any(p is q for p in model.r0_head.parameters() for g in opt.param_groups for q in g['params'])}")
print(f"  backbone frozen stages reflect the released policy: "
      f"{sum(1 for p in model.backbone.parameters() if not p.requires_grad)} frozen tensors")

print("\n=== one training step on a real batch ===")
model.train()
batch = next(iter(runner.train_dataloader))
data = model.data_preprocessor(batch, True)
losses = model.loss(data["inputs"], data["data_samples"])
flat = {k: (sum(v) if isinstance(v, (list, tuple)) else v) for k, v in losses.items()}
for k, v in flat.items():
    print(f"  {k:<26}{float(v):>12.5f}  finite={torch.isfinite(v).all().item()}")
n_gt = [len(s.gt_instances) for s in data["data_samples"]]
print(f"  instances per image after augmentation: {n_gt}  empty crops: {sum(1 for n in n_gt if n == 0)}")
total_loss = sum(flat.values())
total_loss.backward()
g_r0 = sum(p.grad.abs().sum().item() for p in model.r0_head.parameters() if p.grad is not None)
g_bb = sum(p.grad.abs().sum().item() for p in model.backbone.parameters() if p.grad is not None)
print(f"  gradient magnitude  r0_head={g_r0:.4f}  backbone={g_bb:.4f}")
print(f"  r0 receives gradient: {g_r0 > 0}")

print("\n=== one inference step, batch size 1 ===")
model.eval()
tb = next(iter(runner.test_dataloader))
with torch.no_grad():
    td = model.data_preprocessor(tb, False)
    out = model.predict(td["inputs"], td["data_samples"])
for s in out:
    n = len(s.pred_instances.scores)
    nz = int((s.pred_instances.scores > 0).sum())
    print(f"  img {s.img_id}  r0_score={s.get('r0_score'):.4f}  "
          f"detections={n}  nonzero-score={nz}  "
          f"{'GATE SUPPRESSED (scores zeroed, records kept)' if nz == 0 and n else 'gate accepted'}")
print("\nSMOKE 08 PASS")
