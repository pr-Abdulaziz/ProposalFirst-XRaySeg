"""Checkpoints 7-9: integrated cascade training, inference and GPU checks."""
import sys, torch, copy
sys.path.insert(0, "/nfs/ahalqahtani/ProposalFirst-XRaySeg")
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

def build(cfg_path, **over):
    cfg = Config.fromfile(cfg_path)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.work_dir = "/tmp/m09test"
    cfg.train_dataloader["num_workers"] = 0; cfg.train_dataloader["persistent_workers"] = False
    cfg.test_dataloader["num_workers"] = 0; cfg.test_dataloader["persistent_workers"] = False
    for k, v in over.items():
        cfg.model["roi_head"][k] = v
    return cfg, Runner.from_cfg(cfg)

C9 = "projects/proposal_first_pidray_occ/configs/pixray_09_point_refine.py"
C3 = "projects/proposal_first_pidray_occ/configs/pixray_p2_cascade_attfpn.py"
ok = True
def check(label, cond, extra=""):
    global ok; ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + extra if extra else ''}")

print("=== Checkpoint 7: training cascade adapter ===")
cfg9, r9 = build(C9)
m9 = r9.model.cuda().train()
torch.manual_seed(42)

# exactly one shared instance
refiners = [mod for mod in m9.modules() if type(mod).__name__ == "PointRefineHead"]
check("exactly one shared PointRefineHead instance", len(refiners) == 1, f"found {len(refiners)}")
new_params = sum(p.numel() for p in refiners[0].parameters())
check("new parameter count 458,383", new_params == 458383, f"got {new_params:,}")

batch = next(iter(r9.train_dataloader))
data = m9.data_preprocessor(batch, True)
losses = m9.loss(data["inputs"], data["data_samples"])
point_keys = sorted(k for k in losses if "point" in k)
loss_keys = [k for k in point_keys if "loss" in k]
diag_keys = [k for k in point_keys if "loss" not in k]
print(f"    point loss keys      : {loss_keys}")
print(f"    point diagnostic keys: {diag_keys[:4]} ...")
check("one summed point loss per stage", len(loss_keys) == 3, f"{len(loss_keys)}")
check("diagnostics excluded from the loss parser",
      all("loss" not in k for k in diag_keys) and len(diag_keys) > 0)
check("all point losses finite", all(torch.isfinite(losses[k]).all() for k in loss_keys))

# stage weighting applied exactly once: recompute the head's raw value
raw = {}
h = refiners[0]
orig = h.loss_and_refine
def spy(*a, **kw):
    out = orig(*a, **kw)
    raw[kw.get("stage", 0)] = float(out["loss_point"].detach())
    return out
h.loss_and_refine = spy
torch.manual_seed(42)
losses2 = m9.loss(data["inputs"], data["data_samples"])
h.loss_and_refine = orig
weights = cfg9.model["roi_head"]["stage_loss_weights"]
for s in range(3):
    if s in raw and f"s{s}.loss_point" in losses2:
        got = float(losses2[f"s{s}.loss_point"]); want = raw[s] * weights[s]
        check(f"stage {s} weight applied exactly once (w={weights[s]})",
              abs(got - want) < 1e-4, f"raw {raw[s]:.5f} -> {got:.5f}, expected {want:.5f}")

# gradients through the integrated path
m9.zero_grad()
def total_loss(d):
    """Sum every loss entry; cascade values may be tensors or lists of them."""
    out = 0
    for k, v in d.items():
        if "loss" not in k:
            continue
        out = out + (sum(x.sum() for x in v) if isinstance(v, (list, tuple)) else v.sum())
    return out
total_loss(losses).backward()
le = h.embedding.level_embedding.grad
check("both level embeddings get gradient via cascade",
      le is not None and le[0].abs().sum() > 0 and le[1].abs().sum() > 0,
      f"l0={le[0].abs().sum():.3e} l1={le[1].abs().sum():.3e}" if le is not None else "none")
tg = sum(p.grad.abs().sum() for p in h.transformer.parameters() if p.grad is not None)
check("transformer receives gradient", tg > 0, f"{tg:.3e}")
cg = sum(p.grad.abs().sum() for p in m9.roi_head.mask_head.parameters() if p.grad is not None)
check("coarse mask heads still receive gradient", cg > 0, f"{cg:.3e}")
bg = sum(p.grad.abs().sum() for p in m9.backbone.parameters() if p.grad is not None)
check("backbone receives gradient", bg > 0, f"{bg:.3e}")
frozen_ok = all(not p.requires_grad for n, p in m9.backbone.named_parameters()
                if n.startswith(("conv1", "bn1", "layer1")))
check("frozen stem/layer1 stay frozen after train()", frozen_ok)
bn_eval = sum(1 for mod in m9.backbone.modules()
              if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and not mod.training)
check("backbone BN remains in eval mode", bn_eval > 0, f"{bn_eval} modules")

print("\n=== Checkpoint 9: AMP, optimizer updates, memory ===")
# runner.optim_wrapper stays a lazy ConfigDict until the train loop builds it.
optim_wrapper = r9.build_optim_wrapper(cfg9.optim_wrapper)
opt = optim_wrapper.optimizer
in_opt = {id(q) for g in opt.param_groups for q in g["params"]}
refiner_in = sum(1 for q in h.parameters() if id(q) in in_opt)
check("every refinement parameter is in the optimizer",
      refiner_in == len(list(h.parameters())), f"{refiner_in}/{len(list(h.parameters()))}")

scaler = torch.cuda.amp.GradScaler()
before = {n: q.detach().clone() for n, q in h.named_parameters()}
skipped = 0
for step in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        la = m9.loss(data["inputs"], data["data_samples"])
        total = total_loss(la)
    check(f"AMP step {step} forward finite", torch.isfinite(total).all(),
          f"{float(total):.4f}") if step == 0 else None
    scaler.scale(total).backward()
    scaler.unscale_(opt)
    gr = [q.grad for q in h.parameters() if q.grad is not None]
    finite = all(torch.isfinite(g).all() for g in gr)
    prev = scaler.get_scale()
    scaler.step(opt); scaler.update()
    if scaler.get_scale() < prev:
        skipped += 1
    if step == 0:
        check("AMP gradients finite after unscale", finite, f"{len(gr)} tensors")
moved = sum(1 for n, q in h.named_parameters()
            if not torch.equal(q.detach(), before[n]))
check("optimizer actually updated refinement parameters",
      moved > 0, f"{moved}/{len(before)} tensors changed, {skipped}/3 AMP steps skipped")
print(f"    peak GPU memory: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

print("\n=== Checkpoint 8: inference adapter ===")
m9.eval()
tb = next(iter(r9.test_dataloader))
with torch.no_grad():
    td = m9.data_preprocessor(tb, False)
    out9 = m9.predict(td["inputs"], td["data_samples"])
for s in out9:
    hh, ww = s.ori_shape
    mk = s.pred_instances.masks
    check(f"img {s.img_id}: masks at original {ww}x{hh}",
          tuple(mk.shape[-2:]) == (hh, ww) if len(mk) else True,
          f"{tuple(mk.shape)} dtype={mk.dtype}")

print("\n=== refinement-disabled equivalence with Method 03 ===")
cfg_off, r_off = build(C9, refine_enabled=False)
m_off = r_off.model.cuda().eval()
cfg3, r3 = build(C3)
m3 = r3.model.cuda().eval()
sd = m3.state_dict()
missing, unexpected = m_off.load_state_dict(sd, strict=False)
new_only = all("point_refine_head" in k for k in missing)
check("only point_refine_head keys missing when loading 03 weights",
      new_only and len(unexpected) == 0,
      f"{len(missing)} missing (all new: {new_only}), {len(unexpected)} unexpected")
with torch.no_grad():
    a = m_off.predict(copy.deepcopy(td)["inputs"], copy.deepcopy(td)["data_samples"])
    b = m3.predict(copy.deepcopy(td)["inputs"], copy.deepcopy(td)["data_samples"])
same = all(torch.equal(x.pred_instances.masks, y.pred_instances.masks)
           for x, y in zip(a, b))
check("refine_enabled=False is bit-identical to Method 03", same)

print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
sys.exit(0 if ok else 1)
