"""Checkpoint 10: bounded Method 09 pilot, at most 200 optimizer updates.

Software and optimization evidence only. A clean pilot establishes that the
integration trains; it says nothing about segmentation quality.

Initialised from the Method 03 checkpoint, which means these weights have
already seen the full PIXray training split for 60 epochs. That is recorded, and
it is why this pilot cannot serve as a matched comparison against anything.
"""
import hashlib, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/nfs/ahalqahtani/ProposalFirst-XRaySeg")
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

MAX_UPDATES = 200
CFG = "projects/proposal_first_pidray_occ/configs/pixray_09_point_refine.py"
INIT = "work_dirs/pixray_p2/epoch_60.pth"
OUT = Path("output/audit/method09")
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

cfg = Config.fromfile(CFG)
if cfg.get("custom_imports"):
    import_modules_from_strings(**cfg["custom_imports"])
cfg.work_dir = "/tmp/m09pilot"
cfg.train_dataloader["num_workers"] = 2
cfg.train_dataloader["persistent_workers"] = False
runner = Runner.from_cfg(cfg)
model = runner.model.cuda()

prov = {
    "config": CFG, "config_sha256": sha(CFG),
    "initialisation_checkpoint": INIT, "init_sha256": sha(INIT),
    "init_note": "Method 03 epoch 60: already exposed to the full PIXray train "
                 "split. Labels this pilot as continued optimisation, not a "
                 "matched comparison.",
    "source_sha256": {p: sha(p) for p in sorted(
        str(q) for q in Path("projects/proposal_first_pidray_occ/models/point_refine").glob("*.py"))},
    "annotations_sha256": sha("data/PIXray/annotations/train.json"),
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "seed": cfg.randomness["seed"], "max_updates": MAX_UPDATES,
    "precision": cfg.optim_wrapper["type"],
    "accumulative_counts": cfg.optim_wrapper["accumulative_counts"],
}
ck = torch.load(INIT, map_location="cpu")
missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
prov["missing_keys"] = len(missing); prov["unexpected_keys"] = len(unexpected)
prov["missing_all_new"] = all("point_refine_head" in k for k in missing)
print(f"  init: {len(missing)} missing (all new head: {prov['missing_all_new']}), "
      f"{len(unexpected)} unexpected")

refiner = [m for m in model.modules() if type(m).__name__ == "PointRefineHead"][0]
optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)
optim_wrapper.initialize_count_status(model, 0, MAX_UPDATES * cfg.optim_wrapper["accumulative_counts"])
model.train()

def total_loss(d):
    out = 0
    for k, v in d.items():
        if "loss" not in k:
            continue
        out = out + (sum(x.sum() for x in v) if isinstance(v, (list, tuple)) else v.sum())
    return out

print("\n  --- forward/backward only, one batch, before any update ---")
batch = next(iter(runner.train_dataloader))
data = model.data_preprocessor(batch, True)
with torch.no_grad():
    pass
losses = model.loss(data["inputs"], data["data_samples"])
tl = total_loss(losses)
print(f"  total loss {float(tl):.5f} finite={torch.isfinite(tl).item()}")
print(f"  point losses: " + "  ".join(
    f"{k}={float(v):.5f}" for k, v in sorted(losses.items()) if "loss_point" in k))
tl.backward()
model.zero_grad(set_to_none=True)

records, samples_seen = [], []
start = time.perf_counter()
updates = 0
loader = iter(runner.train_dataloader)
step = 0
torch.cuda.reset_peak_memory_stats()
while updates < MAX_UPDATES:
    try:
        batch = next(loader)
    except StopIteration:
        loader = iter(runner.train_dataloader); batch = next(loader)
    data = model.data_preprocessor(batch, True)
    samples_seen.extend(int(s.img_id) for s in data["data_samples"])
    with optim_wrapper.optim_context(model):
        losses = model.loss(data["inputs"], data["data_samples"])
        tl = total_loss(losses)
    if not torch.isfinite(tl):
        print(f"  STOP: non-finite loss at step {step}"); break
    optim_wrapper.update_params(tl)
    step += 1
    updates = optim_wrapper._inner_count // cfg.optim_wrapper["accumulative_counts"]
    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            gn = sum(p.grad.norm().item() ** 2 for p in refiner.parameters()
                     if p.grad is not None) ** 0.5
        rec = {"step": step, "updates": updates, "total": float(tl),
               "grad_norm_refiner": gn,
               **{k: float(v) for k, v in losses.items()
                  if "loss_point" in k or "point_bce" in k}}
        records.append(rec)
        print(f"  step {step:>4} updates {updates:>3}  total {float(tl):.4f}  "
              f"refiner |g| {gn:.4f}  " +
              "  ".join(f"{k.split('.')[-1]}={float(v):.4f}"
                        for k, v in sorted(losses.items()) if "loss_point" in k))

elapsed = time.perf_counter() - start
prov.update({
    "optimizer_steps_run": step, "optimizer_updates": updates,
    "elapsed_seconds": elapsed, "seconds_per_step": elapsed / max(step, 1),
    "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
    "unique_images_seen": len(set(samples_seen)),
    "records": records,
})
(OUT / "pilot_provenance.json").write_text(json.dumps(prov, indent=2))
(OUT / "pilot_sample_manifest.json").write_text(json.dumps(sorted(set(samples_seen))))
print(f"\n  {step} steps -> {updates} optimizer updates in {elapsed:.1f}s "
      f"({elapsed/max(step,1):.3f}s/step)")
print(f"  peak GPU {prov['peak_gpu_mib']:.0f} MiB   unique images {prov['unique_images_seen']}")
print(f"  wrote {OUT}/pilot_provenance.json and pilot_sample_manifest.json")
print("\n  PILOT COMPLETE (software evidence only; no generalisation claim)")
