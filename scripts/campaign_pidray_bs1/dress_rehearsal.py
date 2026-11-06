#!/usr/bin/env python
"""End-to-end rehearsal of one run: campaign hooks, checkpointing, resume, scoring.

preflight.py strips the campaign hooks and never reaches an epoch boundary, so it
cannot see the parts that only happen at the end of a run. This runs the real
resolved config -- hooks included -- over a tiny subset for two epochs and checks
exactly what run_guard.train-end will check after 50 real epochs:

  A  two epochs from scratch: checkpoint written, RNG state inside it,
     update accounting complete and consistent with accumulation 1
  B  one epoch, then resume for the second: CampaignRNGStateHookV2 restores a
     GPU-mapped checkpoint (the original hook raises TypeError here)
  C  inference and COCO scoring on a small slice of each test split

Nothing here is a PIDray result: the subset, the epoch count and the work
directory are all separate from the campaign's.

    python scripts/campaign_pidray_bs1/dress_rehearsal.py <method id> [--images N]
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_multiseed.score_predictions import score  # noqa: E402

OUT = C.OUT / "rehearsal"
EPOCHS = 2


def base_cfg(mid: str, seed: int, n_images: int, work: Path, epochs: int):
    cfg = Config.fromfile(str(C.resolved_config(mid, seed)), import_custom_modules=True)
    import_modules_from_strings(**cfg.custom_imports)
    ds = copy.deepcopy(dict(cfg.train_dataloader.dataset))
    ds["indices"] = list(range(n_images))
    cfg.train_dataloader.dataset = ds
    cfg.train_dataloader["persistent_workers"] = False
    cfg.train_cfg["max_epochs"] = epochs
    cfg.work_dir = str(work)
    return cfg


def accounting(work: Path) -> list[dict]:
    path = work / "update_accounting.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


def main() -> int:
    C.require_leakage_policy()
    mid = sys.argv[1]
    seed = C.SEEDS[0]
    n = int(sys.argv[sys.argv.index("--images") + 1]) if "--images" in sys.argv else 24
    slug = C.BY_ID[mid].slug
    result = dict(method=mid, slug=slug, images=n, epochs=EPOCHS,
                  node=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
                  resolved_config_sha256=C.sha256(C.resolved_config(mid, seed)))
    problems = []

    # -- A: two epochs from scratch, with the campaign hooks -----------------
    expected_updates = -(-n // C.PHYSICAL_BATCH) * EPOCHS
    result["expected_update_opportunities"] = expected_updates
    work_a = OUT / slug / "A_fresh"
    shutil.rmtree(work_a, ignore_errors=True)
    t0 = time.time()
    Runner.from_cfg(base_cfg(mid, seed, n, work_a, EPOCHS)).train()
    ck = work_a / f"epoch_{EPOCHS}.pth"
    rows = accounting(work_a)
    blob = torch.load(ck, map_location="cpu", weights_only=False) if ck.is_file() else {}
    cum = rows[-1].get("cumulative", {}) if rows else {}
    # the exact conditions run_guard.train-end applies, with EPOCHS for MAX_EPOCHS
    guard = dict(checkpoint_exists=ck.is_file(),
                 last_epoch_is_final=bool(rows) and rows[-1].get("epoch") == EPOCHS,
                 update_opportunities=cum.get("update_opportunities"),
                 # n is a count of images; an iteration consumes PHYSICAL_BATCH of
                 # them, exactly as run_guard derives updates() from iters_per_epoch().
                 expected_update_opportunities=expected_updates,
                 opportunities_match=cum.get("update_opportunities") == expected_updates,
                 successful_updates=cum.get("successful_updates"),
                 nonfinite_loss_iterations=cum.get("nonfinite_loss_iterations"),
                 no_nonfinite=cum.get("nonfinite_loss_iterations") == 0)
    result["A_fresh"] = dict(
        seconds=time.time() - t0, epochs_recorded=len(rows), guard_checks=guard,
        checkpoint_has_rng="campaign_rng" in blob,
        checkpoint_run_key=blob.get("meta", {}).get("campaign_run"),
        rng_keys=sorted(blob.get("campaign_rng", {})),
        would_pass_train_end=all([guard["checkpoint_exists"], guard["last_epoch_is_final"],
                                  guard["opportunities_match"], guard["no_nonfinite"]]))
    if not result["A_fresh"]["would_pass_train_end"]:
        problems.append(f"run_guard train-end would reject a complete run: {guard}")
    if not result["A_fresh"]["checkpoint_has_rng"]:
        problems.append("checkpoint carries no campaign_rng block")
    if result["A_fresh"]["checkpoint_run_key"] != C.run_key(mid, seed):
        problems.append("checkpoint run key does not match")

    # -- B: one epoch, then resume for the second ---------------------------
    work_b = OUT / slug / "B_resume"
    shutil.rmtree(work_b, ignore_errors=True)
    Runner.from_cfg(base_cfg(mid, seed, n, work_b, 1)).train()
    cfg_b = base_cfg(mid, seed, n, work_b, EPOCHS)
    cfg_b.resume = True
    cfg_b.load_from = None
    try:
        runner_b = Runner.from_cfg(cfg_b)
        runner_b.train()
        resumed, error = True, None
    except Exception as exc:  # noqa: BLE001
        resumed, error = False, f"{type(exc).__name__}: {exc}"
        problems.append(f"resume failed: {error}")
    rows_b = accounting(work_b)
    result["B_resume"] = dict(
        resumed=resumed, error=error, epochs_recorded=[r.get("epoch") for r in rows_b],
        final_checkpoint=(work_b / f"epoch_{EPOCHS}.pth").is_file(),
        cumulative=rows_b[-1].get("cumulative") if rows_b else None)
    if resumed and not result["B_resume"]["final_checkpoint"]:
        problems.append("resume produced no final checkpoint")

    # -- C: inference and scoring on a slice of each split -------------------
    splits = {}
    if ck.is_file():
        for split in C.TEST_SPLITS:
            cfg = Config.fromfile(str(C.resolved_config(mid, seed)), import_custom_modules=True)
            import_modules_from_strings(**cfg.custom_imports)
            paths = cfg.campaign_test_splits[split]
            ann = Path(paths["evaluator"])
            ds = copy.deepcopy(dict(cfg.test_dataloader.dataset))
            ds["ann_file"] = paths["loader"]
            ds["indices"] = list(range(40))
            cfg.test_dataloader.dataset = ds
            for e in (cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]):
                if "ann_file" in e:
                    e["ann_file"] = str(ann)
            cfg.load_from = str(ck)
            cfg.resume = False
            cfg.work_dir = str(OUT / slug / f"C_{split}")
            pred = Path(cfg.work_dir) / "predictions.pkl"
            Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
            try:
                runner = Runner.from_cfg(cfg)
                from mmdet.evaluation import DumpDetResults
                runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=str(pred)))
                runner.test()
                full = score(pred, ann)
                splits[split] = dict(ok=True, images_scored=full["coverage"]["prediction_images"],
                                     segm_AP=full["segm"]["stats"]["AP"],
                                     bbox_AP=full["bbox"]["stats"]["AP"],
                                     note="40-image slice; coverage mismatch against the full split is expected")
                del runner
            except Exception as exc:  # noqa: BLE001
                splits[split] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
                problems.append(f"scoring {split} failed: {splits[split]['error']}")
    result["C_scoring"] = splits
    result["problems"] = problems
    result["status"] = "PASS" if not problems else "FAIL"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{slug}.json").write_text(json.dumps(result, indent=1, default=float))
    shutil.rmtree(OUT / slug, ignore_errors=True)
    print(json.dumps(dict(method=mid, status=result["status"], problems=problems,
                          train_end_would_pass=result["A_fresh"]["would_pass_train_end"],
                          resume=result["B_resume"]["resumed"],
                          scoring={k: v["ok"] for k, v in splits.items()}), indent=1), flush=True)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
