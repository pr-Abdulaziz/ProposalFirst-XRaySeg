#!/usr/bin/env python
"""Launch guard and append-only records for campaign training jobs.

    run_guard.py train-start <method id> <seed> [--resume]
    run_guard.py train-end   <method id> <seed> <train exit code>

train-start refuses to run unless the frozen manifest exists and the run's resolved
config, the source snapshot, the annotations and the pretrained weights still hash
to their frozen values; a fresh run also needs an empty work directory. It writes a
write-once launch record. train-end appends the outcome and exits non-zero unless
the final-epoch checkpoint exists and the update accounting is complete, so the dependent
evaluation job only runs after a complete training run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    BY_ID, MANIFEST, MAX_EPOCHS, UPDATES, append_jsonl, run_dir, run_key, sha256, snapshot_changes,
    source_snapshot, write_once)


def fail(msg: str) -> None:
    print(f"ABORT: {msg}", flush=True)
    sys.exit(3)


def gpu_info() -> list[dict]:
    q = "name,memory.total,driver_version,uuid,pci.bus_id"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return [dict(zip(q.split(","), [x.strip() for x in line.split(",")])) for line in out]


def gpu_check(max_used_mib: int = 500, wait_s: int = 600) -> dict:
    """Require a clean GPU: no compute process and at most ``max_used_mib`` in use.

    Polls for up to ``wait_s`` seconds, so memory released by a job that has just
    ended on this node is not mistaken for another user's allocation.
    """
    deadline = time.time() + wait_s
    while True:
        used = [int(x) for x in subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split()]
        apps = [line.strip() for line in subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.splitlines() if line.strip()]
        clean = bool(used) and max(used) <= max_used_mib and not apps
        if clean or time.time() >= deadline:
            return dict(clean=clean, memory_used_mib=used, compute_apps=apps, limit_mib=max_used_mib,
                        checked=time.strftime("%Y-%m-%dT%H:%M:%S"))
        time.sleep(15)


def train_start(mid: str, seed: int, resume: bool) -> None:
    if not MANIFEST.is_file():
        fail("campaign manifest is not frozen")
    man = json.loads(MANIFEST.read_text())
    key = run_key(mid, seed)
    run = man["runs"].get(key)
    if run is None or not run["launchable"]:
        fail(f"{key} is not an approved run in the frozen manifest")
    cfg = ROOT / run["resolved_config"]
    if sha256(cfg) != run["resolved_config_sha256"]:
        fail(f"{cfg} changed after the freeze")
    snap = source_snapshot()
    frozen = json.loads((ROOT / man["source_snapshot"]["per_file"]).read_text())["files"]
    changes = snapshot_changes(frozen, snap["files"])
    if changes["modified"] or changes["deleted"]:
        fail(f"frozen source changed: modified={changes['modified'][:10]} deleted={changes['deleted'][:10]}")
    for name, rec in man["annotations"].items():
        if sha256(ROOT / rec["path"]) != rec["sha256"]:
            fail(f"annotation {rec['path']} changed")
    needed = ["resnet101_imagenet"] if BY_ID[mid].family == "detector" else [
        "sam_vit_large_converted", "sam_vit_large_hf", "sam_vit_large_config"]
    for name in needed:
        rec = man["pretrained"][name]
        if sha256(ROOT / rec["path"]) != rec["sha256"]:
            fail(f"pretrained {name} changed")
    text = cfg.read_text()
    if "load_from = None" not in text or "resume = False" not in text:
        fail("resolved config does not start from pretrained initialisation only")
    wd = Path(run["work_dir"])
    ckpts = sorted(wd.glob("*.pth")) if wd.exists() else []
    if not resume and ckpts:
        fail(f"{wd} already holds checkpoints {[c.name for c in ckpts]}; inspect the earlier attempt, "
             "then resubmit with --resume or move the attempt aside explicitly")
    if resume and not (wd / "last_checkpoint").is_file():
        fail("--resume given but no last_checkpoint exists")
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    import torch, mmengine, mmdet, mmcv  # noqa: E401
    record = dict(
        event="train_started", run_key=key, method=mid, seed=seed, resume=resume, job=job,
        node=os.uname().nodename, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        gpus=gpu_info(), gpu_state_at_launch=gpu_check(wait_s=0),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        command=[sys.executable, "scripts/campaign_multiseed/train_run.py", run["resolved_config"]]
        + (["--resume"] if resume else []),
        resolved_config=run["resolved_config"], resolved_config_sha256=run["resolved_config_sha256"],
        source_snapshot_sha256=snap["aggregate_sha256"], frozen_source_unchanged=True,
        source_files_added_since_freeze=changes["added"],
        annotations={k: v["sha256"] for k, v in man["annotations"].items()},
        pretrained={n: man["pretrained"][n]["sha256"] for n in needed},
        precision="AMP float16 training (AmpOptimWrapper), fp32 master weights",
        environment=dict(python=sys.version.split()[0], executable=sys.executable,
                         torch=(torch.__version__, torch.__file__), cuda=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), mmengine=(mmengine.__version__, mmengine.__file__),
                         mmdet=(mmdet.__version__, mmdet.__file__), mmcv=(mmcv.__version__, mmcv.__file__),
                         cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING")),
        existing_checkpoints=[c.name for c in ckpts])
    write_once(run_dir(mid, seed) / f"launch_job{job}.json", record)
    append_jsonl(run_dir(mid, seed) / "status.jsonl", dict(event="train_started", job=job, resume=resume,
                                                            node=record["node"], time=time.time()))
    print(f"guard passed: {key} job={job} node={record['node']} gpu={record['gpus']}", flush=True)


def train_end(mid: str, seed: int, code: int) -> None:
    man = json.loads(MANIFEST.read_text())
    key = run_key(mid, seed)
    wd = Path(man["runs"][key]["work_dir"])
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    acct_file = wd / "update_accounting.jsonl"
    lines = [json.loads(x) for x in acct_file.read_text().splitlines()] if acct_file.is_file() else []
    last = lines[-1] if lines else {}
    cum = last.get("cumulative", {})
    ckpt = wd / f"epoch_{MAX_EPOCHS}.pth"
    complete = (code == 0 and ckpt.is_file() and last.get("epoch") == MAX_EPOCHS
                and cum.get("update_opportunities") == UPDATES and cum.get("nonfinite_loss_iterations") == 0)
    event = dict(event="train_complete" if complete else "train_failed", job=job, exit_code=code,
                 time=time.time(), last_epoch=last.get("epoch"), cumulative=cum,
                 checkpoint_sha256=sha256(ckpt) if ckpt.is_file() else None)
    append_jsonl(run_dir(mid, seed) / "status.jsonl", event)
    print(json.dumps(event, default=str), flush=True)
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gpu-check":
        state = gpu_check()
        print(json.dumps(state), flush=True)
        sys.exit(0 if state["clean"] else 4)
    mid, seed = sys.argv[2], int(sys.argv[3])
    if cmd == "train-start":
        train_start(mid, seed, "--resume" in sys.argv[4:])
    elif cmd == "train-end":
        train_end(mid, seed, int(sys.argv[4]))
    else:
        fail(f"unknown command {cmd}")
