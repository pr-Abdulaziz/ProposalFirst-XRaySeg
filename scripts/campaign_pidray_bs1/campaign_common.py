"""Shared definitions for the PIDray-30 single-seed batch-1 campaign (2026-09-21).

A separate campaign from the frozen PIXray one: different dataset, different
batch size, one seed. It reuses the PIXray campaign's *method* definitions so the
architectures compared on both datasets are literally the same objects, and it
reuses nothing else -- no path, manifest or output of the PIXray campaign is read
or written here.

User instructions recorded for this campaign (2026-09-21):
  * dataset PIDray-30, "a different dataset ... we will compare different architectures"
  * physical batch 1, no gradient accumulation
  * 50 epochs
  * one seed (42)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from scripts.campaign_multiseed.campaign_common import (  # architectures only
    METHODS, BY_ID, Method, sha256, append_jsonl, write_once, snapshot_changes)

ROOT = Path("/nfs/ahalqahtani/ProposalFirst-XRaySeg")
CAMPAIGN = "pidray_bs2_e10_2026-09-22"
OUT = ROOT / "output" / CAMPAIGN
WORK = ROOT / "work_dirs" / CAMPAIGN
MANIFEST = OUT / "campaign_manifest.json"

SEEDS = (42,)                                          # user instruction: one seed
# Physical batch 2 (user instruction, 2026-09-21, after the batch-1 attempt
# diverged). The batch-1 run of method 08 (job 12317) reached a loss of 438 at
# iteration ~511, went 79% non-finite in epoch 1 and 100% in epoch 2, with the
# AMP loss scale collapsed to 0.0; evidence in
# output/pidray_bs1_e50_2026-09-21/. Batch is the only protocol field changed:
# the optimizer, learning rate, schedule shape and epoch count are untouched.
PHYSICAL_BATCH = 2
ACCUMULATION = 1                                       # no gradient accumulation
EFFECTIVE_BATCH = PHYSICAL_BATCH * ACCUMULATION
# 10 epochs (user instruction, 2026-09-22: "rerun all experiments with only 10
# epochs"). The PIDray paper trains two-stage detectors for 12 epochs; at batch 2
# this is 114,950 optimizer updates per run, 2.6x what every run of the PIXray
# campaign received. The 15-epoch attempt (pidray_bs2_e15_2026-09-22: 01 complete,
# 02 cancelled at epoch 11) is superseded and kept as a record.
MAX_EPOCHS = 10
# 500 iterations, the MMDetection default that the PIDray paper states it follows
# ("other hyper-parameters involved in the experiments follow the settings of
# MMDetection"). The earlier 100-image rule gave only 50 iterations here, and the
# measured instability began the moment it ended: method 08 sat at loss 2.0 at
# iteration 50, 4,043 at 250 and 13,657 at 500. The fp16 variant with this warmup
# survived where the 50-iteration one collapsed. 500 is 0.29% of training.
WARMUP_ITERS = 500
NUM_CLASSES = 12
CLASSES = ("Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench",
           "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter")
TEST_SPLITS = ("easy", "hard", "hidden")

DATA = ROOT / "data/PIDray"
DATA_A = ROOT / "data/PIDray-A/annotations"
ANN = dict(
    train=DATA / "annotations/train.json",
    occ_train=DATA_A / "train_occ.json",
    **{f"test_{s}": DATA / f"annotations/test_{s}.json" for s in TEST_SPLITS},
    **{f"occ_test_{s}": DATA_A / f"test_{s}_occ.json" for s in TEST_SPLITS},
)
IMAGE_DIRS = dict(train=DATA / "train", test=DATA / "test")
RELEASED_TRAIN_IMAGES = 23074
TEST_IMAGES = dict(easy=24758, hard=9746, hidden=13069)

# ---------------------------------------------------------------------------
# Leakage gate. scripts/audit_pidray.py --check-content-hashes FAILED on
# 2026-09-21: byte-identical images are shared between train and all three test
# splits in PIDray as released (see output/<campaign>/audit/). Training must not
# start until the user chooses how to handle it, so every entry point refuses to
# run while this is None. The choice is recorded in the frozen manifest.
#
#   "exclude_train_duplicates" -- train on a derived annotation file that drops
#       the train-side duplicates. Test splits stay exactly as released.
#   "as_released"              -- train on the released file and report the
#       leakage as a limitation.
# Set 2026-09-21 by user decision, after the gate was re-verified against the
# derived training file (audit/gate_verification.json: PASS).
LEAKAGE_POLICY: str | None = "exclude_train_duplicates"
# The de-leaked annotations and the leakage audit are campaign-neutral: they were
# produced once and are shared by any PIDray campaign built from this decision.
SHARED = ROOT / "output" / "pidray_deleaked_2026-09-21"
AUDIT = SHARED
DELEAKED_TRAIN_ANN = SHARED / "annotations" / "train_deleaked.json"


def train_annotation(occ: bool) -> Path:
    """The training annotation a run must use, given the recorded leakage policy."""
    require_leakage_policy()
    if occ:
        return DELEAKED_OCC_TRAIN_ANN if LEAKAGE_POLICY == "exclude_train_duplicates" else ANN["occ_train"]
    return DELEAKED_TRAIN_ANN if LEAKAGE_POLICY == "exclude_train_duplicates" else ANN["train"]


DELEAKED_OCC_TRAIN_ANN = SHARED / "annotations" / "train_occ_deleaked.json"


def require_leakage_policy() -> None:
    if LEAKAGE_POLICY not in ("exclude_train_duplicates", "as_released"):
        raise SystemExit(
            "PIDray leakage gate is open: scripts/audit_pidray.py --check-content-hashes "
            "failed and LEAKAGE_POLICY is not set. No training may be built or launched. "
            "See output/pidray_deleaked_2026-09-21/leakage_report.json.")


def train_images() -> int:
    ann = train_annotation(occ=False)
    return len(json.loads(ann.read_text())["images"])


def iters_per_epoch() -> int:
    return -(-train_images() // PHYSICAL_BATCH)


# ---------------------------------------------------------------------------
# All ten architectures, ascending order (user instruction, 2026-09-21, restoring
# 02, 04 and 07 after batch 2 and 20 epochs cut the campaign from 13.5 days to
# under four). Each method is a single run (one seed).
_METHOD_ORDER = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "09M"]
# Node assignment. User instruction (2026-09-21): place each run on a different
# node, round-robin, rather than letting them all land on whichever node is free.
# This distributes *where* the work runs, not *when*: QOS student allows one
# running job per account, so the runs are still sequential. issnode1 is the
# login node (foreign GPU users outside SLURM) and issnode5 is drained.
CAMPAIGN_NODES = ("issnode2", "issnode3", "issnode4")


EXCLUDED_METHODS: dict[str, str] = {}   # nothing excluded


def run_order() -> list[tuple[str, int]]:
    return [(mid, seed) for mid in _METHOD_ORDER for seed in SEEDS]


def point_seed(seed: int) -> int:
    digest = hashlib.sha256(f"{CAMPAIGN}/point_refine/{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run_key(mid: str, seed: int) -> str:
    return f"{BY_ID[mid].slug}/seed_{seed}"


def resolved_config(mid: str, seed: int) -> Path:
    return OUT / "protocol" / "resolved_configs" / BY_ID[mid].slug / f"seed_{seed}.py"


def work_dir(mid: str, seed: int) -> Path:
    return WORK / BY_ID[mid].slug / f"seed_{seed}"


def run_dir(mid: str, seed: int) -> Path:
    return OUT / "runs" / BY_ID[mid].slug / f"seed_{seed}"


def eval_dir(mid: str, seed: int) -> Path:
    return OUT / "evaluations" / BY_ID[mid].slug / f"seed_{seed}"


RUNTIME_FILES = ("scripts/common.sh",
                 "scripts/campaign_pidray_bs1/campaign_common.py",
                 "scripts/campaign_pidray_bs1/build_campaign.py",
                 "scripts/campaign_multiseed/train_run.py",
                 "scripts/campaign_pidray_bs1/run_guard.py",
                 "scripts/campaign_pidray_bs1/evaluate_run.py",
                 "scripts/campaign_multiseed/score_predictions.py",
                 "scripts/campaign_pidray_bs1/slurm_train.sh",
                 "scripts/campaign_pidray_bs1/slurm_eval.sh")


def method_record(m: Method) -> dict:
    return asdict(m)


SOURCE_DIRS = ("projects", "mmdet", "mmpretrain", "tools", "configs")


def source_files() -> list[Path]:
    files = [ROOT / f for f in RUNTIME_FILES]
    for d in SOURCE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"} or ".egg-info" in str(p):
                continue
            files.append(p)
    return files


def source_snapshot() -> dict:
    per_file = {str(p.relative_to(ROOT)): sha256(p) for p in source_files()}
    aggregate = hashlib.sha256(
        "\n".join(f"{k} {v}" for k, v in sorted(per_file.items())).encode()).hexdigest()
    return dict(aggregate_sha256=aggregate, file_count=len(per_file), files=per_file)


def updates() -> int:
    return iters_per_epoch() * MAX_EPOCHS
