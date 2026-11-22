"""Shared definitions for the PIXray multi-seed campaign (2026-09-14).

Every script in this directory imports its paths, method list, seeds and run order
from here, so a run is identified the same way by the builder, preflight, guards,
submission, evaluation and aggregation.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path("/nfs/ahalqahtani/ProposalFirst-XRaySeg")
CAMPAIGN = "pixray_multiseed_bs4_e50_2026-09-14"
OUT = ROOT / "output" / CAMPAIGN
WORK = ROOT / "work_dirs" / CAMPAIGN
MANIFEST = OUT / "campaign_manifest.json"
SEEDS = (42, 123, 2026)
PHYSICAL_BATCH = 4
ACCUMULATION = 1                                       # user instruction: no gradient accumulation
EFFECTIVE_BATCH = PHYSICAL_BATCH * ACCUMULATION        # 4 (the historical protocol used 16)
TRAIN_IMAGES = 3560
ITERS_PER_EPOCH = -(-TRAIN_IMAGES // PHYSICAL_BATCH)   # 890
MAX_EPOCHS = 50                                        # user instruction (2026-09-14): 50 epochs for every method
MICROBATCHES = MAX_EPOCHS * ITERS_PER_EPOCH             # 44,500
UPDATES = MICROBATCHES // ACCUMULATION                  # 44,500
# Historical warmup = 100 images (50 iterations at batch 2) = 25 iterations (25 updates) at batch 4.
WARMUP_ITERS = 25

ANN = dict(
    train=ROOT / "data/PIXray/annotations/train.json",
    test=ROOT / "data/PIXray/annotations/test.json",
    occ_train=ROOT / "data/PIXray-A/annotations/train_occ.json",
    occ_test=ROOT / "data/PIXray-A/annotations/test_occ.json",
)
IMAGE_DIRS = dict(train=ROOT / "data/PIXray/train", test=ROOT / "data/PIXray/test")
PRETRAINED = dict(
    resnet101_imagenet=ROOT / "pretrained/torch/hub/checkpoints/resnet101-63fe2227.pth",
    sam_vit_large_converted=ROOT / "pretrained/sam_vit_large/pytorch_model_new.bin",
    sam_vit_large_hf=ROOT / "pretrained/sam_vit_large/pytorch_model.bin",
    sam_vit_large_config=ROOT / "pretrained/sam_vit_large/config.json",
)

C = "projects/proposal_first_pidray_occ/configs"
O = "configs/occlusion"


@dataclass(frozen=True)
class Method:
    id: str
    slug: str
    name: str
    historical_config: str
    historical_id: str
    supervision: str          # original | occ | conflict
    family: str               # detector | sam
    role: str
    launchable: bool
    note: str = ""


METHODS = [
    Method("01", "m01_mask_rcnn_r101_fpn", "Mask R-CNN, ResNet-101 + FPN",
           f"{C}/pixray_b2_mask_rcnn_r101.py", "B2", "original", "detector",
           "detector baseline", True),
    Method("02", "m02_cascade_r101_fpn", "Cascade Mask R-CNN, ResNet-101 + FPN",
           f"{C}/pixray_p1_cascade_fpn.py", "P1", "original", "detector",
           "detector baseline", True),
    Method("03", "m03_cascade_r101_attfpn", "Cascade Mask R-CNN, ResNet-101 + AttFPN",
           f"{C}/pixray_p2_cascade_attfpn.py", "P2", "original", "detector",
           "parent of 04, 08, 09, 09M", True),
    Method("04", "m04_cascade_attfpn_recall_rpn", "Cascade + AttFPN + recall-enhanced RPN",
           f"{C}/pixray_p3_cascade_attfpn_recall_rpn.py", "P3", "original", "detector",
           "detector baseline", True),
    Method("05", "m05_rsprompter_sam_lora", "RSPrompter-style SAM ViT-L + LoRA adaptation",
           f"{O}/pixray_b5_rsprompter.py", "B5", "original", "sam",
           "SAM comparison (not an untouched SAM release)", True),
    Method("06", "m06_occ_bilayer_sam", "Occ bilayer SAM (Occ paper method)",
           f"{O}/pixray_s2_occ_bisam.py", "S2", "occ", "sam",
           "additional-supervision comparison", True),
    Method("07", "m07_bilayer_fcn_recall_rpn", "Bilayer FCN mask head on Method 04",
           f"{C}/pixray_b8_bilayer_fcn_recall_rpn.py", "B8", "occ", "detector",
           "additional-supervision comparison", True,
           "keeps PIXray-A occluder supervision by user decision (2026-09-14); reported beside 06, not as a target-only method"),
    Method("08", "m08_pidray_r0r1", "PIDray R0/R1 adapted to PIXray",
           f"{C}/pixray_08_pidray_r0r1.py", "", "original", "detector",
           "PIDray mechanism on Method 03", True),
    Method("09", "m09_point_refine_transformer", "Transformer point refinement on Method 03",
           f"{C}/pixray_09_point_refine.py", "", "original", "detector",
           "proposed", True),
    Method("09M", "m09m_point_refine_mlp", "Matched MLP point-refinement control on Method 03",
           f"{C}/pixray_09m_point_refine_mlp.py", "", "original", "detector",
           "matched control for 09", True),
]
BY_ID = {m.id: m for m in METHODS}

# Fixed run order, as instructed by the user (2026-09-14): 01, 02, 03, 04, 07, 08, 09 & 09M,
# then the two SAM methods 05 and 06 last. Each method runs its three seeds before the next
# method; 09 and 09M alternate seed by seed so the matched pair shares the most similar period.
_METHOD_ORDER = ["01", "02", "03", "04", "07", "08", ("09", "09M"), "05", "06"]


def run_order() -> list[tuple[str, int]]:
    order = []
    for item in _METHOD_ORDER:
        group = item if isinstance(item, tuple) else (item,)
        for seed in SEEDS:
            order += [(mid, seed) for mid in group]
    return order


def point_seed(seed: int) -> int:
    """Point-refinement generator seed derived from the run seed (documented policy)."""
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


def sha256(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# The guarded snapshot covers code that training and evaluation execute. Build,
# preflight, submission and reporting scripts are recorded by hash in the manifest
# but are not guarded, so analysis code can be written while runs are queued.
SOURCE_DIRS = ("projects", "mmdet", "mmpretrain", "tools", "configs")
RUNTIME_FILES = ("scripts/common.sh",
                 "scripts/campaign_multiseed/campaign_common.py",
                 "scripts/campaign_multiseed/train_run.py",
                 "scripts/campaign_multiseed/run_guard.py",
                 "scripts/campaign_multiseed/evaluate_run.py",
                 "scripts/campaign_multiseed/score_predictions.py",
                 "scripts/campaign_multiseed/slurm_train.sh",
                 "scripts/campaign_multiseed/slurm_eval.sh")


def snapshot_changes(frozen: dict, current: dict) -> dict:
    """Modified or deleted frozen files break a run; added files cannot change existing imports."""
    return dict(modified=sorted(k for k in frozen if k in current and current[k] != frozen[k]),
                deleted=sorted(k for k in frozen if k not in current),
                added=sorted(k for k in current if k not in frozen))


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


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def write_once(path: Path, obj) -> None:
    """Write a JSON record that must never be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def method_record(m: Method) -> dict:
    return asdict(m)
