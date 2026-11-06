#!/usr/bin/env bash
# Train one full-official-split arm under Method 03's protocol.
#
#   sbatch --job-name=m09 scripts/slurm_full_split_arm.sh <config> [--resume]
#
# The guards refuse anything that would make the arm non-comparable to Method
# 03, because each of those failures would produce a plausible-looking number
# rather than an error.
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --output=output/full_split_campaign/2026-09-12/train_%x_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1

# Method 03 trained with 2 dataloader workers; common.sh defaults to 4. Worker
# count seeds the per-worker augmentation RNG, so this is protocol, not tuning.
export XRAY_NUM_WORKERS=2

CONFIG="${1:?usage: sbatch scripts/slurm_full_split_arm.sh <config> [--resume]}"
shift || true

echo "node=$(hostname) job=${SLURM_JOB_ID} config=${CONFIG} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON_BIN}" - "${CONFIG}" <<'GUARD' || exit 3
import hashlib, json, sys
from pathlib import Path
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
freeze = json.loads(Path("output/full_split_campaign/2026-09-12/freeze.json").read_text())
problems = []

# 1. the official split, unmodified
train = cfg.train_dataloader["dataset"]["ann_file"]
test = cfg.test_dataloader["dataset"]["ann_file"]
if not train.endswith("annotations/train.json"):
    problems.append(f"training annotations are {train}, not the official split")
if not test.endswith("annotations/test.json"):
    problems.append(f"test annotations are {test}, not the official split")
for name, path in (("train.json", "data/PIXray/annotations/train.json"),
                   ("test.json", "data/PIXray/annotations/test.json")):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != freeze["annotation_sha256"][path]:
        problems.append(f"{name} has changed since the freeze")

# 2. Method 03's protocol
if cfg.train_dataloader["num_workers"] != 2:
    problems.append(f"num_workers is {cfg.train_dataloader['num_workers']}, "
                    "not Method 03's 2; the augmentation stream would differ")
if cfg.train_cfg["max_epochs"] != 60:
    problems.append(f"max_epochs is {cfg.train_cfg['max_epochs']}, not 60")
if cfg.randomness["seed"] != 42:
    problems.append(f"seed is {cfg.randomness['seed']}, not 42")
if cfg.load_from is not None:
    problems.append(f"load_from is {cfg.load_from!r}; arms must start from ImageNet")

# 3. the frozen checkpoint-selection rule: final epoch, nothing chosen
if cfg.get("val_cfg") is not None or cfg.get("val_dataloader") is not None:
    problems.append("a validation loop exists; Method 03 had none and the "
                    "frozen rule is the final epoch")
if "save_best" in (cfg.default_hooks["checkpoint"] or {}):
    problems.append("CheckpointHook defines save_best; the frozen rule is the "
                    "final epoch, not a selected one")

# 4. the arm must be one that was frozen, with its frozen config bytes
digest = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
match = [a for a, s in freeze["arms"].items() if s["config_sha256"] == digest]
if not match:
    problems.append(f"{sys.argv[1]} does not match any frozen arm's config hash; "
                    "it was edited after the freeze")
elif match[0] == "03":
    problems.append("Method 03 is reused, not retrained; refusing to overwrite it")

if problems:
    print("ABORT: " + "; ".join(problems))
    sys.exit(1)
print(f"guard passed: arm {match[0]}, train={Path(train).name}, test={Path(test).name}, "
      f"workers=2, epochs=60, seed=42, checkpoint rule = final epoch")
GUARD

"${PYTHON_BIN}" tools/train.py "${CONFIG}" "$@"
status=$?
echo "train exit=${status}  end=$(date -Is)"
exit "${status}"
