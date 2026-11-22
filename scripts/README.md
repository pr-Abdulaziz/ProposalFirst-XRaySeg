# Portable PIXray and PIDray-30 GPU workflows

Run commands from a Linux GPU allocation after copying or cloning the complete
project. Scripts derive the project root from their own location.

## Fresh machine

```bash
cd /path/to/ProposalFirst-XRaySeg
bash scripts/setup.sh
bash scripts/prepare_assets.sh
# Run one dataset workflow at a time:
bash scripts/run_pixray_primary60.sh
# After PIXray completes and the PIDray leakage gate passes:
bash scripts/run_pidray30_primary60.sh
```

Each dataset-specific command performs its read-only dataset audit, configuration
preflights, GPU smoke tests, eleven isolated 60-epoch trainings, evaluation,
error analysis, and qualitative comparison. The PIXray workflow must finish
before PIDray-30 begins. The PIDray content-hash leakage audit remains a hard
gate. Long training cannot begin until all safety gates pass.

The PIDray command performs the read-only dataset audit, eleven-config contract
check, model-build preflights, one-batch GPU smoke tests, eleven isolated
60-epoch trainings, three-split evaluation, error analysis, and qualitative
comparison. Long training cannot begin until all safety gates pass.

## External mounts

```bash
export PIDRAY_DATA_ROOT=/shared/data/PIDray
export PIDRAYA_ANN_ROOT=/shared/data/PIDray-A/annotations
export SAM_VIT_LARGE_DIR=/shared/models/sam_vit_large
export SAM_VIT_LARGE_CKPT="$SAM_VIT_LARGE_DIR/pytorch_model_new.bin"
bash scripts/run_pidray30_primary60.sh
```

## Uniform low-memory profile

If physical batch 4 fails a smoke test, apply the same profile to every run:

```bash
export XRAY_BATCH_SIZE_PER_GPU=2
export XRAY_EFFECTIVE_BATCH_SIZE=16
bash scripts/run_pidray30_primary60.sh
```

Do not combine runs made with different physical-batch or GPU profiles in a
runtime comparison. Checkpoints are isolated under
`work_dirs/pidray30_<experiment>/`. The combined workbook is
`output/reports/pidray30/pidray30_primary60.xlsx` and the log is
`output/logs/pidray30_primary60.log`.

To train or evaluate selected runs:

```bash
python scripts/run_pixray_experiments.py train --experiments P1 P2
python scripts/run_pixray_experiments.py evaluate --experiments P1 P2
python scripts/run_pidray30_experiments.py train --experiments P1 P2
python scripts/run_pidray30_experiments.py evaluate --experiments P1 P2 --splits easy hard hidden
```
