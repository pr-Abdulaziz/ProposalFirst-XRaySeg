# Method 09 — next commands (not launched)

## Measured resource requirements

From pilot job 11044 and integration job 11042:

* peak GPU memory **7335 MiB** (fits the 24 GiB RTX 4090 with wide margin)
* **0.233 s/step** at physical batch 2, accumulation 8
* 1780 steps/epoch x 60 epochs = 106,800 steps
* estimated **~6.9 GPU-hours** per 60-epoch run, against Method 03's measured 6.37

## Development matrix — three matched arms, not yet launched

All share Method 03's schedule, batch, optimizer and evaluator. Only the mask
refinement differs.

```bash
# 1. matched Method 03 baseline (already trained; reuse only if provenance matches)
#    work_dirs/pixray_p2/epoch_60.pth

# 2. per-point MLP control: same pipeline, embeddings, schedule and budgets,
#    transformer replaced. Config not yet written.
# 3. primary Method 09 transformer
EXPERIMENT=09 sbatch scripts/slurm_train_pixray_one.sh
```

Before any of these: register 09 in `registry_seven.py` and
`run_pixray_experiments.py`, which has not been done — Method 09 is currently
config-only and deliberately outside the active registry.

## Development split

No legitimate development split exists in this project; every historical
checkpoint was trained on the full training split and evaluated on the official
test set. Selecting Method 09's design on that test set would contaminate it.
A training-only development split must be proposed and its leakage checked
before the matched comparison. This is unresolved and blocks a fair development
campaign.

## Evaluation

```bash
EXPERIMENTS="09" REPORT=output/reports/pixray_09/method09.xlsx \
  sbatch scripts/slurm_eval_pixray.sh
```

## PIDray

Not prepared, not authorized by this handoff. `C` is a constructor argument
throughout (embedding `Linear(C,64)`, decoder `Linear(128,C)`, class gathers,
scatter expansion), and `D=128` with 4 heads is independent of class count, so
the migration needs a config change and a state-dict shape diff, not code edits.
