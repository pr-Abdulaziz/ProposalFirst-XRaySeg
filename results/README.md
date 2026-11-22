# Results and analysis

Outputs of the PIXray campaign: 10 experiments x 3 seeds (42, 123, 2026), 50 epochs, official test split (1,486 images).

- `evaluations/<experiment>/seed_<n>/`: `metrics_fullprec.json` (COCO box and mask metrics, per class, per IoU)
  and `eval_record.json` (checkpoint and config checksums, software versions, throughput).
- `predictions/<experiment>/seed_<n>.pkl`: raw test-set predictions (MMDetection format); every statistic
  can be recomputed from these without a GPU.
- `statistics/statistics.json`: two-level paired bootstrap, permutation and TOST results, Holm adjustments.
- `analysis/`: paired object-level mask quality (IoU, Dice, boundary IoU).
- `logs/<experiment>/seed_<n>/`: training logs and scalar curves.
- `protocol/`: resolved configs of all runs, campaign manifest with checksums, fairness audit, amendments.
- `figures/`: campaign plots (PDF and PNG). Qualitative figures that show PIXray images are not
  included; obtain the dataset from its authors to regenerate them.
- `supplementary/`: inference-time ablation tables, BCa intervals, parameter counts, small-object and
  boundary-subset analyses, and the per-run results of the inference-time ablations.
