# ProposalFirst-XRaySeg

Paper code for shared uncertainty-guided point refinement in X-ray prohibited-item instance segmentation: Cascade Mask R-CNN with ResNet-101 and AttFPN, plus matched Transformer and MLP refiners.

**Start with [the reproduction guide](docs/REPRODUCIBILITY.md).** It identifies the ten paper configurations, exact experiment budgets, implementation files, environment requirements and safe commands.

```powershell
python scripts/research.py audit --dataset pixray
python -B -m unittest discover -s tests -p test_research_cli.py -v
```

- Main paper campaign: `output/pixray_multiseed_bs4_e50_2026-09-14/` (ten methods, three seeds, batch 4, 50 epochs).
- Separate PIDray campaign: `output/pidray_bs2_e10_2026-09-22/` (ten methods, seed 42, batch 2, 10 epochs).
- Frozen resolved configs and manifests define the historical experiments. Create new portable configs with `scripts/research.py prepare`; preserve frozen files.
- Datasets and weight binaries are excluded from Git. The local working copy retains all 40 main final checkpoints; see [checkpoint distribution](CHECKPOINTS.md) for the separate release inventory.
- Source restoration provenance and validation: `output/repair_2026-11-06/`.
- Superseded material: `archive/cleanup_2026-11-06/`.

## Current local-copy status

Missing/empty code and all 40 frozen configs were restored from the matching originals on 6 November 2026. Their recorded historical hashes now match. Static checks and matching-source Linux CPU runtime checks passed; native Windows GPU execution was not tested.

The local PIDray image copy remains incomplete, and the manuscript folder plus some historical result/report files are missing or empty. Do not interpret a restored code tree as restored data or results. The asset audit reports these gaps and blocks portable preparation when required inputs fail.

The original cluster-oriented README is preserved at `archive/cleanup_2026-11-06/superseded_documents/README_before_handoff.md`. Its earlier eleven-method/60-epoch launch instructions are not the paper protocol.