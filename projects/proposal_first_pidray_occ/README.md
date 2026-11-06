# Proposal-first PIDray segmentation package

This package contains the detector, proposal, SAM-prompting, bilayer decoding,
mask-booster, screening, and evaluation components used by the PIDray-30 study.

## Active data

The active configs load the deterministic 23,074-image training subset from
`data/PIDray`, paired overlap supervision from `data/PIDray-A/annotations`, and
the complete Easy, Hard, and Hidden test splits. Every numerical comparison is
scored against the same main PIDray annotations.

## Experiment matrix

| ID | Controlled architecture |
|---|---|
| P1 | Cascade Mask R-CNN + FPN |
| P2 | P1 + AttFPN |
| P3 | P2 + smaller anchors and a larger proposal budget |
| P4 | P2 + learned two-stage Cascade RPN at P2's proposal budget |
| S1 | Proposal-first single SAM decoder |
| S2 | Occ bilayer SAM baseline |
| S3-P | Recall-enhanced proposals + guided BiSAM |
| S3-MBoost | S3-P + residual mask correction |
| E1 | Recall RPN + prompt adapter + single decoder |
| E2 | E1 + hard-aware target-mask learning |
| E3 | Conditional guided BiSAM + hard-aware loss + mask correction |

Every experiment trains independently for 60 epochs from its declared generic
ImageNet/SAM initialization. Shared optimization, augmentation, batch, seed,
checkpoint, and evaluation settings are enforced by
`tools/check_hybrid_contracts.py`.

Run the complete workflow from the project root:

```bash
bash scripts/run_pidray30_primary60.sh
```

See [`docs/evaluation-protocol.md`](../../docs/evaluation-protocol.md) for the
data and reporting contract.

