# PIDray-30 and PIDray-A evaluation protocol

## Data contract

Training uses a deterministic seed-42 30% subset containing 23,074 PIDray
images. Evaluation uses the complete official test splits:

| Split | Images |
|---|---:|
| Easy | 24,758 |
| Hard | 9,746 |
| Hidden | 13,069 |
| Total | 47,573 |

Main PIDray and PIDray-A files must contain identical image records and target
polygons for each split. PIDray-A may add `bg_object_segmentation` for overlap
supervision. Normal images are retained because they train the R0 image-level
screening head.

The verified training file contains 3,419 instances with non-empty overlap
masks. The three supplied test files contain the overlap field but no non-empty
overlap masks. Therefore, overlap-aware losses are a training ablation, while
test-time difficulty claims must use Easy, Hard, and Hidden rather than an
unsupported overlap-mask subgroup.

## Fair-comparison contract

1. Every experiment starts from the declared generic ImageNet/SAM weights,
   never another experiment's trained checkpoint.
2. All experiments use seed 42, 60 epochs, 512x512 inputs, the same augmentation,
   AdamW (`lr=1e-4`, weight decay `0.05`), AMP, clipping at 1.0, and effective
   batch 16.
3. P1--P4 use main target annotations. Models that require overlap supervision
   load the paired PIDray-A annotations.
4. Every model is scored with the same main PIDray annotations on each split.
5. Easy, Hard, and Hidden results remain separate; COCO AP is not reconstructed
   by averaging split AP values.
6. Report box and mask AP/AP50/AP75, proposal/detection recall, precision/F1 and
   mask-quality diagnostics from error analysis, latency, throughput, peak GPU
   memory, parameter counts, training time, and GPU-hours.

## Registered experiments

| ID | Architecture difference | Training |
|---|---|---|
| P1 | Cascade Mask R-CNN + FPN | independent 60 epochs |
| P2 | P1 + AttFPN | independent 60 epochs |
| P3 | P2 + recall-enhanced anchors/proposal budget | independent 60 epochs |
| P4 | P2 + learned two-stage Cascade RPN | independent 60 epochs |
| S1 | Proposal-first single SAM decoder | independent 60 epochs |
| S2 | Occ bilayer SAM baseline | independent 60 epochs |
| S3-P | Recall-enhanced proposals + guided BiSAM | independent 60 epochs |
| S3-MBoost | S3-P + residual mask booster | independent 60 epochs |
| E1 | Recall RPN + single decoder + prompt adapter | independent 60 epochs |
| E2 | E1 + hard-aware mask loss | independent 60 epochs |
| E3 | Conditional guided BiSAM + hard-aware loss + mask booster | independent 60 epochs |

## Execution and outputs

```bash
bash scripts/run_pidray30_primary60.sh
```

The launcher audits data, checks contracts, builds and smoke-tests every model,
then trains. Outputs are isolated by experiment and split. The report is
`output/reports/pidray30/pidray30_primary60.xlsx`; error samples are under
`output/error_analysis/pidray30/`; qualitative figures are under
`output/reports/pidray30/qualitative/`.
