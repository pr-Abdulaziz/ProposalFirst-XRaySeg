# Latest experiment removed

Completed on 26 October 2026 at the user's explicit request.

Removed the October 24 guided-context Transformer experiment for PIXray and PIDray: trained campaign v4, same-experiment preparation attempts v1–v3 and initial preparation, preflight artifacts, checkpoints, logs, evaluations, exclusive model/utility/test/review files, local deployment copies and the derived comparison report/visualization. This cleanup did not create a backup of the retired experiment.

The cluster queue was empty before deletion. Training jobs 15001/15002 and evaluation jobs 15003/15004/15005/15051 were already completed; no active job required cancellation. The campaign monitor had already been removed, and no matching local automation configuration was found during this cleanup.

## Preservation checks

| Location | Preserved inventory entries checked | Content hashes checked | Unexpected changes | Remaining removal targets |
|---|---:|---:|---:|---:|
| Cluster project | 79,363 | 1,511 | 0 | 0 |
| Local project | 3,237 | 1,145 | 0 | 0 |

SHA-256 comparisons cover selected small source, configuration and record files. Large checkpoint and dataset files were checked by path, size and modification time, with directory inventories and symlink targets also compared; they were not all rehashed. Local checks cover the retained project tree. Cluster checks cover projects, scripts, configs, tests, reviews, docs, output, work_dirs, data, splits, checkpoints and pretrained, without following directory symlinks.

All earlier experiments, their configurations and results, the Parent checkpoints, original Transformer/MLP models, earlier diagnostics, datasets and shared evaluator/audit repairs were retained. A post-cleanup cluster import check successfully resolved the historical PointRefineHead, PointRefineCascadeRoIHead, PIDrayCascadeRCNN and AttFPN registry entries, and verified the two Parent checkpoint paths and shared repaired scorer paths still exist.

The earlier architecture proposal and project-wide architecture/runtime audits were preserved as historical documents. New planning is recorded separately; no replacement architecture or training run has been launched.

## Receipts

- Local verification: `output/cleanup_context_refiner_2026-10-26/local_cleanup_receipt.json`
- Fetched cluster verification: `output/cleanup_context_refiner_2026-10-26/remote_cleanup_receipt.json`
- Local before/after preserved inventories are in the same directory.
- Cluster before/after inventories and import-check receipt: `/nfs/ahalqahtani/ProposalFirst-XRaySeg/output/cleanup_context_refiner_2026-10-26/`
- Reviewed removal scripts: `tmp/cleanup_context_refiner_2026-10-26/`
- New plan: `docs/SEGMENTATION_RESEARCH_PLAN_2026-10-26.md`

These receipts record authorized removal and verification, not experimental results or model artifacts.
