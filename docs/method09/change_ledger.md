# Method 09 — change ledger

Every departure from the implementation handoff, with its reason.

| Departure | Reason |
|---|---|
| Padded input observed as `[2,3,480,512]`, not 512x512 | Measured. Batches pad to the batch maximum on a 32 divisor; the handoff labelled 512x512 as an example. |
| `N_s` observed as 14/9/7, not 10 | Measured. Each cascade stage reassigns and resamples. |
| New parameters 458,383, not 212,976 | The handoff states its figure described an MLP. Reported as measured. |
| Integer grid points rather than continuous sampling | Specified by the handoff as an intentional adaptation. |
| `sample_fine_features` rewritten to loop over images | Defect fix: per-ROI indexing of the feature map allocated ~3 GiB and caused OOM. Follows the repository's own PointRend pattern. No budget reduced. |
| Point loss returned as a single `loss_point` key | `CascadeRoIHead` weights every `loss_mask` entry containing `loss`. One key means the stage weight is applied exactly once; per-level values are reported under keys without `loss` so MMEngine's parser does not sum them. |
| `merge_aug_masks` / `empty_instances` imported from `mmdet.models.test_time_augs` and `mmdet.models.utils.misc` | The paths in the handoff sketch do not exist in mmdet 3.2; these are the paths the parent class uses. |

## Not implemented, by instruction

224 output level; relative-position attention bias; stage embeddings (built but
disabled); Dice, boundary, consistency or amodal losses; residual-logit
correction; cross-ROI attention.

## Untested

Multi-rank DDP. Production training here is single-GPU (world size 1 confirmed
from the Method 03 summary), so no DDP path is exercised. The empty-ROI
graph-connected zero reaching all 41 parameters is implemented for it, but the
multi-rank case itself is unrun.
