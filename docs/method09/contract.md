# Method 09 — tensor, coordinate, embedding, loss and freezing contract

Status of 2026-09-12. **All ten checkpoints verified.** Every shape below is
observed, not assumed.
Every shape below is **observed**, not assumed.

## Runtime

torch 2.1.2+cu118 · mmcv 2.1.0 · mmengine 0.10.7 · mmdet 3.2.0 (vendored at
`/nfs/ahalqahtani/ProposalFirst-XRaySeg/mmdet/__init__.py`) · python 3.10.21.

## Checkpoint 1: baseline trace (job 11038, GPU)

The as-run Method 03 config (`work_dirs/pixray_p2/20260903_194802/vis_data/config.py`)
agrees with the current resolution on every contract field: 15 classes,
AttFPN 256ch x 5 outputs, mask_size 28 at all stages, stage weights
[1, 0.5, 0.25], frozen_stages=1, norm_eval=True, batch 2 x accumulation 8,
AdamW 1e-4, 60 epochs, seed 42, world size 1.

| Step | Observed | Note |
|---|---|---|
| preprocessed input | `[2, 3, 480, 512]` | **not 512x512**; padded to the batch max on a 32 divisor |
| per-image `img_shape` | `(449, 512)`, `(267, 512)` | real content differs; padding is present |
| backbone C2..C5 | `[2,256,120,128]`, `[2,512,60,64]`, `[2,1024,30,32]`, `[2,2048,15,16]` | strides 4/8/16/32 |
| AttFPN P2..P6 | all 256 ch; `120x128`, `60x64`, `30x32`, `15x16`, `8x8` | strides 4/8/16/32/64 |
| stage positive ROIs | `[14,5]`, `[9,5]`, `[7,5]` | **N_s differs per stage**; `[img_idx,x1,y1,x2,y2]` |
| mask ROI features | `[N_s, 256, 14, 14]` | extractor output_size (14,14) |
| stage mask logits | `[N_s, 15, 28, 28]` | foreground channels, no background |
| frozen parameters | 33 tensors | stem + layer1 |
| loss keys | `s0/s1/s2.loss_mask` etc. | cascade wrapper already prefixes per stage |

**Departure from the handoff's worked example:** it assumed a padded 512x512
input and `N_s = 10`. Neither holds. The example was explicitly labelled as
example dimensions, so this is recorded rather than treated as a conflict.

## Checkpoint 2: token embedding — verified

`[N,K,256] + [N,K,C] + [N,K,2] -> [N,K,128]`, observed `(10,196,128)`.
Four branches summed then LayerNorm, per specification. Level embedding has two
entries and **both receive gradient**. Stage conditioning is implemented but
**off**. Class count is a constructor argument: a `C=12` build has
`mask_proj[0].in_features == 12`, and a mismatched input is rejected.
Coarse logits are fed unnormalised. Parameters: **59,648**.

Branch norms on random input: image 13.013, mask 3.257, pos 0.542, level 0.229.
Logged in the pilot; not rescaled.

## Checkpoint 3: transformer — verified

`[10,196,128] -> [10,196,15]`. Two pre-LayerNorm blocks, 4 heads x 32,
FFN 128->512->128, dropout 0, `batch_first=True`, `need_weights=False`.

* **ROI isolation exact**: perturbing ROI 0's tokens changes other ROIs by
  `0.00e+00`.
* **Permutation equivariant** in point order (max diff 1.86e-08), so nothing
  depends on sampling order.
* An all-masked ROI raises rather than producing a NaN softmax.
* Decoder init `std = 0.00097` (target 0.001), bias 0 — small but nonzero.
* Parameters: **398,735**.

**Total new parameters: 458,383.** The handoff's 212,976 figure applied to an
MLP and does not describe this head.

## Checkpoint 4: sampling and coordinates — verified

Delegates to `mmcv.ops.point_sample` (`align_corners=False`) and
`rel_roi_point_to_rel_img_point`; uncertainty is mmdet's `get_uncertainty`,
which is `-|logit|` on the label channel.

* Pixel centres: index 0,1,4,5 on a 2x4 grid give
  `(0.125,0.25) (0.375,0.25) (0.125,0.75) (0.375,0.75)`, i.e.
  `x=(c+0.5)/W`, `y=(r+0.5)/H`.
* On a **non-square 8x12** coordinate-coded map, centre samples `x=5.500
  y=3.500`, first pixel `0.000, 0.000`, last `11.000, 7.000` — exact.
* On an **asymmetric** mask (stripe in the left quarter, top half):
  left-top 1.00, right-top 0.00, left-bottom 0.00. x and y are not swapped; a
  square symmetric fixture would have hidden this.
* Training selection: 196 unique indices per ROI, deterministic under a seeded
  generator. Inference selection: 784 unique, deterministic, and guarded on
  small grids (784 requested on 4x4 returns 16).
* Scatter writes only selected positions and **does not mutate** its input.

## Departures from the specification so far

| Departure | Reason |
|---|---|
| Padded input is 480x512, not 512x512 | Observed; the handoff's figure was an example |
| N_s is 14/9/7, not 10 | Observed; assignments are recomputed per stage |
| New parameters 458,383, not 212,976 | The handoff's figure described an MLP |
| Integer grid points, not continuous | Specified by the handoff as an intentional adaptation |

## Checkpoints 5-6: refinement levels — verified

28x28 -> 56 -> 112, observed `(6, 15, 112, 112)`. The coarse input is not
mutated. Exactly **784 positions change per ROI** at the 56 grid, the rest
keeping their interpolated values. Both level embeddings receive gradient
through the integrated path. Uncertainty selection reads the *current* grid
while token conditioning samples the *original* 28x28 logits, as specified.

Empty batch: a graph-connected zero reaching **41/41 parameters**, so a rank
with no positive ROIs still participates under DDP.

## Checkpoints 7-9: integration on GPU (job 11042) — all passed

* **Exactly one shared `PointRefineHead`**, 458,383 parameters.
* Loss keys `s0/s1/s2.loss_point` only; every diagnostic key omits `loss`, so
  MMEngine's parser never sums a term twice.
* **Stage weight applied exactly once**, verified numerically against the head's
  raw value: 0.69204 -> 0.69204 (w=1), 0.69306 -> 0.34653 (w=0.5),
  0.69219 -> 0.17305 (w=0.25).
* Gradients reach the level embeddings (l0 1.27e-2, l1 1.25e-2), transformer
  (5.1e1), coarse mask heads (3.6e2) and backbone (6.2e3). Frozen stem and
  layer1 stay frozen; 104 backbone BN modules stay in eval.
* AMP: forward finite, gradients finite after unscale, **41/41 refinement
  tensors actually updated**, 1 of 3 early steps skipped by the scaler.
* Inference returns masks at the original resolution (448x880, bool).
* Loading Method 03 weights leaves **41 missing keys, all `point_refine_head`,
  0 unexpected**, and `refine_enabled=False` is **bit-identical to Method 03**.

## Checkpoint 10: bounded pilot (job 11044)

200 optimizer updates, 1600 steps at accumulation 8, 372.8 s (0.233 s/step),
peak **7335 MiB**, 3200 sampled images recorded in `pilot_sample_manifest.json`.

Point loss by stage, first to last logged step:

| stage | start | end |
|---|---|---|
| s0 | 0.6926 | 0.3849 |
| s1 | 0.3462 | 0.2032 |
| s2 | 0.1730 | 0.0999 |

Initialised from Method 03 epoch 60, which has already seen the full PIXray
training split. This is continued optimisation, **not** a matched comparison,
and the pilot is software evidence only.

`refiner |g|` reads 0.0000 at steps 200/400/600/800/1600 because those land on
accumulation boundaries where gradients are zeroed after the update. Not a
failure.

## Defect found and fixed during the pilot

The first pilot attempt died with CUDA OOM trying to allocate 3.19 GiB.
`sample_fine_features` indexed the feature map with a per-ROI image index,
materialising one full-resolution P2 copy per ROI — about 3 GiB at roughly 100
inference detections. Rewritten to sample one image at a time, following
`PointRendRoIHead._get_fine_grained_point_feats`, writing results back by index
so ROI order survives interleaved image indices (re-verified on a fixture with
ROIs from two images interleaved).

**No batch size, output resolution, precision or point budget was reduced.**
This was an implementation defect, not a resource limit.
