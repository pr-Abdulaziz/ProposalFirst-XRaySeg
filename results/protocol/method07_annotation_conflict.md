# Method 07 (bilayer FCN mask head): annotation-policy conflict

**Status: decided 2026-09-14. Option B: keep historical 07 with PIXray-A occluder supervision; reported beside 06 as an additional-supervision comparison.**

## The conflict

The campaign rule is that only the Occ-paper method (06) may consume Occ occluder annotations.

The historical Method 07 (`pixray_b8_bilayer_fcn_recall_rpn.py`) is Method 04 plus `BilayerFCNMaskHead`. It trains on `data/PIXray-A/annotations/train_occ.json` through the `Oc*` pipeline, and it supervises a second, class-agnostic occluder mask. That occluder loss (weight 0.25) comes from `bg_object_segmentation`. Occluder supervision is what defines it, so it cannot run under the rule as it stands.

## Can it run target-only?

Not unchanged, and not as the same method. From `projects/proposal_first_pidray_occ/models/bilayer_fcn_mask_head.py` and `bilayer_cascade_roi_head.py`:

1. **It needs occluder masks.** During training the RoI head always requests occluder logits. `get_bo_targets` then raises `KeyError` unless `gt_instances.bo_masks` exists. Pointing the loader at `train.json` would crash. Setting `bo_loss_weight=0` still reads the masks. Making it run target-only needs a code change.
2. **Without occluder targets, part of the head gets no gradient at all.** The occluder upsample and 1×1 occluder classifier (262,657 parameters per stage) would be dead weight. Keeping them would be exactly the "unused branch" the brief rules out.
3. **The rest of the branch collapses into extra capacity.** The four occluder convolutions and the residual projection (2,426,112 parameters per stage, 7.28 M over three stages) would still get gradient, but only through `target_feat + residual_proj(bo_feat)`. They become a second 4-conv trunk summed into the target features, which makes the head wider, not occlusion-aware.

So a target-only 07 is a meaningful model, but a different one. It is a capacity or dual-trunk mask-head variant of Method 04, not the historical bilayer method.

## Options

| Option | What runs | Scientific identity | Rule |
|---|---|---|---|
| **A. Target-only replacement "07T"** | Method 04 + dual-trunk FCN mask head. The occluder upsample and classifier are removed and the loss reads nothing but target masks; `train.json` with the standard pipeline. Needs a small, reviewed code path (a flag that builds the head without the occluder output and never requests occluder targets), plus its own preflight. | New name and description: "04 + extra parallel mask-head trunk (+7.28 M params)". 04 → 07T isolates added mask-head capacity. Not comparable to historical 07 results. | Complies |
| **B. Keep historical 07** | Unchanged bilayer FCN with PIXray-A occluder supervision | Same method as before; 04 → 07 tests occluder supervision without SAM | Requires an explicit exception: a second additional-supervision method beside 06 |
| **C. Exclude 07 from this campaign** | Nothing | Historical 07 results stay in the archive only | Complies, but drops the arm, so it needs your explicit approval |

**Recommendation.** Choose **B** if the question is still "does occluder supervision help a conventional mask head?", reported next to 06 as an additional-supervision comparison. Choose **A** only if you want a target-only capacity control for 04. It should then be named and described as a different method.

## Effect on the queue

The account's QOS runs one job at a time. Three 07 runs can be appended to the queue after your decision without touching any frozen run. They would go in a versioned manifest addendum; the frozen manifest is never edited.
