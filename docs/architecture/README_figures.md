# Architecture figures (corrected versions)

Deterministic vector graphics. Every coordinate is a literal in the scripts; re-running
them reproduces byte-identical SVG. No image-generation model was used, and no model
code, configuration or result file was modified.

## Files

| Figure | Source | Exports |
|---|---|---|
| 1. Overall framework | `fig1_overall_framework.py` | `fig1_overall_framework.{svg,pdf,png}` |
| 2. Transformer point refiner | `fig23_point_refiners.py` | `fig2_transformer_point_refiner.{svg,pdf,png}` |
| 3. MLP point refiner | `fig23_point_refiners.py` | `fig3_mlp_point_refiner.{svg,pdf,png}` |
| 4. ResNet-101 + AttFPN hierarchy | `fig4_backbone_attfpn.py` | `fig4_backbone_attfpn.{svg,pdf,png}` |
| 5. Segmentation framework (inference) | `fig5_segmentation_framework.py`, `make_fig5_assets.py` | `fig5_segmentation_framework.{svg,pdf,png}` |
| 6. Transformer point processor | `fig67_point_processors.py` | `fig6_transformer_point_processor.{svg,pdf,png}` |
| 7. MLP point processor | `fig67_point_processors.py` | `fig7_mlp_point_processor.{svg,pdf,png}` |

Shared primitives: `draw_common.py`. Originals preserved unchanged in `original/`.

Regenerate:

    .venv/bin/python fig1_overall_framework.py
    .venv/bin/python fig23_point_refiners.py

All figures are 7.0 in wide (504 pt). SVG keeps live `<text>` elements, so it stays
editable in Inkscape/Illustrator; PDF embeds TrueType (`pdf.fonttype 42`) and keeps text
selectable; PNG is 600 dpi (4200 px wide). Word: insert the PDF for print, the PNG when
a raster is required.

## Source of truth for every label

`projects/proposal_first_pidray_occ/models/point_refine/`: `head.py` (refinement loop,
`refine()`), `embeddings.py` (token construction), `transformer.py` (`_EncoderBlock`,
decoder), `mlp_control.py` (`_MixerBlock`), `sampling.py` (`scatter_points`,
`select_*_points`).

## Corrections made against the previous versions

**Figure 1 — overall framework**

1. The refinement procedure is now shown explicitly instead of as a single "point
   refinement" box: upsample → select → sample → embed → mix → decode → scatter.
2. Point selection is fed by the **evolving** mask `M̃^(s,r)`, while token conditioning
   is fed by the **original** coarse logits `M^(s)`. The old figure left both implicit;
   the two sources are now separate arrows, and the detector feed is labelled
   `M^(s) (original)`.
3. Added the refinement **loop**: the updated mask `M^(s,r)` returns to the bilinear
   upsampling step (`r: 56 → 112`). It does not pass through the embedding summation.
4. `u_i` now feeds sampling and the coordinate embedding only — it no longer implies any
   connection to scatter. Scatter receives exactly three inputs: `M̃^(s,r)`, `S`, `ℓ`.
5. Scatter is labelled "replace selected cells", not an addition or a residual update.
6. Added the explicit `P_2` route from AttFPN to the refiner, and marked that the cascade
   has **three separate** coarse mask heads (stacked outline).
7. Selection is labelled "Uncertainty sampling", not "Top-P": training mixes uncertain
   and random locations (196 points, 3× oversample, 75 % uncertain).
8. Dimensions added where they matter: `M^(s)` 15×28×28, `M^(s,112)` 15×112×112,
   `Z ∈ R^(P×128)`, `f_i·256`, `m_i·15`.
9. Removed the parameter counts, the point-budget sentence and the long footer; the
   footer is now `28² → 56² → 112² · shared across s and r`.
10. Transformer and MLP are drawn as two dashed alternatives joined by "or"
    ("alternatives"), instead of a single ambiguous box.

**Figures 2 and 3 — refiner variants**

11. Identical layouts and component positions; only the point-mixing sublayer differs.
12. The embedding no longer sums raw inputs of different widths. Each branch shows its
    projection — `φ_f: 256→128`, `φ_m: 15→64→128` (GELU), `φ_p: 2→128→128` (GELU) — and
    every line into `⊕` is labelled **128**. `e_r ∈ R^128` goes straight to the
    summation, with no identity layer. No cascade-stage embedding is drawn (it exists in
    code but is off in the reported configuration).
13. `m_i` is annotated "from `M^(s)`", distinguishing it from the evolving mask.
14. One complete block is drawn, with the container labelled "×2 sequential"; the old
    figures showed two boxes that could be read as parallel branches.
15. Both residual connections now start **before** their LayerNorm (junction dot),
    bypass the complete sublayer, and end at their own `⊕` with a visible arrowhead.
16. Sublayer widths are explicit: MHSA `4×32`, FFN `128→512→128`, point-wise MLP
    `128→256→128`, decoder `LN → Linear 128→15`. No sigmoid in the decoder: it emits
    logits, and scatter operates on logits.
17. Scatter's three inputs are labelled (`ℓ`, `S`, `M̃^(s,r)`) with `S` and `M̃^(s,r)`
    entering as compact external arrows; the output is `M^(s,r)`.
18. Removed: parameter counts, time-complexity expressions, the repeated
    selection/refinement flow and full-sentence footers. One point-budget note remains:
    `P = 196 train / 784 test, per RoI and level`.

## Suggested captions

**Figure 1.** Cascade Mask R-CNN with a shared point refiner applied at two resolutions.
Original coarse logits condition point embeddings, while the evolving mask guides point
selection.

**Figures 2–3.** Transformer and MLP refiners use the same embedding, decoder and
refinement procedure. The transformer exchanges information among selected points within
an RoI; the MLP processes points independently.

## Validation performed

1. Every input to `⊕` is 128-dimensional and labelled as such.
2. The original coarse-mask input `M^(s)` is the embedding source at both levels; the
   evolving mask is drawn on a separate path.
3. The two internal blocks are sequential ("×2 sequential" on the container).
4. Each residual bypasses a complete sublayer (pre-LN → sublayer → add).
5. Scatter receives the grid, the indices and the decoded logits — nothing else.
6. The second level starts from the updated first-level mask (explicit loop arrow).
7. Transformer and MLP are separate alternatives (dashed, "or", separate figures).
8. Rendered at 7 in and inspected; no clipped or overlapping labels, no connector
   crossings in Figure 1.

**Figure 4 — ResNet-101 + AttFPN** (checked against the resolved config of method 03 and `projects/proposal_first_pidray_occ/models/attfpn.py`)

19. ResNet-101 stages drawn with their real block counts (3, 4, 23, 3); stem and res2 shaded as frozen (`frozen_stages=1`), plus the missing fact that BatchNorm statistics are frozen in every stage (`norm_eval=True`).
20. "MaxPool ×2" replaced by the actual stem: one 7×7 stride-2 conv and one stride-2 max-pool. C1 is stated as unused (`out_indices=(0,1,2,3)`), instead of being drawn as a pyramid input.
21. AttFPN split into its two real stages: an mmdet FPN top-down pass, then five selective-attention modules, one per output level, each gathering all five levels (resize, level softmax over channel and spatial attention, GC context block, residual).
22. P6 is drawn as what it is — a stride-2 max-pool of P5 (`add_extra_convs=False`) — rather than an independent output.
23. Added C2–C5 channel counts and, for a 512×512 input, spatial sizes for every feature map.
24. Added consumers: RPN uses P2–P6, RoIAlign (7×7 box, 14×14 mask) uses P2–P5 only, the point refiner (09, 09M) samples P2.
25. Removed the malformed C5 block and the "P1: not produced" note from the image-generated original.

**Figure 4, revision 2 (labelling pass; previous version kept as `original/v1_fig4_backbone_attfpn.*`)**

26. Block interiors reduced to module names: Stem, Res2–Res5, FPN fusion, Selective attention. Convolution specifications, block counts and the attention paragraph removed from this overview.
27. Tensor identifiers are the main labels ($C_2$–$C_5$, $P_2$–$P_6$); properties sit outside in smaller text as "256 ch" / "stride 4". The output group states "256 channels per level" once.
28. Every number is labelled; no "s4" shorthand and no bare numbers. The input-dependent spatial sizes (128², 64², …) and their 512 × 512 footnote are removed.
29. Usage notes reduced to three lines (RPN $P_2$–$P_6$, RoIAlign $P_2$–$P_5$, Point refiner $P_2$), with experiment IDs removed. The fixed RoIAlign output sizes are kept as a separate note: box RoI 7 × 7, mask RoI 14 × 14.
30. FPN and selective-attention blocks shrunk to fit their labels. Selective attention now feeds the five levels through one distribution line. The "C1 is not used" sentence is dropped: C₁ is no longer drawn, and the missing connection from the stem shows it.

31. The "Image" block is replaced by a real PIXray training image containing a prohibited item (a pistol): `data/PIXray/train/102036201.png`, cropped to the bag. The crop is stored in `assets/fig4_input_pixray_train_102036201.png` so the figure does not depend on the dataset path. SVG and PDF embed it as a raster; everything else stays vector.

Suggested caption: "ResNet-101 + AttFPN feature hierarchy. Grey stages are frozen. AttFPN applies a standard FPN top-down pass followed by five selective-attention modules, one per output level, each attending over all five levels. P6 is a stride-2 max-pool of P5; RoIAlign reads P2–P5 and the RPN reads P2–P6. Input: a PIXray training image containing a pistol."


## Figures 5-7

**Revision 2 (review feedback, 2026-09-23; previous versions kept as `original/v1_*`).** Figure 5: tagged "inference workflow"; point refiner annotated "shared across stages and refinement levels" and "56 × 56 → 112 × 112", with "Transformer or MLP" moved inside the block; "Paste" renamed "Select class + paste"; output labelled "predicted mask overlay"; C defined in the caption. Figures 6-7: attention equation removed, leaving "4 heads · 32 ch per head"; containers shortened by about 13%.

Colour code, shared by the three figures and stated in each legend: peach = learned module, white = parameter-free operation, blue = tensor. No in-figure titles, no 3D slabs, no decorative blobs. Both images in Figure 5 are real data.

**Figure 5 — segmentation framework** (checked against `PointRefineCascadeRoIHead.predict_mask` and mmdet `CascadeRoIHead`)

32. Mask heads are fed as the code feeds them: **RoIAlign (256 ch · 14 × 14) on P₂–P₅ at the final detection boxes b**. The original drew them fed directly by the "RPN + cascade" block and omitted RoIAlign.
33. The three stage branches are drawn **once**, inside a dashed plate "for each stage s = 1, 2, 3", instead of in triplicate. The mask head is marked separate per stage; the point refiner is marked shared across stages.
34. The inference order is exactly the code's: refine each stage → σ per stage → mean over s → paste the channel of the predicted class c into box b → threshold p ≥ 0.5.
35. Detections (b, c) feed RoIAlign (boxes) and Paste (boxes and class). The one unavoidable crossing, boxes over the P₂ line, is drawn as a line hop.
36. The fixed class count is written as C (15 on PIXray, 12 on PIDray) rather than hard-coding 15.
37. Real input and output: PIXray test image `101627037.png` and the predicted mask of the MLP variant (09M, seed 42), mask IoU 0.949 with the ground truth. The original's schematic bottle and fuzzy probability blobs are removed. The unused ⊕ legend entry is removed.

**Figures 6-7 — point processors** (checked against `transformer.py` `_EncoderBlock` and `mlp_control.py` `_MixerBlock`)

38. Identical geometry; only the mixing sublayer differs.
39. The attention sublayer shows what `nn.MultiheadAttention` computes: separate W_Q, W_K, W_V projections, scaled dot-product attention softmax(QKᵀ/√d_h)V with 4 heads of d_h = 32 ch (stacked sheets), concatenation, then W_O. The original fed LayerNorm output straight into the heads and had no Q/K/V projections.
40. Attention is labelled "among the P points of one RoI". In the code the RoI is the batch axis, so points of different RoIs never attend to each other.
41. The MLP mixer and the FFN are marked "each point independently". That makes the matched-control logic visible: attention is the only place points interact.
42. Pre-LN residuals start at a junction before each LayerNorm and bypass the whole sublayer. Only hidden widths that change are annotated (256 for the MLP, 512 for the FFN), and the 128-channel token width is stated at Z and Z′.
43. Removed: 3D tensor slabs, in-figure titles, the legend entries for self-evident symbols, and the "Points processed independently" text that overflowed its box.

Suggested captions:

**Figure 5.** Inference workflow of the instance-segmentation framework. Detections (b, c) from the RPN and three-stage cascade define the RoIs for RoIAlign on P₂–P₅. For each stage s, a separate mask head predicts coarse logits M^(s) (C × 28 × 28). The point refiner, shared across stages and refinement levels, refines them at 56 × 56 and then 112 × 112 using fine features from P₂. Stage probabilities are averaged; the channel of the predicted class is selected, pasted into the detection box and thresholded. C = 15 classes on PIXray (12 on PIDray). During training each stage instead uses its own positive RoIs. Input and output are a PIXray test image and the predicted mask of the MLP variant, shown as an overlay.

**Figure 6.** Transformer point processor. Two pre-LayerNorm blocks with separate parameters. Multi-head self-attention (4 heads, d_h = 32) exchanges information among the P selected points of one RoI; the feed-forward network acts on each point independently.

**Figure 7.** MLP point processor, the matched control. Identical to Figure 6 except that self-attention is replaced by a point-wise MLP, so no information passes between points.
