"""B5: RSPrompter on PIXray under this repository's protocol.

The occlusion-aware paper compares its model against RSPrompter and attributes
the difference to occlusion-aware bilayer decoding. B5 is that comparison run
here: it inherits S2 and reverts only the occlusion-specific components, so
B5 against S2 differs by the bilayer decoder and its supervision and nothing
else.

RSPrompter has no occluder branch, so B5 reads the target annotations
(``train.json``) rather than the occlusion track. The two files describe the
same instances -- byte-identical category, box and mask for all 3,560 train and
1,486 test images -- so this changes what is loaded, not what is supervised.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_s2_occ_bisam import *

work_dir = f"{project_root}/work_dirs/pixray_b5"

# --- revert the occlusion-aware heads to their RSPrompter originals ---------
model["roi_head"]["type"] = "RSPrompterAnchorRoIPromptHead"
model["roi_head"]["mask_head"]["type"] = "RSPrompterAnchorMaskHead"
model["roi_head"]["mask_head"]["mask_decoder"]["type"] = "RSSamMaskDecoder"
# Inert upstream, but leaving it would misdescribe the model.
# RSPrompter has no occluder branch, so the bilayer-only keys inherited from
# S2 are dropped rather than left to be swallowed by **kwargs.
model["roi_head"]["mask_head"].pop("loss_mask_bo", None)
model["roi_head"]["mask_head"].pop("bo_loss_weight", None)

# --- revert the occlusion-aware data path ----------------------------------
model["data_preprocessor"]["type"] = "DetDataPreprocessor"
model["data_preprocessor"]["batch_augments"] = [dict(
    type="BatchFixedSizePad",
    size=crop_size,
    img_pad_value=0,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
)]

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomResize",
        scale=crop_size,
        ratio_range=(0.5, 1.5),
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=crop_size,
        crop_type="absolute",
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(
        type="FilterAnnotations",
        min_gt_bbox_wh=(1e-5, 1e-5),
        by_mask=True,
        keep_empty=True,
    ),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="Resize", scale=crop_size, keep_ratio=True),
    dict(
        type="Pad",
        size=crop_size,
        pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0),
    ),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id", "img_path", "ori_shape", "img_shape", "pad_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader["dataset"] = dict(
    type="PIXrayDataset",
    metainfo=dict(classes=CLASSES),
    data_root=data_root,
    ann_file="annotations/train.json",
    data_prefix=dict(img="train/"),
    pipeline=train_pipeline,
)
test_dataloader["dataset"] = dict(
    type="PIXrayDataset",
    metainfo=dict(classes=CLASSES),
    data_root=data_root,
    ann_file="annotations/test.json",
    data_prefix=dict(img="test/"),
    test_mode=True,
    pipeline=test_pipeline,
)
