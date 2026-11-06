#!/usr/bin/env python
"""Generate the three PIXray development-arm configs from one shared overlay.

Three arms that are meant to differ only in architecture must not be three
hand-written files that drift. The overlay -- split, validation loop, evaluator,
checkpoint policy, initialisation -- is one string here and is written verbatim
into each arm, and the script verifies afterwards that the three overlays are
byte-identical.

Re-running this overwrites the three ``pixray_dev_*.py`` configs and nothing
else. It does not touch methods 01-08, their configs or their checkpoints.
"""
from __future__ import annotations

from pathlib import Path

OVERLAY = '''
import os as _os
from pathlib import Path as _Path

# The split lives outside data_root on purpose: the released dataset directory
# stays exactly as distributed, and the split is an external artefact that can
# be regenerated from scripts/build_dev_split.py and checked against its
# manifest. Absolute paths, so ann_file is not joined onto data_root.
dev_split_root = str(
    _Path(
        _os.getenv("PIXRAY_DEV_SPLIT_ROOT", f"{project_root}/splits/pixray_dev_v1")
    ).expanduser().resolve()
)

uses_validation = True

# dev-train only. The 534 dev-val images are absent from training for every arm.
train_dataloader["dataset"]["ann_file"] = f"{dev_split_root}/dev_train.json"

# dev-val images are physically in data/PIXray/train/, so data_prefix is "train/".
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PIXrayDataset",
        data_root=data_root,
        ann_file=f"{dev_split_root}/dev_val.json",
        data_prefix=dict(img="train/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
# Model selection uses mask AP on dev-val and nothing else. Runtime metrics
# belong to the test protocol; measuring them here would only add noise to a
# selection signal. dev_val.json happens to contain no annotation with id 0, so
# the compat keys coincide with the official ones -- enabled anyway so the same
# evaluator code path runs everywhere.
val_evaluator = dict(
    type="XRayCocoMetric",
    ann_file=f"{dev_split_root}/dev_val.json",
    metric=["bbox", "segm"],
    classwise=True,
    zero_id_compat=True,
)
val_cfg = dict(type="ValLoop")
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=2)

# A development run must be structurally incapable of reading the official test
# split. Removing the loader and the loop is that guarantee; it is not a matter
# of remembering to pass the right subcommand.
test_dataloader = None
test_evaluator = None
test_cfg = None

default_hooks["checkpoint"] = dict(
    type="CheckpointHook",
    interval=1,
    max_keep_ckpts=1,
    save_last=True,
    save_best="coco/segm_mAP",
    rule="greater",
)

# Every arm starts from the same ImageNet ResNet-101 initialisation. The
# historical epoch-60 checkpoints were trained on all 3560 training images,
# including the 534 now held out, so using one as a development starting point
# would leak dev-val into all three arms before the first step.
load_from = None
resume = False

del _os, _Path
'''

SHARED = ("All three development arms share one overlay, generated from a single "
          "template by\nscripts/write_dev_configs.py, so they can differ only in the "
          "architecture they\ninherit. Data, schedule, optimiser, batch, augmentation, "
          "evaluation and\ncheckpoint policy are identical by construction.")

ARMS = {
    "pixray_dev_a_cascade_attfpn.py": dict(
        base=".pixray_p2_cascade_attfpn",
        work="pixray_dev_a_cascade_attfpn",
        title="Development arm A: Method 03, no point refinement.",
        body='''Reference arm of the three-way development comparison. Identical to Method 03
in architecture; the only differences from ``pixray_p2_cascade_attfpn`` are the
development split, the dev-val validation loop and the removal of the official
test loader.

This arm is *retrained* on dev-train. The historical Method 03 checkpoint
cannot serve as the development baseline: it saw all 3560 training images,
including the 534 dev-val images, so comparing a dev-train arm against it would
compare two different training sets rather than two architectures.

A vs B isolates the effect of adding point refinement at all.''',
    ),
    "pixray_dev_b_point_refine_mlp.py": dict(
        base=".pixray_09m_point_refine_mlp",
        work="pixray_dev_b_point_refine_mlp",
        title="Development arm B: point refinement with the capacity-matched MLP.",
        body='''Control arm. Adds the refinement head with no attention: every point is
processed independently, 458,127 refinement parameters.

A vs B isolates the effect of adding point refinement at all.
B vs C isolates the effect of attention between points, and nothing else.''',
    ),
    "pixray_dev_c_point_refine_transformer.py": dict(
        base=".pixray_09_point_refine",
        work="pixray_dev_c_point_refine_transformer",
        title="Development arm C: point refinement with the transformer (Method 09).",
        body='''The proposed candidate. Adds the refinement head with two-block self-attention
over the sampled points of each ROI, 458,383 refinement parameters -- 256 more
than arm B, a 0.06 percent difference.

B vs C isolates the effect of attention between points, and nothing else.''',
    ),
}


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "projects/proposal_first_pidray_occ/configs"
    for filename, spec in ARMS.items():
        text = (f'"""{spec["title"]}\n\n{spec["body"]}\n\n{SHARED}\n"""\n\n'
                "from mmengine.config import read_base\n\n"
                f"with read_base():\n    from {spec['base']} import *\n"
                f"{OVERLAY}\n"
                f'work_dir = f"{{project_root}}/work_dirs/{spec["work"]}"\n')
        (out / filename).write_text(text)
        print("wrote", filename)

    blocks = {(out / f).read_text().split("import *\n", 1)[1].rsplit("work_dir =", 1)[0]
              for f in ARMS}
    if len(blocks) != 1:
        print("FAIL overlays are not byte-identical")
        return 1
    print("overlay byte-identical across all three arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
