"""PIDray paper protocol: full official train set, then one final test.

This alias retains the explicit paper-reproduction work directory.
"""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_legacy import *

train_dataloader['dataset']['ann_file'] = 'annotations/train.json'

# The paper trains on the complete official training set. Do not use the
# official test set for checkpoint selection during this final fit.
val_dataloader = None
val_evaluator = None
val_cfg = None
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=12,
)
default_hooks['checkpoint'] = dict(
    type='CheckpointHook',
    interval=1,
    max_keep_ckpts=1,
    save_last=True,
)
custom_hooks = [dict(
    type='PIDrayCheckpointProvenanceHook',
    training_scope='official_full_train',
    annotation_file='annotations/train.json',
)]
load_from = None
resume = False
work_dir = './work_dirs/pidray_r101_attfpn_r0_paper_full'
