"""Build the Hugging Face release folder: seed-42 weights of the 10 PIXray experiments.

Each checkpoint keeps only `state_dict` and `meta` (optimizer, scheduler and RNG state are dropped).
Absolute project paths in configs and metadata are rewritten relative to the repository root.
"""
import glob
import hashlib
import json
import os
import shutil

import torch

ROOT = '.'
CAMPAIGN = 'pixray_multiseed_bs4_e50_2026-09-14'
WORK = f'{ROOT}/work_dirs/{CAMPAIGN}'
EVAL = f'{ROOT}/output/{CAMPAIGN}/evaluations'
OUT = f'{ROOT}/release/huggingface'
SEED = 42
PREFIX = ROOT + '/'

NAMES = {
    'm01_mask_rcnn_r101_fpn': 'Mask R-CNN R101-FPN',
    'm02_cascade_r101_fpn': 'Cascade Mask R-CNN R101-FPN',
    'm03_cascade_r101_attfpn': 'Cascade Mask R-CNN R101-AttFPN (Parent)',
    'm04_cascade_attfpn_recall_rpn': 'Parent + recall RPN',
    'm05_rsprompter_sam_lora': 'RSPrompter-style SAM (LoRA)',
    'm06_occ_bilayer_sam': 'Occluder-aware bilayer SAM',
    'm07_bilayer_fcn_recall_rpn': 'Bilayer FCN + recall RPN',
    'm08_pidray_r0r1': 'PIDray-style R0/R1 heads',
    'm09_point_refine_transformer': 'Parent + shared point refiner (transformer)',
    'm09m_point_refine_mlp': 'Parent + shared point refiner (MLP)',
}


def rel(obj):
    if isinstance(obj, str):
        return obj.replace(PREFIX, '')
    if isinstance(obj, dict):
        return {k: rel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rel(v) for v in obj]
    return obj


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


rows = []
for exp, name in NAMES.items():
    src_dir = f'{WORK}/{exp}/seed_{SEED}'
    dst = f'{OUT}/{exp}'
    os.makedirs(dst, exist_ok=True)

    ck = torch.load(f'{src_dir}/epoch_50.pth', map_location='cpu')
    torch.save({'meta': rel(ck['meta']), 'state_dict': ck['state_dict']}, f'{dst}/model.pth')
    n_params = sum(v.numel() for v in ck['state_dict'].values())
    del ck

    with open(f'{src_dir}/seed_{SEED}.py') as f:
        cfg = f.read()
    with open(f'{dst}/config.py', 'w') as f:
        f.write(rel(cfg))

    rec = sorted(glob.glob(f'{EVAL}/{exp}/seed_{SEED}/eval_record_job*.json'))[-1]
    met = sorted(glob.glob(f'{EVAL}/{exp}/seed_{SEED}/metrics_fullprec_job*.json'))[-1]
    for src, name_out in ((rec, 'eval_record.json'), (met, 'metrics.json')):
        with open(src) as f:
            data = json.load(f)
        with open(f'{dst}/{name_out}', 'w') as f:
            json.dump(rel(data), f, indent=1)

    with open(met) as f:
        m = json.load(f)
    rows.append((exp, name, 100 * m['segm']['stats']['AP'], 100 * m['bbox']['stats']['AP'],
                 n_params, sha256(f'{dst}/model.pth')))
    print(exp, 'done', flush=True)

with open(f'{OUT}/checkpoints.json', 'w') as f:
    json.dump([dict(experiment=r[0], name=r[1], mask_ap=round(r[2], 2), box_ap=round(r[3], 2),
                    parameters=r[4], model_sha256=r[5]) for r in rows], f, indent=1)
for r in rows:
    print(f'{r[0]:32s} mask {r[2]:.2f} box {r[3]:.2f}')
