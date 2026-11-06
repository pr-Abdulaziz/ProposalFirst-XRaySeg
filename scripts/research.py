#!/usr/bin/env python3
"""Safe paper entry point: audit assets or prepare a new portable run config.

Uses the standard library. Does not submit jobs, train, install packages, or edit
frozen experiment records. Run --help, then read docs/REPRODUCIBILITY.md.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pprint
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = '/nfs/ahalqahtani/ProposalFirst-XRaySeg'
CAMPAIGNS = {'pixray': 'pixray_multiseed_bs4_e50_2026-09-14',
             'pidray': 'pidray_bs2_e10_2026-09-22'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def local_path(root, value):
    value = str(value).replace('\\', '/')
    if value == OLD_ROOT or value.startswith(OLD_ROOT + '/'):
        value = value[len(OLD_ROOT):].lstrip('/')
    p = Path(value)
    result = p if p.is_absolute() else root / p
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Project path escapes root: {value}')
    return result


def literal(node):
    """Read resolved literals/dict(...) without executing Python configuration."""
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [literal(v) for v in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys): raise ValueError('Dictionary unpacking unsupported')
        return {literal(k): literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub,ast.UAdd)):
        v = literal(node.operand)
        if type(v) not in (int,float): raise ValueError('Non-numeric unary expression')
        return -v if isinstance(node.op,ast.USub) else v
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id=='dict' and not node.args:
        if any(k.arg is None for k in node.keywords): raise ValueError('Keyword unpacking unsupported')
        return {k.arg: literal(k.value) for k in node.keywords}
    raise ValueError(f'Nonliteral config expression rejected: {type(node).__name__}')


def read_config(path):
    result = {}
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    for n in tree.body:
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):
            raise ValueError('Only resolved assignment configs are supported')
        result[n.targets[0].id] = literal(n.value)
    if not result or 'model' not in result: raise ValueError('Empty or unresolved model config')
    return result


def rebase(value, root):
    if isinstance(value, dict): return {k:rebase(v,root) for k,v in value.items()}
    if isinstance(value,list): return [rebase(v,root) for v in value]
    if isinstance(value,tuple): return tuple(rebase(v,root) for v in value)
    if isinstance(value,str) and (value==OLD_ROOT or value.startswith(OLD_ROOT+'/')):
        return str(local_path(root,value))
    return value


def manifest(root, dataset):
    p = root/'output'/CAMPAIGNS[dataset]/'campaign_manifest.json'
    return json.loads(p.read_text()),p


def image_check(root, dataset, annotation):
    data = json.loads(annotation.read_text(encoding='utf-8-sig'))
    train = annotation.name.startswith('train')
    prefixes = ['train'] if train else ['test']
    if dataset=='pidray' and train: prefixes = ['train','train/train0','train/train1']
    missing=[];empty=[]
    for row in data['images']:
        paths=[root/'data'/('PIXray' if dataset=='pixray' else 'PIDray')/prefix/row['file_name'] for prefix in prefixes]
        existing=next((p for p in paths if p.is_file()),None)
        if existing is None: missing.append(row['file_name'])
        elif existing.stat().st_size==0: empty.append(row['file_name'])
    return {'annotation':str(annotation),'images':len(data['images']),
            'missing_count':len(missing),'empty_count':len(empty),
            'missing_examples':missing[:10],'empty_examples':empty[:10]}


def audit(root, dataset):
    man,mp = manifest(root,dataset)
    bad_configs=[]
    for key,run in man['runs'].items():
        p=local_path(root,run['resolved_config'])
        if not p.is_file() or sha(p)!=run['resolved_config_sha256']:bad_configs.append(key)
    snap=json.loads(local_path(root,man['source_snapshot']['per_file']).read_text())['files']
    bad_sources=[p for p,h in snap.items() if not local_path(root,p).is_file() or sha(local_path(root,p))!=h]
    ann=[];bad_ann=[]
    for key,v in man['annotations'].items():
        p=local_path(root,v['path'])
        if not p.is_file() or sha(p)!=v['sha256']:bad_ann.append(key);continue
        if key.startswith('occ_'):continue
        ann.append(image_check(root,dataset,p))
    weights=[]
    for key,v in man['pretrained'].items():
        p=local_path(root,v['path'])
        if not p.is_file() or p.stat().st_size==0:weights.append(key)
    ready=not (bad_configs or bad_sources or bad_ann or weights or any(a['missing_count'] or a['empty_count'] for a in ann))
    return dict(dataset=dataset,manifest=str(mp),asset_status='PASS' if ready else 'INCOMPLETE',
                configs_checked=len(man['runs']),bad_configs=bad_configs,bad_sources=bad_sources,
                bad_annotations=bad_ann,missing_or_empty_weights=weights,images=ann,
                limits='Image paths/sizes and frozen source/config/annotation hashes checked. Weights presence only. Runtime, image decoding and accuracy are separate checks.')


def prepare(root,dataset,method,seed,name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):raise ValueError('Unsafe run name')
    man,mp=manifest(root,dataset)
    matches=[r for r in man['runs'].values() if r['method']==method and r['seed']==seed and r.get('launchable',True)]
    if len(matches)!=1:raise ValueError('Method/seed not uniquely approved in this campaign')
    readiness=audit(root,dataset)
    if readiness['asset_status']!='PASS':raise ValueError('Dataset/source audit failed; run audit and restore missing assets first')
    run=matches[0];src=local_path(root,run['resolved_config']);original=read_config(src)
    cfg=rebase(original,root)
    if cfg.get('resume') is not False or cfg.get('load_from') is not None:
        raise ValueError('Frozen config is not a fresh-start protocol')
    work=root/'work_dirs'/'reproductions'/name/dataset/f"method_{method}"/f"seed_{seed}"
    dest=root/'output'/'reproductions'/name/dataset/f"method_{method}"/f"seed_{seed}"
    if work.exists() or dest.exists():raise FileExistsError('Run name already used; choose a new one')
    cfg['work_dir']=str(work)
    text='\n\n'.join(f'{k} = {pprint.pformat(v,sort_dicts=False,width=100)}' for k,v in cfg.items())+'\n'
    compile(text,'prepared_config.py','exec')
    dest.mkdir(parents=True)
    target=dest/'config.py';target.write_text(text,encoding='utf-8')
    record=dict(source_manifest=str(mp),source_config=str(src),source_sha256=sha(src),
                prepared_sha256=sha(target),changes='Project-root paths and fresh work_dir only',
                training_started=False,work_dir=str(work))
    (dest/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return dict(status='PREPARED_ONLY',config=str(target),work_dir=str(work))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit',help='Read-only paper asset checks; works without ML packages')
    a.add_argument('--dataset',choices=['pixray','pidray','all'],default='pixray')
    p=sub.add_parser('prepare',help='Write a portable derivative config in a new run folder')
    p.add_argument('--dataset',choices=CAMPAIGNS,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--name',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='audit':
            names=list(CAMPAIGNS) if args.dataset=='all' else [args.dataset]
            result=[audit(ROOT,n) for n in names]
            print(json.dumps(result,indent=2));return 0 if all(x['asset_status']=='PASS' for x in result) else 2
        print(json.dumps(prepare(ROOT,args.dataset,args.method,args.seed,args.name),indent=2));return 0
    except (ValueError,FileNotFoundError,FileExistsError,KeyError) as e:
        print(f'ABORT: {e}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())