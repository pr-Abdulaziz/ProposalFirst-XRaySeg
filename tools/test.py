# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import torch

sys.path.append(sys.path[0] + '/..')

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--report',
        help='append evaluation metrics and raw JSON to an Excel workbook')
    parser.add_argument(
        '--report-dataset',
        default='unspecified',
        help='dataset label stored in --report (for example PIDray-A)')
    parser.add_argument(
        '--report-notes', default='', help='provenance note stored in --report')
    parser.add_argument(
        '--test-split',
        choices=('easy', 'hard', 'hidden'),
        help=(
            'select a named annotation file from config.test_split_ann_files; '
            'used by the PIDray protocol'
        ),
    )
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config, import_custom_modules=True)
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.test_split is not None:
        split_files = cfg.get('test_split_ann_files', {})
        if args.test_split not in split_files:
            raise KeyError(
                f'{args.config} does not define test split '
                f'{args.test_split!r}'
            )
        split_ann_file = split_files[args.test_split]
        metric_split_files = cfg.get('metric_split_ann_files', split_files)
        if args.test_split not in metric_split_files:
            raise KeyError(
                f'{args.config} does not define metric split '
                f'{args.test_split!r}'
            )
        metric_ann_file = metric_split_files[args.test_split]
        test_dataset = cfg.test_dataloader.dataset
        while 'dataset' in test_dataset:
            test_dataset = test_dataset.dataset
        test_dataset.ann_file = split_ann_file
        evaluators = (
            cfg.test_evaluator
            if isinstance(cfg.test_evaluator, list)
            else [cfg.test_evaluator]
        )
        for evaluator in evaluators:
            if 'ann_file' in evaluator:
                evaluator.ann_file = metric_ann_file

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # Start testing. MMEngine returns the evaluator metric dictionary.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    evaluation_started = time.perf_counter()
    metrics = runner.test()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_started
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    metrics['efficiency/parameters_total'] = sum(
        parameter.numel() for parameter in model.parameters())
    metrics['efficiency/parameters_trainable'] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    evaluation_images = len(runner.test_dataloader.dataset)
    metrics['efficiency/evaluation_seconds'] = evaluation_seconds
    metrics['efficiency/evaluation_images'] = evaluation_images
    metrics['efficiency/end_to_end_images_per_second'] = (
        evaluation_images / evaluation_seconds if evaluation_seconds else 0.0
    )
    if torch.cuda.is_available():
        metrics['efficiency/peak_gpu_memory_mb'] = (
            torch.cuda.max_memory_allocated() / 1024**2
        )
    training_summary = Path(args.checkpoint).resolve().parent / 'training_summary.json'
    if training_summary.is_file():
        summary = json.loads(training_summary.read_text(encoding='utf-8'))
        for name in ('training_seconds', 'gpu_count', 'gpu_hours'):
            if name in summary:
                metrics[f'efficiency/{name}'] = summary[name]
    if args.report:
        from experiment_reporting import write_experiment_report

        seed = cfg.get('randomness', {}).get('seed')
        report, raw = write_experiment_report(
            args.report,
            metrics,
            config=args.config,
            checkpoint=args.checkpoint,
            seed=seed,
            dataset=args.report_dataset,
            split=args.test_split or 'test',
            notes=args.report_notes,
        )
        print(f'Excel report: {report}')
        print(f'Raw metrics: {raw}')


if __name__ == '__main__':
    main()
