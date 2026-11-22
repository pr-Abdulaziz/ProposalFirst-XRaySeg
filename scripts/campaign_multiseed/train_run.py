#!/usr/bin/env python
"""Train one campaign run from its frozen resolved config.

Identical to tools/train.py except that CUDA_LAUNCH_BLOCKING is 0 (the historical
entry point forced synchronous kernel launches, which slows training and changes
no computation) and that no config value can be overridden from the command line.

    python scripts/campaign_multiseed/train_run.py <resolved_config.py> [--resume]
"""
import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import gc  # noqa: E402

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402


def main() -> int:
    config = sys.argv[1]
    resume = "--resume" in sys.argv[2:]
    cfg = Config.fromfile(config, import_custom_modules=True)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.launcher = "none"
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory before training: {free / 2**20:.0f} MiB free of {total / 2**20:.0f} MiB; "
              f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
    if resume:
        cfg.resume = True
        cfg.load_from = None  # MMEngine then resumes from work_dir/last_checkpoint only
    runner = Runner.from_cfg(cfg)
    runner.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
