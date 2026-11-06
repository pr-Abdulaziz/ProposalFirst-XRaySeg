#!/usr/bin/env python
"""Campaign amendment: activation checkpointing in Method 06's SAM encoder (user decision A, 2026-09-17).

06 seed 42 (job 11331) ran out of GPU memory at batch 4. Nothing frozen is edited. This script:

  --configs    writes write-once amended resolved configs for 06 (all three seeds) under
               protocol/amendments/06_activation_checkpointing/. Each differs from its frozen config
               in exactly two places: the encoder type (CheckpointedMMPretrainSamVisionEncoder) and
               one extra custom import. This is verified on the loaded configs.
  --finalize   after the GPU job (equivalence test and preflight on the amended config) passed,
               writes a write-once amended manifest: the frozen manifest with 06's three resolved
               configs and 06's preflight replaced, and an amendment record. Training of 06 and
               evaluation of all runs then read the amended manifest through the *_amended wrappers.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    BY_ID, MANIFEST, OUT, SEEDS, resolved_config, run_key, sha256, write_once)

AMEND_DIR = OUT / "protocol" / "amendments" / "06_activation_checkpointing"
AMENDED_MANIFEST = OUT / "summaries" / "amended_manifest_06_checkpointing.json"
RECORD = OUT / "summaries" / "amendment_06_activation_checkpointing.json"
EQUIV = AMEND_DIR / "equivalence_check.json"
PREFLIGHT = OUT / "preflight" / "m06_occ_bilayer_sam_checkpointed" / "preflight.json"
OLD_TYPE = "type='MMPretrainSamVisionEncoder'"
NEW_TYPE = "type='CheckpointedMMPretrainSamVisionEncoder'"
OLD_IMPORT = "        'mmdet.occlu',\n"
NEW_IMPORT = "        'mmdet.occlu',\n        'projects.proposal_first_pidray_occ.models.checkpointed_sam_encoder',\n"


def amended_config(seed: int) -> Path:
    return AMEND_DIR / f"seed_{seed}.py"


def flat(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, repr(obj)


def write_configs() -> None:
    from mmengine.config import Config
    AMEND_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        src = resolved_config("06", seed)
        text = src.read_text()
        assert text.count(OLD_TYPE) == 1 and text.count(OLD_IMPORT) == 1, f"unexpected frozen config {src}"
        new = text.replace(OLD_TYPE, NEW_TYPE).replace(OLD_IMPORT, NEW_IMPORT)
        dst = amended_config(seed)
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w") as f:
            f.write(new)
        a = dict(flat(Config.fromfile(str(src), import_custom_modules=False).to_dict()))
        b = dict(flat(Config.fromfile(str(dst), import_custom_modules=False).to_dict()))
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        expected = ["custom_imports.imports[1]", "custom_imports.imports[2]", "model.backbone.type"]
        if diff != expected:
            raise SystemExit(f"ABORT: seed {seed} differs in {diff}, expected {expected}")
        print(f"seed {seed}: {dst} sha256={sha256(dst)} differences={diff}")


def finalize() -> None:
    eq = json.loads(EQUIV.read_text())
    pre = json.loads(PREFLIGHT.read_text())
    problems = []
    if eq.get("status") != "PASS":
        problems.append(f"equivalence check {eq.get('status')}")
    if pre["summary"]["failed"]:
        problems.append(f"preflight failed checks: {[c['name'] for c in pre['checks'] if c['status'] == 'FAIL']}")
    if pre["config_sha256"] != sha256(amended_config(42)):
        problems.append("preflight ran on a different config")
    if problems:
        raise SystemExit("ABORT: " + "; ".join(problems))
    frozen = json.loads(MANIFEST.read_text())
    man = copy.deepcopy(frozen)
    for seed in SEEDS:
        run = man["runs"][run_key("06", seed)]
        run["frozen_resolved_config"], run["frozen_resolved_config_sha256"] = run["resolved_config"], run["resolved_config_sha256"]
        run["resolved_config"] = str(amended_config(seed).relative_to(ROOT))
        run["resolved_config_sha256"] = sha256(amended_config(seed))
    man["preflight"]["06_frozen_config"] = man["preflight"]["06"]
    man["preflight"]["06"] = dict(result=str(PREFLIGHT.relative_to(ROOT)), job=pre["job"], node=pre["node"], gpu=pre["gpu"],
                                  passed=pre["summary"]["passed"], waived_checks=[],
                                  peak_allocated_mb=pre["peak_allocated_mb_max"], peak_reserved_mb=pre["peak_reserved_mb_max"],
                                  timing=pre["timing"], equivalence_check=str(EQUIV.relative_to(ROOT)))
    man["amendments"] = [
        dict(file="output/pixray_multiseed_bs4_e50_2026-09-14/summaries/scope_amendment_seed42.json", status="superseded by restore_3seeds"),
        dict(file="output/pixray_multiseed_bs4_e50_2026-09-14/summaries/scope_amendment_restore_3seeds.json"),
        dict(file=str(RECORD.relative_to(ROOT)))]
    write_once(AMENDED_MANIFEST, man)
    write_once(RECORD, dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        instruction="User (2026-09-17) chose option A: activation checkpointing in Method 06's SAM image encoder "
                    "after 06 seed 42 (job 11331) ran out of GPU memory at batch 4.",
        change="model.backbone.type MMPretrainSamVisionEncoder -> CheckpointedMMPretrainSamVisionEncoder, plus its import; "
               "no other configuration difference (verified on the loaded configs)",
        unchanged="batch 4, no accumulation, 50 epochs, optimiser, schedule, data, annotations, seeds, evaluation; "
                  "every other method; the frozen manifest itself",
        effect="memory and speed only; recomputation replays RNG and autocast state; equivalence test on real data below",
        failed_attempt=dict(job="11331", error="torch.cuda.OutOfMemoryError in mask_cross_entropy (22.56 GiB allocated)"),
        waiver_withdrawn="the frozen ~95% memory waiver for 06 no longer applies; the amended preflight passes the 90% criterion",
        equivalence=eq, preflight=dict(result=str(PREFLIGHT.relative_to(ROOT)), summary=pre["summary"],
                                       peak_allocated_mb=pre["peak_allocated_mb_max"], timing=pre["timing"]),
        frozen_manifest_sha256=sha256(MANIFEST), amended_manifest=str(AMENDED_MANIFEST.relative_to(ROOT)),
        amended_manifest_sha256=sha256(AMENDED_MANIFEST),
        amended_configs={s: dict(path=str(amended_config(s).relative_to(ROOT)), sha256=sha256(amended_config(s))) for s in SEEDS}))
    print(f"amended manifest {AMENDED_MANIFEST} sha256={sha256(AMENDED_MANIFEST)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--configs", action="store_true")
    g.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    write_configs() if a.configs else finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
