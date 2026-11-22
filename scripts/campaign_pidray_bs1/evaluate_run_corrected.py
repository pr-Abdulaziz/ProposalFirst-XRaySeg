#!/usr/bin/env python
"""Opt-in PIDray evaluation with corrected labels and strict validation.

Run inside an allocated GPU job, using the project environment:
    python scripts/campaign_pidray_bs1/evaluate_run_corrected.py 03 42

Reuses the frozen evaluator and verified easy-split amendment without editing
them. Results/status go to evaluations_corrected_v1/<method>/seed_<seed>/job<id>.
Prediction dumps retain the frozen evaluator's work-directory convention; the
wrapper refuses an existing dump or output directory before inference starts.
This entrypoint does not submit a job, train, or modify historical evaluation
records. Its image-weighted summary retains the legacy name; it is not pooled AP.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO  # noqa: E402
import scripts.campaign_multiseed.score_predictions as SP  # noqa: E402
from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_pidray_bs1 import evaluate_run as E  # noqa: E402
from scripts.campaign_pidray_bs1 import evaluate_run_amended as A  # noqa: E402
from scripts.campaign_pidray_bs1.coco_resample_mapped import label_to_category, to_coco_mapped  # noqa: E402
from scripts.campaign_pidray_bs1.evaluation_checks import diagnostic_errors  # noqa: E402


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: evaluate_run_corrected.py <method id> [<seed>]")
    mid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) == 3 else C.SEEDS[0]
    manifest = json.loads(C.MANIFEST.read_text())
    run = manifest["runs"][C.run_key(mid, seed)]
    # The frozen evaluator checks only evaluator hashes. Also protect all model
    # source bytes before re-evaluating a historical checkpoint.
    frozen = json.loads((ROOT / manifest["source_snapshot"]["per_file"]).read_text())["files"]
    changes = C.snapshot_changes(frozen, C.source_snapshot()["files"])
    if changes["modified"] or changes["deleted"]:
        raise SystemExit(f"ABORT: frozen source changed: {changes}")
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    out = C.OUT / "evaluations_corrected_v1" / C.BY_ID[mid].slug / f"seed_{seed}" / f"job{job}"
    if out.exists():
        raise SystemExit(f"ABORT: output already exists: {out}; use a fresh evaluation job")
    for split in C.TEST_SPLITS:
        pred = Path(run["work_dir"]) / "test" / f"predictions_job{job}_{split}.pkl"
        if pred.exists():
            raise SystemExit(f"ABORT: refusing to overwrite {pred}; use a fresh evaluation job")
    # Resolve/hash-check the amendment before any inference or output mutation.
    A.amended_paths()
    original_runner, original_score = E.Runner, E.score
    original_eval_dir, original_append = C.eval_dir, C.append_jsonl
    original_sources, original_config = E.EVALUATOR_SOURCES, E.Config
    context = {}

    class CheckedRunner:
        @staticmethod
        def from_cfg(cfg):
            runner = original_runner.from_cfg(cfg)
            dataset = runner.test_dataloader.dataset
            # These are the actual loader's label assignments, not sorted IDs.
            context.clear()
            context["mapping"] = {int(label): int(category) for category, label in dataset.cat2label.items()}
            context["classes"] = tuple(dataset.metainfo["classes"])
            original_test = runner.test

            def test():
                metrics = original_test()
                context["metrics"] = metrics
                return metrics

            runner.test = test
            return runner

    def checked_score(pred, ann):
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(ann))
        mapping = label_to_category(gt, context["classes"])
        if mapping != context["mapping"]:
            raise RuntimeError("loader and evaluation annotation label/category mappings differ")
        old_converter = SP.to_coco
        try:
            SP.to_coco = lambda samples, kind: to_coco_mapped(samples, kind, mapping)
            full = SP.score(pred, ann)
        finally:
            SP.to_coco = old_converter
        checks = {}
        for kind in ("segm", "bbox"):
            rounded = context["metrics"][f"coco/{kind}_mAP"]
            official, safe = full[kind]["stats"]["AP"], full[kind]["stats_zero_id_safe"]["AP"]
            checks[kind] = dict(mmdet=rounded, official=official, zero_id_safe=safe,
                                matches=bool(min(abs(round(official, 3) - rounded),
                                                 abs(round(safe, 3) - rounded)) < 1.5e-3))
        errors = diagnostic_errors({"current": dict(coverage=full["coverage"], check=checks)}, ("current",))
        full["category_mapping"] = mapping
        full["validation"] = dict(passed=not errors, errors=errors, check=checks)
        if errors:
            raise RuntimeError("corrected evaluation failed: " + "; ".join(errors))
        return full

    out.mkdir(parents=True, exist_ok=False)
    try:
        E.Runner, E.score = CheckedRunner, checked_score
        C.eval_dir = lambda method, run_seed: out
        C.append_jsonl = lambda path, record: original_append(out / "status.jsonl", record)
        E.EVALUATOR_SOURCES = original_sources + [
            "scripts/campaign_pidray_bs1/evaluate_run_corrected.py",
            "scripts/campaign_pidray_bs1/evaluate_run_amended.py",
            "scripts/campaign_pidray_bs1/evaluation_checks.py",
            "scripts/campaign_pidray_bs1/coco_resample_mapped.py"]
        return A.main()
    finally:
        E.Runner, E.score, E.Config = original_runner, original_score, original_config
        E.EVALUATOR_SOURCES = original_sources
        C.eval_dir, C.append_jsonl = original_eval_dir, original_append


if __name__ == "__main__":
    sys.exit(main())
