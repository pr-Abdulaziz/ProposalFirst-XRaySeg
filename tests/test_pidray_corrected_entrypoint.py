"""Exercise the additive wrapper orchestration with lightweight runtime stubs."""
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.campaign_pidray_bs1.evaluation_checks import diagnostic_errors


class CorrectedEntrypointTest(unittest.TestCase):
    def exercise(self, *, missing_image=False, wrong_mapping=False, existing_dump=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"files": {}}))
            manifest = root / "manifest.json"
            work = root / "work"
            manifest.write_text(json.dumps({"runs": {"03:42": {"work_dir": str(work)}},
                                            "source_snapshot": {"per_file": "snapshot.json"}}))
            seen = {}
            C = types.SimpleNamespace(
                SEEDS=(42,), MANIFEST=manifest, OUT=root / "output", TEST_SPLITS=("easy", "hard", "hidden"),
                BY_ID={"03": types.SimpleNamespace(slug="parent")}, run_key=lambda mid, seed: f"{mid}:{seed}",
                snapshot_changes=lambda a, b: dict(modified=[], deleted=[], added=[]),
                source_snapshot=lambda: {"files": {}}, eval_dir=lambda mid, seed: root / "old_evaluations",
                append_jsonl=lambda path, record: seen.update(status_path=path))

            class Runner:
                @staticmethod
                def from_cfg(cfg):
                    seen["runner_created"] = True
                    dataset = types.SimpleNamespace(cat2label={8: 0, 3: 1}, metainfo={"classes": ("b", "a")})
                    return types.SimpleNamespace(test_dataloader=types.SimpleNamespace(dataset=dataset),
                                                 test=lambda: {"coco/segm_mAP": 0.123, "coco/bbox_mAP": 0.123})

            E = types.SimpleNamespace(Runner=Runner, score=lambda *a: None, EVALUATOR_SOURCES=[], Config=object())
            SP = types.SimpleNamespace(to_coco=lambda *a: "original")

            def score(pred, ann):
                seen["converter_mapping"] = SP.to_coco([], "bbox")
                coverage = dict(images_match=not missing_image, prediction_images=2 if missing_image else 3,
                                unique_images=2 if missing_image else 3, test_images=3)
                return dict(coverage=coverage, **{kind: dict(stats={"AP": 0.1234}, stats_zero_id_safe={"AP": 0.1234})
                                                 for kind in ("segm", "bbox")})

            SP.score = score

            def amended_main():
                runner = E.Runner.from_cfg({})
                runner.test()
                seen["full"] = E.score(root / "pred.pkl", root / "ann.json")
                C.append_jsonl(root / "historical_status.jsonl", {})
                return 0

            A = types.SimpleNamespace(main=amended_main, amended_paths=lambda: {})
            namespace = dict(ROOT=root, Path=Path, json=json, os=os, sys=sys, io=io, contextlib=contextlib,
                             COCO=lambda path: object(), C=C, E=E, SP=SP, A=A,
                             diagnostic_errors=diagnostic_errors,
                             label_to_category=lambda gt, classes: {0: 3, 1: 8} if wrong_mapping else {0: 8, 1: 3},
                             to_coco_mapped=lambda samples, kind, mapping: mapping)
            source = Path(__file__).resolve().parents[1] / "scripts/campaign_pidray_bs1/evaluate_run_corrected.py"
            tree = ast.parse(source.read_text())
            # Compile the real entrypoint function, without importing its ML dependencies.
            main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            exec(compile(ast.Module(body=[main_node], type_ignores=[]), str(source), "exec"), namespace)
            originals = (E.Runner, E.score, C.eval_dir, C.append_jsonl, SP.to_coco)
            if existing_dump:
                prediction = work / "test/predictions_jobsynthetic_easy.pkl"
                prediction.parent.mkdir(parents=True)
                prediction.write_bytes(b"original")
            try:
                with patch.object(sys, "argv", [str(source), "03", "42"]), patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic"}):
                    seen["exit_code"] = namespace["main"]()
            finally:
                self.assertEqual((E.Runner, E.score, C.eval_dir, C.append_jsonl, SP.to_coco), originals)
                if existing_dump:
                    self.assertEqual(prediction.read_bytes(), b"original")
                    self.assertNotIn("runner_created", seen)
            return seen

    def test_uses_runtime_category_order_and_separate_status_location(self):
        seen = self.exercise()
        self.assertEqual(seen["exit_code"], 0)
        self.assertEqual(seen["converter_mapping"], {0: 8, 1: 3})
        self.assertIn("evaluations_corrected_v1", str(seen["status_path"]))
        self.assertTrue(seen["full"]["validation"]["passed"])

    def test_missing_image_aborts_despite_matching_ap(self):
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            self.exercise(missing_image=True)

    def test_loader_annotation_mapping_disagreement_aborts(self):
        with self.assertRaisesRegex(RuntimeError, "mappings differ"):
            self.exercise(wrong_mapping=True)

    def test_existing_prediction_dump_is_preserved(self):
        with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
            self.exercise(existing_dump=True)


if __name__ == "__main__":
    unittest.main()
