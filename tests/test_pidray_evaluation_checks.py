"""Synthetic CPU regressions; no training libraries, data, checkpoints, or jobs."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.campaign_pidray_bs1.evaluation_checks import (
    AP_TOLERANCE, PIDRAY_VALIDATION_CASES, coverage_errors, diagnostic_errors,
    mapped_analysis_errors, write_new_report)


def valid_coverage():
    return dict(images_match=True, prediction_images=3, unique_images=3, test_images=3)


def valid_splits():
    return {split: dict(coverage=valid_coverage(),
                        check={kind: dict(matches=True) for kind in ("segm", "bbox")})
            for split in ("easy", "hard", "hidden")}


def valid_report():
    pixray = dict.fromkeys(("mapping_is_identity", "errors_identical", "mask_quality_identical",
                           "pr_curves_identical", "per_gt_identical"), True)
    pixray.update(ap_cache_matches_scorer=dict(segm=0.0, bbox=0.0),
                  ap_cache_self_check=dict(segm=0.0, bbox=0.0))
    rows = [dict(method=method, split=split, ap_cache_vs_rescored=dict(segm=0.0, bbox=0.0),
                 ap_cache_self_check=dict(segm=0.0, bbox=0.0),
                 tp_mapped_analysis=10, tp_corrected_scorer=10.0)
            for method, split in PIDRAY_VALIDATION_CASES]
    return dict(pixray_regression=pixray, pidray=rows)


class DiagnosticChecksTest(unittest.TestCase):
    def test_complete_matching_splits_pass(self):
        self.assertEqual(diagnostic_errors(valid_splits()), [])

    def test_matching_metrics_cannot_hide_missing_images(self):
        splits = valid_splits()
        splits["hidden"]["coverage"].update(images_match=False, prediction_images=2, unique_images=2)
        self.assertTrue(diagnostic_errors(splits))

    def test_duplicate_images_fail_even_if_id_set_matches(self):
        record = valid_coverage()
        record["prediction_images"] = 4
        self.assertTrue(coverage_errors(record))

    def test_incomplete_or_unexpected_splits_fail(self):
        for name in ("easy", "hard", "hidden"):
            with self.subTest(split=name):
                splits = valid_splits()
                del splits[name]
                self.assertTrue(diagnostic_errors(splits))
        splits = valid_splits()
        splits["unexpected"] = copy.deepcopy(splits["easy"])
        self.assertTrue(diagnostic_errors(splits))

    def test_each_metric_agreement_is_required(self):
        for kind in ("segm", "bbox"):
            with self.subTest(kind=kind):
                splits = valid_splits()
                splits["hard"]["check"][kind]["matches"] = False
                self.assertTrue(diagnostic_errors(splits))

    def test_missing_coverage_is_not_success(self):
        splits = valid_splits()
        del splits["easy"]["coverage"]
        self.assertTrue(diagnostic_errors(splits))

    def test_invalid_count_is_not_success(self):
        for value in (None, -1, 3.0, True):
            with self.subTest(value=value):
                coverage = valid_coverage()
                coverage["unique_images"] = value
                self.assertTrue(coverage_errors(coverage))


class MappedAnalysisChecksTest(unittest.TestCase):
    def test_valid_report_passes(self):
        self.assertEqual(mapped_analysis_errors(valid_report()), [])

    def test_each_false_regression_flag_fails(self):
        for name, value in valid_report()["pixray_regression"].items():
            if value is True:
                with self.subTest(flag=name):
                    report = valid_report()
                    report["pixray_regression"][name] = False
                    self.assertTrue(mapped_analysis_errors(report))

    def test_each_ap_difference_and_self_check_is_enforced(self):
        cases = [("pixray_regression", field) for field in
                 ("ap_cache_matches_scorer", "ap_cache_self_check")]
        cases += [("pidray", field) for field in ("ap_cache_vs_rescored", "ap_cache_self_check")]
        for group, field in cases:
            for kind in ("segm", "bbox"):
                with self.subTest(group=group, field=field, kind=kind):
                    report = valid_report()
                    row = report[group] if group == "pixray_regression" else report[group][0]
                    row[field][kind] = 2 * AP_TOLERANCE
                    self.assertTrue(mapped_analysis_errors(report))

    def test_small_roundoff_is_permitted(self):
        report = valid_report()
        report["pixray_regression"]["ap_cache_self_check"]["segm"] = AP_TOLERANCE / 2
        self.assertEqual(mapped_analysis_errors(report), [])

    def test_nan_infinity_negative_and_missing_differences_fail(self):
        for value in (float("nan"), float("inf"), -1e-12, None, True):
            with self.subTest(value=value):
                report = valid_report()
                report["pidray"][0]["ap_cache_self_check"]["segm"] = value
                self.assertTrue(mapped_analysis_errors(report))

    def test_tp_mismatch_fails(self):
        report = valid_report()
        report["pidray"][1]["tp_mapped_analysis"] += 1
        self.assertTrue(mapped_analysis_errors(report))

    def test_invalid_tp_counts_fail(self):
        for value in (-1, 10.5, float("nan"), float("inf"), True, None):
            with self.subTest(value=value):
                report = valid_report()
                report["pidray"][0]["tp_mapped_analysis"] = value
                report["pidray"][0]["tp_corrected_scorer"] = value
                self.assertTrue(mapped_analysis_errors(report))

    def test_missing_and_duplicate_validation_cases_fail(self):
        report = valid_report()
        report["pidray"].pop()
        self.assertTrue(mapped_analysis_errors(report))
        report = valid_report()
        report["pidray"][2] = copy.deepcopy(report["pidray"][0])
        self.assertTrue(mapped_analysis_errors(report))

    def test_missing_report_fields_fail(self):
        self.assertTrue(mapped_analysis_errors({}))
        report = valid_report()
        del report["pixray_regression"]["per_gt_identical"]
        self.assertTrue(mapped_analysis_errors(report))

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_new_report(path, {"original": True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new_report(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(path.read_text()), {"original": True})


if __name__ == "__main__":
    unittest.main()
