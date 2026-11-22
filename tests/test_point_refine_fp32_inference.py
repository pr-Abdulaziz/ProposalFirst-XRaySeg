"""CPU contracts for opt-in FP32 prediction; no claim about AMP training.

Run from the deployed project root:
    PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover \
        -s tests -p test_point_refine_fp32_inference.py -v
"""
from __future__ import annotations

import importlib
import unittest
from unittest import mock

import torch

PREFIX = "projects.proposal_first_pidray_occ.models.point_refine"
stable = importlib.import_module(PREFIX + ".stable_inference")
fp32 = importlib.import_module(PREFIX + ".fp32_inference")
HEAD_PAIRS = (
    (stable.PointRefineHeadStableInference, fp32.PointRefineHeadFP32Inference),
    (stable.PointRefineMLPHeadStableInference, fp32.PointRefineMLPHeadFP32Inference),
    (stable.PointRefineHeadDetachedStableInference,
     fp32.PointRefineHeadDetachedFP32Inference),
    (stable.PointRefineMLPHeadDetachedStableInference,
     fp32.PointRefineMLPHeadDetachedFP32Inference),
)


class FP32PredictionTests(unittest.TestCase):
    @staticmethod
    def heads(base_cls, new_cls, classes=12):
        kwargs = dict(num_classes=classes, train_points=8, infer_points=12)
        torch.manual_seed(42)
        base = base_cls(**kwargs)
        torch.manual_seed(42)
        new = new_cls(**kwargs)
        return base, new

    @staticmethod
    def inputs(classes=12):
        return (
            torch.randn(1, classes, 28, 28),
            torch.tensor([[0., 4., 5., 52., 53.]]),
            torch.tensor([classes - 1], dtype=torch.long),
            torch.randn(1, 256, 16, 16),
            torch.rand(1, 64, 64),
        )

    def same(self, left, right):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_state_dict_and_fp32_predictions_match_stable(self):
        for classes in (12, 15):
            for base_cls, new_cls in HEAD_PAIRS:
                with self.subTest(head=new_cls.__name__, classes=classes):
                    base, new = self.heads(base_cls, new_cls, classes)
                    self.assertEqual(list(base.state_dict()), list(new.state_dict()))
                    for name, value in base.state_dict().items():
                        self.same(value, new.state_dict()[name])
                    base.load_state_dict(new.state_dict(), strict=True)
                    new.load_state_dict(base.state_dict(), strict=True)
                    base.eval()
                    new.eval()
                    coarse, rois, labels, fine, _ = self.inputs(classes)
                    with torch.no_grad():
                        expected = base.predict_refine(coarse, rois, labels, fine)
                        actual = new.predict_refine(coarse, rois, labels, fine)
                    self.same(expected, actual)
                    self.assertEqual(actual.dtype, torch.float32)

    def test_half_inputs_cpu_autocast_and_direct_refine(self):
        for base_cls, new_cls in HEAD_PAIRS:
            with self.subTest(head=new_cls.__name__):
                base, new = self.heads(base_cls, new_cls)
                base.eval()
                new.eval()
                coarse, rois, labels, fine, _ = self.inputs()
                # Compare the exact half-quantized values promoted to FP32,
                # not the higher-precision values lost by initial quantization.
                coarse, rois, fine = coarse.half(), rois.half(), fine.half()
                with torch.no_grad():
                    expected = base.predict_refine(
                        coarse.float(), rois.float(), labels, fine.float())
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        actual = new.predict_refine(coarse, rois, labels, fine)
                        refined, losses, diagnostics = new.refine(
                            coarse, rois, labels, fine, training=False)
                self.same(expected, actual)
                self.same(expected, refined)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(losses, [])
                self.assertEqual(diagnostics, {"point_count_l0": 12.,
                                               "point_count_l1": 12.})

    def test_training_outputs_gradients_and_rng_are_unchanged(self):
        for base_cls, new_cls in HEAD_PAIRS:
            with self.subTest(head=new_cls.__name__):
                base, new = self.heads(base_cls, new_cls)
                coarse, rois, labels, fine, gt = self.inputs()
                coarse.requires_grad_()
                fine.requires_grad_()
                new_coarse = coarse.detach().clone().requires_grad_()
                new_fine = fine.detach().clone().requires_grad_()
                gen_base = torch.Generator().manual_seed(123)
                gen_new = torch.Generator().manual_seed(123)
                left = base.loss_and_refine(
                    coarse, rois, labels, fine, gt, generator=gen_base)
                with mock.patch.object(
                    new, "_require_fp32_prediction_weights",
                    side_effect=AssertionError("training entered prediction path"),
                ):
                    right = new.loss_and_refine(
                        new_coarse, rois, labels, new_fine, gt, generator=gen_new)
                self.assertEqual(set(left), set(right))
                for name in left:
                    self.same(left[name], right[name])
                self.assertTrue(torch.equal(gen_base.get_state(), gen_new.get_state()))
                left["loss_point"].backward()
                right["loss_point"].backward()
                for (name, p), (new_name, q) in zip(
                    base.named_parameters(), new.named_parameters(),
                ):
                    self.assertEqual(name, new_name)
                    self.assertEqual(p.grad is None, q.grad is None)
                    if p.grad is not None:
                        self.same(p.grad, q.grad)
                for p, q in ((coarse, new_coarse), (fine, new_fine)):
                    self.assertEqual(p.grad is None, q.grad is None)
                    if p.grad is not None:
                        self.same(p.grad, q.grad)

    def test_empty_predictions_are_fp32_and_half_weights_fail_clearly(self):
        for base_cls, new_cls in HEAD_PAIRS:
            with self.subTest(head=new_cls.__name__):
                _, new = self.heads(base_cls, new_cls)
                coarse = torch.empty(0, 12, 28, 28, dtype=torch.float16)
                rois = torch.empty(0, 5, dtype=torch.float16)
                labels = torch.empty(0, dtype=torch.long)
                fine = torch.randn(1, 256, 16, 16, dtype=torch.float16)
                actual = new.predict_refine(coarse, rois, labels, fine)
                self.assertEqual(actual.shape, coarse.shape)
                self.assertEqual(actual.dtype, torch.float32)
                new.half()
                with self.assertRaisesRegex(TypeError, "float32 head weights"):
                    new.predict_refine(coarse, rois, labels, fine)


if __name__ == "__main__":
    unittest.main()
