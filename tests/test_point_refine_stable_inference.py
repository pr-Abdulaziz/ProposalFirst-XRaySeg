"""CPU regression checks for the opt-in inference selector (stdlib unittest).

From the deployed project root:
    PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover \
        -s tests -p test_point_refine_stable_inference.py -v

No datasets, checkpoints, CUDA allocation, or output files are required.
The normal project runtime dependencies must be installed.
"""
from __future__ import annotations

import importlib
import unittest
from unittest import mock

import torch

PREFIX = "projects.proposal_first_pidray_occ.models.point_refine"
stable = importlib.import_module(PREFIX + ".stable_inference")
original = importlib.import_module(PREFIX + ".head")
mlp = importlib.import_module(PREFIX + ".mlp_control")
detached = importlib.import_module(PREFIX + ".detached")

HEAD_PAIRS = (
    (original.PointRefineHead, stable.PointRefineHeadStableInference),
    (mlp.PointRefineMLPHead, stable.PointRefineMLPHeadStableInference),
    (detached.PointRefineHeadDetached,
     stable.PointRefineHeadDetachedStableInference),
    (detached.PointRefineMLPHeadDetached,
     stable.PointRefineMLPHeadDetachedStableInference),
)


class StableSelectionTests(unittest.TestCase):
    def test_unequal_scores_keep_true_ranking_and_class_channels(self):
        for size in (56, 112):
            for dtype in (torch.float16, torch.bfloat16, torch.float32,
                          torch.float64):
                with self.subTest(size=size, dtype=dtype):
                    # Unselected classes are more uncertain everywhere: they
                    # must have no influence on the selected class's ranking.
                    logits = torch.zeros(2, 3, size, size, dtype=dtype)
                    labels = torch.tensor([0, 2])
                    for roi, channel in enumerate(labels.tolist()):
                        flat = logits[roi, channel].view(-1)
                        flat.fill_(20)
                        flat[0] = 1
                        # FP32 difference is below the legacy ramp even at56.
                        # FP64 difference detects any accidental float32 cast.
                        difference = (1e-10 if dtype == torch.float64 else
                                      1e-4 if dtype == torch.float32 else 1)
                        flat[-1 - roi] = 1 - difference
                    before = logits.clone()
                    selected = stable.select_inference_points_stable(
                        logits, labels, num_points=2)
                    expected = torch.tensor([[size * size - 1, 0],
                                             [size * size - 2, 0]])
                    self.assertTrue(torch.equal(selected, expected))
                    self.assertTrue(torch.equal(logits, before))

    def test_exact_ties_are_ascending_and_repeatable(self):
        for size in (56, 112):
            for dtype in (torch.float16, torch.float32, torch.float64):
                with self.subTest(size=size, dtype=dtype):
                    logits = torch.full((2, 3, size, size), 0.25, dtype=dtype)
                    labels = torch.tensor([2, 1])
                    expected = torch.arange(19).expand(2, -1)
                    rng_before = torch.get_rng_state().clone()
                    for _ in range(2):
                        result = stable.select_inference_points_stable(
                            logits, labels, num_points=19)
                        self.assertTrue(torch.equal(result, expected))
                    self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_empty_rois_and_point_budgets(self):
        for size in (56, 112):
            for dtype in (torch.float16, torch.float32):
                with self.subTest(size=size, dtype=dtype):
                    result = stable.select_inference_points_stable(
                        torch.empty(0, 3, size, size, dtype=dtype),
                        torch.empty(0, dtype=torch.long))
                    self.assertEqual(result.shape, (0, 784))
                    self.assertEqual(result.dtype, torch.long)
        logits = torch.ones(1, 1, 2, 2)
        labels = torch.tensor([0])
        self.assertEqual(stable.select_inference_points_stable(
            logits, labels, 0).shape, (1, 0))
        self.assertTrue(torch.equal(stable.select_inference_points_stable(
            logits, labels, 99), torch.arange(4).view(1, 4)))
        with self.assertRaises(ValueError):
            stable.select_inference_points_stable(logits, labels, -1)


class HeadCompatibilityTests(unittest.TestCase):
    @staticmethod
    def make_pair(base_cls, stable_cls, classes=12):
        # Exercise actual two-level defaults, reducing only point budgets.
        kwargs = dict(num_classes=classes, train_points=8, infer_points=12)
        torch.manual_seed(42)
        base = base_cls(**kwargs)
        torch.manual_seed(42)
        new = stable_cls(**kwargs)
        return base, new

    @staticmethod
    def inputs(classes=12, empty=False):
        n = 0 if empty else 1
        coarse = torch.randn(n, classes, 28, 28, requires_grad=True)
        fine = torch.randn(1, 256, 16, 16, requires_grad=True)
        rois = torch.tensor([[0., 4., 5., 52., 53.]])[:n]
        labels = torch.tensor([classes - 1], dtype=torch.long)[:n]
        gt = torch.rand(n, 64, 64)
        return coarse, rois, labels, fine, gt

    def assert_tensor_equal(self, left, right):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_default_state_dict_names_values_and_strict_loading(self):
        for classes in (12, 15):
            for base_cls, stable_cls in HEAD_PAIRS:
                with self.subTest(head=stable_cls.__name__, classes=classes):
                    base, new = self.make_pair(base_cls, stable_cls, classes)
                    self.assertEqual(list(base.state_dict()), list(new.state_dict()))
                    for key, value in base.state_dict().items():
                        self.assert_tensor_equal(value, new.state_dict()[key])
                    new.load_state_dict(base.state_dict(), strict=True)
                    base.load_state_dict(new.state_dict(), strict=True)

    def test_training_outputs_losses_gradients_and_rng_unchanged(self):
        for base_cls, stable_cls in HEAD_PAIRS:
            with self.subTest(head=stable_cls.__name__):
                base, new = self.make_pair(base_cls, stable_cls)
                coarse, rois, labels, fine, gt = self.inputs()
                coarse_new = coarse.detach().clone().requires_grad_()
                fine_new = fine.detach().clone().requires_grad_()
                gen_base = torch.Generator().manual_seed(123)
                gen_new = torch.Generator().manual_seed(123)
                left = base.loss_and_refine(
                    coarse, rois, labels, fine, gt, stage=2, generator=gen_base)
                with mock.patch.object(
                    stable, "select_inference_points_stable",
                    side_effect=AssertionError("training used inference selector"),
                ):
                    right = new.loss_and_refine(
                        coarse_new, rois, labels, fine_new, gt, stage=2,
                        generator=gen_new)
                self.assertEqual(set(left), set(right))
                for key in left:
                    self.assert_tensor_equal(left[key], right[key])
                self.assertTrue(torch.equal(gen_base.get_state(), gen_new.get_state()))
                left["loss_point"].backward()
                right["loss_point"].backward()
                for (name, p), (new_name, q) in zip(
                    base.named_parameters(), new.named_parameters(),
                ):
                    self.assertEqual(name, new_name)
                    self.assertEqual(p.grad is None, q.grad is None)
                    if p.grad is not None:
                        self.assert_tensor_equal(p.grad, q.grad)
                for p, q in ((coarse, coarse_new), (fine, fine_new)):
                    self.assertEqual(p.grad is None, q.grad is None)
                    if p.grad is not None:
                        self.assert_tensor_equal(p.grad, q.grad)

    def test_inference_loop_matches_original_with_only_sampler_replaced(self):
        for base_cls, stable_cls in HEAD_PAIRS:
            with self.subTest(head=stable_cls.__name__):
                base, new = self.make_pair(base_cls, stable_cls)
                base.eval()
                new.eval()
                coarse, rois, labels, fine, _ = self.inputs()
                before = coarse.detach().clone()
                with torch.no_grad():
                    with mock.patch.object(
                        original, "select_inference_points",
                        side_effect=stable.select_inference_points_stable,
                    ):
                        left = base.refine(coarse, rois, labels, fine, False, stage=1)
                    right = new.refine(coarse, rois, labels, fine, False, stage=1)
                self.assert_tensor_equal(left[0], right[0])
                self.assertEqual(right[0].shape, (1, 12, 112, 112))
                self.assertEqual(left[1:], right[1:])
                self.assertEqual(right[1], [])
                self.assertEqual(right[2], {"point_count_l0": 12.,
                                           "point_count_l1": 12.})
                self.assert_tensor_equal(coarse, before)

    def test_empty_entry_points_preserve_contract(self):
        for base_cls, stable_cls in HEAD_PAIRS:
            with self.subTest(head=stable_cls.__name__):
                base, new = self.make_pair(base_cls, stable_cls)
                coarse, rois, labels, fine, gt = self.inputs(empty=True)
                self.assertIs(new.predict_refine(coarse, rois, labels, fine), coarse)
                expected = base.loss_and_refine(coarse, rois, labels, fine, gt)
                actual = new.loss_and_refine(coarse, rois, labels, fine, gt)
                self.assertEqual(set(actual), {"loss_point"})
                self.assert_tensor_equal(expected["loss_point"], actual["loss_point"])
                actual["loss_point"].backward()
                self.assertTrue(all(p.grad is not None for p in new.parameters()))


if __name__ == "__main__":
    unittest.main()
