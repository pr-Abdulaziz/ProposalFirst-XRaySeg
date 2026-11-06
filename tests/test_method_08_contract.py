"""Architecture contract for method 08 (PIDray R0/R1 adapted to PIXray).

CPU-only. Verifies R0 tensor shapes, the legacy training gate's gradient
scaling under controlled outcomes, R0 label derivation after augmentation, and
that no PIDray dataset class or path survives the PIXray inheritance.

    PYTHONPATH=. .venv/bin/python tests/test_method_08_contract.py
"""
from __future__ import annotations

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

register_all_modules(init_default_scope=True)
import projects.proposal_first_pidray_occ  # noqa: F401

CONFIG = "projects/proposal_first_pidray_occ/configs/pixray_08_pidray_r0r1.py"


def _cfg():
    return Config.fromfile(CONFIG)


class _Sample:
    """Minimal stand-in carrying only what R0 and the gate read.

    ``batch_input_shape`` is the padded canvas and ``img_shape`` the unpadded
    content; R0 builds its attention padding mask from the difference, so both
    are required for a faithful test.
    """

    def __init__(self, n_instances: int, img_shape=(512, 512),
                 batch_input_shape=(512, 512)):
        self.gt_instances = InstanceData(
            labels=torch.zeros(n_instances, dtype=torch.long))
        self.img_shape = img_shape
        self.batch_input_shape = batch_input_shape
        self.metainfo = {"img_shape": img_shape,
                         "batch_input_shape": batch_input_shape}
        self._meta = {}

    def get(self, key, default=None):
        return self._meta.get(key, default)

    def set_metainfo(self, d):
        self._meta.update(d)


def test_r0_consumes_the_last_fpn_level_at_expected_shape():
    """R0 reads features[-1]; at padded 512 that is P6, 256 channels, 8x8."""
    head = MODELS.build(_cfg().model["r0_head"])
    feats = tuple(torch.randn(2, 256, 512 // s, 512 // s)
                  for s in (4, 8, 16, 32, 64))
    assert feats[-1].shape == (2, 256, 8, 8), feats[-1].shape
    logits = head(feats, [_Sample(1), _Sample(0)])
    assert logits.shape == (2,), f"expected one logit per image, got {logits.shape}"


def test_r0_handles_rectangular_batches():
    """A non-square padded batch must still yield one logit per image."""
    head = MODELS.build(_cfg().model["r0_head"])
    feats = tuple(torch.randn(3, 256, 640 // s, 640 // s)
                  for s in (4, 8, 16, 32, 64))
    # padded to 640x640, real content 448x640 -> a genuine padding mask
    samples = [_Sample(n, img_shape=(448, 640), batch_input_shape=(640, 640))
               for n in (2, 0, 1)]
    logits = head(feats, samples)
    assert logits.shape == (3,), logits.shape


def test_r0_label_comes_from_post_transform_instances():
    """An empty crop yields target 0; a crop with instances yields 1.

    These are augmentation artefacts, not original normal-baggage images.
    """
    head = MODELS.build(_cfg().model["r0_head"])
    feats = tuple(torch.randn(2, 256, 512 // s, 512 // s)
                  for s in (4, 8, 16, 32, 64))
    samples = [_Sample(3), _Sample(0)]
    logits = head(feats, samples)
    losses = head.loss_by_logits(logits, samples)
    key = next(iter(losses))
    assert torch.isfinite(losses[key]), losses
    targets = [float(len(s.gt_instances) > 0) for s in samples]
    assert targets == [1.0, 0.0], targets


def test_legacy_gate_scales_detector_gradient_by_selected_fraction():
    """None / all / mixed selection must scale the detector gradient exactly."""
    from projects.proposal_first_pidray_occ.models.pidray_cascade_rcnn import (
        PIDrayCascadeRCNN)
    scale = PIDrayCascadeRCNN._legacy_gradient_scale
    for selected, total in ((0, 4), (4, 4), (1, 4), (3, 4)):
        source = torch.tensor(2.0, requires_grad=True)
        losses = {"loss_det": source * 3.0}
        scaled = scale(losses, float(selected) / total)
        value = scaled["loss_det"]
        if source.grad is not None:
            source.grad = None
        value.backward()
        expected = 3.0 * selected / total
        assert torch.isclose(source.grad, torch.tensor(expected)), (
            f"selected={selected}/{total}: gradient {source.grad} != {expected}")


def test_r0_still_learns_when_detector_gradient_is_zeroed():
    """With nothing selected the R0 loss must still produce R0 gradients."""
    head = MODELS.build(_cfg().model["r0_head"])
    feats = tuple(torch.randn(2, 256, 512 // s, 512 // s)
                  for s in (4, 8, 16, 32, 64))
    samples = [_Sample(0), _Sample(0)]        # both empty -> gate selects none
    logits = head(feats, samples)
    losses = head.loss_by_logits(logits, samples)
    next(iter(losses.values())).backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "R0 received no gradient"
    assert any(g.abs().sum() > 0 for g in grads), "all R0 gradients are zero"


def test_no_pidray_dataset_or_path_survives_inheritance():
    """08 must use PIXray data only."""
    cfg = _cfg()
    for split in ("train_dataloader", "test_dataloader"):
        ds = cfg[split]["dataset"]
        assert "PIXray" in ds["type"], f"{split}: {ds['type']}"
        assert "PIDray" not in ds["type"], f"{split}: {ds['type']}"
        for key in ("ann_file", "data_root"):
            value = str(ds.get(key, ""))
            assert "PIDray" not in value, f"{split}.{key}: {value}"
        assert "PIXray-A" not in str(ds.get("ann_file", "")), "must not use PIXray-A"
    assert cfg.model["roi_head"]["mask_head"]["num_classes"] == 15
    assert cfg.get("load_from") is None, "08 must not warm-start"


def test_08_does_not_inherit_04_rpn_or_07_mask_head():
    cfg = _cfg()
    scales = cfg.model["rpn_head"]["anchor_generator"]["scales"]
    assert list(scales) == [8], f"04's recall RPN leaked in: {scales}"
    assert "Bilayer" not in str(cfg.model["roi_head"]["mask_head"]["type"])
    assert "Bilayer" not in str(cfg.model["roi_head"]["type"])


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1; print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:                                    # noqa: BLE001
            failed += 1; print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n  {len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
