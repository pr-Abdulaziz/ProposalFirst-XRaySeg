"""Bilayer mask head whose occluder-loss weight is configurable.

The upstream occlusion-aware head hardcodes its occluder term. In
``mmdet/occlu/occlu.py`` the loss is computed as ``self.loss_mask(...)`` --
the target-mask loss object, not a separate one -- and scaled by a literal
``0.25``. The ``loss_mask_bo`` entry a config supplies is inert: it is
swallowed by ``**kwargs`` in ``RSPrompterAnchorMaskHead.__init__`` and
``self.loss_mask_bo`` is referenced nowhere in the repository.

Exposing the weight makes the occluder supervision a controlled variable. The
default reproduces the upstream behaviour exactly, so a checkpoint trained
before this class existed stays comparable; ``bo_loss_weight=0`` removes the
occluder gradient while leaving the decoder, and its residual guidance path
into the target decoder, fully in place.
"""

from mmdet.occlu.occlu import OcPrompterAnchorMaskHead
from mmdet.registry import MODELS

# The literal applied at mmdet/occlu/occlu.py, in
# ``OcPrompterAnchorMaskHead.loss_and_target``.
UPSTREAM_BO_LOSS_WEIGHT = 0.25


@MODELS.register_module()
class WeightedOcPrompterAnchorMaskHead(OcPrompterAnchorMaskHead):
    """``OcPrompterAnchorMaskHead`` with a settable occluder-loss weight.

    Args:
        bo_loss_weight: Weight applied to the occluder-mask loss. Defaults to
            ``0.25``, the value the upstream head hardcodes. ``0`` disables
            occluder supervision without removing the occluder decoder.
    """

    def __init__(self, *args, bo_loss_weight: float = UPSTREAM_BO_LOSS_WEIGHT, **kwargs):
        # Validated before the parent runs so a bad weight fails on config
        # parse rather than after the SAM decoder has been constructed.
        if bo_loss_weight < 0:
            raise ValueError(
                f"bo_loss_weight must be non-negative, got {bo_loss_weight}"
            )
        self.bo_loss_weight = float(bo_loss_weight)
        super().__init__(*args, **kwargs)

    def loss_and_target(self, *args, **kwargs) -> dict:
        outputs = super().loss_and_target(*args, **kwargs)
        if self.bo_loss_weight == UPSTREAM_BO_LOSS_WEIGHT:
            return outputs
        # The parent has already applied UPSTREAM_BO_LOSS_WEIGHT. Rescaling
        # keeps the occluder decoder in the graph -- at weight 0 it still
        # receives gradient through the residual guidance path that feeds the
        # target decoder, which is the point of the ablation.
        scale = self.bo_loss_weight / UPSTREAM_BO_LOSS_WEIGHT
        outputs["loss_mask_bo"]["loss_mask_bo"] = (
            outputs["loss_mask_bo"]["loss_mask_bo"] * scale
        )
        return outputs
