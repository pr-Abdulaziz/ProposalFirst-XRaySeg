"""Point selection, coordinate conversion and feature/target sampling.

Coordinate handling delegates to the repository's own utilities
(``mmcv.ops.point_sample`` and ``rel_roi_point_to_rel_img_point``) rather than
to formulas written here, because an independently derived normalisation can
disagree with MMCV's ``align_corners`` and pixel-centre conventions in ways that
only show up as a systematic half-pixel shift.

Selection uses unique integer grid locations at each output resolution,
converted to pixel centres as ``x = (column + 0.5) / W`` and
``y = (row + 0.5) / H``. That is a deliberate adaptation from PointRend's
continuous training sampling: it makes the scatter step exact and its targets
unambiguous. It is recorded as an adaptation, not as a reproduction.
"""
from __future__ import annotations

import torch
from mmcv.ops import point_sample, rel_roi_point_to_rel_img_point
from torch import Tensor

from mmdet.models.utils.point_sample import get_uncertainty


def grid_indices_to_roi_coords(indices: Tensor, height: int, width: int) -> Tensor:
    """Flattened grid indices to pixel-centre ROI coordinates.

    ``idx = row * width + column`` with ``row`` along y. Returns ``(x, y)`` in
    ``[0, 1]``, matching the ordering ``point_sample`` expects.
    """
    row = torch.div(indices, width, rounding_mode="floor")
    column = indices % width
    x = (column.float() + 0.5) / width
    y = (row.float() + 0.5) / height
    return torch.stack([x, y], dim=-1)


def select_training_points(
    logits: Tensor,
    labels: Tensor,
    num_points: int = 196,
    oversample_ratio: int = 3,
    uncertain_fraction: float = 0.75,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uncertain-plus-random unique grid indices for supervision.

    Draws ``oversample_ratio * num_points`` unique candidates, keeps the most
    uncertain ``uncertain_fraction``, and fills the remainder uniformly from
    locations not already chosen. Uncertainty is ``-|logit|`` on the label's
    channel, i.e. distance from the decision boundary, not error against ground
    truth: ground truth selects the channel and nothing more.

    Returns ``[N, num_points]`` unique flattened indices.
    """
    num_rois, _, height, width = logits.shape
    total = height * width
    device = logits.device
    num_uncertain = min(int(num_points * uncertain_fraction), num_points)
    num_random = num_points - num_uncertain
    if total < num_points:
        raise ValueError(
            f"grid has {total} locations but {num_points} unique points were "
            "requested; reduce the point budget for this resolution")

    candidates = min(oversample_ratio * num_points, total)
    # Unique candidates per ROI, drawn with the controlled generator.
    scores = torch.rand(num_rois, total, device=device, generator=generator)
    candidate_idx = scores.topk(candidates, dim=1).indices           # [N, candidates]

    uncertainty = get_uncertainty(logits, labels).view(num_rois, total)
    candidate_unc = uncertainty.gather(1, candidate_idx)
    top = candidate_unc.topk(num_uncertain, dim=1).indices
    chosen = candidate_idx.gather(1, top)                            # [N, num_uncertain]

    if num_random > 0:
        # Exclude what is already chosen, then draw uniformly from the rest.
        mask = torch.zeros(num_rois, total, dtype=torch.bool, device=device)
        mask.scatter_(1, chosen, True)
        free = torch.rand(num_rois, total, device=device, generator=generator)
        free = free.masked_fill(mask, -1.0)
        extra = free.topk(num_random, dim=1).indices
        chosen = torch.cat([chosen, extra], dim=1)
    return chosen


def select_inference_points(
    logits: Tensor, labels: Tensor, num_points: int = 784
) -> Tensor:
    """The most uncertain unique grid indices, no randomness, no ground truth.

    Ties are broken by ascending flattened index so repeated runs agree.
    """
    num_rois, _, height, width = logits.shape
    total = height * width
    k = min(num_points, total)
    uncertainty = get_uncertainty(logits, labels).view(num_rois, total)
    # Deterministic tie-break: subtract a monotone ramp far below the spacing
    # of real uncertainty values, so equal scores resolve by index order.
    ramp = torch.arange(total, device=logits.device, dtype=uncertainty.dtype)
    ramp = ramp * torch.finfo(uncertainty.dtype).eps
    return (uncertainty - ramp).topk(k, dim=1).indices


def sample_fine_features(
    feature_map: Tensor, rois: Tensor, roi_coords: Tensor, spatial_scale: float
) -> Tensor:
    """Sample ``[N, K, C]`` fine features at ROI-relative points.

    ``rois`` are ``[N, 5]`` as ``(image_index, x1, y1, x2, y2)`` in transformed
    image coordinates. Conversion and sampling both use MMCV, so stride,
    padding and ``align_corners`` follow the detector's own convention.

    Sampling runs one image at a time, following the pattern in
    ``PointRendRoIHead._get_fine_grained_point_feats``. Indexing the feature map
    with a per-ROI image index instead would materialise one full-resolution
    copy per ROI: at roughly 100 inference detections on a 120x128x256 P2 that
    is about 3 GiB, which is what exhausted the GPU on the first pilot attempt.
    Results are written back by index, so the output order matches ``rois``
    regardless of how ROIs are grouped.
    """
    num_rois, num_points = roi_coords.shape[:2]
    channels = feature_map.shape[1]
    image_index = rois[:, 0].long()
    out = feature_map.new_zeros(num_rois, num_points, channels)
    for batch_ind in range(feature_map.shape[0]):
        mask = image_index == batch_ind
        if not bool(mask.any()):
            continue
        feat = feature_map[batch_ind].unsqueeze(0)
        image_points = rel_roi_point_to_rel_img_point(
            rois[mask], roi_coords[mask], feat.shape[2:], spatial_scale)
        # One batch element carrying every point of this image's ROIs.
        sampled = point_sample(
            feat, image_points.reshape(1, -1, 2), align_corners=False)
        out[mask] = sampled.squeeze(0).transpose(0, 1).reshape(
            int(mask.sum()), num_points, channels)
    return out


def sample_coarse_logits(coarse_logits: Tensor, roi_coords: Tensor) -> Tensor:
    """Sample ``[N, K, C]`` from the stage's original coarse output.

    The input is never modified: conditioning reads the same tensor that the
    baseline mask loss uses.
    """
    sampled = point_sample(coarse_logits, roi_coords, align_corners=False)
    return sampled.permute(0, 2, 1).contiguous()


def sample_mask_targets(
    gt_masks: Tensor, rois: Tensor, roi_coords: Tensor
) -> Tensor:
    """Sample ``[N, K]`` soft targets from full-resolution transformed masks.

    ``gt_masks`` is ``[N, H, W]`` in transformed image coordinates, one mask per
    ROI, already matched to its assigned instance. Values are bilinear and lie
    in ``[0, 1]``; ``BCEWithLogitsLoss`` accepts soft targets directly, so they
    are not thresholded.
    """
    image_points = rel_roi_point_to_rel_img_point(
        rois, roi_coords, gt_masks.shape[1:], 1.0)
    return point_sample(
        gt_masks.unsqueeze(1).float(), image_points, align_corners=False).squeeze(1)


def scatter_points(grid: Tensor, indices: Tensor, values: Tensor) -> Tensor:
    """Write predicted point logits into a grid without mutating the input.

    Args:
        grid: ``[N, C, H, W]`` current logits.
        indices: ``[N, K]`` flattened locations.
        values: ``[N, K, C]`` predicted logits.

    Returns a new tensor; unselected positions keep their interpolated values.
    """
    num_rois, channels, height, width = grid.shape
    flat = grid.reshape(num_rois, channels, height * width).clone()
    expanded = indices.unsqueeze(1).expand(-1, channels, -1)          # [N, C, K]
    flat = flat.scatter(2, expanded, values.permute(0, 2, 1))
    return flat.reshape(num_rois, channels, height, width)
