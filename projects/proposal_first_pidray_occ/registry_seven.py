"""Authoritative registry for the eight active PIXray methods.

Canonical identifiers are the strings ``"01"`` .. ``"08"``. They are strings,
not integers: the leading zero is part of the identifier and must survive shell
arguments, JSON, and spreadsheet cells.

Historical aliases resolve to canonical ids so older commands keep working, but
they never introduce a second entry or a second run. Everything else --
excluded arms, their configs and checkpoints -- is retained on disk and listed
in ARCHIVED for supplementary analysis; nothing here runs them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    id: str
    historical_id: str
    name: str
    role: str
    config: str
    work_dir: str
    supervision: str
    initialization: str
    comparisons: tuple = field(default=())


C = "projects/proposal_first_pidray_occ/configs"
O = "configs/occlusion"

METHODS: dict[str, Method] = {
    "01": Method("01", "B2", "Mask R-CNN, ResNet-101 + FPN", "Baseline 1",
                 f"{C}/pixray_b2_mask_rcnn_r101.py", "work_dirs/pixray_b2",
                 "target_masks_only", "ImageNet_ResNet101"),
    "02": Method("02", "P1", "Cascade Mask R-CNN, ResNet-101 + FPN", "Baseline 2",
                 f"{C}/pixray_p1_cascade_fpn.py", "work_dirs/pixray_p1",
                 "target_masks_only", "ImageNet_ResNet101",
                 ("01->02 adds cascade refinement; note rpn_max_per_img also "
                  "changes 1000->2000, so this is not a single-variable step",)),
    "03": Method("03", "P2", "Cascade Mask R-CNN, ResNet-101 + AttFPN", "Baseline 3",
                 f"{C}/pixray_p2_cascade_attfpn.py", "work_dirs/pixray_p2",
                 "target_masks_only", "ImageNet_ResNet101",
                 ("02->03 changes only the neck",)),
    "04": Method("04", "P3", "Cascade Mask R-CNN, R-101 + AttFPN + recall-enhanced RPN",
                 "Baseline 4", f"{C}/pixray_p3_cascade_attfpn_recall_rpn.py",
                 "work_dirs/pixray_p3", "target_masks_only", "ImageNet_ResNet101",
                 ("03->04 changes anchor scales and both proposal budgets together",)),
    "05": Method("05", "B5", "RSPrompter, SAM ViT-L with LoRA", "SAM comparison",
                 f"{O}/pixray_b5_rsprompter.py", "work_dirs/pixray_b5",
                 "target_masks_only", "SAM_ViT_L",
                 ("adapted single-decoder control built by reverting S2; not a "
                  "certified reproduction of the independent RSPrompter release",)),
    "06": Method("06", "S2", "Occ bilayer SAM built on RSPrompter",
                 "Previous-paper comparison", f"{O}/pixray_s2_occ_bisam.py",
                 "work_dirs/pixray_s2", "target_plus_overlap_masks", "SAM_ViT_L",
                 ("05->06 adds the occluder decoder and residual guidance and "
                  "their supervision together; it isolates neither",)),
    "07": Method("07", "B8", "Method 04 with a bilayer FCN mask head",
                 "Proposed candidate", f"{C}/pixray_b8_bilayer_fcn_recall_rpn.py",
                 "work_dirs/pixray_b8", "target_plus_overlap_masks", "ImageNet_ResNet101",
                 ("04->07 adds an occluder branch, extra capacity and residual "
                  "features together; it does not isolate supervision",)),
    "08": Method("08", "", "PIDray R0/R1, adapted to PIXray", "PIDray mechanism",
                 f"{C}/pixray_08_pidray_r0r1.py", "work_dirs/pixray_08_pidray_r0r1",
                 "target_masks_only", "ImageNet_ResNet101",
                 ("03->08 adds the R0/R1 image-level gate to method 03; no other "
                  "architectural change. Not a PIDray benchmark reproduction: "
                  "same PIXray images and target annotations, 15 classes.",)),
}

# 08 has no historical id: it has never been run before, so fabricating one
# would imply prior results that do not exist.
ALIASES: dict[str, str] = {
    m.historical_id.upper(): k for k, m in METHODS.items() if m.historical_id}

# Retained on disk, excluded from the active workflow and the headline table.
ARCHIVED = {
    "B1": "Mask R-CNN R-50 (backbone variant)",
    "B3": "Cascade R-CNN R-101, detection only",
    "B4": "Cascade Mask R-CNN R-50 (backbone variant)",
    "B6": "P1 + bilayer FCN",
    "B7": "P2 + bilayer FCN",
    "P4": "Cascade RPN variant",
    "S1": "ProposalFirst single decoder (separate hybrid direction)",
    "S3-P": "RecallRPN + guided bilayer SAM",
    "S3-MBoost": "S3-P + mask booster",
    "E1": "RecallRPN + prompt adapter",
    "E2": "E1 + hard-aware loss",
    "E3": "Conditional bilayer + booster",
}


def resolve(identifier: str) -> str:
    """Map a canonical id or historical alias to a canonical id."""
    key = str(identifier).strip()
    if key in METHODS:
        return key
    upper = key.upper()
    if upper in ALIASES:
        return ALIASES[upper]
    if upper in ARCHIVED:
        raise KeyError(
            f"{identifier!r} is archived ({ARCHIVED[upper]}). It is retained on "
            "disk for supplementary analysis but is not part of the active seven.")
    raise KeyError(f"unknown method {identifier!r}; active ids are {sorted(METHODS)}")


def resolve_all(identifiers) -> list[str]:
    """Resolve a list, preserving order and rejecting duplicates after aliasing."""
    out: list[str] = []
    for item in identifiers:
        canonical = resolve(item)
        if canonical in out:
            raise ValueError(
                f"{item!r} resolves to {canonical}, which is already selected; "
                "an alias must not schedule the same run twice")
        out.append(canonical)
    return out


def validate() -> None:
    ids = list(METHODS)
    assert ids == [f"{i:02d}" for i in range(1, 9)], f"ids must be 01..08, got {ids}"
    for key, m in METHODS.items():
        assert m.id == key, f"{key} disagrees with its own id {m.id}"
    for attr in ("config", "work_dir"):
        values = [getattr(m, attr) for m in METHODS.values()]
        assert len(values) == len(set(values)), f"duplicate {attr}"
    hist = [m.historical_id for m in METHODS.values() if m.historical_id]
    assert len(hist) == len(set(hist)), "duplicate historical id"
    overlap = set(ALIASES) & set(ARCHIVED)
    assert not overlap, f"alias also archived: {overlap}"
