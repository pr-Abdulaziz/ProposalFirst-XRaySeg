# Upstream source provenance

## Occ runtime

- Repository: https://github.com/Ryh1218/Occ
- Preserved revision: `7d8c5fe1c04a730e957b6195e14067a84485f72c`
- Unified locations: `mmdet`, `mmpretrain`, `configs/occlusion`, and `tools`
- Original README: `references/occ/README.md`

## PIDray baseline

- Repository: https://github.com/lutao2021/PIDray.git
- Reference revision: `2acc52493804422b492c04159a7ac65c6a7ee4c2`
- Modern port: `projects/proposal_first_pidray_occ`

The PIDray repository publishes no license, so its files are not redistributed
here. The port was checked against the released Cascade `R0/R1` configuration,
AttFPN and modified one-stage/two-stage detectors at the revision above
([`configs/cascade_mask_rcnn_r101_with_R0R1.py`](https://github.com/lutao2021/PIDray/blob/2acc52493804422b492c04159a7ac65c6a7ee4c2/configs/cascade_mask_rcnn_r101_with_R0R1.py),
[`mmdet/models/necks/attfpn.py`](https://github.com/lutao2021/PIDray/blob/2acc52493804422b492c04159a7ac65c6a7ee4c2/mmdet/models/necks/attfpn.py),
[`mmdet/models/detectors/`](https://github.com/lutao2021/PIDray/tree/2acc52493804422b492c04159a7ac65c6a7ee4c2/mmdet/models/detectors)).
Frozen source files that mention `references/legacy_pidray/` refer to those
upstream files; they are kept byte-identical because their hashes are recorded.
