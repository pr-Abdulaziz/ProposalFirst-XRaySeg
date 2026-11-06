# Upstream source provenance

## Occ runtime

- Repository: https://github.com/Ryh1218/Occ
- Preserved revision: `7d8c5fe1c04a730e957b6195e14067a84485f72c`
- Unified locations: `mmdet`, `mmpretrain`, `configs/occlusion`, and `tools`
- Original README: `references/occ/README.md`

## PIDray baseline

- Repository: https://github.com/lutao2021/PIDray.git
- Preserved revision: `2acc52493804422b492c04159a7ac65c6a7ee4c2`
- Original method-specific files: `references/legacy_pidray`
- Modern port: `projects/proposal_first_pidray_occ`

The method-specific PIDray snapshot includes the released Cascade `R0/R1`
configuration, AttFPN, modified one-stage/two-stage detectors, and legacy
requirements. Generic MMDetection 2.x files were intentionally not duplicated.
