# Third-party notices

This repository contains material from other projects. It is **not** covered by the
all-rights-reserved notice in [`LICENSE`](LICENSE) and stays under its owners' licenses.
License texts are in [`licenses/`](licenses/). Provenance details are in
[`references/SOURCES.md`](references/SOURCES.md).

| Component | Where in this repository | Upstream | License |
|---|---|---|---|
| MMDetection | `mmdet/`, parts of `tools/`, packaging files | https://github.com/open-mmlab/mmdetection | Apache License 2.0, "Copyright (c) OpenMMLab" ([text](licenses/Apache-2.0.txt)) |
| MMPretrain | `mmpretrain/` | https://github.com/open-mmlab/mmpretrain | Apache License 2.0, "Copyright (c) OpenMMLab" ([text](licenses/Apache-2.0.txt)) |
| Occ: Prohibited Items Segmentation via Occlusion-aware Bilayer Modeling (ICME 2025) | `mmdet/occlu/`, `configs/occlusion/`, adaptations in `tools/`, `mmdet/`, `mmpretrain/` | https://github.com/Ryh1218/Occ (revision `7d8c5fe1c04a730e957b6195e14067a84485f72c`) | MIT License, "Copyright (c) 2025 RYH" ([text](licenses/MIT-Ryh1218-Occ.txt)) |
| PIDray baseline (R0/R1 configuration, AttFPN, modified detectors) | `references/legacy_pidray/`; a port of it is in `projects/proposal_first_pidray_occ/` | https://github.com/lutao2021/PIDray (revision `2acc52493804422b492c04159a7ac65c6a7ee4c2`) | No license is published upstream, so all rights remain with its authors. Portions derived from it are not covered by `LICENSE`. |
| Hugging Face logo | `site/assets/hf-logo.svg` | https://huggingface.co | Trademark of Hugging Face, Inc.; used only to point to their platform. |
| PIXray dataset example images | Figures in the paper and on the project page | PIXray (Ma et al., cited in the paper) | Subject to the dataset's own terms. The dataset is not redistributed here. |

## Notes

- The Apache License 2.0 requires that recipients receive a copy of the license
  ([`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)) and that copyright notices in the
  source files are kept. Modifications made in this repository are recorded in its git history.
- The MIT License requires that its copyright and permission notice be included in copies
  or substantial portions of the covered code. The unmodified text is in
  [`licenses/MIT-Ryh1218-Occ.txt`](licenses/MIT-Ryh1218-Occ.txt).
- If you believe something here is attributed incorrectly, please open an issue.
