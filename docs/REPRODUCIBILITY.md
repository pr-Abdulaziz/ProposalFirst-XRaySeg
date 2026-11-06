# Reproducing the paper experiments

This is the successor guide for the historical PIXray point-refinement study. For the research context and model equations, start with [the project context guide](README.md). The main entry point is `scripts/research.py`. It audits assets or prepares a portable config; it never submits a job or starts training automatically.

## 1. Understand the scope

The paper compares ten configurations on the official PIXray split. The main detector is ImageNet-initialized ResNet-101 + AttFPN + three-stage Cascade Mask R-CNN. Transformer and MLP point refiners are separate variants of the same detector. Historical comparisons and proposed refiners use their recorded configurations and seeds, not settings copied from newer experiments.

| ID | Configuration | Original source config |
|---|---|---|
| 01 | Mask R-CNN, R101 + FPN | `projects/proposal_first_pidray_occ/configs/pixray_b2_mask_rcnn_r101.py` |
| 02 | Cascade Mask R-CNN, R101 + FPN | `projects/proposal_first_pidray_occ/configs/pixray_p1_cascade_fpn.py` |
| 03 | Cascade Mask R-CNN, R101 + AttFPN (Parent) | `projects/proposal_first_pidray_occ/configs/pixray_p2_cascade_attfpn.py` |
| 04 | Parent with recall-enhanced RPN | `projects/proposal_first_pidray_occ/configs/pixray_p3_cascade_attfpn_recall_rpn.py` |
| 05 | RSPrompter-style SAM ViT-L + LoRA | `configs/occlusion/pixray_b5_rsprompter.py` |
| 06 | Occlusion-aware bilayer SAM | `configs/occlusion/pixray_s2_occ_bisam.py` |
| 07 | Bilayer FCN on recall-RPN detector | `projects/proposal_first_pidray_occ/configs/pixray_b8_bilayer_fcn_recall_rpn.py` |
| 08 | Parent with PIDray R0/R1 adaptation | `projects/proposal_first_pidray_occ/configs/pixray_08_pidray_r0r1.py` |
| 09 | Parent with Transformer point refinement | `projects/proposal_first_pidray_occ/configs/pixray_09_point_refine.py` |
| 09M | Parent with matched MLP point refinement | `projects/proposal_first_pidray_occ/configs/pixray_09m_point_refine_mlp.py` |

Use the frozen **resolved** configs for actual reproduction; the source configs above explain the implementation. Methods 06/07 use extra occluder supervision. Their comparison is not a pure architecture ablation.

| Protocol | PIXray paper campaign | Separate PIDray campaign |
|---|---|---|
| Campaign | `pixray_multiseed_bs4_e50_2026-09-14` | `pidray_bs2_e10_2026-09-22` |
| Training batch / accumulation | 4 / 1 | 2 / 1 |
| Epochs | 50 | 10 |
| Seeds | 42, 123, 2026 | 42 |
| Precision | Historical AMP float16 | FP32 |
| Optimizer | AdamW, lr 1e-4, weight decay 0.05 | Same |
| Gradient clipping | Global L2 norm 1.0 | Same |
| Warm-up | 25 iterations | 500 iterations |
| Checkpoint selection | Final epoch; no validation selection | Same |

The older eleven-method/60-epoch launchers are retained as historical source dependencies. They do not define the paper protocol. The retired Mask2Former campaigns are outside this paper workflow.

## 2. Know the files

- `projects/proposal_first_pidray_occ/models/point_refine/`: embedding, sampling, Transformer/MLP processing and cascade adapter.
- `projects/proposal_first_pidray_occ/models/attfpn.py`: attention feature pyramid.
- `mmdet/`, `mmpretrain/`: vendored framework source; preserve these versions.
- `output/<campaign>/campaign_manifest.json`: authoritative methods, seeds, annotations, checksums and run records.
- `output/<campaign>/protocol/resolved_configs/`: original frozen configs. Their absolute cluster paths are historical; do not edit these files.
- `work_dirs/<campaign>/`: main final checkpoints. Separate derivative runs belong under `work_dirs/reproductions/`.
- `output/reproductions/`: new portable configs and provenance, separated from historical evidence.
- `docs/PROJECT_ARCHITECTURE_2026-10-24.md`: model equations and implementation map.
- `docs/PROJECT_RUNTIME_AUDIT_2026-10-24.md`: historical runtime audit and opt-in inference/evaluation repairs.
- `output/repair_2026-11-06/`: local source restoration and validation evidence.
- `archive/cleanup_2026-11-06/`: superseded files; outside the active reproduction workflow.

## 3. Audit before using a GPU

From the project root on Windows, any compatible Python with the standard library can run:

```powershell
python scripts/research.py audit --dataset pixray
python scripts/research.py audit --dataset pidray
python -B -m unittest discover -s tests -p test_research_cli.py -v
```

Exit status 0 means the checked assets pass. Exit status 2 means assets are incomplete or arguments are invalid. Audit checks config/source/annotation hashes, weight presence, and missing/zero-byte image files. It does not decode images, rehash large weights by default, establish accuracy or validate a GPU runtime.

At this handoff: PIXray assets pass; PIDray remains incomplete because many image files are zero bytes. The supplied paper PDFs are now included under `paper/`; editable DOCX files and private review notes are excluded. Some historical result/report payloads in the D: copy remain incomplete. Code restoration does not reconstruct those data or results. Use the exact-file lists in `output/cleanup_2026-11-06/` and restore verified originals rather than creating replacement labels or fabricated metrics.

## 4. Use a supported execution environment

The verified runtime is Linux, Python 3.10, PyTorch 2.1.2 + CUDA 11.8, torchvision 0.16.2, MMCV 2.1.0 and MMEngine 0.10.7. Native Windows CUDA execution was not tested in this repair. A copied Linux `.venv` is not a Windows Python environment. Use a fresh Linux/WSL environment or an equivalent cluster checkout.

For WSL, this same project normally appears at `/mnt/d/ProposalFirst-XRaySeg`. For another Linux machine, run from its local project path. The portable config helper derives the project root from its own location.

Example for a new Linux environment with Python 3.10 and a compatible NVIDIA driver:

```bash
cd /mnt/d/ProposalFirst-XRaySeg
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install openmim==0.3.9 mmengine==0.10.7
mim install mmcv==2.1.0
python -m pip install -r requirements/cloud.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$PWD/pretrained/torch"
python -m pip check
python scripts/check_project_runtime.py --device cpu --output output/new_runtime_cpu.json
```

The output path for runtime checks must be fresh. CUDA checks require an existing Slurm allocation in the current audit utility; do not run GPU work on a login node. Use that utility with `--device cuda` in an allocation before full training. Inspect all required checks, not only successful imports. A source build may be attempted if no compatible MMCV wheel is found; confirm compiled operators before proceeding.

The full measured environment is recorded in `output/environment_audit_2026-10-24/requirements.freeze.txt`; it is a historical record, not an instruction to blindly reinstall every optional package. See the official [PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/) and [MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for wheel/build details. No package installation was performed in this repair.

## 5. Prepare a new portable configuration

Run this in the actual Linux environment where training will occur, so that generated paths match that environment:

```bash
python scripts/research.py prepare --dataset pixray --method 09 --seed 42 --name my_reproduction
```

This command:

1. Requires the recorded source/config/annotation hashes and dataset checks to pass.
2. Selects only an approved method and seed from the frozen manifest.
3. Reads the resolved config without executing arbitrary config code.
4. Rebases the old cluster project-root paths and chooses a fresh work directory.
5. Preserves model, preprocessing, losses, precision, schedule, batch size and random seed.
6. Writes `output/reproductions/my_reproduction/pixray/method_09/seed_42/config.py` and `provenance.json`.

It refuses an existing reproduction directory and starts no training. The historic `load_from=None` initializes the backbone from ImageNet and new heads freshly; it does not load another experiment's trained checkpoint. The existing final checkpoints are for evaluation, not initial training weights.

The existing training entry point can then be used explicitly in an appropriate GPU session:

```bash
python scripts/campaign_multiseed/train_run.py output/reproductions/my_reproduction/pixray/method_09/seed_42/config.py
```

Use a new reproduction name for each fresh attempt. Do not change the recorded batch, precision or learning rate merely to fit another GPU and still call it the same comparison. Inspect the first full epoch for finite loss/weights, real optimizer updates and frozen BN. The preserved historical campaign launchers/guards expect original cluster paths and should not be invoked unchanged on D: or WSL.

## 6. Evaluate carefully

Use the final checkpoint and the corresponding prepared config. Run native MMDetection `tools/test.py CONFIG CHECKPOINT --out NEW_PREDICTIONS.pkl` to obtain predictions in a Linux GPU environment. Keep evaluation outputs outside historical campaign output directories and use fresh names.

For historical evaluation semantics, read `scripts/campaign_multiseed/evaluate_run.py`, `score_predictions.py`, and `docs/evaluation-protocol.md`. PIXray contains a known zero-annotation-ID convention; report whether official or corrected matching was used. PIDray category mapping must follow the dataset's actual `cat2label` and the amended Easy annotations. The corrected PIDray evaluator contains extra coverage and scoring checks, but its frozen-campaign entry point also assumes original cluster paths; it is not a portable one-line evaluator for a derivative run. Do not remove its validation gates to get a score.

Report mask AP/AP50/AP75, recall and boundary/matched-mask diagnostics where recorded. Separate mask-derived boxes from independently predicted detector boxes. Image-weighted split AP is not pooled PIDray AP. Compare multiple seeds and preserve per-seed results before claiming an architecture gain. No new model training, GPU evaluation or accuracy claim was made during this code restoration.

## 7. Change the code without destroying the evidence

Keep frozen config/source manifests and historic metrics unchanged. Develop new models in a separate module/config and create a new experiment name. Record the single intended difference, rerun static/asset checks and the relevant runtime tests, and compare controls with the same protocol. Keep paper assets and historical checkpoints; regenerate disposable caches. Large copied runtimes and preliminary runs are not required in the active study tree.
