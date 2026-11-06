# ProposalFirst-XRaySeg: project and runtime audit

Verified on 24 October 2026 against the live cluster project:
`/nfs/ahalqahtani/ProposalFirst-XRaySeg`.

The core environment works on the tested RTX 4090. The audit also found and
reproduced inference and evaluation defects; the repairs below are deployed as
separate, opt-in paths. Historical models, frozen configurations, checkpoints
and experiment results were preserved. No packages were upgraded and no model
training or full dataset evaluation was launched.

See [the architecture guide](PROJECT_ARCHITECTURE_2026-10-24.md) for the diagram,
equations, implementation files and configuration instructions.

## Current project

| Component | Authoritative location |
|---|---|
| Custom models, datasets, metrics and hooks | `projects/proposal_first_pidray_occ/` |
| Point-refinement implementation | `projects/proposal_first_pidray_occ/models/point_refine/` |
| Vendored frameworks | `mmdet/`, `mmpretrain/` |
| Active Python interpreter | `.venv/bin/python` |
| Runtime selection and environment paths | `scripts/common.sh` |
| PIXray campaign | `output/pixray_multiseed_bs4_e50_2026-09-14/` |
| PIDray campaign | `output/pidray_bs2_e10_2026-09-22/` |
| Completed detached-MLP diagnostic | `output/pidray_diag_gradiso_2026-09-26/` |
| This audit's logs, hashes, backups and checks | `output/environment_audit_2026-10-24/` |

The implemented main branch is ResNet-101 → AttFPN → three-stage Cascade
Mask R-CNN → shared point refiner. Its Transformer has two 128-dimensional,
four-head self-attention blocks and 512-wide FFNs. It predicts replacement
class logits at selected points as masks grow from 28² to 56² to 112².
The decoder is configured for 15 PIXray classes or 12 PIDray classes.
The context/cross-attention architecture discussed separately is not implemented
in this audited model. Detaching refiner inputs does not freeze the detector.

All 40 frozen resolved configs parsed, matched their recorded SHA256 hashes,
registered their main detector types, and referenced existing training annotations.
PIXray has 30 runs: batch 4, 50 epochs, seeds 42/123/2026, `AmpOptimWrapper`.
PIDray has 10 runs: batch 2, 10 epochs, seed 42, FP32 `OptimWrapper`.
The older README's eleven-method/60-epoch protocol and the historical `bs1`
directory name do not describe these current resolved campaigns.

## Environment and extensions

| Dependency | Verified version |
|---|---|
| Python | 3.10.21 |
| PyTorch / torchvision | 2.1.2+cu118 / 0.16.2+cu118 |
| CUDA runtime / cuDNN | 11.8 / 8700 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| Vendored MMDetection | 3.2.0 |
| NumPy / SciPy | 1.24.0 / 1.12.0 |
| Transformers / PEFT | 4.38.1 / 0.8.2 |
| OpenCV / pycocotools | 4.8.1.78 / 2.0.7 |

`pip check` passed. The live environment was captured in `requirements.freeze.txt`.
Core framework, SAM-related dependency and custom registry imports passed.
There is no project-owned CUDA extension to compile: compiled operators come
from MMCV/torchvision. `mmcv-lite` is not a substitute for the installed MMCV.
Neither FlashAttention nor xFormers is required by this implementation.
Missing optional `timm` and `pytest` did not block the active configurations or
the executed tests; the selected tests run with the standard-library unittest
runner. Do not install every optional test/tracking requirement into this
working environment, or use `scripts/setup.sh` as a health check: it installs
packages and can recreate an incompatible environment.

The shell's bare `python` points to Miniconda, not necessarily this `.venv`.
Use the explicit project interpreter or `source scripts/common.sh`.

## Defects and deployed repairs

| Finding | Repair / boundary |
|---|---|
| Original inference tie-breaking subtracts `index × dtype.eps`, changing unequal uncertainty rankings | New `stable_inference.py` heads sort unchanged scores stably. FP32 misselection reproduced on CPU; stable ordering verified on GPU in four dtypes at both resolutions. Original heads remain historical defaults. |
| Legacy refiner fails when half feature tensors enter point sampling under CUDA autocast | New `fp32_inference.py` heads retain stable selection and execute prediction refinement in FP32. Tested with half inputs under CUDA autocast. Training is inherited unchanged; this is not an AMP-training repair. |
| Frozen PIDray evaluation entrypoint uses category-unsafe contiguous-label scoring | New `scripts/campaign_pidray_bs1/evaluate_run_corrected.py` checks runtime category mapping, coverage and evaluator agreement, and writes separate results. Existing corrected historical scores are not rewritten. |
| Diagnostic success did not enforce complete image coverage | `evaluate_diag.py` now rejects missing/duplicate images and metric disagreement, and refuses existing outputs. |
| Mapped-analysis validation could exit successfully despite regression failures | `validate_mapped_analysis.py` now enforces its checks and supports a fresh `--output` path without overwriting old reports. |
| README directs users to an older protocol | A current-guide banner points to this audit and the architecture guide. |

The two modified existing evaluation scripts were outside both frozen source
manifests; their original bytes were backed up. Model repairs are additive and
require explicit `custom_imports` and head-type selection in a new diagnostic
configuration. Training outputs, gradients and RNG states were regression-tested
against the corresponding original heads. Strict state-dict loading passed for
12 and 15 classes. Any new inference scores must carry a separate identity.

## Verification completed

| Check | Result |
|---|---|
| CPU runtime suite | 23 required checks passed |
| CUDA runtime suite on RTX 4090, `issnode5`, Slurm job 14995 | 27 required checks passed; job completed successfully in 24 seconds |
| Actual compiled operators | RoIAlign forward/backward, MMCV NMS, torchvision NMS passed on CPU and CUDA; optional CUDA deformable convolution passed |
| Sampling / annotation operations | Point-coordinate conversion and gradients; COCO RLE, area and IoU passed |
| Original Transformer, MLP and detached heads | FP32 loss/backward, inference and empty-RoI checks passed |
| New inference heads | Four FP32 compatibility heads passed CUDA autocast prediction checks, including equality with their FP32 reference on the same quantized inputs |
| Regression cases | 7 stable-selector + 4 FP32-inference + 21 evaluation + 7 image-gate + 10 synthetic zero-ID tests passed |
| Stable GPU ranking | Eight dtype/resolution combinations passed; exact ties retain ascending grid order |
| Frozen source | Both campaigns' original source hashes unchanged after repairs; added modules recorded separately |

Initial job 14994 failed and is retained as evidence. One failure was in the new
test harness's keyword call to MMCV RoIAlign, corrected to positional arguments.
Four others exposed the legacy half-input dtype defect. The final report retains
those four failures explicitly as optional legacy diagnostics; a core PASS does
not mean legacy AMP inference works. `runtime_cuda.json` and
`runtime_cuda_v2.json` distinguish the initial and final checks.

## Data and experiment observations

Annotation records and local pretrained asset existence were inspected, not every
image or checkpoint tensor. PIDray's campaign uses 22,990 de-leaked training
images; the released training annotation has 23,074. The amended easy test split
has 24,757 images, with 9,746 hard and 13,069 hidden images. Do not replace these
campaign decisions with the unamended release files. PIXray remains 3,560
training and 1,486 test images.

The detached-MLP diagnostic is completed: its saved evaluation record reports
49.2929108160107 pooled mask AP and 58.111914145853916 pooled box AP, with all
three split-coverage and MMDetection-agreement flags passing. These are inspected
saved results from job 12700, not a new evaluation, and remain one-seed evidence.
Split AP averages are not pooled AP; the corrected legacy wrapper retains an
explicitly image-weighted summary, whereas the diagnostic computes pooled AP.

## Recheck commands

From the cluster project root, select a new output filename for each invocation:

```bash
PYTHONPATH=. .venv/bin/python -B scripts/check_project_runtime.py \
  --device cpu --output output/runtime_check_NEW_cpu.json

# Inside a scheduler allocation, never on an unallocated login GPU:
PYTHONPATH=. .venv/bin/python -B scripts/check_project_runtime.py \
  --device cuda --include-legacy-deform-conv \
  --output output/runtime_check_NEW_cuda.json

PYTHONPATH=. OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B \
  -m unittest discover -s tests -p 'test_point_refine_*inference.py' -v
PYTHONPATH=. .venv/bin/python -B \
  -m unittest discover -s tests -p 'test_pidray_*.py' -v
```

Add `--include-legacy-amp-inference` only when intentionally reproducing the
documented legacy limitation. Keep FP32 compatibility head weights in FP32;
calling `model.half()` is explicitly unsupported by those types.

For future separately authorized PIDray evaluation, the additive corrected
entrypoint is `scripts/campaign_pidray_bs1/evaluate_run_corrected.py <method> <seed>`
inside an allocated GPU job. It retains frozen model/config semantics; it does
not automatically select the new stable/FP32 heads. The old frozen
`slurm_eval.sh` still invokes the legacy scorer and must not be mistaken for this
corrected entrypoint. Full mapped-analysis regression is a separate, larger CPU
task; its entrypoint now requires a fresh report path.

## Scope limits and next development step

This was source, configuration and bounded runtime verification. It did not run
full SAM forwards, all ten complete detectors, AMP training, every dataset image,
all checkpoint loads, or complete AP reevaluations. Compatibility is established
for the tested RTX 4090 environment, not every GPU partition or future package
version. The new inference behavior has no measured AP effect yet.

Use a separately versioned diagnostic to measure the sampler repair first. Keep
that change distinct from context-attention architecture experiments. Background
false detections, missed proposals and bad boxes still require detector-level
analysis; a point mask refiner alone cannot resolve all three.
