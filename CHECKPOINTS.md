# Paper experiment checkpoints

Large checkpoint binaries are excluded from Git. `checkpoints_manifest.json` identifies the 40 final checkpoints associated with the main historical campaigns: 30 PIXray files (ten methods, three seeds, epoch 50) and ten PIDray files (ten methods, seed 42, epoch 10).

The [paper-checkpoints-v1 GitHub release](https://github.com/pr-Abdulaziz/ProposalFirst-XRaySeg/releases/tag/paper-checkpoints-v1) lists the inventory of the original checkpoints; their binaries were never uploaded because the transfer was interrupted. The seed-42 models are on Hugging Face (see below). Preliminary, retired Mask2Former, generic downloaded model and unrelated diagnostic checkpoints are excluded.

The binaries are original experiment checkpoints and may contain optimizer/training state. Use their matching frozen configurations. Do not use the final trained checkpoints to initialize a purported fresh training comparison. The original datasets are not redistributed in this repository.

PIXray is the paper's three-seed campaign. PIDray is a separate single-seed campaign and should not be merged into the PIXray results. The code's reproduction guide documents protocols and remaining local-data limitations.

## Hugging Face release (seed 42, weights only)

The seed-42 run of each of the ten PIXray experiments is available at [abdulaziz-alqahtani/xray-point-refinement](https://huggingface.co/abdulaziz-alqahtani/xray-point-refinement). These files contain model weights and MMEngine metadata only (optimizer and scheduler state removed), so they cannot be used to resume training. Their SHA-256 checksums are in that repository's `checkpoints.json` and differ from the original checkpoints listed in `checkpoints_manifest.json`, which include training state. Seeds 123 and 2026 and the PIDray checkpoints are not part of that release.
