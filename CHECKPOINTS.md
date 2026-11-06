# Paper experiment checkpoints

Large checkpoint binaries are excluded from Git. `checkpoints_manifest.json` identifies the 40 final checkpoints associated with the main historical campaigns: 30 PIXray files (ten methods, three seeds, epoch 50) and ten PIDray files (ten methods, seed 42, epoch 10).

They are intended for the `paper-checkpoints-v1` GitHub release. An asset is available only when its named file is listed on that release; this manifest alone is not confirmation of an uploaded binary. Preliminary, retired Mask2Former, generic downloaded model and unrelated diagnostic checkpoints are excluded from this release.

The binaries are original experiment checkpoints and may contain optimizer/training state. Use their matching frozen configurations. Do not use the final trained checkpoints to initialize a purported fresh training comparison. The original datasets are not redistributed in this repository.

PIXray is the paper's three-seed campaign. PIDray is a separate single-seed campaign and should not be merged into the PIXray results. The code's reproduction guide documents protocols and remaining local-data limitations.
