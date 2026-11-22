# Side-by-side configuration (pixray_multiseed_bs4_e50_2026-09-14)

Generated from the resolved seed-42 configs and preflight parameter counts. Every protocol field (optimizer, schedule, AMP, clipping, batch, workers, sampler, epochs, checkpoint rule, evaluator, augmentation parameters) is identical across methods and is listed once in the shared table.

| Item | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 09M |
|---|---|---|---|---|---|---|---|---|---|---|
| Training annotations | train.json | train.json | train.json | train.json | train.json | train_occ.json | train_occ.json | train.json | train.json | train.json |
| Test-loader annotations | test.json | test.json | test.json | test.json | test.json | test_occ.json | test_occ.json | test.json | test.json | test.json |
| Scored against | test.json | test.json | test.json | test.json | test.json | test.json | test.json | test.json | test.json | test.json |
| Detector | MaskRCNN | CascadeRCNN | CascadeRCNN | CascadeRCNN | RSPrompterAnchor | RSPrompterAnchor | CascadeRCNN | PIDrayCascadeRCNN | CascadeRCNN | CascadeRCNN |
| Backbone | ResNet-101 | ResNet-101 | ResNet-101 | ResNet-101 | SAM ViT-L + LoRA r16 (qkv) | SAM ViT-L + LoRA r16 (qkv) | ResNet-101 | ResNet-101 | ResNet-101 | ResNet-101 |
| Initialisation | torchvision ImageNet | torchvision ImageNet | torchvision ImageNet | torchvision ImageNet | SAM ViT-L weights | SAM ViT-L weights | torchvision ImageNet | torchvision ImageNet | torchvision ImageNet | torchvision ImageNet |
| Frozen | stem + stage 1; BN statistics | stem + stage 1; BN statistics | stem + stage 1; BN statistics | stem + stage 1; BN statistics | image encoder except LoRA | image encoder except LoRA | stem + stage 1; BN statistics | stem + stage 1; BN statistics | stem + stage 1; BN statistics | stem + stage 1; BN statistics |
| Neck | FPN | FPN | AttFPN | AttFPN | RSFPN | RSFPN | AttFPN | AttFPN | AttFPN | AttFPN |
| Batch-one normalisation | backbone BN frozen | backbone BN frozen | backbone BN frozen; AttFPN attention BN uses running-stat fallback | backbone BN frozen; AttFPN attention BN uses running-stat fallback | LayerNorm; prompt-head BN over RoIs | LayerNorm; prompt-head BN over RoIs | backbone BN frozen; AttFPN attention BN uses running-stat fallback | backbone BN frozen; AttFPN attention BN uses running-stat fallback | backbone BN frozen; AttFPN attention BN uses running-stat fallback | backbone BN frozen; AttFPN attention BN uses running-stat fallback |
| Anchor scales | [8] | [8] | [8] | [4, 8] | [4, 8] | [4, 8] | [4, 8] | [8] | [8] | [8] |
| Proposals train (pre/kept) | 2000/1000 | 2000/2000 | 2000/2000 | 3000/3000 | 2000/1000 | 2000/1000 | 3000/3000 | 2000/2000 | 2000/2000 | 2000/2000 |
| Proposals test (pre/kept) | 1000/1000 | 1000/1000 | 1000/1000 | 2000/2000 | 1000/1000 | 1000/1000 | 2000/2000 | 1000/1000 | 1000/1000 | 1000/1000 |
| RoI head | StandardRoIHead | CascadeRoIHead | CascadeRoIHead | CascadeRoIHead | RSPrompterAnchorRoIPromptHead | OcPrompterAnchorRoIPromptHead | BilayerCascadeRoIHead | CascadeRoIHead | PointRefineCascadeRoIHead | PointRefineCascadeRoIHead |
| Stages / loss weights | 1 / 1 | 3 / [1, 0.5, 0.25] | 3 / [1, 0.5, 0.25] | 3 / [1, 0.5, 0.25] | 1 / 1 | 1 / 1 | 3 / [1, 0.5, 0.25] | 3 / [1, 0.5, 0.25] | 3 / [1, 0.5, 0.25] | 3 / [1, 0.5, 0.25] |
| Mask head | FCNMaskHead | FCNMaskHead | FCNMaskHead | FCNMaskHead | RSPrompterAnchorMaskHead | WeightedOcPrompterAnchorMaskHead | BilayerFCNMaskHead | FCNMaskHead | FCNMaskHead | FCNMaskHead |
| Mask target size | 28 | 28 | 28 | 28 | (512, 512) | (512, 512) | 28 | 28 | 28 | 28 |
| Occluder loss weight | - | - | - | - | - | 0.25 | 0.25 | - | - | - |
| R0/R1 gate | - | - | - | - | - | - | - | yes | - | - |
| Point refinement | - | - | - | - | - | - | - | - | PointRefineHead; pts 196/784; grids (56, 112) | PointRefineMLPHead; pts 196/784; grids (56, 112) |
| Input normalisation | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] | mean [123.7, 116.3, 103.5] |
| Test: score / NMS / max det / mask thr | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 | 0.05 / 0.5 / 100 / 0.5 |
| Parameters total (M) | 63.04 | 96.07 | 97.10 | 97.11 | 332.67 | 336.73 | 105.17 | 98.68 | 97.56 | 97.56 |
| Trainable (M) | 62.81 | 95.84 | 96.88 | 96.88 | 27.57 | 31.63 | 104.95 | 98.46 | 97.34 | 97.34 |
| Frozen (M) | 0.23 | 0.23 | 0.23 | 0.23 | 305.10 | 305.10 | 0.23 | 0.23 | 0.23 | 0.23 |

## Shared protocol

| Setting | Value |
|---|---|
| dataset | official PIXray split: 3,560 train / 1,486 test images |
| physical_batch_per_gpu | 4 |
| world_size | 1 |
| accumulative_counts | 1 |
| effective_batch | 4 |
| optimizer | AdamW lr=1e-4 weight_decay=0.05, betas/eps torch defaults (0.9, 0.999) / 1e-8, one parameter group; learning rate kept from the batch-16 protocol, not rescaled for the effective batch of 4 |
| amp | AmpOptimWrapper float16, dynamic GradScaler (defaults) |
| clip_grad | global L2 max_norm=1.0 (after unscaling) |
| warmup | LinearLR start_factor=0.001, iteration-based, 25 iterations x 4 = 100 images (historical: 50 iterations x 2 = 100 images) |
| cosine | CosineAnnealingLR epoch-based, begin=1 end=50 T_max=50 eta_min=1e-7 (historical end=60 T_max=60; shortened with the schedule at the user's instruction, same shape) |
| max_epochs | 50 |
| validation | none |
| checkpoint_rule | final epoch_50.pth; no best-checkpoint selection |
| train_loader | DefaultSampler shuffle=True, no batch sampler, num_workers=2, persistent workers |
| test_loader | batch 1, num_workers=2, DefaultSampler shuffle=False |
| randomness | randomness.seed=<run seed>, deterministic=False, diff_rank_seed=False, cudnn_benchmark=False |
| point_refine_seed | sha256('<campaign>/point_refine/<seed>')[:4] & 0x7fffffff |
| cuda_launch_blocking | 0 for every campaign run (historical tools/train.py forced 1; no effect on computation) |
| Microbatch iterations | 44,500 (890 per epoch) |
| Optimizer-update opportunities | 44,500 (remainder 0) |
| Updates below base LR during warmup | 24 |
