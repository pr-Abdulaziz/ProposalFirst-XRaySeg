# PIDray-30 batch-1 campaign (`pidray_bs1_e50_2026-09-21`)

Ten architectures, one seed, physical batch 1, no gradient accumulation, 50 epochs.
A separate campaign from the PIXray one: it reuses only the *method definitions*
(so the architectures compared on both datasets are the same objects) and neither
reads nor writes anything under `output/pixray_multiseed_bs4_e50_2026-09-14/`.

## Protocol

| | |
|---|---|
| dataset | PIDray-30 train, de-leaked (22,990 images, 12,014 instances, 61.6 % empty) |
| test | released `easy` 24,758 / `hard` 9,746 / `hidden` 13,069 — untouched |
| classes | 12 |
| batch | 1 physical, accumulation 1 |
| epochs | 50 → 22,990 iterations/epoch → 1,149,500 iterations/run |
| optimizer | AdamW lr 1e-4, wd 0.05 (inherited, **not** rescaled for batch 1) |
| schedule | linear warmup 100 iterations (start factor 0.001), epoch cosine T_max 50 → 1e-7 |
| precision | AMP float16, grad clip 1.0 |
| validation | disabled; final-epoch checkpoint only |
| seed | 42 |
| order | 01, 02, 03, 04, 07, 08, 09, 09M, 05, 06 |

## Leakage gate

`scripts/audit_pidray.py --check-content-hashes` fails on PIDray as released: 84
training images are byte-identical to test images (50 easy, 19 hard, 15 hidden;
11 annotated, 11 instances). User decision, 2026-09-21: exclude them from
training. `build_deleaked_annotations.py` writes read-only derived files under
`output/<campaign>/annotations/`; the released files under `data/` are never
modified and the test splits are untouched. `verify_gate.py` re-checks and must
report PASS. `campaign_common.LEAKAGE_POLICY` gates every entry point in code.

Also recorded, not acted on: 38 duplicate groups *within* the test splits
(easy/hard 11, easy/hidden 21, hard/hidden 6).

## Pipeline

    python scripts/audit_pidray.py --check-content-hashes    # dataset gate (fails as released)
    python scripts/campaign_pidray_bs1/leakage_report.py     # full duplicate enumeration
    python scripts/campaign_pidray_bs1/build_deleaked_annotations.py
    python scripts/campaign_pidray_bs1/verify_gate.py        # must be PASS
    sbatch  scripts/campaign_pidray_bs1/slurm_profile_bs1.sh 01 02 03 04 07 08 09 09M 05 06
    python scripts/campaign_pidray_bs1/build_campaign.py     # resolve + audit + hash configs
    sbatch  scripts/campaign_pidray_bs1/slurm_preflight.sh 01 02 03 04 07 08 09 09M 05 06
    python scripts/campaign_pidray_bs1/freeze_manifest.py    # write-once
    python scripts/campaign_pidray_bs1/submit_campaign.py    # afterok-chained training
    python scripts/campaign_pidray_bs1/submit_campaign.py --submit-evals

`summarize_profile.py` turns the profile into the schedule table and the per-job
time limits (profile mean x 1.35 + 1 h, capped at the 7-day partition limit).

## Notes

* **Learning rate is not rescaled for batch 1.** Every method keeps the inherited
  AdamW 1e-4, so the comparison is internally fair; absolute AP will sit below a
  batch-1-tuned setup. Changing it would be a silent hyperparameter change.
* **PIDray `area` is the mask area** (median 0.39 x box area), unlike PIXray where
  it is the box area, so COCO small/medium/large here are true mask-area bins.
  `evaluate_run.py` records this in the evaluator metadata.
* `test_easy.json` contains annotation id 0, so `ZeroIdSafeCOCOeval` is reported
  beside the official values, as on PIXray.
* Empty ("normal") images are kept, `filter_empty_gt=False, min_size=0`, matching
  the project's established PIDray-30 configs.
* Resume works here: `CampaignRNGStateHookV2`
  (`projects/proposal_first_pidray_occ/hooks/campaign_hooks_v2.py`) restores RNG
  state from a GPU-mapped checkpoint, which the original hook could not. It is a
  new file registered through `custom_imports`, so the PIXray campaign's frozen
  source snapshot stays byte-identical.
