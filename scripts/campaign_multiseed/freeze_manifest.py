#!/usr/bin/env python
"""Freeze the campaign manifest once the build audit and every approved preflight pass.

Re-verifies that nothing changed since the build (resolved configs, source
snapshot, annotations, pretrained weights), collects the preflight summaries, writes
the protocol tables, and writes campaign_manifest.json exactly once (read-only).
Later status changes go to append-only records under runs/ and summaries/.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    BY_ID, CAMPAIGN, MANIFEST, MAX_EPOCHS, METHODS, OUT, SEEDS, eval_dir, method_record, run_dir,
    run_order, sha256, source_snapshot)

PROT = OUT / "protocol"
MEMORY_CHECK = "peak allocated memory below 90% of GPU memory (reserved recorded)"
# User decision (2026-09-14): 06 runs at batch 4 without activation checkpointing, accepting ~95%
# peak allocated memory. An out-of-memory failure is reported, never fixed by changing settings.
MEMORY_WAIVERS = {"06": "user accepted ~95% peak allocated memory at batch 4 without activation checkpointing (2026-09-14)"}


def g(d, *path, default="-"):
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        elif isinstance(d, list) and isinstance(p, int) and p < len(d):
            d = d[p]
        else:
            return default
    return d


def method_rows(build, pre) -> tuple[list[str], dict[str, list[str]]]:
    from mmengine.config import Config
    cols = [m.id for m in METHODS if m.launchable]
    rows: dict[str, list[str]] = {}

    def add(label, fn):
        rows[label] = [str(fn(Config.fromfile(str(ROOT / build["runs"][f"{BY_ID[c].slug}/seed_42"]["resolved_config"])).to_dict(), c))
                       for c in cols]
    add("Training annotations", lambda c, m: Path(c["train_dataloader"]["dataset"]["ann_file"]).name)
    add("Test-loader annotations", lambda c, m: Path(c["test_dataloader"]["dataset"]["ann_file"]).name)
    add("Scored against", lambda c, m: ", ".join(Path(e["ann_file"]).name for e in c["test_evaluator"] if "ann_file" in e))
    add("Detector", lambda c, m: c["model"]["type"])
    add("Backbone", lambda c, m: (f"ResNet-{g(c, 'model', 'backbone', 'depth')}" if "depth" in c["model"]["backbone"]
                                  else "SAM ViT-L + LoRA r16 (qkv)"))
    add("Initialisation", lambda c, m: ("torchvision ImageNet" if BY_ID[m].family == "detector" else "SAM ViT-L weights"))
    add("Frozen", lambda c, m: ("stem + stage 1; BN statistics" if BY_ID[m].family == "detector"
                                else "image encoder except LoRA"))
    add("Neck", lambda c, m: c["model"]["neck"]["type"])
    add("Batch-one normalisation", lambda c, m: (
        "backbone BN frozen; AttFPN attention BN uses running-stat fallback" if c["model"]["neck"]["type"] == "AttFPN"
        else "backbone BN frozen" if BY_ID[m].family == "detector" else "LayerNorm; prompt-head BN over RoIs"))
    add("Anchor scales", lambda c, m: g(c, "model", "rpn_head", "anchor_generator", "scales"))
    add("Proposals train (pre/kept)", lambda c, m: f"{g(c, 'model', 'train_cfg', 'rpn_proposal', 'nms_pre')}/{g(c, 'model', 'train_cfg', 'rpn_proposal', 'max_per_img')}")
    add("Proposals test (pre/kept)", lambda c, m: f"{g(c, 'model', 'test_cfg', 'rpn', 'nms_pre')}/{g(c, 'model', 'test_cfg', 'rpn', 'max_per_img')}")
    add("RoI head", lambda c, m: c["model"]["roi_head"]["type"])
    add("Stages / loss weights", lambda c, m: f"{g(c, 'model', 'roi_head', 'num_stages', default=1)} / {g(c, 'model', 'roi_head', 'stage_loss_weights', default='1')}")
    add("Mask head", lambda c, m: (c["model"]["roi_head"]["mask_head"][0]["type"] if isinstance(c["model"]["roi_head"]["mask_head"], list)
                                   else c["model"]["roi_head"]["mask_head"]["type"]))
    add("Mask target size", lambda c, m: (g(c, "model", "train_cfg", "rcnn", 0, "mask_size") if isinstance(c["model"]["train_cfg"]["rcnn"], list)
                                          else g(c, "model", "train_cfg", "rcnn", "mask_size")))
    add("Occluder loss weight", lambda c, m: g(c, "model", "roi_head", "mask_head", "bo_loss_weight"))
    add("R0/R1 gate", lambda c, m: "yes" if "r0_head" in c["model"] else "-")
    add("Point refinement", lambda c, m: (f"{c['model']['roi_head']['point_refine_head']['type']}; pts {c['model']['roi_head']['point_refine_head']['train_points']}/{c['model']['roi_head']['point_refine_head']['infer_points']}; grids {c['model']['roi_head']['point_refine_head']['output_sizes']}"
                                          if "point_refine_head" in c["model"]["roi_head"] else "-"))
    add("Input normalisation", lambda c, m: f"mean {[round(x, 1) for x in c['model']['data_preprocessor']['mean']]}")
    add("Test: score / NMS / max det / mask thr", lambda c, m: f"{g(c, 'model', 'test_cfg', 'rcnn', 'score_thr')} / {g(c, 'model', 'test_cfg', 'rcnn', 'nms', 'iou_threshold')} / {g(c, 'model', 'test_cfg', 'rcnn', 'max_per_img')} / {g(c, 'model', 'test_cfg', 'rcnn', 'mask_thr_binary')}")
    rows["Parameters total (M)"] = [f"{pre[c]['parameters']['total']/1e6:.2f}" for c in cols]
    rows["Trainable (M)"] = [f"{pre[c]['parameters']['trainable']/1e6:.2f}" for c in cols]
    rows["Frozen (M)"] = [f"{pre[c]['parameters']['frozen']/1e6:.2f}" for c in cols]
    return cols, rows


def main() -> int:
    if MANIFEST.exists():
        raise SystemExit(f"ABORT: {MANIFEST} already exists; the frozen manifest is never overwritten")
    build = json.loads((PROT / "build_report.json").read_text())
    problems = list(build["problems"])
    for key, run in build["runs"].items():
        if sha256(ROOT / run["resolved_config"]) != run["resolved_config_sha256"]:
            problems.append(f"{key}: resolved config changed since the build")
    snap = source_snapshot()
    if snap["aggregate_sha256"] != build["source_snapshot"]["aggregate_sha256"]:
        problems.append("source changed since the build; rebuild before freezing")
    pre = {}
    for m in METHODS:
        if not m.launchable:
            continue
        path = OUT / "preflight" / m.slug / "preflight.json"
        if not path.is_file():
            problems.append(f"{m.id}: no preflight result")
            continue
        r = json.loads(path.read_text())
        pre[m.id] = r
        failed = [c for c in r["checks"] if c["status"] == "FAIL"]
        waived = [c for c in failed if m.id in MEMORY_WAIVERS and c.get("name", c.get("check")) == MEMORY_CHECK]
        r["waived_checks"] = [dict(check=MEMORY_CHECK, reason=MEMORY_WAIVERS[m.id], record=c) for c in waived]
        if len(failed) > len(waived):
            problems.append(f"{m.id}: preflight has {len(failed) - len(waived)} failed checks that are not waived")
        if r["config_sha256"] != build["runs"][f"{m.slug}/seed_42"]["resolved_config_sha256"]:
            problems.append(f"{m.id}: preflight ran on a different config")
        geo_path = OUT / "preflight" / m.slug / "inference_geometry.json"
        geo = json.loads(geo_path.read_text()) if geo_path.is_file() else None
        if geo is None or geo["status"] != "PASS" or geo["config_sha256"] != r["config_sha256"]:
            problems.append(f"{m.id}: inference-geometry preflight missing or not passed")
        r["inference_geometry"] = geo
    if problems:
        print("\n".join(problems))
        return 2

    cols, rows = method_rows(build, pre)
    lines = [f"# Side-by-side configuration ({CAMPAIGN})", "",
             "Generated from the resolved seed-42 configs and preflight parameter counts. Every protocol field "
             "(optimizer, schedule, AMP, clipping, batch, workers, sampler, epochs, checkpoint rule, evaluator, "
             "augmentation parameters) is identical across methods and is listed once in the shared table.", "",
             "| Item | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows.items()]
    lines += ["", "## Shared protocol", "", "| Setting | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in build["protocol"].items()]
    s = build["schedule"]
    lines += [f"| Microbatch iterations | {s['microbatch_iterations']:,} ({s['iterations_per_epoch']:,} per epoch) |",
              f"| Optimizer-update opportunities | {s['update_opportunities']:,} (remainder {s['remainder_microbatches']}) |",
              f"| Updates below base LR during warmup | {s['updates_below_base_lr_in_warmup']} |"]
    (PROT / "fairness_table.md").write_text("\n".join(lines) + "\n")
    diffs = json.loads((PROT / "fairness_differences.json").read_text())
    dl = ["# Classified configuration differences", "", "| Key | Category | Values | Rationale |", "|---|---|---|---|"]
    for k, v in diffs["classified"].items():
        inv = {}
        for mid, val in v["values"].items():
            inv.setdefault(val, []).append(mid)
        vals = "; ".join(f"{','.join(ms)}: {val[:60]}" for val, ms in inv.items()).replace("|", "/")
        dl.append(f"| `{k}` | {v['category']} | {vals} | {v['rationale'][:160]} |")
    (PROT / "fairness_differences.md").write_text("\n".join(dl) + "\n")

    approved = [(mid, seed) for mid, seed in run_order() if BY_ID[mid].launchable]
    manifest = dict(
        campaign=CAMPAIGN, status="frozen", frozen_at=dt.datetime.now().isoformat(timespec="seconds"),
        dataset="PIXray official split only; PIDray not included", seeds=list(SEEDS),
        protocol=build["protocol"], schedule={k: v for k, v in build["schedule"].items() if k != "per_epoch"},
        methods=[method_record(m) for m in METHODS],
        run_order=[dict(index=i, method=mid, seed=seed, run_key=f"{BY_ID[mid].slug}/seed_{seed}")
                   for i, (mid, seed) in enumerate(approved)],
        runs={k: dict(v, run_dir=str(run_dir(v["method"], v["seed"]).relative_to(ROOT)),
                      eval_dir=str(eval_dir(v["method"], v["seed"]).relative_to(ROOT)))
              for k, v in build["runs"].items()},
        held=[],
        annotation_decisions=dict(
            rule="05 and every other detector train on data/PIXray/annotations/train.json; only 06 and 07 train on "
                 "PIXray-A occluder annotations; every method is scored against data/PIXray/annotations/test.json",
            method_07="kept with occluder supervision by user decision (2026-09-14); reported as an additional-"
                      "supervision comparison beside 06, not as a target-only method",
            method_06_memory=MEMORY_WAIVERS["06"],
            decision_document=str((PROT / "method07_annotation_conflict.md").relative_to(ROOT))),
        annotation_policy={m.id: dict(supervision=m.supervision, **{k: build["runs"][f"{m.slug}/seed_42"]["policy"].get(k)
                                                                    for k in ("train_annotations", "test_loader_annotations",
                                                                              "evaluator_annotations", "occlusion_components",
                                                                              "inherited_passthrough", "conflict")})
                           for m in METHODS},
        annotations=build["annotations"], annotation_identity=build["annotation_identity"], images=build["images"],
        pretrained=build["pretrained"],
        source_snapshot=dict(build["source_snapshot"], per_file=str((PROT / "source_snapshot.json").relative_to(ROOT))),
        environment=build["environment"], registry_mapping=build["registry_mapping"],
        fairness=dict(build["fairness"], table=str((PROT / "fairness_table.md").relative_to(ROOT)),
                      differences_sha256=sha256(PROT / "fairness_differences.json")),
        preflight={mid: dict(result=str((OUT / "preflight" / BY_ID[mid].slug / "preflight.json").relative_to(ROOT)),
                             job=r["job"], node=r["node"], gpu=r["gpu"], passed=r["summary"]["passed"], waived_checks=r["waived_checks"],
                             parameters={k: r["parameters"][k] for k in ("total", "trainable", "frozen", "buffers")},
                             peak_reserved_mb=r["peak_reserved_mb_max"], headroom_mb=r["headroom_mb"],
                             inference_geometry=dict(status=r["inference_geometry"]["status"],
                                                     nonempty_masks_checked=r["inference_geometry"]["nonempty_masks_checked"],
                                                     result=str((OUT / "preflight" / BY_ID[mid].slug / "inference_geometry.json").relative_to(ROOT))),
                             scorer_validation=str((PROT / "scorer_validation.json").relative_to(ROOT)),
                             timing=r["timing"]) for mid, r in pre.items()},
        evaluation_policy=dict(checkpoint=f"epoch_{MAX_EPOCHS}.pth of the run's own work_dir",
                               timing="no evaluation starts until every approved training run has completed", split="data/PIXray/annotations/test.json",
                               scorer="scripts/campaign_multiseed/score_predictions.py",
                               primary="official pycocotools COCOeval", secondary="ZeroIdSafeCOCOeval (separately labelled)",
                               size_ranges="COCO area ranges on PIXray 'area' (bounding-box area)"),
        reporting_policy=("every seed reported; mean and sample standard deviation over three seeds; paired per-seed "
                          "differences against the stated baseline; no best-seed reporting, no prediction merging, "
                          "no threshold/checkpoint/architecture selection on test results"),
        superseded_campaigns=[
            dict(path="output/pixray_multiseed_2026-09-14", physical_batch=1,
                 status="cancelled at the user's instruction before any epoch completed",
                 manifest_sha256=sha256(ROOT / "output/pixray_multiseed_2026-09-14/campaign_manifest.json")),
            dict(path="output/pixray_multiseed_bs8_2026-09-14", physical_batch=8,
                 status="preflight stopped at the user's instruction to use batch 4 without accumulation; never frozen or launched"),
            dict(path="output/pixray_multiseed_bs4_2026-09-14", physical_batch=4, max_epochs=60,
                 status="preflight only; superseded by the user's instruction to train 50 epochs; never frozen or launched")],
        historical_archive=dict(path="archive/historical_output_2026-09-14/historical_output_2026-09-14.tar.gz",
                                manifest_sha256=sha256(ROOT / "archive/historical_output_2026-09-14/MANIFEST.json")),
        build_report_sha256=sha256(PROT / "build_report.json"))
    from scripts.campaign_multiseed.campaign_common import write_once
    write_once(MANIFEST, manifest)
    (OUT / "campaign_manifest.sha256").write_text(f"{sha256(MANIFEST)}  campaign_manifest.json\n")
    print(f"frozen {MANIFEST} ({len(approved)} approved runs) sha256={sha256(MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
