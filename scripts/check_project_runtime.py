#!/usr/bin/env python3
"""Bounded runtime/extension checks; no downloads, training jobs or installs.

Run from the project environment. CPU mode hides all CUDA devices before imports.
CUDA mode requires an existing SLURM allocation. The JSON output must not exist.
The model probes use tiny synthetic RoIs and reduced point budgets; they do not
validate trained checkpoints, full detectors, dataset integrity or model accuracy.
Historical heads are required to pass FP32 checks. CUDA AMP checks use explicitly
selected FP32-inference compatibility heads; historical AMP is an optional
diagnostic and is not a supported path implied by a successful core report.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback


LEGACY_HEADS = ("PointRefineHead", "PointRefineMLPHead", "PointRefineHeadDetached",
                "PointRefineMLPHeadDetached")
FP32_INFERENCE_HEADS = ("PointRefineHeadFP32Inference", "PointRefineMLPHeadFP32Inference",
                        "PointRefineHeadDetachedFP32Inference",
                        "PointRefineMLPHeadDetachedFP32Inference")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-legacy-deform-conv", action="store_true",
                        help="Also execute a tiny CUDA deformable-convolution check.")
    parser.add_argument("--include-legacy-amp-inference", action="store_true",
                        help="Diagnose historical CUDA AMP heads; optional failures remain visible.")
    return parser.parse_args()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


class Report:
    def __init__(self, stream, metadata):
        self.stream = stream
        self.data = {"schema_version": 2, "metadata": metadata, "checks": []}
        self.flush()

    def flush(self):
        required = [c for c in self.data["checks"] if c["required"]]
        self.data["summary"] = {
            "required_passed": sum(c["status"] == "pass" for c in required),
            "required_not_passed": sum(c["status"] != "pass" for c in required),
            "optional_not_passed": sum(c["status"] != "pass" for c in self.data["checks"]
                                       if not c["required"]),
        }
        self.stream.seek(0)
        json.dump(self.data, self.stream, indent=2, allow_nan=False)
        self.stream.write("\n")
        self.stream.truncate()
        self.stream.flush()

    def check(self, name, function, required=True):
        start = time.monotonic()
        item = {"name": name, "required": required}
        try:
            item.update(status="pass", details=function())
        except Exception as exc:
            item.update(status="fail", error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc()[-14000:])
        item["seconds"] = round(time.monotonic() - start, 4)
        self.data["checks"].append(item)
        self.flush()
        print(f"{item['status'].upper()}: {name}", flush=True)
        return item["status"] == "pass"

    def skip(self, name, reason, required=True):
        self.data["checks"].append({"name": name, "required": required,
                                    "status": "skip", "reason": reason})
        self.flush()
        print(f"SKIP: {name}: {reason}", flush=True)


def run(args, report):
    loaded = {}

    def import_one(name):
        module = importlib.import_module(name)
        loaded[name] = module
        if name == "torch":
            # Set these before later imports can execute parallel torch work.
            module.set_num_threads(1)
            module.set_num_interop_threads(1)
        distributions = importlib.metadata.packages_distributions().get(name.split('.')[0], [])
        versions = {}
        for distribution in distributions:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        return {"module_version": str(getattr(module, "__version__", "not_exposed")),
                "module_path": str(getattr(module, "__file__", "not_exposed")),
                "distributions": versions}

    # Collect every import outcome even when another package fails.
    for name in ("torch", "torchvision", "mmcv", "mmengine", "numpy",
                 "mmdet", "mmpretrain", "transformers", "peft", "cv2", "pycocotools"):
        report.check(f"import:{name}", lambda name=name: import_one(name))

    def pip_check():
        result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                capture_output=True, text=True, timeout=60)
        # Metadata conflicts can concern optional tools; they do not fail the
        # required runtime checks but must not be labelled a successful pip check.
        require(result.returncode == 0,
                "pip check reported metadata conflicts (inspect against the core checks):\n"
                + result.stdout[-12000:] + result.stderr[-12000:])
        return {"returncode": result.returncode, "stdout": result.stdout[-12000:],
                "stderr": result.stderr[-12000:],
                "interpretation": "Inspect conflicts against the required import/operation checks."}

    report.check("package_metadata:pip_check", pip_check, required=False)

    def custom_registry():
        from mmdet.utils import register_all_modules
        from mmdet.registry import MODELS
        register_all_modules(init_default_scope=True)
        import_one("projects.proposal_first_pidray_occ")
        import_one("projects.proposal_first_pidray_occ.models.point_refine.mlp_control")
        import_one("projects.proposal_first_pidray_occ.models.point_refine.detached")
        import_one("projects.proposal_first_pidray_occ.models.point_refine.fp32_inference")
        expected = ("AttFPN", "PointRefineCascadeRoIHead") + LEGACY_HEADS + FP32_INFERENCE_HEADS
        found = {name: MODELS.get(name) is not None for name in expected}
        require(all(found.values()), f"Unregistered required custom modules: {found}")
        return {"registered": found,
                "project_module_path": loaded["projects.proposal_first_pidray_occ"].__file__}

    registry_ok = report.check("custom_registry", custom_registry)

    def extension_metadata():
        ext = importlib.import_module("mmcv._ext")
        ops = importlib.import_module("mmcv.ops")
        details = {"binary_path": ext.__file__}
        for name in ("get_compiling_cuda_version", "get_compiler_version"):
            function = getattr(ops, name, None)
            details[name] = str(function()) if function else "not_exposed"
        return details

    report.check("mmcv_extension_metadata", extension_metadata)
    if "torch" not in loaded:
        report.skip("tensor_operations", "PyTorch import failed")
        return

    torch = loaded["torch"]
    torch.manual_seed(729)

    def device_check():
        details = {"torch_cuda_build": torch.version.cuda,
                   "cudnn_version": torch.backends.cudnn.version(),
                   "device_requested": args.device}
        if args.device == "cuda":
            require(bool(os.environ.get("SLURM_JOB_ID")), "CUDA checks require SLURM_JOB_ID")
            require(torch.cuda.is_available(), "CUDA is unavailable in this allocation")
            details.update(name=torch.cuda.get_device_name(0),
                           capability=list(torch.cuda.get_device_capability(0)))
            probe = torch.ones(4, device="cuda") * 2
            require(float(probe.sum()) == 8, "Basic CUDA tensor arithmetic failed")
            torch.cuda.synchronize()
        else:
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU mode did not hide CUDA")
        return details

    if not report.check("device", device_check):
        report.skip("tensor_operations", "Requested device check failed")
        return
    device = torch.device(args.device)

    def finite(tensor, label):
        require(torch.isfinite(tensor).all().item(), f"Non-finite tensor: {label}")

    def roi_align_check():
        from mmcv.ops import roi_align
        features = (torch.arange(128, device=device, dtype=torch.float32)
                    .reshape(1, 2, 8, 8) / 10).requires_grad_()
        rois = torch.tensor([[0, 1, 1, 7, 7]], device=device, dtype=torch.float32)
        # MMCV 2.1 exports Function.apply here, which accepts positional args.
        result = roi_align(features, rois, (3, 3), 1.0, 2, "avg", True)
        require(tuple(result.shape) == (1, 2, 3, 3), "Unexpected RoIAlign shape")
        finite(result, "RoIAlign output")
        result.sum().backward()
        finite(features.grad, "RoIAlign gradient")
        require(features.grad.abs().sum().item() > 0, "RoIAlign gradient is zero")
        return {"output_shape": list(result.shape), "gradient_sum": features.grad.sum().item()}

    report.check("mmcv_roi_align_forward_backward", roi_align_check)

    def nms_check(provider):
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]],
                             device=device, dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        if provider == "mmcv":
            from mmcv.ops import nms
            _, keep = nms(boxes, scores, 0.5)
        else:
            from torchvision.ops import nms
            keep = nms(boxes, scores, 0.5)
        require(keep.cpu().tolist() == [0, 2], f"Unexpected {provider} NMS indices")
        return {"indices": keep.cpu().tolist()}

    for provider in ("mmcv", "torchvision"):
        report.check(f"{provider}_nms", lambda provider=provider: nms_check(provider))

    def point_sample_check():
        from mmcv.ops import point_sample, rel_roi_point_to_rel_img_point
        features = torch.randn(1, 2, 8, 8, device=device, requires_grad=True)
        rois = torch.tensor([[0, 0, 0, 8, 8]], device=device, dtype=torch.float32)
        coords = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]], device=device)
        image_coords = rel_roi_point_to_rel_img_point(rois, coords, features.shape[2:], 1.0)
        finite(image_coords, "converted point coordinates")
        require(torch.allclose(image_coords, coords), "Full-image RoI coordinate conversion differs")
        result = point_sample(features, image_coords, align_corners=False)
        finite(result, "point samples")
        require(tuple(result.shape) == (1, 2, 2), "Unexpected point_sample shape")
        result.sum().backward()
        finite(features.grad, "point-sampling gradient")
        require(features.grad.abs().sum().item() > 0, "Point-sampling gradient is zero")
        return {"shape": list(result.shape), "coordinates": image_coords.cpu().tolist()}

    report.check("point_sample_coordinates_backward", point_sample_check)

    def coco_rle_check():
        import numpy as np
        from pycocotools import mask as mask_utils
        mask = np.zeros((8, 9), dtype=np.uint8)
        mask[2:6, 3:7] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        decoded = mask_utils.decode(rle)
        require(np.array_equal(mask, decoded), "COCO RLE round-trip differs")
        area = int(mask_utils.area(rle))
        iou = float(mask_utils.iou([rle], [rle], [0])[0, 0])
        require(area == 16 and iou == 1.0, "COCO mask area/IoU differs")
        return {"area": area, "self_iou": iou}

    report.check("pycocotools_rle_area_iou", coco_rle_check)

    def head_check(class_name, amp=False):
        from mmdet.registry import MODELS
        detached = class_name.endswith("Detached")
        head = MODELS.build(dict(type=class_name, num_classes=12,
                                 output_sizes=(56, 112), train_points=16,
                                 infer_points=32, oversample_ratio=3,
                                 uncertain_fraction=0.75, embed_dim=128,
                                 num_heads=4, num_blocks=2, ffn_dim=512)).to(device)
        coarse = torch.randn(2, 12, 28, 28, device=device, requires_grad=True)
        features = torch.randn(1, 256, 16, 16, device=device, requires_grad=True)
        rois = torch.tensor([[0, 4, 4, 44, 52], [0, 16, 8, 60, 60]],
                            device=device, dtype=torch.float32)
        labels = torch.tensor([0, 11], device=device, dtype=torch.long)
        targets = torch.zeros(2, 64, 64, device=device)
        targets[0, 10:40, 10:34] = 1
        targets[1, 20:52, 24:54] = 1
        original = coarse.detach().clone()

        if amp:
            require(args.device == "cuda", "AMP inference check requires CUDA")
            head.eval()
            # Coarse/fine outputs would be half precision in a full AMP detector.
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                predicted = head.predict_refine(coarse.detach().half(), rois, labels,
                                                features.detach().half())
            finite(predicted, "AMP refined logits")
            require(tuple(predicted.shape) == (2, 12, 112, 112), "Wrong AMP inference shape")
            compatibility_head = class_name in FP32_INFERENCE_HEADS
            if compatibility_head:
                require(predicted.dtype == torch.float32,
                        "FP32-inference compatibility head did not return FP32")
                # Compare identical quantized inputs with/without outer AMP.
                # This checks the FP32 island, not model accuracy or FP16 training.
                with torch.no_grad():
                    reference = head.predict_refine(coarse.detach().half(), rois, labels,
                                                    features.detach().half())
                require(torch.equal(predicted, reference),
                        "Compatibility-head output changed under outer autocast")
            return {"shape": list(predicted.shape), "dtype": str(predicted.dtype),
                    "fp32_inference_compatibility_head": compatibility_head,
                    "note": ("FP32 head weights; half inputs; FP32 inference island under outer AMP. "
                             "No AMP-training, historical-head AMP, or accuracy claim."
                             if compatibility_head else
                             "Optional historical AMP diagnostic; not required supported behavior.")}

        head.train()
        generator = torch.Generator(device=device).manual_seed(1234)
        result = head.loss_and_refine(coarse, rois, labels, features, targets,
                                      stage=0, generator=generator)
        require([key for key in result if "loss" in key] == ["loss_point"],
                "Point loss would be counted more than once")
        loss = result["loss_point"]
        finite(loss, "point loss")
        finite(result["_refined"], "training refined logits")
        require(tuple(result["_refined"].shape) == (2, 12, 112, 112), "Wrong training shape")
        loss.backward()
        gradients = [p.grad for p in head.parameters() if p.grad is not None]
        require(bool(gradients), "Refiner received no gradients")
        for grad in gradients:
            finite(grad, "refiner gradient")
        require(any(g.abs().sum().item() > 0 for g in gradients), "All refiner gradients are zero")
        input_gradients = {}
        for name, tensor in (("coarse", coarse), ("features", features)):
            grad = tensor.grad
            if grad is not None:
                finite(grad, f"{name} gradient")
            amount = 0.0 if grad is None else grad.abs().sum().item()
            input_gradients[name] = amount
            require(amount == 0 if detached else amount > 0,
                    f"Unexpected {'detached' if detached else 'ordinary'} gradient policy for {name}")
        require(torch.equal(coarse.detach(), original), "Refinement mutated original coarse logits")

        head.eval()
        with torch.no_grad():
            predicted = head.predict_refine(coarse.detach(), rois, labels, features.detach())
        finite(predicted, "inference refined logits")
        require(tuple(predicted.shape) == (2, 12, 112, 112), "Wrong inference shape")

        head.zero_grad(set_to_none=True)
        empty_coarse = torch.empty(0, 12, 28, 28, device=device, requires_grad=True)
        empty = head.loss_and_refine(empty_coarse, rois[:0], labels[:0], features.detach(), targets[:0])
        finite(empty["loss_point"], "empty point loss")
        require(empty["loss_point"].item() == 0, "Empty loss is not zero")
        empty["loss_point"].backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                    and p.grad.abs().sum().item() == 0 for p in head.parameters()),
                "Empty branch did not give finite connected zero gradients")
        with torch.no_grad():
            empty_prediction = head.predict_refine(empty_coarse.detach(), rois[:0], labels[:0],
                                                   features.detach())
        require(empty_prediction.shape[0] == 0, "Empty inference produced instances")
        return {"loss": loss.item(), "shape": list(predicted.shape),
                "input_gradient_abs_sums": input_gradients,
                "empty_prediction_shape": list(empty_prediction.shape),
                "train_points": 16, "infer_points": 32,
                "note": "Synthetic head checks; no optimiser step or full detector build."}

    for name in LEGACY_HEADS:
        if registry_ok:
            report.check(f"head_fp32:{name}", lambda name=name: head_check(name))
        else:
            report.skip(f"head_fp32:{name}", "Custom registry check failed")
    if args.device == "cuda":
        for name in FP32_INFERENCE_HEADS:
            if registry_ok:
                report.check(f"head_compat_amp_inference:{name}", lambda name=name: head_check(name, amp=True))
            else:
                report.skip(f"head_compat_amp_inference:{name}", "Custom registry check failed")
    if args.include_legacy_amp_inference:
        if args.device != "cuda":
            report.skip("optional_legacy_amp_inference", "CUDA-only optional diagnostic", required=False)
        else:
            for name in LEGACY_HEADS:
                if registry_ok:
                    report.check(f"optional_legacy_amp_inference:{name}",
                                 lambda name=name: head_check(name, amp=True), required=False)
                else:
                    report.skip(f"optional_legacy_amp_inference:{name}",
                                "Custom registry check failed", required=False)

    if args.include_legacy_deform_conv:
        if args.device != "cuda":
            report.skip("optional_legacy_deform_conv", "CUDA-only optional check", required=False)
        else:
            def deform_conv_check():
                from mmcv.ops import DeformConv2d
                module = DeformConv2d(2, 2, 3, padding=1, bias=False).to(device)
                features = torch.randn(1, 2, 8, 8, device=device, requires_grad=True)
                offsets = torch.zeros(1, 18, 8, 8, device=device, requires_grad=True)
                output = module(features, offsets)
                finite(output, "deformable convolution output")
                output.sum().backward()
                finite(features.grad, "deformable convolution input gradient")
                finite(offsets.grad, "deformable convolution offset gradient")
                return {"shape": list(output.shape)}
            report.check("optional_legacy_deform_conv", deform_conv_check, required=False)


def main():
    args = parse_args()
    # Set before any third-party import, including imports that transitively load torch.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = args.output.open("x", encoding="utf-8")
    except FileExistsError:
        print(f"Refusing to overwrite existing report: {args.output}", file=sys.stderr)
        return 2
    with stream:
        report = Report(stream, {"started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "hostname": socket.gethostname(), "project_root": str(root),
                                 "python_executable": sys.executable, "python_version": platform.python_version(),
                                 "device": args.device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                 "mode_boundaries": {
                                     "required_fp32_heads": list(LEGACY_HEADS),
                                     "required_cuda_amp_inference_heads": (
                                         list(FP32_INFERENCE_HEADS) if args.device == "cuda" else []),
                                     "legacy_amp_diagnostic_requested": args.include_legacy_amp_inference,
                                     "amp_training_tested": False,
                                     "note": "Core PASS does not establish historical AMP support or full-detector accuracy."},
                                 "scope": "Bounded synthetic import/binary/head checks, not full experiment validation."})
        if args.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
            report.check("cuda_allocation_guard",
                         lambda: require(False, "CUDA checks require an existing SLURM allocation"))
        else:
            try:
                run(args, report)
            except Exception as exc:
                report.check("unexpected_runner_failure", lambda: (_ for _ in ()).throw(exc))
        report.data["metadata"]["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report.flush()
        failed = report.data["summary"]["required_not_passed"]
        print(json.dumps(report.data["summary"], sort_keys=True), flush=True)
        print(f"Report: {args.output}", flush=True)
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
