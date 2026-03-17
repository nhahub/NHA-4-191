#!/usr/bin/env python3
"""Benchmark object detection models for Road-Sense real-time trade-off analysis.

Models:
- YOLOv8s (Ultralytics)
- YOLOv11m (Ultralytics)
- SSD300 VGG16 (torchvision)
- Faster R-CNN ResNet50 FPN v2 (torchvision)

Outputs:
- experiments/model_benchmarks/model_benchmark_results.csv
- experiments/model_benchmarks/model_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
import requests
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn_v2,
    ssd300_vgg16,
)
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO


COCO128_ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

# COCO 91-class IDs to COCO 80-class IDs (index is 91-class id; value is 80-class id or -1).
COCO91_TO_COCO80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


@dataclass
class BenchResult:
    model: str
    family: str
    params_m: float
    model_size_mb: float
    device: str
    imgsz: int
    num_images: int
    map50: float
    map5095: float
    latency_ms: float
    fps: float
    cpu_mem_delta_mb: float
    gpu_mem_peak_mb: float


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    """Download file from URL to destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  {dest} already exists, skipping download")
        return
    print(f"  Downloading {url}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Downloaded to {dest}")


def image_to_label_path(rel_image: str) -> str:
    """Convert image path to corresponding label path."""
    return rel_image.replace("images/", "labels/").replace(".jpg", ".txt")


def prepare_sample_dataset(root: Path) -> List[Path]:
    """Prepare COCO128 sample dataset for benchmarking."""
    coco_root = root / "coco128"
    images_root = coco_root / "images" / "train2017"
    labels_root = coco_root / "labels" / "train2017"

    if not images_root.exists() or not labels_root.exists():
        zip_path = root / "coco128.zip"
        download_file(COCO128_ZIP_URL, zip_path)
        print(f"  Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

    image_paths_all = sorted(images_root.glob("*.jpg"))
    image_paths: List[Path] = []
    for img_path in image_paths_all:
        lbl_path = labels_root / f"{img_path.stem}.txt"
        if lbl_path.exists():
            image_paths.append(img_path)
        if len(image_paths) >= 12:
            break

    if len(image_paths) < 6:
        raise RuntimeError(f"Insufficient labeled sample images in coco128: {len(image_paths)}")

    print(f"  Found {len(image_paths)} labeled images")
    return image_paths


def yolo_txt_to_boxes_labels(label_path: Path, width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse YOLO format label file to boxes and labels tensors."""
    if not label_path.exists():
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)
    
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)

    boxes = []
    labels = []
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = map(float, parts)
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls))

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)
    
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def load_ground_truths(image_paths: List[Path], dataset_root: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load ground truth annotations for all images."""
    gts: Dict[str, Dict[str, torch.Tensor]] = {}
    labels_root = dataset_root / "coco128" / "labels" / "train2017"
    
    for img_path in image_paths:
        with Image.open(img_path) as img:
            w, h = img.size
        lbl_path = labels_root / f"{img_path.stem}.txt"
        boxes, labels = yolo_txt_to_boxes_labels(lbl_path, w, h)
        gts[str(img_path)] = {"boxes": boxes, "labels": labels}
    
    return gts


def count_parameters(model: torch.nn.Module) -> float:
    """Count model parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def state_dict_size_mb(model: torch.nn.Module) -> float:
    """Calculate model state dict size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return size_mb


def yolo_weight_size_mb(model: YOLO) -> float:
    """Get YOLO model weight file size in MB."""
    try:
        # Try different attribute names for model path
        for attr in ["ckpt_path", "model_path", "weights"]:
            if hasattr(model, attr):
                path = getattr(model, attr)
                if path and isinstance(path, (str, Path)):
                    path = Path(path)
                    if path.exists():
                        return path.stat().st_size / (1024 * 1024)
    except Exception:
        pass
    # Fallback: estimate from model state
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        model.model.save(tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        os.remove(tmp_path)
        return size_mb
    except Exception:
        return float("nan")


def yolo_predict(
    model: YOLO,
    image_path: Path,
    device: str,
    imgsz: int,
    conf: float,
) -> Dict[str, torch.Tensor]:
    """Run YOLO inference and return predictions."""
    out = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False,
        augment=False,
    )[0]
    
    if out.boxes is None or len(out.boxes) == 0:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        }
    
    boxes = out.boxes.xyxy.detach().cpu().to(torch.float32)
    scores = out.boxes.conf.detach().cpu().to(torch.float32)
    labels = out.boxes.cls.detach().cpu().to(torch.int64)
    return {"boxes": boxes, "scores": scores, "labels": labels}


def torchvision_predict(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    score_thresh: float,
) -> Dict[str, torch.Tensor]:
    """Run torchvision model inference and return predictions."""
    with Image.open(image_path).convert("RGB") as img:
        t = pil_to_tensor(img).float() / 255.0
    
    with torch.no_grad():
        out = model([t.to(device)])[0]

    boxes = out["boxes"].detach().cpu().to(torch.float32)
    scores = out["scores"].detach().cpu().to(torch.float32)
    labels_91 = out["labels"].detach().cpu().to(torch.int64)

    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels_91 = labels_91[keep]

    if len(labels_91) == 0:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        }

    # Map COCO 91-class to 80-class labels
    mapped_labels = []
    mapped_keep = []
    for i, lab in enumerate(labels_91.tolist()):
        mapped = COCO91_TO_COCO80.get(lab, -1)
        if mapped >= 0:
            mapped_keep.append(i)
            mapped_labels.append(mapped)

    if mapped_keep:
        idx = torch.tensor(mapped_keep, dtype=torch.int64)
        boxes = boxes[idx]
        scores = scores[idx]
        labels = torch.tensor(mapped_labels, dtype=torch.int64)
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        scores = torch.empty((0,), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.int64)

    return {"boxes": boxes, "scores": scores, "labels": labels}


def benchmark_model(
    name: str,
    family: str,
    image_paths: List[Path],
    gt_map: Dict[str, Dict[str, torch.Tensor]],
    imgsz: int,
    conf: float,
    warmup: int,
    repeats: int,
    device_str: str,
) -> BenchResult:
    """Benchmark a single model and return results."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    proc = psutil.Process(os.getpid())
    cpu_before = proc.memory_info().rss

    # Load model
    if family == "yolo":
        model = YOLO(name)
        try:
            info = model.info(verbose=False)
            if isinstance(info, (list, tuple)) and len(info) > 0:
                params_m = float(info[0].get("params", info[0].get("parameters", 0))) / 1e6
            elif isinstance(info, dict):
                params_m = float(info.get("params", info.get("parameters", 0))) / 1e6
            else:
                params_m = count_parameters(model.model) if hasattr(model, "model") else 0.0
        except Exception:
            params_m = count_parameters(model.model) if hasattr(model, "model") else 0.0
        size_mb = yolo_weight_size_mb(model)
    elif family == "ssd":
        weights = SSD300_VGG16_Weights.DEFAULT
        model = ssd300_vgg16(weights=weights)
        params_m = count_parameters(model)
        size_mb = state_dict_size_mb(model)
    elif family == "fasterrcnn":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        params_m = count_parameters(model)
        size_mb = state_dict_size_mb(model)
    else:
        raise ValueError(f"Unknown family: {family}")

    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(device_str if use_cuda else "cpu")

    if family in {"ssd", "fasterrcnn"}:
        model = model.to(device)
        model.eval()

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup pass to stabilize latency
    print(f"  Running {warmup} warmup iterations...")
    for _ in range(warmup):
        for image_path in image_paths:
            if family == "yolo":
                _ = yolo_predict(model, image_path, device_str, imgsz, conf)
            else:
                _ = torchvision_predict(model, image_path, device, conf)

    # Benchmark iterations
    print(f"  Running {repeats} benchmark iterations...")
    latencies: List[float] = []
    for rep in range(repeats):
        for idx, image_path in enumerate(image_paths):
            start = time.perf_counter()
            if family == "yolo":
                pred = yolo_predict(model, image_path, device_str, imgsz, conf)
            else:
                pred = torchvision_predict(model, image_path, device, conf)
            if use_cuda:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            gt = gt_map[str(image_path)]
            # Filter out empty ground truths for metric calculation
            if len(gt["boxes"]) > 0 or len(pred["boxes"]) > 0:
                metric.update([pred], [gt])

    # Compute metrics
    try:
        maps = metric.compute()
        map50 = float(maps.get("map_50", torch.tensor(0.0)).item())
        map5095 = float(maps.get("map", torch.tensor(0.0)).item())
    except Exception as e:
        print(f"  Warning: mAP computation failed: {e}")
        map50 = 0.0
        map5095 = 0.0

    # Memory tracking
    cpu_after = proc.memory_info().rss
    cpu_delta_mb = (cpu_after - cpu_before) / (1024 * 1024)
    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if use_cuda
        else 0.0
    )

    latency_ms = float(np.mean(latencies)) if latencies else float("nan")
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    return BenchResult(
        model=name,
        family=family,
        params_m=round(params_m, 3),
        model_size_mb=round(size_mb, 2),
        device=str(device),
        imgsz=imgsz,
        num_images=len(image_paths),
        map50=round(map50, 4),
        map5095=round(map5095, 4),
        latency_ms=round(latency_ms, 2),
        fps=round(fps, 2),
        cpu_mem_delta_mb=round(cpu_delta_mb, 2),
        gpu_mem_peak_mb=round(gpu_peak_mb, 2),
    )


def run(args: argparse.Namespace) -> None:
    """Main benchmark runner."""
    root = Path(args.project_root).resolve()
    artifacts = root / "experiments" / "model_benchmarks"
    sample_root = artifacts / "samples"
    ensure_dir(artifacts)

    print("=" * 60)
    print("Road-Sense Model Benchmark")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Image Size: {args.imgsz}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"Warmup Iterations: {args.warmup}")
    print(f"Benchmark Repeats: {args.repeats}")
    print("=" * 60)

    print("\nPreparing sample dataset...")
    image_paths = prepare_sample_dataset(sample_root)
    gt_map = load_ground_truths(image_paths, sample_root)

    models = [
        ("yolov8s.pt", "yolo"),
        ("yolo11m.pt", "yolo"),
        ("ssd300_vgg16", "ssd"),
        ("fasterrcnn_resnet50_fpn_v2", "fasterrcnn"),
    ]

    results: List[BenchResult] = []
    for model_name, family in models:
        print(f"\nBenchmarking {model_name} ({family})...")
        try:
            result = benchmark_model(
                name=model_name,
                family=family,
                image_paths=image_paths,
                gt_map=gt_map,
                imgsz=args.imgsz,
                conf=args.conf,
                warmup=args.warmup,
                repeats=args.repeats,
                device_str=args.device,
            )
            results.append(result)
            print(
                f"  ✓ map50={result.map50:.4f} map5095={result.map5095:.4f} "
                f"latency={result.latency_ms:.2f}ms fps={result.fps:.2f}"
            )
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            import traceback
            traceback.print_exc()

    if not results:
        raise RuntimeError("All benchmarks failed")

    # Create DataFrame and sort
    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values(by=["map5095", "fps"], ascending=[False, False]).reset_index(drop=True)

    # Save results
    csv_path = artifacts / "model_benchmark_results.csv"
    json_path = artifacts / "model_benchmark_results.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print("\n" + df.to_string(index=False))
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Road-Sense Model Benchmark Runner")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repeats")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())