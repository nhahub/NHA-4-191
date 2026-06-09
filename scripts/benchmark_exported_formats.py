#!/usr/bin/env python3
"""Benchmark exported YOLO model formats.

Benchmarks all successfully exported format files from models/exports/.
Uses KITTI validation set for ground truth.

Outputs:
- experiments/format_comparison/format_comparison_results.csv
- experiments/format_comparison/format_comparison_results.json
- experiments/format_comparison/*.png visualizations
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Disable ultralytics auto-install of dependencies
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.environ["YOLO_VERBOSE"] = "False"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import BENCHMARK_NUM_IMAGES, compute_precision_recall, ensure_dir, yolo_txt_to_boxes_labels

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "models/exports"
VAL_IMAGES_DIR = PROJECT_ROOT / "data/processed/kitti/images/val"
VAL_LABELS_DIR = PROJECT_ROOT / "data/processed/kitti/labels/val"
BENCHMARK_DIR = PROJECT_ROOT / "experiments/format_comparison"

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
NUM_IMAGES = BENCHMARK_NUM_IMAGES
WARMUP = 1  # Reduced for speed
REPEATS = 3  # Reduced for speed

CUSTOM_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]


@dataclass
class BenchmarkResult:
    format_name: str
    model_path: Path
    model_size_mb: float
    map50: float
    map5095: float
    precision: float
    recall: float
    latency_ms: float
    fps: float
    per_class: dict[str, float]
    error: str = ""


def load_validation_set(
    images_dir: Path, labels_dir: Path, num_images: int = 50
) -> tuple[list[Path], dict[str, dict[str, torch.Tensor]]]:
    """Load validation images and ground truths."""
    print(f"Loading validation images from: {images_dir}")
    image_paths = []
    gts = {}

    for ext in [".jpg", ".jpeg", ".png"]:
        image_paths.extend(sorted(images_dir.glob(f"*{ext}")))

    count = 0
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with Image.open(img_path) as img:
                w, h = img.size
            boxes, labels = yolo_txt_to_boxes_labels(label_path, w, h)
            gts[str(img_path)] = {"boxes": boxes, "labels": labels}
            count += 1
            if count >= num_images:
                break

    print(f"  Loaded {len(gts)} images with ground truth")
    return list(gts.keys()), gts


def run_predict(model: YOLO, image_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Run inference and return predictions."""
    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        device=device,
        conf=CONF_THRESHOLD,
        verbose=False,
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        }

    boxes = r.boxes.xyxy.detach().cpu().to(torch.float32)
    scores = r.boxes.conf.detach().cpu().to(torch.float32)
    labels = r.boxes.cls.detach().cpu().to(torch.int64)
    # Map native classes to 0,1,2
    mapped_labels = []
    mapped_boxes = []
    mapped_scores = []
    for i, lbl in enumerate(labels.tolist()):
        if lbl in [0, 1, 2]:
            mapped_labels.append(lbl)
            mapped_boxes.append(boxes[i])
            mapped_scores.append(scores[i])
    if mapped_boxes:
        return {
            "boxes": torch.stack(mapped_boxes),
            "scores": torch.tensor(mapped_scores),
            "labels": torch.tensor(mapped_labels, dtype=torch.int64),
        }
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
    }


def benchmark_format(
    model: YOLO,
    format_name: str,
    model_file: Path,
    image_paths: list[Path],
    gts: dict[str, dict[str, torch.Tensor]],
    device: str,
    size_mb: float,
) -> BenchmarkResult:
    """Benchmark a single model format."""
    print(f"\nBenchmarking {format_name}...")
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5])
    all_preds = []
    all_gts = []
    latencies = []

    # Warmup
    print(f"  Warmup ({WARMUP} iterations)...")
    for _ in range(WARMUP):
        for img_path in image_paths:
            _ = run_predict(model, img_path, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    # Benchmark
    print(f"  Running {REPEATS} iterations...")
    for rep in range(REPEATS):
        for img_path in tqdm(image_paths, desc=f"  Iter {rep + 1}/{REPEATS}", leave=False):
            start = time.perf_counter()
            pred = run_predict(model, img_path, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

            gt = gts[str(img_path)]
            all_preds.append(pred)
            all_gts.append(gt)
            metric.update([pred], [gt])

    # Compute metrics
    maps = metric.compute()
    map50 = float(maps.get("map_50", torch.tensor(0.0)).item())
    map5095 = float(maps.get("map", torch.tensor(0.0)).item())

    precision, recall = compute_precision_recall(all_preds, all_gts)

    # Per-class mAP50
    per_class_map = {}
    for i, cls_name in enumerate(CUSTOM_CLASSES):
        key = f"map_50_per_class/{i}"
        per_class_map[cls_name] = float(maps[key].item()) if key in maps else 0.0

    latency_ms = np.mean(latencies) if latencies else float("nan")
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    print(f"  ✓ mAP@50: {map50:.4f}, mAP@50-95: {map5095:.4f}")
    print(f"  ✓ Latency: {latency_ms:.2f}ms, FPS: {fps:.2f}")

    result = BenchmarkResult(
        format_name=format_name,
        model_path=model_file,
        model_size_mb=size_mb,
        map50=round(map50, 4),
        map5095=round(map5095, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        latency_ms=round(latency_ms, 2),
        fps=round(fps, 2),
        per_class=per_class_map,
    )
    return result  # noqa: RET504


def get_model_size_mb(model: YOLO) -> float:
    """Get model file size in MB."""
    try:
        path = Path(model.ckpt_path)
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
    except Exception:  # noqa: S110
        pass
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model.model.save(tmp.name)
            size = Path(tmp.name).stat().st_size / (1024 * 1024)
            os.remove(tmp.name)
            return size
    except Exception:
        return 0.0


def create_visualizations(results: list[BenchmarkResult], output_dir: Path, device: str) -> None:
    """Create comparison plots."""
    print("\nCreating visualizations...")
    ensure_dir(output_dir)

    valid = [r for r in results if r.map50 > 0]
    if len(valid) < 2:
        print("  Not enough successful benchmarks for comparison")
        return

    formats = [r.format_name for r in valid]
    map50s = [r.map50 for r in valid]
    fps_list = [r.fps for r in valid]
    sizes = [r.model_size_mb for r in valid]

    # Accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(formats, map50s, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    plt.ylabel("mAP@50")
    plt.title("Accuracy Comparison")
    plt.ylim(0, 1.05)
    for bar, val in zip(bars, map50s):
        plt.annotate(
            f"{val:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()
    print("  ✓ accuracy_comparison.png")

    # Speed
    plt.figure(figsize=(10, 6))
    bars = plt.bar(formats, fps_list, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    plt.ylabel("FPS")
    plt.title("Speed Comparison")
    plt.ylim(0, max(fps_list) * 1.2)
    for bar, val in zip(bars, fps_list):
        plt.annotate(
            f"{val:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "speed_comparison.png", dpi=150)
    plt.close()
    print("  ✓ speed_comparison.png")

    # Size
    plt.figure(figsize=(10, 6))
    bars = plt.bar(formats, sizes, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    plt.ylabel("Size (MB)")
    plt.title("Model Size Comparison")
    plt.ylim(0, max(sizes) * 1.2)
    for bar, val in zip(bars, sizes):
        plt.annotate(
            f"{val:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "size_comparison.png", dpi=150)
    plt.close()
    print("  ✓ size_comparison.png")

    # Speed vs Accuracy
    plt.figure(figsize=(10, 6))
    for fmt, acc, spd in zip(formats, map50s, fps_list):
        plt.scatter(spd, acc, s=200, edgecolors="black", linewidth=2)
        plt.annotate(fmt.upper(), (spd, acc), xytext=(10, 5), textcoords="offset points")
    plt.xlabel("FPS")
    plt.ylabel("mAP@50")
    plt.title("Speed vs Accuracy Trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_accuracy.png", dpi=150)
    plt.close()
    print("  ✓ speed_vs_accuracy.png")

    # Sample predictions
    create_sample_predictions(valid, output_dir, device)


def create_sample_predictions(results: list[BenchmarkResult], output_dir: Path, device: str) -> None:
    """Show predictions from each format on sample images."""
    print("\nCreating sample predictions...")
    samples = list(VAL_IMAGES_DIR.glob("*.jpg"))[:4]
    if not samples:
        return

    n_fmt = len(results)
    n_img = len(samples)
    fig, axes = plt.subplots(n_fmt, n_img, figsize=(4 * n_img, 3 * n_fmt))
    if n_fmt == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        model = YOLO(str(res.model_path))
        for col, img_path in enumerate(samples):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = model.predict(
                source=str(img_path),
                imgsz=IMAGE_SIZE,
                device=device,
                conf=CONF_THRESHOLD,
                verbose=False,
            )[0]
            if out.boxes is not None:
                for box in out.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    if cls_id in [0, 1, 2]:
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            img,
                            f"{CUSTOM_CLASSES[cls_id]}:{box.conf[0].item():.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(res.format_name.upper(), fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ sample_predictions.png")


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Save results to CSV and JSON."""
    ensure_dir(output_dir)

    rows = []
    for r in results:
        rows.append(
            {
                "Format": r.format_name,
                "Model Path": str(r.model_path),
                "Model Size (MB)": round(r.model_size_mb, 2),
                "mAP@50": r.map50,
                "mAP@50-95": r.map5095,
                "Precision": r.precision,
                "Recall": r.recall,
                "Latency (ms)": r.latency_ms,
                "FPS": r.fps,
                "Error": r.error,
            }
        )

    df = pd.DataFrame(rows)
    valid = df[df["mAP@50"] > 0].sort_values(["mAP@50", "Latency (ms)"], ascending=[False, True])
    df = pd.concat([valid, df[df["mAP@50"] == 0]])

    csv_path = output_dir / "format_comparison_results.csv"
    json_path = output_dir / "format_comparison_results.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(
            {
                "results": rows,
                "settings": {
                    "image_size": IMAGE_SIZE,
                    "num_images": NUM_IMAGES,
                    "warmup": WARMUP,
                    "repeats": REPEATS,
                },
            },
            f,
            indent=2,
        )

    print(f"\n✓ Saved CSV: {csv_path}")
    print(f"✓ Saved JSON: {json_path}")
    print("\n" + "=" * 60)
    print("FORMAT COMPARISON SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda:0":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    ensure_dir(EXPORT_DIR)
    ensure_dir(BENCHMARK_DIR)

    # Find exported models and filter by availability
    all_model_files = (
        list(EXPORT_DIR.glob("*.pt"))
        + list(EXPORT_DIR.glob("*.onnx"))
        + list(EXPORT_DIR.glob("*.torchscript"))
        + list(EXPORT_DIR.glob("*.engine"))
    )
    model_files = []

    for mf in all_model_files:
        fmt = mf.suffix[1:].lower()
        if fmt == "onnx":
            try:
                import onnxruntime as ort

                providers = ort.get_available_providers()
                # Use ONNX on CPU if it works, or GPU if CUDA provider available
                # ONNX Runtime CPU can still run, just slower
                model_files.append(mf)
            except ImportError:
                print(f"Skipping {mf.name} - onnxruntime not available")
                continue
        elif fmt == "tensorrt":
            try:
                import tensorrt as trt  # noqa: F401

                model_files.append(mf)
            except ImportError:
                print(f"Skipping {mf.name} - TensorRT not available")
                continue
        else:
            model_files.append(mf)

    if not model_files:
        raise FileNotFoundError(f"No exportable models found in {EXPORT_DIR}")

    print(f"\nFound {len(model_files)} exportable models:")
    for mf in model_files:
        print(f"  - {mf.name}")

    # Load validation data
    image_paths, gts = load_validation_set(VAL_IMAGES_DIR, VAL_LABELS_DIR, NUM_IMAGES)

    # Benchmark each format
    results = []
    for model_file in model_files:
        fmt_name = model_file.suffix[1:]  # remove dot

        # Skip ONNX if onnxruntime-gpu not available (will fall back to very slow CPU)
        if fmt_name == "onnx":
            try:
                import onnxruntime as ort

                # Check if CUDA Execution Provider is available
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" not in providers and "CUDA" not in device:
                    # CPU is fine if device is CPU
                    pass
                elif "CUDAExecutionProvider" not in providers and "CUDA" in device:
                    print(f"  Skipping ONNX - CUDAExecutionProvider not available (using {providers})")
                    continue
            except ImportError:
                print("  Skipping ONNX - onnxruntime not properly installed")
                continue

        try:
            model = YOLO(str(model_file))
            size_mb = model_file.stat().st_size / (1024 * 1024)

            # Run benchmark
            res = benchmark_format(model, fmt_name, model_file, image_paths, gts, device, size_mb)
            results.append(res)
        except Exception as e:
            print(f"  ✗ Failed to benchmark {model_file.name}: {e}")
            results.append(
                BenchmarkResult(
                    format_name=fmt_name,
                    model_path=model_file,
                    model_size_mb=model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0.0,
                    map50=0.0,
                    map5095=0.0,
                    precision=0.0,
                    recall=0.0,
                    latency_ms=float("nan"),
                    fps=0.0,
                    per_class={},
                    error=str(e),
                )
            )

    # Visualize and save
    create_visualizations(results, BENCHMARK_DIR, device)
    save_results(results, BENCHMARK_DIR)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)
    print(f"Models: {EXPORT_DIR}")
    print(f"Results: {BENCHMARK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
