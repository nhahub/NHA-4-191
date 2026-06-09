#!/usr/bin/env python3
"""Export custom YOLO model to multiple formats and benchmark them.

Export formats:
- PyTorch (.pt) - original
- ONNX (.onnx) - with dynamic axes, FP16, simplification
- TensorRT (.engine) - if available
- TorchScript (.torchscript) - if available

Benchmarks:
- mAP50, mAP50-95 on KITTI validation set
- Latency (ms), FPS
- Model size (MB)

Outputs:
- models/exports/ - exported model files
- experiments/format_comparison/ - benchmark results and visualizations
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import ensure_dir

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models/checkpoints/best-3classes-exp34332.pt"
VAL_IMAGES_DIR = PROJECT_ROOT / "data/processed/kitti/images/val"
VAL_LABELS_DIR = PROJECT_ROOT / "data/processed/kitti/labels/val"
EXPORT_DIR = PROJECT_ROOT / "models/exports"
BENCHMARK_DIR = PROJECT_ROOT / "experiments/format_comparison"

# Classes
CUSTOM_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]
NATIVE_CLASS_MAP = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}
NATIVE_COCO_MAP = {
    "Vehicle": [2, 7, 5, 3],
    "Pedestrian": [0],
    "Cyclist": [1],
}

# Benchmark settings
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
NUM_BENCHMARK_IMAGES = 50  # At least 50 as requested
WARMUP_ITERATIONS = 2
BENCHMARK_REPEATS = 10


@dataclass
class FormatResult:
    format_name: str
    model_path: Path
    model_size_mb: float
    map50: float
    map5095: float
    latency_ms: float
    fps: float
    export_success: bool
    benchmark_success: bool
    error_msg: str = ""


def yolo_txt_to_boxes_labels(label_path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def load_validation_set(
    images_dir: Path, labels_dir: Path, num_images: int = 50
) -> tuple[list[Path], dict[str, dict[str, torch.Tensor]]]:
    """Load validation images and ground truth annotations."""
    print(f"Loading validation images from: {images_dir}")
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = []
    for ext in image_extensions:
        all_images.extend(sorted(images_dir.glob(f"*{ext}")))

    if not all_images:
        raise RuntimeError(f"No images found in {images_dir}")

    # Take first N images with labels
    image_paths = []
    gts = {}
    for img_path in all_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            image_paths.append(img_path)
            with Image.open(img_path) as img:
                w, h = img.size
            boxes, labels = yolo_txt_to_boxes_labels(label_path, w, h)
            gts[str(img_path)] = {"boxes": boxes, "labels": labels}
            if len(image_paths) >= num_images:
                break

    print(f"  Loaded {len(image_paths)} images with ground truth")
    return image_paths, gts


def check_dependencies() -> dict[str, bool]:
    """Check which export formats are available."""
    deps = {
        "onnx": False,
        "tensorrt": False,
        "torchscript": True,  # Built into PyTorch
        "onnxsim": False,
    }

    try:
        import onnx  # noqa: F401

        deps["onnx"] = True
    except ImportError:
        pass

    try:
        import tensorrt  # noqa: F401

        deps["tensorrt"] = True
    except ImportError:
        pass

    try:
        import onnxsim  # noqa: F401

        deps["onnxsim"] = True
    except ImportError:
        pass

    return deps


def _format_result(
    format_name: str,
    path: Path,
    size_mb: float = 0.0,
    success: bool = False,
    error: str = "",
) -> FormatResult:
    return FormatResult(
        format_name=format_name,
        model_path=path,
        model_size_mb=size_mb,
        map50=0.0,
        map5095=0.0,
        latency_ms=0.0,
        fps=0.0,
        export_success=success,
        benchmark_success=False,
        error_msg=error,
    )


def _copy_pytorch_model() -> FormatResult:
    original_path = EXPORT_DIR / "best-3classes-exp34332-original.pt"
    shutil.copy2(MODEL_PATH, original_path)
    size_mb = original_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to {original_path} ({size_mb:.2f} MB)")
    return _format_result("pytorch", original_path, size_mb, success=True)


def _export_onnx(model: YOLO, device: str, deps: dict) -> FormatResult:
    onnx_path = model.export(
        format="onnx",
        dynamic=True,
        half=True,
        simplify=deps["onnxsim"],
        imgsz=IMAGE_SIZE,
        device=device,
    )
    onnx_path = Path(onnx_path)
    if onnx_path.parent != EXPORT_DIR:
        dest = EXPORT_DIR / onnx_path.name
        shutil.move(str(onnx_path), dest)
        onnx_path = dest
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to {onnx_path} ({size_mb:.2f} MB)")
    return _format_result("onnx", onnx_path, size_mb, success=True)


def _export_tensorrt(model: YOLO, device: str) -> FormatResult:
    engine_path = model.export(
        format="engine",
        half=True,
        imgsz=IMAGE_SIZE,
        device=device,
        workspace=4,
    )
    engine_path = Path(engine_path)
    if engine_path.parent != EXPORT_DIR:
        dest = EXPORT_DIR / engine_path.name
        shutil.move(str(engine_path), dest)
        engine_path = dest
    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to {engine_path} ({size_mb:.2f} MB)")
    return _format_result("tensorrt", engine_path, size_mb, success=True)


def _export_torchscript(model: YOLO, device: str) -> FormatResult:
    torchscript_path = model.export(
        format="torchscript",
        imgsz=IMAGE_SIZE,
        device=device,
    )
    torchscript_path = Path(torchscript_path)
    if torchscript_path.parent != EXPORT_DIR:
        dest = EXPORT_DIR / torchscript_path.name
        shutil.move(str(torchscript_path), dest)
        torchscript_path = dest
    size_mb = torchscript_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to {torchscript_path} ({size_mb:.2f} MB)")
    return _format_result("torchscript", torchscript_path, size_mb, success=True)


_FORMAT_EXPORTERS = [
    ("pytorch", _copy_pytorch_model, None),
    ("onnx", _export_onnx, "onnx"),
    ("tensorrt", _export_tensorrt, "tensorrt"),
    ("torchscript", _export_torchscript, None),
]


def export_models(model: YOLO, device: str) -> list[FormatResult]:
    ensure_dir(EXPORT_DIR)
    results: list[FormatResult] = []

    print("\n" + "=" * 60)
    print("EXPORTING MODELS")
    print("=" * 60)

    deps = check_dependencies()

    for idx, (fmt_name, export_fn, dep_key) in enumerate(_FORMAT_EXPORTERS, 1):
        print(f"\n[{idx}/{len(_FORMAT_EXPORTERS)}] {fmt_name.title()}")

        if dep_key is not None and not deps.get(dep_key):
            print(f"  ⚠ Skipped - {fmt_name} dependencies not installed")
            results.append(_format_result(fmt_name, Path(""), error=f"{fmt_name} dependencies not installed"))
            continue

        try:
            if fmt_name == "pytorch":
                results.append(export_fn())
            else:
                results.append(export_fn(model, device, deps) if dep_key == "onnx" else export_fn(model, device))
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append(_format_result(fmt_name, Path(""), error=str(e)))

    return results


def load_model_for_format(format_name: str, model_path: Path, device: str) -> YOLO | None:
    """Load model in specified format for inference."""
    if format_name == "pytorch":
        return YOLO(str(model_path))
    if format_name in ["onnx", "torchscript"]:
        # For exported formats, we use ONNX Runtime or back to PyTorch via YOLO
        # ONNX Runtime is used automatically by YOLO when loading .onnx
        # For TorchScript, YOLO wraps it
        return YOLO(str(model_path))
    if format_name == "tensorrt":
        # TensorRT engines are loaded via YOLO as well
        return YOLO(str(model_path))
    return None


def benchmark_format(
    format_name: str,
    model_path: Path,
    image_paths: list[Path],
    gts: dict[str, dict[str, torch.Tensor]],
    device: str,
) -> tuple[float, float, float, float]:
    """Benchmark a single model format."""
    print(f"\nBenchmarking {format_name}...")

    # Load model
    try:
        model = load_model_for_format(format_name, model_path, device)
        if model is None:
            raise RuntimeError(f"Failed to load {format_name} model")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        raise

    # Setup metric
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5])
    latencies = []

    # Warmup
    print(f"  Running {WARMUP_ITERATIONS} warmup iterations...")
    for _ in range(WARMUP_ITERATIONS):
        for img_path in image_paths:
            _ = model.predict(
                source=str(img_path),
                imgsz=IMAGE_SIZE,
                device=device,
                conf=CONF_THRESHOLD,
                verbose=False,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    # Benchmark
    print(f"  Running {BENCHMARK_REPEATS} benchmark iterations...")
    for rep in range(BENCHMARK_REPEATS):
        for img_path in tqdm(image_paths, desc=f"  Iter {rep + 1}/{BENCHMARK_REPEATS}", leave=False):
            start = time.perf_counter()
            results = model.predict(
                source=str(img_path),
                imgsz=IMAGE_SIZE,
                device=device,
                conf=CONF_THRESHOLD,
                verbose=False,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            # Get predictions
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                pred_boxes = torch.empty((0, 4), dtype=torch.float32)
                pred_scores = torch.empty((0,), dtype=torch.float32)
                pred_labels = torch.empty((0,), dtype=torch.int64)
            else:
                pred_boxes = result.boxes.xyxy.detach().cpu().to(torch.float32)
                pred_scores = result.boxes.conf.detach().cpu().to(torch.float32)
                pred_labels = result.boxes.cls.detach().cpu().to(torch.int64)

            gt = gts[str(img_path)]
            metric.update(
                [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
                [{"boxes": gt["boxes"], "labels": gt["labels"]}],
            )

    # Compute metrics
    maps = metric.compute()
    map50 = float(maps.get("map_50", torch.tensor(0.0)).item())
    map5095 = float(maps.get("map", torch.tensor(0.0)).item())

    latency_ms = float(np.mean(latencies)) if latencies else float("nan")
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    print(f"  ✓ mAP@50: {map50:.4f}, mAP@50-95: {map5095:.4f}")
    print(f"  ✓ Latency: {latency_ms:.2f}ms, FPS: {fps:.2f}")

    return map50, map5095, latency_ms, fps


def benchmark_all_formats(
    results: list[FormatResult],
    image_paths: list[Path],
    gts: dict[str, dict[str, torch.Tensor]],
    device: str,
) -> list[FormatResult]:
    """Benchmark all successfully exported formats."""
    updated_results = []

    print("\n" + "=" * 60)
    print("BENCHMARKING ALL FORMATS")
    print("=" * 60)

    for res in results:
        if res.export_success:
            try:
                map50, map5095, latency_ms, fps = benchmark_format(
                    res.format_name,
                    res.model_path,
                    image_paths,
                    gts,
                    device,
                )
                res.map50 = map50
                res.map5095 = map5095
                res.latency_ms = latency_ms
                res.fps = fps
                res.benchmark_success = True
            except Exception as e:
                print(f"  ✗ Benchmark failed for {res.format_name}: {e}")
                res.benchmark_success = False
                res.error_msg = str(e)
        else:
            print(f"  ⚠ Skipping {res.format_name} - export failed")

        updated_results.append(res)

    return updated_results


def create_visualizations(results: list[FormatResult], output_dir: Path, device: str = "cpu") -> None:
    """Create comparison visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    ensure_dir(output_dir)

    # Filter successful benchmarks
    valid_results = [r for r in results if r.benchmark_success]
    if not valid_results:
        print("  No successful benchmarks to visualize")
        return

    formats = [r.format_name for r in valid_results]
    map50s = [r.map50 for r in valid_results]
    latencies = [r.latency_ms for r in valid_results]
    fps_list = [r.fps for r in valid_results]
    fps_list = [r.fps for r in valid_results]
    sizes = [r.model_size_mb for r in valid_results]

    # 1. Accuracy comparison (mAP50)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(formats, map50s, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    plt.ylabel("mAP@50")
    plt.title("Accuracy Comparison Across Formats")
    plt.ylim(0, max(map50s) * 1.2)
    for bar, val in zip(bars, map50s):
        plt.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()
    print("  ✓ Saved: accuracy_comparison.png")

    # 2. Speed comparison (FPS)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(formats, fps_list, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    plt.ylabel("FPS")
    plt.title("Speed Comparison Across Formats")
    plt.ylim(0, max(fps_list) * 1.2)
    for bar, val in zip(bars, fps_list):
        plt.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "speed_comparison.png", dpi=150)
    plt.close()
    print("  ✓ Saved: speed_comparison.png")

    # 3. Model size comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(formats, sizes, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    plt.ylabel("Size (MB)")
    plt.title("Model Size Comparison")
    plt.ylim(0, max(sizes) * 1.2)
    for bar, val in zip(bars, sizes):
        plt.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "size_comparison.png", dpi=150)
    plt.close()
    print("  ✓ Saved: size_comparison.png")

    # 4. Speed vs Accuracy tradeoff
    plt.figure(figsize=(10, 6))
    for fmt, acc, spd in zip(formats, map50s, fps_list):
        plt.scatter(spd, acc, s=200, edgecolors="black", linewidth=2)
        plt.annotate(
            fmt.upper(),
            (spd, acc),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=10,
        )
    plt.xlabel("FPS (higher is better)")
    plt.ylabel("mAP@50 (higher is better)")
    plt.title("Speed vs Accuracy Trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_accuracy.png", dpi=150)
    plt.close()
    print("  ✓ Saved: speed_vs_accuracy.png")

    # 5. Latency comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(formats, latencies, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency Comparison")
    plt.ylim(0, max(latencies) * 1.2)
    for bar, val in zip(bars, latencies):
        plt.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=150)
    plt.close()
    print("  ✓ Saved: latency_comparison.png")

    # 6. Sample predictions from each format (if all formats work)
    if len(valid_results) >= 2:
        create_sample_predictions(valid_results, output_dir, device)


def create_sample_predictions(
    results: list[FormatResult],
    output_dir: Path,
    device: str,
) -> None:
    """Create visual comparison of sample predictions from each format."""
    print("\nCreating sample predictions comparison...")

    # Load a sample image
    sample_images = list(VAL_IMAGES_DIR.glob("*.jpg"))[:6]
    if not sample_images:
        return

    n_models = len(results)
    n_images = len(sample_images)

    fig, axes = plt.subplots(n_models, n_images, figsize=(4 * n_images, 3 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        model = YOLO(str(res.model_path))
        for col, img_path in enumerate(sample_images):
            results = model.predict(
                source=str(img_path),
                imgsz=IMAGE_SIZE,
                device=device,
                conf=CONF_THRESHOLD,
                verbose=False,
            )[0]

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    # Only show our 3 classes
                    if cls_id in [0, 1, 2]:
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{CUSTOM_CLASSES[cls_id]}:{conf:.2f}"
                        cv2.putText(
                            img,
                            label,
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
    plt.savefig(output_dir / "sample_predictions_all_formats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: sample_predictions_all_formats.png")


def save_results(
    results: list[FormatResult],
    output_dir: Path,
) -> None:
    """Save benchmark results to CSV and JSON."""
    ensure_dir(output_dir)

    # Create DataFrame
    data = []
    for r in results:
        data.append(
            {
                "Format": r.format_name,
                "Model Path": str(r.model_path) if r.model_path else "",
                "Model Size (MB)": round(r.model_size_mb, 2),
                "Export Success": r.export_success,
                "Benchmark Success": r.benchmark_success,
                "mAP@50": round(r.map50, 4) if r.benchmark_success else None,
                "mAP@50-95": round(r.map5095, 4) if r.benchmark_success else None,
                "Latency (ms)": round(r.latency_ms, 2) if r.benchmark_success else None,
                "FPS": round(r.fps, 2) if r.benchmark_success else None,
                "Error": r.error_msg,
            }
        )

    df = pd.DataFrame(data)

    # Sort by mAP50 (desc) then latency (asc)
    df_valid = df[df["Benchmark Success"]].copy()
    if not df_valid.empty:
        df_valid = df_valid.sort_values(["mAP@50", "Latency (ms)"], ascending=[False, True])
        df = pd.concat([df_valid, df[~df["Benchmark Success"]]])  # noqa: E712

    # Save CSV
    csv_path = output_dir / "format_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    # Save JSON
    json_data = {
        "model_path": str(MODEL_PATH),
        "image_size": IMAGE_SIZE,
        "num_benchmark_images": NUM_BENCHMARK_IMAGES,
        "warmup_iterations": WARMUP_ITERATIONS,
        "benchmark_repeats": BENCHMARK_REPEATS,
        "results": [
            {
                "format": r.format_name,
                "model_path": str(r.model_path) if r.model_path else None,
                "model_size_mb": round(r.model_size_mb, 2),
                "export_success": r.export_success,
                "benchmark_success": r.benchmark_success,
                "mAP50": round(r.map50, 4) if r.benchmark_success else None,
                "mAP5095": round(r.map5095, 4) if r.benchmark_success else None,
                "latency_ms": round(r.latency_ms, 2) if r.benchmark_success else None,
                "fps": round(r.fps, 2) if r.benchmark_success else None,
                "error": r.error_msg,
            }
            for r in results
        ],
    }

    json_path = output_dir / "format_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FORMAT COMPARISON SUMMARY")
    print("=" * 60)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))
    print("=" * 60)


def main():
    """Main pipeline."""
    # Check CUDA
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda:0":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Check model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Create directories
    ensure_dir(EXPORT_DIR)
    ensure_dir(BENCHMARK_DIR)

    # Load base model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"  ✓ Model loaded ({model.model.yaml.get('nc', 3)} classes)")

    # Step 1: Export all formats
    export_results = export_models(model, device)

    # Step 2: Load validation data
    image_paths, gts = load_validation_set(
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
        num_images=NUM_BENCHMARK_IMAGES,
    )

    # Step 3: Benchmark all formats
    benchmark_results = benchmark_all_formats(export_results, image_paths, gts, device)

    # Step 4: Create visualizations
    create_visualizations(benchmark_results, BENCHMARK_DIR, device)

    # Step 5: Save results
    save_results(benchmark_results, BENCHMARK_DIR)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Exported models: {EXPORT_DIR}")
    print(f"Benchmark results: {BENCHMARK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        import cv2  # Check OpenCV availability
    except ImportError:
        print("Warning: OpenCV not available - sample predictions may fail")
        # Define dummy cv2 functions
        import types

        cv2 = types.SimpleNamespace()
        cv2.imread = lambda x: None
        cv2.cvtColor = lambda x, y: None
        cv2.rectangle = lambda *args, **kwargs: None
        cv2.putText = lambda *args, **kwargs: None

    main()
