#!/usr/bin/env python3
"""INT8 quantization pipeline for Road-Sense models.

Exports trained PyTorch models to TFLite INT8 with calibration,
then benchmarks accuracy vs speed against FP32 and FP16 baselines.

Usage:
    python scripts/quantize_model.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt

    # With accuracy evaluation
    python scripts/quantize_model.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --data data/processed/kitti/data.yaml

    # Full pipeline with benchmark
    python scripts/quantize_model.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --data data/processed/kitti/data.yaml \\
        --benchmark
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("quantize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT8 Quantization Pipeline for Road-Sense")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained PyTorch model weights")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML for calibration and evaluation")
    parser.add_argument("--output-dir", type=str, default="models/exports/quantized", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark after export")
    parser.add_argument("--benchmark-images", type=int, default=100, help="Number of images for benchmark")
    parser.add_argument("--device", type=str, default="0", help="Device for FP32 baseline evaluation")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    return parser.parse_args()


def export_tflite_formats(model, output_dir: Path, imgsz: int) -> dict:
    logger.info(f"\n--- Exporting TFLite formats to: {output_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    logger.info("Exporting TFLite FP16...")
    try:
        fp16_path = model.export(format="tflite", imgsz=imgsz, int8=False, ncalib=0)
        results["tflite_fp16"] = {"path": str(Path(fp16_path).resolve()), "status": "exported"}
        logger.info(f"  TFLite FP16 saved to: {fp16_path}")
    except Exception as e:
        logger.error(f"  TFLite FP16 export failed: {e}")
        results["tflite_fp16"] = {"status": "failed", "error": str(e)}

    logger.info("Exporting TFLite INT8 (quantized)...")
    try:
        int8_path = model.export(format="tflite", imgsz=imgsz, int8=True)
        results["tflite_int8"] = {"path": str(Path(int8_path).resolve()), "status": "exported"}
        logger.info(f"  TFLite INT8 saved to: {int8_path}")
    except Exception as e:
        logger.error(f"  TFLite INT8 export failed: {e}")
        results["tflite_int8"] = {"status": "failed", "error": str(e)}

    return results


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).resolve()

    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        return 1

    from ultralytics import YOLO

    logger.info(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    output_dir = Path(args.output_dir).resolve()
    results = {
        "source_weights": str(weights_path),
        "source_size_mb": round(weights_path.stat().st_size / (1024 * 1024), 2),
        "tflite_results": {},
    }

    def extract_metrics(metrics) -> dict:
        result = {}
        try:
            if hasattr(metrics, "box"):
                if hasattr(metrics.box, "map50") and metrics.box.map50 is not None:
                    result["mAP50"] = float(metrics.box.map50)
                if hasattr(metrics.box, "map") and metrics.box.map is not None:
                    result["mAP50-95"] = float(metrics.box.map)
                if hasattr(metrics.box, "mp") and metrics.box.mp is not None:
                    result["precision"] = float(metrics.box.mp)
                if hasattr(metrics.box, "mr") and metrics.box.mr is not None:
                    result["recall"] = float(metrics.box.mr)
        except Exception as e:
            logger.warning(f"Could not extract some metrics: {e}")
        return result

    if args.data:
        data_path = Path(args.data)
        if data_path.exists():
            logger.info("\n--- Evaluating FP32 baseline (mAP) ---")
            try:
                fp32_metrics = model.val(data=str(data_path))
                results["fp32_baseline"] = extract_metrics(fp32_metrics)
                logger.info(f"  FP32 mAP@50: {results['fp32_baseline'].get('mAP50', 0):.4f}")
                logger.info(f"  FP32 mAP@50:95: {results['fp32_baseline'].get('mAP50-95', 0):.4f}")
            except Exception as e:
                logger.warning(f"  FP32 evaluation failed: {e}")
        else:
            logger.warning(f"Data YAML not found: {data_path}")

    tflite_results = export_tflite_formats(model, output_dir, args.imgsz)
    results["tflite_results"] = tflite_results

    for fmt, info in tflite_results.items():
        if info.get("status") == "exported":
            fmt_path = Path(info["path"])
            results["tflite_results"][fmt]["size_mb"] = round(fmt_path.stat().st_size / (1024 * 1024), 2)

    if args.data and any(info.get("status") == "exported" for info in tflite_results.values()):
        logger.info("\n--- Evaluating TFLite format accuracy ---")
        for fmt, info in tflite_results.items():
            if info.get("status") != "exported":
                continue
            fmt_path = info["path"]
            logger.info(f"  Evaluating {fmt} at {fmt_path}...")
            try:
                tflite_model = YOLO(str(fmt_path))
                tflite_metrics = tflite_model.val(data=str(data_path), verbose=False)
                metrics = extract_metrics(tflite_metrics)
                results["tflite_results"][fmt]["metrics"] = metrics
                logger.info(f"    mAP@50: {metrics.get('mAP50', 0):.4f}, mAP@50:95: {metrics.get('mAP50-95', 0):.4f}")
            except Exception as e:
                logger.warning(f"  {fmt} evaluation failed: {e}")
                results["tflite_results"][fmt]["val_error"] = str(e)

    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"\nSource model: {results['source_weights']}")
    print(f"Source size: {results['source_size_mb']} MB")

    has_accuracy = "fp32_baseline" in results

    if has_accuracy:
        header = f"{'Format':<20} {'Size':<10} {'mAP@50':<10} {'mAP@50:95':<12} {'Drop (mAP50-95)':<16}"
        print(f"\n{header}")
        print("-" * 68)

        fp32_map = results["fp32_baseline"].get("mAP50-95", 0)
        print(f"{'FP32 (baseline)':<20} {results['source_size_mb']:<10.1f} "
              f"{results['fp32_baseline'].get('mAP50', 0):<10.4f} {fp32_map:<12.4f} {'—':<16}")

        for fmt, info in tflite_results.items():
            if info.get("status") != "exported":
                continue
            size = info.get("size_mb", 0)
            metrics = info.get("metrics", {})
            fmt_map = metrics.get("mAP50-95", 0)
            drop = fp32_map - fmt_map
            status = "✅" if drop <= 0.02 else "⚠️"
            print(f"{fmt:<20} {size:<10.1f} {metrics.get('mAP50', 0):<10.4f} {fmt_map:<12.4f} {drop:<14.4f} {status}")
    else:
        for fmt, info in tflite_results.items():
            status_icon = "✅" if info.get("status") == "exported" else "❌"
            size = info.get("size_mb", "N/A")
            size_str = f"{size} MB" if isinstance(size, (int, float)) else size
            print(f"{status_icon} {fmt:<15} → {size_str}")



    if args.benchmark:
        logger.info(f"\n--- Running speed benchmark ({args.benchmark_images} images) ---")
        dummy_image = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
        formats_to_bench = {"pytorch_fp32": model}

        for fmt, info in tflite_results.items():
            if info.get("status") == "exported":
                try:
                    formats_to_bench[fmt] = YOLO(str(info["path"]))
                except Exception as e:
                    logger.warning(f"  Could not load {fmt} for benchmark: {e}")

        print(f"\n{'Format':<25} {'Avg (ms)':<12} {'FPS':<10}")
        print("-" * 47)

        for name, bench_model in formats_to_bench.items():
            times = []
            warmup = 5
            total = args.benchmark_images

            for i in range(total + warmup):
                start = time.perf_counter()
                bench_model.predict(dummy_image, verbose=False, device=args.device if name == "pytorch_fp32" else "cpu")
                elapsed = (time.perf_counter() - start) * 1000
                if i >= warmup:
                    times.append(elapsed)

            avg_ms = float(np.mean(times))
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
            results.setdefault("benchmark", {})[name] = {"avg_ms": round(avg_ms, 2), "fps": round(fps, 2)}
            print(f"{name:<25} {avg_ms:<12.2f} {fps:<10.1f}")

    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
