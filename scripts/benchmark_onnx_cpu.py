#!/usr/bin/env python3
"""Benchmark ONNX Runtime vs PyTorch CPU inference speed.

Measures FPS and latency for both backends across single/batch inference.

Usage:
    python scripts/benchmark_onnx_cpu.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --onnx models/exports/best.onnx

    python scripts/benchmark_onnx_cpu.py \\
        --weights models/checkpoints/best-3classes-exp34332.pt \\
        --num-images 200 --device cpu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime vs PyTorch CPU inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to PyTorch model weights")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model (auto-derived if not set)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--num-images", type=int, default=200, help="Number of images for benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device for PyTorch inference (cpu or cuda:0)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    return parser.parse_args()


def benchmark_pytorch(weights_path: str, imgsz: int, num_images: int, warmup: int, device: str) -> dict:
    from ultralytics import YOLO

    logger.info(f"Loading PyTorch model: {weights_path}")
    model = YOLO(weights_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    logger.info(f"Warming up ({warmup} iters)...")
    for _ in range(warmup):
        model.predict(dummy, verbose=False, device=device)

    logger.info(f"Benchmarking ({num_images} iters)...")
    times = []
    for _ in range(num_images):
        start = time.perf_counter()
        model.predict(dummy, verbose=False, device=device)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))

    return {
        "backend": f"PyTorch ({device})",
        "avg_ms": round(avg_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
    }


def benchmark_onnx(onnx_path: str, imgsz: int, num_images: int, warmup: int) -> dict:
    from src.deployment.onnx_runner import ONNXRunner

    logger.info(f"Loading ONNX model: {onnx_path}")
    runner = ONNXRunner(onnx_path, imgsz=imgsz)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    logger.info(f"Warming up ({warmup} iters)...")
    for _ in range(warmup):
        runner.predict(dummy)

    logger.info(f"Benchmarking ({num_images} iters)...")
    times = []
    for _ in range(num_images):
        start = time.perf_counter()
        runner.predict(dummy)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))

    provider = runner.providers[0] if runner.providers else "unknown"

    return {
        "backend": f"ONNX Runtime ({provider})",
        "avg_ms": round(avg_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
    }


def resolve_onnx_path(weights_path: str) -> str | None:
    pt_path = Path(weights_path)
    candidates = [
        pt_path.with_suffix(".onnx"),
        pt_path.parent / pt_path.stem.replace(".pt", "") / "best.onnx",
        Path("models/exports") / f"{pt_path.stem}.onnx",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main() -> int:
    args = parse_args()
    weights_path = str(Path(args.weights).resolve())

    if not Path(weights_path).exists():
        logger.error(f"Weights not found: {weights_path}")
        return 1

    onnx_path = args.onnx or resolve_onnx_path(weights_path)
    if onnx_path and not Path(onnx_path).exists():
        logger.warning(f"ONNX model not found at {onnx_path}, skipping ONNX benchmark")
        onnx_path = None

    results = {"config": {"imgsz": args.imgsz, "num_images": args.num_images, "warmup": args.warmup}, "benchmarks": []}

    logger.info(f"\n{'='*60}")
    logger.info("CPU INFERENCE BENCHMARK")
    logger.info(f"{'='*60}")
    logger.info(f"Image size: {args.imgsz}×{args.imgsz}")
    logger.info(f"Benchmark images: {args.num_images}")

    pt_result = benchmark_pytorch(weights_path, args.imgsz, args.num_images, args.warmup, args.device)
    results["benchmarks"].append(pt_result)
    logger.info(f"  {pt_result['backend']}: {pt_result['avg_ms']} ms avg, {pt_result['fps']} FPS")

    if onnx_path:
        onnx_result = benchmark_onnx(onnx_path, args.imgsz, args.num_images, args.warmup)
        results["benchmarks"].append(onnx_result)
        logger.info(f"  {onnx_result['backend']}: {onnx_result['avg_ms']} ms avg, {onnx_result['fps']} FPS")

        if pt_result["avg_ms"] > 0:
            speedup = pt_result["avg_ms"] / onnx_result["avg_ms"]
            results["speedup_vs_pytorch"] = round(speedup, 2)
            logger.info(f"  Speedup: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Backend':<35} {'Avg (ms)':<10} {'FPS':<10} {'P50 (ms)':<10} {'P95 (ms)':<10}")
    print("-" * 75)
    for b in results["benchmarks"]:
        print(f"{b['backend']:<35} {b['avg_ms']:<10.2f} {b['fps']:<10.1f} {b['p50_ms']:<10.2f} {b['p95_ms']:<10.2f}")
    if "speedup_vs_pytorch" in results:
        print(f"\nONNX Runtime speedup: {results['speedup_vs_pytorch']}x")
    print("=" * 60)

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
