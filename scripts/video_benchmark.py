#!/usr/bin/env python3
"""Benchmark real-time video processing performance.

Measures average FPS, frame drop rate, and memory trend over time.

Usage:
    python scripts/video_benchmark.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --source video.mp4

    python scripts/video_benchmark.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --source 0 --duration 60
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video processing benchmark")
    parser.add_argument("--weights", type=str, required=True, help="Model weights")
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--duration", type=int, default=0, help="Max seconds (0 = all frames)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.weights)
    process = psutil.Process(os.getpid())
    src = int(args.source) if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.source}")
        return 1

    total_frames = 0
    dropped_frames = 0
    inference_times = []
    ram_samples = []
    start_time = time.time()
    expected_frame_interval = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30)

    print(f"Benchmarking video: {src}")
    print(f"  Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"  Expected FPS: {1.0/expected_frame_interval:.1f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000
        inference_times.append(elapsed)

        if elapsed > expected_frame_interval * 1000 * 2:
            dropped_frames += 1

        total_frames += 1
        ram_samples.append(process.memory_info().rss / (1024 * 1024))

        elapsed_total = time.time() - start_time
        if args.duration and elapsed_total >= args.duration:
            break

    cap.release()

    avg_ms = float(np.mean(inference_times))
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    p95_ms = float(np.percentile(inference_times, 95))
    p99_ms = float(np.percentile(inference_times, 99))
    drop_rate = dropped_frames / total_frames if total_frames else 0
    avg_ram = float(np.mean(ram_samples))
    peak_ram = float(max(ram_samples))

    print("\n" + "=" * 60)
    print("VIDEO BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total frames:        {total_frames}")
    print(f"  Avg inference:       {avg_ms:.1f} ms")
    print(f"  Effective FPS:       {avg_fps:.1f}")
    print(f"  p95 latency:         {p95_ms:.1f} ms")
    print(f"  p99 latency:         {p99_ms:.1f} ms")
    print(f"  Frame drop rate:     {drop_rate:.2%}")
    print(f"  Avg RAM:             {avg_ram:.0f} MB")
    print(f"  Peak RAM:            {peak_ram:.0f} MB")
    print("=" * 60)

    results = {
        "total_frames": total_frames,
        "avg_inference_ms": round(avg_ms, 1),
        "effective_fps": round(avg_fps, 1),
        "p95_ms": round(p95_ms, 1),
        "p99_ms": round(p99_ms, 1),
        "drop_rate": round(drop_rate, 4),
        "avg_ram_mb": round(avg_ram, 0),
        "peak_ram_mb": round(peak_ram, 0),
    }

    if args.output:
        import json
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
