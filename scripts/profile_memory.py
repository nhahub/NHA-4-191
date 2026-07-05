#!/usr/bin/env python3
"""Profile memory usage during inference and detect leaks.

Measures RAM across repeated inference iterations, checks for growth
trends, and reports pass/fail against the 2 GB budget (KPI SP-11).

Usage:
    python scripts/profile_memory.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt

    python scripts/profile_memory.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --iterations 200 --device cpu
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import psutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("profile_memory")

MEMORY_BUDGET_MB = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile inference memory usage")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--iterations", type=int, default=100, help="Number of inference iterations")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="Device (cpu or cuda:0)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process = psutil.Process(os.getpid())
    from ultralytics import YOLO

    logger.info("Loading model: %s", args.weights)
    model = YOLO(args.weights)
    device = args.device

    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    logger.info("Running %d inference iterations...", args.iterations)
    ram_history = []
    baseline_ram = process.memory_info().rss / (1024 * 1024)
    logger.info("  Baseline RAM (after load): %.1f MB", baseline_ram)

    for i in range(args.iterations):
        model.predict(dummy, verbose=False, device=device)
        current_ram = process.memory_info().rss / (1024 * 1024)
        ram_history.append(current_ram)

        if (i + 1) % 25 == 0:
            logger.info("  Iteration %d/%d — RAM: %.1f MB", i + 1, args.iterations, current_ram)

    final_ram = process.memory_info().rss / (1024 * 1024)
    peak_ram = max(ram_history)
    growth = final_ram - baseline_ram
    growth_rate = growth / args.iterations if args.iterations > 0 else 0

    early_avg = np.mean(ram_history[: args.iterations // 2]) if args.iterations >= 2 else 0
    late_avg = np.mean(ram_history[args.iterations // 2 :]) if args.iterations >= 2 else 0
    drift = late_avg - early_avg

    within_budget = peak_ram <= MEMORY_BUDGET_MB
    leak_suspected = drift > 10 and growth_rate > 0.5

    print("\n" + "=" * 60)
    print("MEMORY PROFILE RESULTS")
    print("=" * 60)
    print(f"  Iterations:            {args.iterations}")
    print(f"  Device:                {device}")
    print(f"  Baseline RAM:          {baseline_ram:.1f} MB")
    print(f"  Peak RAM:              {peak_ram:.1f} MB")
    print(f"  Final RAM:             {final_ram:.1f} MB")
    print(f"  Growth:                {growth:+.1f} MB ({growth_rate:.2f} MB/iter)")
    print(f"  Drift (late-early):    {drift:+.1f} MB")
    print(f"  Budget (KPI SP-11):    {'✅ PASS' if within_budget else '❌ FAIL'} ({MEMORY_BUDGET_MB} MB)")
    if leak_suspected:
        print(f"  ⚠️  Memory leak suspected: +{drift:.1f} MB drift across {args.iterations} iterations")
    else:
        print(f"  ✅ No leak detected (drift: {drift:+.1f} MB)")
    print("=" * 60)

    results = {
        "iterations": args.iterations,
        "device": device,
        "baseline_ram_mb": round(baseline_ram, 1),
        "peak_ram_mb": round(peak_ram, 1),
        "final_ram_mb": round(final_ram, 1),
        "growth_mb": round(growth, 1),
        "growth_per_iter_mb": round(growth_rate, 3),
        "drift_mb": round(drift, 1),
        "within_budget": within_budget,
        "leak_suspected": leak_suspected,
    }

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to: %s", output_path)

    return 0 if within_budget and not leak_suspected else 1


if __name__ == "__main__":
    sys.exit(main())
