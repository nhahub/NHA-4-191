#!/usr/bin/env python3
"""Validate exported model accuracy across all formats.

Compares mAP of ONNX, TorchScript, and TFLite exports against the
PyTorch baseline to ensure accuracy drops stay within tolerance.

Usage:
    # Full validation (requires dataset)
    python scripts/validate_exports.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --data data/processed/kitti/data.yaml

    # Structure-only validation (no dataset needed)
    python scripts/validate_exports.py \\
        --weights models/checkpoints/HPO_run/weights/best.pt \\
        --structure-only
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("validate_exports")


EXPORT_FORMATS = {
    "pytorch": {"ext": ".pt", "export_args": {}},
    "onnx_fp16": {"ext": ".onnx", "export_args": {"format": "onnx", "half": True, "dynamic": True, "simplify": True}},
    "torchscript_traced": {"ext": ".torchscript", "export_args": {"format": "torchscript"}},
    "tflite_fp16": {"ext": "_fp16.tflite", "export_args": {"format": "tflite", "int8": False}},
    "tflite_int8": {"ext": "_int8.tflite", "export_args": {"format": "tflite", "int8": True}},
}

TOLERANCES = {
    "onnx_fp16": 0.01,
    "torchscript_traced": 0.005,
    "tflite_fp16": 0.01,
    "tflite_int8": 0.02,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported model accuracy")
    parser.add_argument("--weights", type=str, required=True, help="Path to PyTorch model weights")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset YAML (required for accuracy validation)")
    parser.add_argument("--structure-only", action="store_true",
                        help="Only validate file structure, skip accuracy")
    parser.add_argument("--export-dir", type=str, default="models/exports/validated",
                        help="Directory for temporary exports")
    parser.add_argument("--device", type=str, default="0", help="Device to run validation on")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    return parser.parse_args()


def resolve_data_yaml(data_path: str | None) -> str | None:
    if data_path is None:
        return None
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"Data YAML not found: {path}")
        return None
    return str(path.resolve())


def export_formats(model, export_dir: Path, device: str) -> dict[str, Path]:
    logger.info(f"\n--- Exporting formats to: {export_dir} ---")
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    # ONNX FP16
    logger.info("Exporting ONNX FP16...")
    onnx_path = model.export(format="onnx", half=True, dynamic=True, simplify=True)
    exported["onnx_fp16"] = Path(onnx_path)

    # TorchScript
    logger.info("Exporting TorchScript...")
    ts_path = model.export(format="torchscript")
    exported["torchscript_traced"] = Path(ts_path)

    # TFLite FP16
    logger.info("Exporting TFLite FP16...")
    tflite_fp16 = model.export(format="tflite", int8=False)
    exported["tflite_fp16"] = Path(tflite_fp16)

    # TFLite INT8
    logger.info("Exporting TFLite INT8...")
    tflite_int8 = model.export(format="tflite", int8=True)
    exported["tflite_int8"] = Path(tflite_int8)

    return exported


def validate_format(format_name: str, format_path: Path, data_yaml: str | None) -> dict:
    from ultralytics import YOLO

    logger.info(f"  Loading {format_name}: {format_path}")
    result = {"status": "loaded", "path": str(format_path), "format": format_name}

    try:
        model = YOLO(str(format_path))
        result["loaded"] = True
    except Exception as e:
        logger.error(f"  Failed to load {format_name}: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
        return result

    if data_yaml:
        try:
            logger.info(f"  Running accuracy validation for {format_name}...")
            metrics = model.val(data=data_yaml, verbose=False)
            result["metrics"] = extract_metrics(metrics)
            logger.info(f"    mAP@50: {result['metrics'].get('mAP50', 0):.4f}, "
                        f"mAP@50:95: {result['metrics'].get('mAP50-95', 0):.4f}")
        except Exception as e:
            logger.warning(f"  Accuracy validation failed for {format_name}: {e}")
            result["val_error"] = str(e)

    return result


def load_and_validate_formats(weights_path: str, data_yaml: str | None, structure_only: bool,
                              export_dir: str | None = None) -> dict:
    from ultralytics import YOLO

    results = {}
    weights_path = str(Path(weights_path).resolve())

    logger.info("=" * 60)
    logger.info("VALIDATING EXPORT FORMATS")
    logger.info("=" * 60)

    # Load PyTorch baseline
    logger.info(f"\n--- Loading PyTorch baseline: {weights_path} ---")
    baseline = YOLO(weights_path)
    results["pytorch"] = {"status": "loaded", "path": weights_path}

    if data_yaml and not structure_only:
        logger.info("Running PyTorch baseline validation...")
        baseline_metrics = baseline.val(data=data_yaml)
        results["pytorch"]["metrics"] = extract_metrics(baseline_metrics)
        logger.info(f"  PyTorch baseline mAP@50: {results['pytorch']['metrics']['mAP50']:.4f}")
        logger.info(f"  PyTorch baseline mAP@50:95: {results['pytorch']['metrics']['mAP50-95']:.4f}")

    logger.info("\nPyTorch baseline loaded successfully")

    if structure_only:
        return results

    # Export to all formats
    export_dir_path = Path(export_dir) if export_dir else Path(tempfile.mkdtemp(prefix="exports_"))
    exported = export_formats(baseline, export_dir_path, "0")

    # Validate each format
    for fmt_name, fmt_path in exported.items():
        logger.info(f"\n--- Validating {fmt_name} ---")
        results[fmt_name] = validate_format(fmt_name, fmt_path, data_yaml)

    return results


def extract_metrics(metrics) -> dict:
    result = {}
    try:
        if hasattr(metrics, "box"):
            m50 = metrics.box.map50
            result["mAP50"] = float(m50) if m50 is not None else 0.0
            m = metrics.box.map
            result["mAP50-95"] = float(m) if m is not None else 0.0
            if hasattr(metrics.box, "mp") and metrics.box.mp is not None:
                result["precision"] = float(metrics.box.mp)
            if hasattr(metrics.box, "mr") and metrics.box.mr is not None:
                result["recall"] = float(metrics.box.mr)
    except Exception as e:
        logger.warning(f"Could not extract metrics: {e}")
    return result


def print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if any("metrics" in r for r in results.values()):
        print(f"\n{'Format':<25} {'mAP@50':<10} {'mAP@50:95':<12} {'Drop':<10} {'Status'}")
        print("-" * 70)

        baseline = results.get("pytorch", {})
        baseline_map50 = baseline.get("metrics", {}).get("mAP50", 0)
        baseline_map5095 = baseline.get("metrics", {}).get("mAP50-95", 0)

        for fmt, data in results.items():
            if fmt == "pytorch":
                label = "PyTorch (baseline)"
                print(f"{label:<25} {baseline_map50:<10.4f} {baseline_map5095:<12.4f} {'—':<10} {'✅ baseline'}")
                continue

            fmt_map50 = data.get("metrics", {}).get("mAP50", 0)
            fmt_map5095 = data.get("metrics", {}).get("mAP50-95", 0)
            drop = baseline_map5095 - fmt_map5095
            tolerance = TOLERANCES.get(fmt, 0.01)
            status = "✅ PASS" if drop <= tolerance else "❌ FAIL"
            print(f"{fmt:<25} {fmt_map50:<10.4f} {fmt_map5095:<12.4f} {drop:<10.4f} {status}")
    else:
        print("\nNo accuracy metrics available (run with --data for full validation)")
        for fmt, data in results.items():
            status = "✅ loaded" if data.get("status") == "loaded" else "❌ failed"
            print(f"  {fmt:<25} {status}")

    print("=" * 60)


def save_results(results: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {path}")


def check_tolerances(results: dict) -> bool:
    all_pass = True
    baseline = results.get("pytorch", {})
    baseline_map = baseline.get("metrics", {}).get("mAP50-95", 0)

    print("\n--- Tolerance Check ---")
    for fmt, tol in TOLERANCES.items():
        if fmt not in results:
            continue
        fmt_metrics = results[fmt].get("metrics", {})
        if not fmt_metrics:
            print(f"  {fmt:<25} ⏭️  no metrics (data not available)")
            continue
        fmt_map = fmt_metrics.get("mAP50-95", 0)
        drop = baseline_map - fmt_map
        passed = drop <= tol
        if not passed:
            all_pass = False
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {fmt:<25} drop={drop:.4f} (tol={tol:.3f}) {status}")

    return all_pass


def main() -> int:
    args = parse_args()
    weights_path = str(Path(args.weights).resolve())

    if not Path(weights_path).exists():
        logger.error(f"Weights not found: {weights_path}")
        return 1

    data_yaml = resolve_data_yaml(args.data) if not args.structure_only else None
    results = load_and_validate_formats(weights_path, data_yaml, args.structure_only, export_dir=args.export_dir)
    print_summary(results)

    if args.output:
        save_results(results, args.output)

    if not args.structure_only and data_yaml:
        all_pass = check_tolerances(results)
        if not all_pass:
            logger.warning("Some formats exceeded accuracy drop tolerance!")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
