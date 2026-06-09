#!/usr/bin/env python3
"""
Model Export Script - Road-Sense

Export trained YOLO model to deployment-friendly formats:
- ONNX (for cross-platform inference)
- TensorFlow Lite (for mobile/edge devices)
- TorchScript (for PyTorch-only environments)

Usage:
    # Export to ONNX
    python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format onnx

    # Export to TFLite
    python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format tflite

    # Export to multiple formats
    python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format onnx tflite

    # Specify output directory
    python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format onnx --output models/exports/
"""

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained Road-Sense YOLO model to deployment formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  onnx    - ONNX format (cross-platform inference with ONNX Runtime)
  tflite  - TensorFlow Lite (mobile/edge deployment)
  torchscript - TorchScript (PyTorch-only, no Python dependency)

Examples:
  # Export to ONNX only
  python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format onnx

  # Export to TFLite
  python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format tflite

  # Export to all supported formats
  python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format onnx tflite

  # Custom output directory
  python src/models/export.py --weights models/checkpoints/best-3classes-exp34332.pt --format onnx --output models/deploy/
        """,
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )

    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        choices=["onnx", "tflite", "torchscript"],
        default=["onnx"],
        help="Export format(s): onnx, tflite, torchscript (default: onnx)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/exports",
        help="Output directory for exported models (default: models/exports)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for export (default: 640)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device for export: 0 (GPU), cpu (default: auto)",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Export in half-precision (FP16) where supported",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress logging output",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    from src.utils import setup_logging as _setup_logging

    _setup_logging(verbose=verbose)
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)


def load_model(weights_path: str, device: str = "") -> YOLO:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    logger.info(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    if device:
        model.to(device)
        logger.info(f"Model moved to device: {device}")

    return model


def export_model(
    model: YOLO,
    formats: list[str],
    output_dir: Path,
    imgsz: int = 640,
    half: bool = False,
) -> dict[str, Path]:
    """
    Export model to specified formats.

    Args:
        model: YOLO model instance.
        formats: List of format names (onnx, tflite, torchscript).
        output_dir: Directory to save exported models.
        imgsz: Input image size.
        half: Whether to use FP16.

    Returns:
        Dictionary mapping format -> exported file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    for fmt in formats:
        logger.info(f"Exporting to {fmt}...")
        try:
            # ONNX-specific options
            export_kwargs = {"imgsz": imgsz, "half": half}
            if fmt == "onnx":
                export_kwargs["simplify"] = False  # Avoid onnxslim dependency

            export_path = model.export(format=fmt, **export_kwargs)
            export_path = Path(export_path)
            logger.info(f"Exported {fmt}: {export_path}")
            exported[fmt] = export_path
        except Exception as e:
            logger.error(f"Failed to export {fmt}: {e}", exc_info=True)

    return exported


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        # Load model
        model = load_model(args.weights, device=args.device)

        # Prepare output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export
        logger.info(f"Exporting to: {', '.join(args.format)}")
        exported = export_model(
            model=model,
            formats=args.format,
            output_dir=output_dir,
            imgsz=args.imgsz,
            half=args.half,
        )

        # Summary
        if exported:
            print("\n" + "=" * 60)
            print("EXPORT SUMMARY")
            print("=" * 60)
            for fmt, path in exported.items():
                size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
                print(f"  {fmt.upper():12s}  {path}  ({size_mb:.1f} MB)")
            print("=" * 60)
            return 0
        logger.error("No models were exported successfully")
        return 1

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
