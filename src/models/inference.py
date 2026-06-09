#!/usr/bin/env python3
"""
Inference Script - Road-Sense

CLI interface for running predictions with trained YOLO models.
Supports single images, directories, and video files.

Usage:
    # Predict on single image
    python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source data/sample.jpg

    # Predict on directory of images
    python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source data/samples/ --output predictions/

    # Predict on video file
    python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source video.mp4 --output output_video.avi

    # Adjust confidence threshold
    python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source image.jpg --conf 0.5
"""

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

from src.utils import DEFAULT_CONFIDENCE

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained Road-Sense YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source image.jpg

  # Batch prediction on directory
  python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source data/images/ --output predictions/

  # Video prediction
  python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source video.mp4 --output result.avi

  # Adjust confidence threshold
  python src/models/inference.py --weights models/checkpoints/best-3classes-exp34332.pt --source image.jpg --conf 0.25
        """,
    )

    # Model arguments
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (e.g., models/checkpoints/best-3classes-exp34332.pt)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training config to load class names from (not required)",
    )

    # Input source
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image file, directory of images, or video file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for annotated results (file or directory). Default: runs/predict/",
    )

    # Inference parameters
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold for detections (default: {DEFAULT_CONFIDENCE})",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for inference (default: 640)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device for inference: 0 (GPU), cpu, 0,1,2,3 (multi-GPU). Default: auto",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated images/videos to output directory (default: True)",
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=False,
        help="Save detection results as text files (YOLO format)",
    )

    parser.add_argument(
        "--save-conf",
        action="store_true",
        default=False,
        help="Include confidence scores in saved txt files",
    )

    parser.add_argument(
        "--view",
        action="store_true",
        default=False,
        help="Display results in a window during inference",
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
        help="Suppress logging output (only errors)",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging."""
    from src.utils import setup_logging as _setup_logging

    _setup_logging(verbose=verbose)
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)


def load_model(weights_path: str | Path, device: str = "") -> YOLO:
    """
    Load YOLO model from checkpoint.

    Args:
        weights_path: Path to .pt weights file.
        device: Device to load model on (auto-detected if empty).

    Returns:
        Loaded YOLO model.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    # Move to device if specified
    if device:
        model.to(device)
        logger.info(f"Model moved to device: {device}")

    # Log model info
    if hasattr(model, "names"):
        logger.info(f"Model classes: {model.names}")
    logger.info("Model loaded successfully")

    return model


def predict(
    model: YOLO,
    source: str | Path,
    output_dir: str | Path | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    save: bool = True,
    save_txt: bool = False,
    save_conf: bool = False,
    view: bool = False,
    device: str = "",
) -> list:
    """
    Run prediction on input source.

    Args:
        model: YOLO model instance.
        source: Input source path (image, directory, or video).
        output_dir: Directory to save results.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        imgsz: Input image size.
        save: Whether to save annotated outputs.
        save_txt: Save detections as txt files.
        save_conf: Include confidence in txt files.
        view: Display results in window.
        device: Device for inference.

    Returns:
        List of results objects.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Build predict args
    predict_args = {
        "source": str(source),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "save": save,
        "save_txt": save_txt,
        "save_conf": save_conf,
        "view": view,
        "device": device,
        "verbose": logger.isEnabledFor(logging.DEBUG),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predict_args["project"] = str(output_dir.parent)
        predict_args["name"] = output_dir.name

    logger.info(f"Running inference on: {source}")
    logger.info(f"Parameters: conf={conf}, iou={iou}, imgsz={imgsz}, device={device or 'auto'}")

    results = model.predict(**predict_args)

    result_count = len(results) if results else 0
    logger.info(f"Inference complete. {result_count} result(s) returned.")

    return results


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        # Load model
        model = load_model(args.weights, device=args.device)

        # Run prediction
        results = predict(
            model=model,
            source=args.source,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            view=args.view,
            device=args.device,
        )

        # Summary
        if results:
            total_detections = sum(len(r.boxes) for r in results if hasattr(r, "boxes"))
            logger.info(f"Total detections across all frames/images: {total_detections}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
