#!/usr/bin/env python3
"""
Real-time Detection - Road-Sense

Real-time object detection from webcam or video stream.
Displays live predictions with bounding boxes, class labels, and FPS.

Usage:
    # Webcam detection (default camera 0)
    python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt

    # Specify camera index
    python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt --source 1

    # Video file detection
    python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt --source video.mp4

    # Adjust confidence
    python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt --conf 0.3
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import CLASS_COLORS_LIST, DEFAULT_CONFIDENCE

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time object detection with Road-Sense YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam detection (camera 0)
  python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt

  # Use external camera (index 1)
  python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt --source 1

  # Video file detection
  python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt --source input_video.mp4

  # Save output to file
  python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt --source 0 --output recording.avi

  # Adjust confidence threshold
  python src/models/realtime.py --weights models/checkpoints/best-3classes-exp34332.pt --conf 0.3
        """,
    )

    # Model arguments
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (e.g., models/checkpoints/best-3classes-exp34332.pt)",
    )

    # Source arguments
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source: camera index (0, 1, ...) or video file path (default: 0)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path to save recording (e.g., output.avi)",
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
        help="Device: 0 (GPU), cpu, 0,1 (multi-GPU). Default: auto",
    )

    parser.add_argument(
        "--no-view",
        action="store_true",
        default=False,
        help="Disable display window (useful for headless processing)",
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
    """Configure logging."""
    from src.utils import setup_logging as _setup_logging

    _setup_logging(verbose=verbose)
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)


def load_model(weights_path: str, device: str = "") -> YOLO:
    """Load YOLO model from checkpoint."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    if device:
        model.to(device)
        logger.info(f"Model moved to device: {device}")

    return model


def parse_source(source: str) -> tuple[str | None, int]:
    """
    Parse source argument to determine video capture source.

    Returns:
        Tuple of (video_file_path or None, camera_index)
    """
    # Try to parse as integer for camera index
    try:
        cam_idx = int(source)
        return None, cam_idx
    except ValueError:
        # Treat as file path
        return source, -1


def init_video_writer(
    frame_shape: tuple[int, int], fps: float, output_path: str, codec: str = "XVID"
) -> cv2.VideoWriter | None:
    """Initialize video writer for saving output."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_shape[::-1])
    if not out.isOpened():
        logger.error(f"Failed to open video writer for: {output_path}")
        return None
    logger.info(f"Recording video to: {output_path}")
    return out


def draw_detections(
    frame: np.ndarray,
    results: Any,  # noqa: ANN401
    class_names: dict,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_conf: bool = True,
) -> None:
    """
    Draw bounding boxes and labels on frame in-place.

    Args:
        frame: OpenCV image (modified in-place).
        results: YOLO prediction results.
        class_names: Mapping of class ID to name.
        line_thickness: Box line thickness.
        font_scale: Text font size.
        show_conf: Whether to show confidence score.
    """
    if not hasattr(results, "boxes") or len(results.boxes) == 0:
        return

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    # Color palette for classes (BGR format)
    colors = CLASS_COLORS_LIST

    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        class_name = class_names.get(cls_id, f"class_{cls_id}")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=line_thickness)

        # Prepare label
        if show_conf:
            label = f"{class_name}: {conf:.2f}"
        else:
            label = class_name

        # Calculate text size
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w + 4, y1),
            color,
            thickness=cv2.FILLED,
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw FPS counter on frame."""
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        # Load model
        model = load_model(args.weights, device=args.device)

        # Get class names
        class_names = getattr(model, "names", {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"})
        logger.info(f"Model classes: {class_names}")

        # Parse source
        video_file, camera_idx = parse_source(args.source)

        if video_file:
            video_path = Path(video_file)
            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return 1
            cap = cv2.VideoCapture(str(video_path))
            logger.info(f"Opening video file: {video_path}")
        else:
            cap = cv2.VideoCapture(camera_idx)
            logger.info(f"Opening camera {camera_idx}")

        if not cap.isOpened():
            logger.error("Failed to open video source")
            return 1

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        logger.info(f"Video: {frame_width}x{frame_height} @ {input_fps:.1f} FPS")

        # Initialize video writer if output requested
        video_writer = None
        if args.output:
            video_writer = init_video_writer(
                frame_shape=(frame_width, frame_height),
                fps=input_fps,
                output_path=args.output,
            )
            if video_writer is None:
                return 1

        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0.0

        logger.info("Starting real-time detection. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or failed to read frame")
                break

            # Run inference
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )

            # results is a list, take first (single image)
            if results:
                draw_detections(frame, results[0], class_names)

            # Update FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()

            draw_fps(frame, current_fps)

            # Display frame
            if not args.no_view:
                cv2.imshow("Road-Sense Real-time Detection", frame)

            # Write to output video
            if video_writer:
                video_writer.write(frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Exiting on user request")
                break

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        logger.info("Real-time detection completed")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Real-time detection failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
