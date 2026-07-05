#!/usr/bin/env python3
"""Process video with real-time detection overlay.

Displays bounding boxes, class labels, confidence scores, and FPS counter.
Can save processed video to file.

Usage:
    # Webcam
    python src/deployment/video_demo.py --weights models/checkpoints/HPO_run/weights/best.pt

    # Video file
    python src/deployment/video_demo.py --weights models/checkpoints/HPO_run/weights/best.pt \\
        --source video.mp4 --output result.mp4

    # ONNX model
    python src/deployment/video_demo.py --weights models/exports/best.onnx
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]
COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 0)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video detection demo")
    parser.add_argument("--weights", type=str, required=True, help="Model weights")
    parser.add_argument("--source", type=str, default="0", help="Video source (path or 0 for webcam)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--display", action="store_true", default=True, help="Show display window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.weights)
    device = args.device

    src = int(args.source) if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Cannot open source {args.source}")
        return 1

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, fps_in, (w, h))

    print(f"Processing: {src} ({w}x{h} @ {fps_in:.1f} FPS)")
    print("Controls: q=quit, p=pause")

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]

            annotated = results.plot()
            fps_text = f"FPS: {fps_in:.0f}"

            if results.boxes:
                cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                class_counts = {}
                for cls in results.boxes.cls:
                    name = results.names[int(cls)]
                    class_counts[name] = class_counts.get(name, 0) + 1
                y_offset = 60
                for name, count in class_counts.items():
                    cv2.putText(annotated, f"{name}: {count}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 25

            if args.display:
                cv2.imshow("Road-Sense Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused

            if out:
                out.write(annotated)
        else:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("p"):
                paused = False
            if key == ord("q"):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
