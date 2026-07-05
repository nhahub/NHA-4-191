#!/usr/bin/env python3
"""Generate Grad-CAM heatmaps for model predictions.

Shows where the model focuses for each detected object.

Usage:
    python scripts/gradcam_visualization.py \
        --weights models/checkpoints/HPO_run/weights/best.pt \
        --source data/processed/kitti/images/val

    python scripts/gradcam_visualization.py \
        --weights models/checkpoints/HPO_run/weights/best.pt \
        --source data/samples/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--weights", type=str, required=True, help="Model weights")
    parser.add_argument("--source", type=str, default="data/samples", help="Image directory")
    parser.add_argument("--output", type=str, default="reports/gradcam_examples", help="Output directory")
    parser.add_argument("--num-images", type=int, default=5, help="Number of examples")
    parser.add_argument("--device", type=str, default="0", help="Device")
    return parser.parse_args()


def generate_heatmap(model, image: np.ndarray, target_class: int) -> np.ndarray:
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(next(model.model.parameters()).device)

    img_tensor.requires_grad_(True)
    pred = model.model(img_tensor)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    score = pred[0, target_class, :, :].mean() if pred.dim() == 4 else pred[0, target_class].mean()
    model.model.zero_grad()
    score.backward()

    grad = img_tensor.grad[0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * img_tensor[0]).sum(dim=0)
    cam = torch.clamp(cam, min=0)
    cam = cam.detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    return cam


def main() -> int:
    args = parse_args()
    model = YOLO(args.weights)
    device = args.device
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.source)
    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in source.iterdir() if p.suffix.lower() in exts][:args.num_images]

    if not images:
        print(f"No images found in {args.source}")
        return 1

    for img_path in images:
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(img_path, device=device, verbose=False)[0]

        if results.boxes and len(results.boxes) > 0:
            top_cls = int(results.boxes.cls[0].item())
            cam = generate_heatmap(model, image_rgb, top_cls)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            out_path = output_dir / f"gradcam_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), overlay_bgr)
            print(f"  Saved: {out_path}")
        else:
            print(f"  No detections: {img_path.name}")

    print(f"\nGrad-CAM examples saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
