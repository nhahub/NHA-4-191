#!/usr/bin/env python3
"""Compare custom YOLO model against YOLOv11m on KITTI validation set.

Outputs:
- experiments/model_comparison/results.csv
- experiments/model_comparison/results.json
- experiments/model_comparison/*.png (visualizations)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import CUSTOM_CLASSES, compute_precision_recall, ensure_dir, yolo_txt_to_boxes_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MODEL_PATH = PROJECT_ROOT / "models/checkpoints/best-3classes-exp34332.pt"
DATA_YAML = PROJECT_ROOT / "data/processed/kitti/data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments/model_comparison"

NATIVE_CLASS_MAP = {
    0: "Vehicle",
    1: "Pedestrian",
    2: "Cyclist",
}
NATIVE_COCO_MAP = {
    "Vehicle": [2, 7, 5, 3],
    "Pedestrian": [0],
    "Cyclist": [1],
}


@dataclass
class MetricsResult:
    model_name: str
    map50: float
    map50_95: float
    precision: float
    recall: float
    latency_ms: float
    fps: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ConfusionEntry:
    true_class: int
    pred_class: int
    count: int


def load_validation_images(val_dir: Path) -> list[Path]:
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = []
    if val_dir.exists():
        for ext in image_extensions:
            images.extend(sorted(val_dir.glob(f"*{ext}")))
    return images


def load_ground_truths(image_paths: list[Path], labels_root: Path) -> dict[str, dict[str, torch.Tensor]]:
    gts = {}
    for img_path in image_paths:
        with Image.open(img_path) as img:
            w, h = img.size
        lbl_path = labels_root / f"{img_path.stem}.txt"
        boxes, labels = yolo_txt_to_boxes_labels(lbl_path, w, h)
        gts[str(img_path)] = {"boxes": boxes, "labels": labels}
    return gts


def map_native_predictions(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    conf_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(labels) == 0:
        return boxes, labels, scores

    keep_mask = scores >= conf_threshold
    boxes = boxes[keep_mask]
    labels = labels[keep_mask]
    scores = scores[keep_mask]

    mapped_boxes = []
    mapped_labels = []
    mapped_scores = []

    for i, label in enumerate(labels.tolist()):
        for mapped_cls, coco_classes in NATIVE_COCO_MAP.items():
            if label in coco_classes:
                mapped_boxes.append(boxes[i].tolist())
                mapped_labels.append(list(NATIVE_CLASS_MAP.keys())[list(NATIVE_CLASS_MAP.values()).index(mapped_cls)])
                mapped_scores.append(scores[i].item())
                break

    if not mapped_boxes:
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.float32),
        )

    return (
        torch.tensor(mapped_boxes, dtype=torch.float32),
        torch.tensor(mapped_labels, dtype=torch.int64),
        torch.tensor(mapped_scores, dtype=torch.float32),
    )


def evaluate_model(
    model: YOLO,
    image_paths: list[Path],
    gts: dict[str, dict[str, torch.Tensor]],
    is_native: bool,
    device: str,
    imgsz: int = 640,
    conf: float = 0.25,
) -> tuple[MetricsResult, list[ConfusionEntry]]:
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    latencies = []
    confusion_entries = []
    all_predictions = []
    all_ground_truths = []

    print(f"  Running inference on {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="  Inference", leave=False):
        start = time.perf_counter()
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            device=device,
            conf=conf,
            verbose=False,
            augment=False,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

        result = results[0]
        gt = gts[str(img_path)]

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = torch.empty((0, 4), dtype=torch.float32)
            pred_labels = torch.empty((0,), dtype=torch.int64)
            pred_scores = torch.empty((0,), dtype=torch.float32)
        else:
            pred_boxes = result.boxes.xyxy.detach().cpu().to(torch.float32)
            pred_scores = result.boxes.conf.detach().cpu().to(torch.float32)
            pred_labels = result.boxes.cls.detach().cpu().to(torch.int64)

        if is_native:
            pred_boxes, pred_labels, pred_scores = map_native_predictions(pred_boxes, pred_labels, pred_scores, conf)

        all_predictions.append({"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores})
        all_ground_truths.append({"boxes": gt["boxes"], "labels": gt["labels"]})

        if len(pred_boxes) > 0 or len(gt["boxes"]) > 0:
            metric.update(
                [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
                [{"boxes": gt["boxes"], "labels": gt["labels"]}],
            )

        for true_label in gt["labels"].tolist():
            if len(pred_labels) > 0:
                best_pred_idx = pred_scores.argmax().item()
                pred_label = pred_labels[best_pred_idx].item()
                confusion_entries.append(ConfusionEntry(true_class=true_label, pred_class=pred_label, count=1))

    maps = metric.compute()
    map50 = float(maps.get("map_50", torch.tensor(0.0)).item())
    map50_95 = float(maps.get("map", torch.tensor(0.0)).item())

    per_class_map50 = {}
    per_class_precision = {}
    per_class_recall = {}

    for cls_idx, cls_name in enumerate(CUSTOM_CLASSES):
        key = f"map_50_per_class/{cls_idx}"
        if key in maps:
            per_class_map50[cls_name] = float(maps[key].item())
        elif "map_per_class" in maps:
            try:
                per_class_map50[cls_name] = float(maps["map_per_class"][cls_idx].item())
            except Exception:
                per_class_map50[cls_name] = 0.0
        else:
            per_class_map50[cls_name] = 0.0

    precision_val, recall_val = compute_precision_recall(all_predictions, all_ground_truths)

    latency_ms = float(np.mean(latencies)) if latencies else float("nan")
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    result = MetricsResult(
        model_name="",
        map50=round(map50, 4),
        map50_95=round(map50_95, 4),
        precision=round(precision_val, 4),
        recall=round(recall_val, 4),
        latency_ms=round(latency_ms, 2),
        fps=round(fps, 2),
        per_class={
            "mAP50": per_class_map50,
            "Precision": per_class_precision,
            "Recall": per_class_recall,
        },
    )

    return result, confusion_entries


def build_confusion_matrix(
    confusion_entries: list[ConfusionEntry],
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for entry in confusion_entries:
        if entry.true_class < num_classes and entry.pred_class < num_classes:
            cm[entry.true_class, entry.pred_class] += entry.count
    return cm


def create_visualizations(
    custom_result: MetricsResult,
    native_result: MetricsResult,
    custom_confusion: list[ConfusionEntry],
    native_confusion: list[ConfusionEntry],
    image_paths: list[Path],
    custom_model: YOLO,
    native_model: YOLO,
    device: str,
    imgsz: int,
    output_dir: Path,
) -> None:
    print("\nCreating visualizations...")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CUSTOM_CLASSES))
    width = 0.35

    custom_map50 = [custom_result.per_class["mAP50"].get(cls, 0) for cls in CUSTOM_CLASSES]
    native_map50 = [native_result.per_class["mAP50"].get(cls, 0) for cls in CUSTOM_CLASSES]

    bars1 = ax.bar(x - width / 2, custom_map50, width, label="Custom Model", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, native_map50, width, label="YOLOv11m", color="#3498db")

    ax.set_xlabel("Class")
    ax.set_ylabel("mAP@50")
    ax.set_title("Per-Class mAP@50 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(CUSTOM_CLASSES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "map50_comparison.png", dpi=150)
    plt.close()
    print("  Saved: map50_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models_data = [
        (
            "Custom Model",
            custom_result.map50,
            custom_result.map50_95,
            custom_result.latency_ms,
        ),
        (
            "YOLOv11m",
            native_result.map50,
            native_result.map50_95,
            native_result.latency_ms,
        ),
    ]

    for idx, (model_name, map50, map5095, latency) in enumerate(models_data):
        ax = axes[idx]
        metrics = ["mAP@50", "mAP@50:95"]
        values = [map50, map5095]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.bar(metrics, values, color=colors)
        ax.set_title(f"{model_name}\nLatency: {latency:.1f}ms | FPS: {1000 / latency:.1f}")
        ax.set_ylim(0, 1)
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
    plt.close()
    print("  Saved: metrics_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_data = [
        (custom_result.latency_ms, custom_result.map50, "Custom Model", "#2ecc71"),
        (native_result.latency_ms, native_result.map50, "YOLOv11m", "#3498db"),
    ]

    for lat, map_val, label, color in scatter_data:
        ax.scatter(lat, map_val, s=200, c=color, label=label, edgecolors="black", linewidth=2)
        ax.annotate(
            label,
            (lat, map_val),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("mAP@50")
    ax.set_title("Speed vs Accuracy Trade-off")
    ax.legend()
    ax.grid(alpha=0.3)

    ax.set_xlim(0, max(custom_result.latency_ms, native_result.latency_ms) * 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_accuracy.png", dpi=150)
    plt.close()
    print("  Saved: speed_vs_accuracy.png")

    print("  Creating sample predictions...")
    sample_images = image_paths[:6]
    n_images = len(sample_images)

    fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))

    for idx, img_path in enumerate(sample_images):
        img = cv2.imread(str(img_path))

        custom_result_img = custom_model.predict(
            source=str(img_path), imgsz=imgsz, device=device, conf=0.25, verbose=False
        )[0]
        native_result_img = native_model.predict(
            source=str(img_path), imgsz=imgsz, device=device, conf=0.25, verbose=False
        )[0]

        img_custom = img.copy()
        if custom_result_img.boxes is not None:
            for box in custom_result_img.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                cv2.rectangle(img_custom, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        axes[0, idx].imshow(cv2.cvtColor(img_custom, cv2.COLOR_BGR2RGB))
        axes[0, idx].axis("off")
        axes[0, idx].set_title("Custom", fontsize=10)

        img_native = cv2.imread(str(img_path))
        if native_result_img.boxes is not None:
            for box in native_result_img.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                if cls_id in [2, 7, 5, 3]:
                    pass
                elif cls_id == 0:
                    pass
                elif cls_id == 1:
                    pass
                else:
                    continue
                cv2.rectangle(img_native, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        axes[1, idx].imshow(cv2.cvtColor(img_native, cv2.COLOR_BGR2RGB))
        axes[1, idx].axis("off")
        axes[1, idx].set_title("YOLOv11m", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: sample_predictions.png")

    cm_custom = build_confusion_matrix(custom_confusion, len(CUSTOM_CLASSES))
    cm_native = build_confusion_matrix(native_confusion, len(CUSTOM_CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axes[0].imshow(cm_custom, cmap="Blues")
    axes[0].set_title("Custom Model Confusion Matrix")
    axes[0].set_xticks(np.arange(len(CUSTOM_CLASSES)))
    axes[0].set_yticks(np.arange(len(CUSTOM_CLASSES)))
    axes[0].set_xticklabels(CUSTOM_CLASSES)
    axes[0].set_yticklabels(CUSTOM_CLASSES)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    for i in range(len(CUSTOM_CLASSES)):
        for j in range(len(CUSTOM_CLASSES)):
            axes[0].text(
                j,
                i,
                cm_custom[i, j],
                ha="center",
                va="center",
                color="white" if cm_custom[i, j] > cm_custom.max() / 2 else "black",
            )
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cm_native, cmap="Blues")
    axes[1].set_title("YOLOv11m Confusion Matrix")
    axes[1].set_xticks(np.arange(len(CUSTOM_CLASSES)))
    axes[1].set_yticks(np.arange(len(CUSTOM_CLASSES)))
    axes[1].set_xticklabels(CUSTOM_CLASSES)
    axes[1].set_yticklabels(CUSTOM_CLASSES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    for i in range(len(CUSTOM_CLASSES)):
        for j in range(len(CUSTOM_CLASSES)):
            axes[1].text(
                j,
                i,
                cm_native[i, j],
                ha="center",
                va="center",
                color="white" if cm_native[i, j] > cm_native.max() / 2 else "black",
            )
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: confusion_matrix.png")


def main():
    ensure_dir(OUTPUT_DIR)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading models...")
    print(f"  Custom model: {CUSTOM_MODEL_PATH}")
    custom_model = YOLO(str(CUSTOM_MODEL_PATH))
    print("  Native YOLOv11m: yolo11m.pt")
    native_model = YOLO("yolo11m.pt")

    val_images_dir = PROJECT_ROOT / "data/processed/kitti/images/val"
    val_labels_dir = PROJECT_ROOT / "data/processed/kitti/labels/val"

    print(f"\nLoading validation images from: {val_images_dir}")
    image_paths = load_validation_images(val_images_dir)
    print(f"  Found {len(image_paths)} images")

    print(f"Loading ground truth labels from: {val_labels_dir}")
    gts = load_ground_truths(image_paths, val_labels_dir)

    print("\n" + "=" * 60)
    print("Evaluating Custom Model")
    print("=" * 60)
    custom_result, custom_confusion = evaluate_model(custom_model, image_paths, gts, is_native=False, device=device)
    custom_result.model_name = "Custom Model"
    print(f"  mAP@50: {custom_result.map50:.4f}")
    print(f"  mAP@50:95: {custom_result.map50_95:.4f}")
    print(f"  Latency: {custom_result.latency_ms:.2f}ms")
    print(f"  FPS: {custom_result.fps:.2f}")

    print("\n" + "=" * 60)
    print("Evaluating YOLOv11m (Native)")
    print("=" * 60)
    native_result, native_confusion = evaluate_model(native_model, image_paths, gts, is_native=True, device=device)
    native_result.model_name = "YOLOv11m"
    print(f"  mAP@50: {native_result.map50:.4f}")
    print(f"  mAP@50:95: {native_result.map50_95:.4f}")
    print(f"  Latency: {native_result.latency_ms:.2f}ms")
    print(f"  FPS: {native_result.fps:.2f}")

    results_df = pd.DataFrame(
        [
            {
                "Model": custom_result.model_name,
                "mAP@50": custom_result.map50,
                "mAP@50-95": custom_result.map50_95,
                "Precision": custom_result.precision,
                "Recall": custom_result.recall,
                "Latency (ms)": custom_result.latency_ms,
                "FPS": custom_result.fps,
            },
            {
                "Model": native_result.model_name,
                "mAP@50": native_result.map50,
                "mAP@50-95": native_result.map50_95,
                "Precision": native_result.precision,
                "Recall": native_result.recall,
                "Latency (ms)": native_result.latency_ms,
                "FPS": native_result.fps,
            },
        ]
    )

    results_csv_path = OUTPUT_DIR / "results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved: {results_csv_path}")

    results_json = {
        "custom_model": {
            "name": "Custom Model",
            "path": str(CUSTOM_MODEL_PATH),
            "mAP50": custom_result.map50,
            "mAP50-95": custom_result.map50_95,
            "precision": custom_result.precision,
            "recall": custom_result.recall,
            "latency_ms": custom_result.latency_ms,
            "fps": custom_result.fps,
            "per_class": custom_result.per_class,
        },
        "native_model": {
            "name": "YOLOv11m",
            "path": "yolo11m.pt",
            "mAP50": native_result.map50,
            "mAP50-95": native_result.map50_95,
            "precision": native_result.precision,
            "recall": native_result.recall,
            "latency_ms": native_result.latency_ms,
            "fps": native_result.fps,
            "per_class": native_result.per_class,
        },
        "class_mapping": {
            "custom": CUSTOM_CLASSES,
            "native_to_custom": NATIVE_COCO_MAP,
        },
    }

    results_json_path = OUTPUT_DIR / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {results_json_path}")

    create_visualizations(
        custom_result,
        native_result,
        custom_confusion,
        native_confusion,
        image_paths,
        custom_model,
        native_model,
        device,
        640,
        OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
