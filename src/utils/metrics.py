from pathlib import Path

import torch


def yolo_txt_to_boxes_labels(label_path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not label_path.exists():
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)

    boxes = []
    labels = []
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = map(float, parts)
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls))

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    return inter / union


def compute_precision_recall(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]

        gt_matched = set()
        pred_matched = set()

        for i, (pb, pl, ps) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            best_iou = 0
            best_gt_idx = -1

            for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if j in gt_matched or pl != gl:
                    continue

                iou = box_iou(pb.unsqueeze(0), gb.unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                gt_matched.add(best_gt_idx)
                pred_matched.add(i)
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(gt_matched)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return precision, recall
