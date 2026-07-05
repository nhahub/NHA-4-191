import os

import cv2
import numpy as np

KITTI_CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


# 1. resize_images


def resize_images(image_paths: list, target_size: tuple = (640, 640)) -> list:
    resized = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        resized.append(img_resized)
    return resized


# 2. convert_labels
def convert_labels(kitti_lines: list, img_w: int, img_h: int) -> list:
    yolo_labels = []
    for line in kitti_lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        obj_type = parts[0]
        if obj_type not in KITTI_CLASSES:
            continue

        cls_id = KITTI_CLASSES[obj_type]
        x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

        # Clamp to image bounds
        x1 = max(0.0, min(x1, img_w))
        x2 = max(0.0, min(x2, img_w))
        y1 = max(0.0, min(y1, img_h))
        y2 = max(0.0, min(y2, img_h))

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        if w <= 0 or h <= 0:
            continue

        yolo_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_labels


# 3. filter_classes


def filter_classes(kitti_lines: list, allowed_classes: list) -> list:
    return [line for line in kitti_lines if line.strip().split()[0] in allowed_classes] if kitti_lines else []


# 4. split_dataset


def split_dataset(file_list: list, val_ratio: float = 0.2, seed: int = 42) -> tuple:
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if len(file_list) == 0:
        return [], []

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(file_list))

    val_size = max(1, int(len(file_list) * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_files = [file_list[i] for i in sorted(train_idx)]
    val_files = [file_list[i] for i in sorted(val_idx)]

    return train_files, val_files


# 5. save_preprocessed_dataset


def save_preprocessed_dataset(
    images: list,
    labels: list,
    filenames: list,
    output_dir: str,
    split: str = "train",
) -> dict:
    images_dir = os.path.join(output_dir, "images", split)
    labels_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    assert len(images) == len(labels) == len(filenames), "images, labels, and filenames must have the same length"

    saved = 0
    for img, lbl_lines, name in zip(images, labels, filenames):
        img_path = os.path.join(images_dir, f"{name}.jpg")
        lbl_path = os.path.join(labels_dir, f"{name}.txt")

        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            f.write("\n".join(lbl_lines))
        saved += 1

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "saved_count": saved,
    }
