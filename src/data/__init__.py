"""
Data module for Road-Sense project.

This module provides utilities for loading, augmenting, and processing
KITTI dataset for object detection tasks.
"""

from .augment_dataset import augment_dataset
from .augmentations import (
    AUGMENTATION_PRESETS,
    get_custom_augmentation,
    get_inference_augmentation,
    get_training_augmentation,
    get_training_augmentation_with_bbox,
    get_validation_augmentation,
    get_validation_augmentation_with_bbox,
)
from .kitti_dataset import KITTIDataset, KITTIDatasetTorch, collate_fn, create_data_loaders
from .kitti_utils import (
    CLASS_ID_TO_NAME,
    KITTI_CLASSES,
    get_dataset_statistics,
    load_kitti_image,
    load_kitti_labels,
    load_yolo_labels,
    print_dataset_statistics,
    save_yolo_labels,
    visualize_bboxes,
    yolo_to_pixel,
)
from .preprocess_dataset import preprocess_dataset

__all__ = [
    # Augmentations
    "get_training_augmentation",
    "get_training_augmentation_with_bbox",
    "get_validation_augmentation",
    "get_validation_augmentation_with_bbox",
    "get_inference_augmentation",
    "get_custom_augmentation",
    "AUGMENTATION_PRESETS",
    # KITTI utilities
    "KITTI_CLASSES",
    "CLASS_ID_TO_NAME",
    "load_kitti_image",
    "load_kitti_labels",
    "save_yolo_labels",
    "load_yolo_labels",
    "yolo_to_pixel",
    "visualize_bboxes",
    "get_dataset_statistics",
    "print_dataset_statistics",
    # Dataset classes
    "KITTIDataset",
    "KITTIDatasetTorch",
    "collate_fn",
    "create_data_loaders",
    # Batch augmentation
    "augment_dataset",
    # Dataset preprocessing
    "preprocess_dataset",
]
