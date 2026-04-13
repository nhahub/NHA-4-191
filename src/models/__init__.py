"""
Models Package - Road-Sense

Training, evaluation, and inference utilities for YOLO object detection models.
"""

from .model_factory import (
    load_model,
    get_model_info,
    list_available_models,
)

from .trainer import (
    YOLOTrainer,
    load_config,
    train,
)

__all__ = [
    # Model factory
    "load_model",
    "get_model_info",
    "list_available_models",
    # Trainer
    "YOLOTrainer",
    "load_config",
    "train",
]
