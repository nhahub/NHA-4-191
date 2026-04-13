"""
Model Factory - Road-Sense

Utilities for loading, inspecting, and managing YOLO models.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO

# Available YOLO model variants for detection
AVAILABLE_MODELS = {
    # YOLOv11 models
    "yolo11n": {"weights": "yolo11n.pt", "params_m": 2.6, "size_mb": 5.4},
    "yolo11s": {"weights": "yolo11s.pt", "params_m": 9.4, "size_mb": 19.0},
    "yolo11m": {"weights": "yolo11m.pt", "params_m": 20.1, "size_mb": 38.8},
    "yolo11l": {"weights": "yolo11l.pt", "params_m": 25.3, "size_mb": 51.4},
    "yolo11x": {"weights": "yolo11x.pt", "params_m": 56.9, "size_mb": 115.0},
    # YOLOv8 models
    "yolov8n": {"weights": "yolov8n.pt", "params_m": 3.2, "size_mb": 6.5},
    "yolov8s": {"weights": "yolov8s.pt", "params_m": 11.2, "size_mb": 21.5},
    "yolov8m": {"weights": "yolov8m.pt", "params_m": 25.9, "size_mb": 52.0},
    "yolov8l": {"weights": "yolov8l.pt", "params_m": 43.7, "size_mb": 87.7},
    "yolov8x": {"weights": "yolov8x.pt", "params_m": 68.2, "size_mb": 136.5},
}


def load_model(
    model_name: str = "yolo11m",
    weights_path: Optional[str] = None,
    pretrained: bool = True,
) -> YOLO:
    """
    Load a YOLO model.

    Args:
        model_name: Model variant name (e.g., 'yolo11m', 'yolov8s').
        weights_path: Path to custom weights file. If None, uses pretrained or
            model_name to auto-download.
        pretrained: Whether to use pretrained COCO weights. Ignored if
            weights_path is provided.

    Returns:
        Ultralytics YOLO model instance.

    Raises:
        ValueError: If model_name is not in AVAILABLE_MODELS.
        FileNotFoundError: If weights_path does not exist.
    """
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )

    if weights_path is not None:
        weights_path = str(Path(weights_path))
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        model = YOLO(weights_path)
    elif pretrained:
        model = YOLO(AVAILABLE_MODELS[model_name]["weights"])
    else:
        # Create a new model from YAML config (untrained)
        model = YOLO(AVAILABLE_MODELS[model_name]["weights"])

    return model


def get_model_info(model: YOLO) -> Dict[str, Any]:
    """
    Extract summary information from a YOLO model.

    Args:
        model: Ultralytics YOLO model instance.

    Returns:
        Dictionary with model info (name, parameters, size, classes).
    """
    import torch

    info = {
        "model_type": type(model).__name__,
        "num_classes": len(model.names) if hasattr(model, "names") else None,
        "class_names": model.names if hasattr(model, "names") else None,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    # Estimate model size by counting parameters
    total_params = info["num_parameters"]
    # FP32 = 4 bytes per parameter
    info["size_mb"] = (total_params * 4) / (1024 * 1024)

    return info


def list_available_models() -> list:
    """
    Return a list of all available model variants with their specs.

    Returns:
        List of dicts with model name and specifications.
    """
    return [
        {
            "name": name,
            "params_m": specs["params_m"],
            "size_mb": specs["size_mb"],
        }
        for name, specs in AVAILABLE_MODELS.items()
    ]
