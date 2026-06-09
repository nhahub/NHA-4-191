import os
from pathlib import Path


def env_or_default(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_path(key: str, default: str) -> Path:
    return Path(env_or_default(key, default))


DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
TRACKING_IOU = 0.35
TRACKING_MAX_MISSED = 3
TRACKING_BBOX_ALPHA = 0.7
TRACKING_CONF_ALPHA = 0.6

CLASS_COLORS = {
    "Vehicle": (0, 255, 0),
    "Pedestrian": (255, 0, 0),
    "Cyclist": (0, 0, 255),
}

CLASS_COLORS_LIST = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
]

CUSTOM_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]

BENCHMARK_IMAGE_SIZE = 640
BENCHMARK_NUM_IMAGES = 50
BENCHMARK_WARMUP = 1
BENCHMARK_REPEATS = 3
