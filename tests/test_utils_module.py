from pathlib import Path

import pytest
import torch

from src.utils import (
    BENCHMARK_NUM_IMAGES,
    CLASS_COLORS,
    CLASS_COLORS_LIST,
    CUSTOM_CLASSES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    ensure_dir,
    env_or_default,
)
from src.utils.constants import env_path
from src.utils.exceptions import (
    APIError,
    ConfigError,
    DataError,
    InferenceError,
    ModelLoadError,
    RoadSenseError,
    retry,
)
from src.utils.metrics import box_iou, compute_precision_recall, yolo_txt_to_boxes_labels


class TestConstants:
    def test_default_confidence(self):
        assert DEFAULT_CONFIDENCE == 0.25

    def test_default_iou(self):
        assert DEFAULT_IOU == 0.45

    def test_class_colors(self):
        assert "Vehicle" in CLASS_COLORS
        assert "Pedestrian" in CLASS_COLORS
        assert "Cyclist" in CLASS_COLORS
        assert len(CLASS_COLORS_LIST) == 3

    def test_custom_classes(self):
        assert CUSTOM_CLASSES == ["Vehicle", "Pedestrian", "Cyclist"]

    def test_benchmark_constants(self):
        assert BENCHMARK_NUM_IMAGES == 50

    def test_env_or_default(self):
        assert env_or_default("NONEXISTENT_VAR_12345", "fallback") == "fallback"
        assert env_or_default("PATH", "fallback") != "fallback"

    def test_env_path(self):
        result = env_path("NONEXISTENT_VAR_12345", "/nonexistent/test")
        assert isinstance(result, Path)
        assert str(result) == "/nonexistent/test"


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(ConfigError, RoadSenseError)
        assert issubclass(ModelLoadError, RoadSenseError)
        assert issubclass(InferenceError, RoadSenseError)
        assert issubclass(DataError, RoadSenseError)
        assert issubclass(APIError, RoadSenseError)

    def test_api_error_status(self):
        err = APIError("bad request", status_code=400, detail="invalid image")
        assert err.status_code == 400
        assert err.detail == "invalid image"
        assert "bad request" in str(err)

    def test_retry_success(self):
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def works():
            nonlocal call_count
            call_count += 1
            return "done"

        assert works() == "done"
        assert call_count == 1

    def test_retry_eventually_succeeds(self):
        call_count = 0

        @retry(max_attempts=5, delay=0.01, backoff=1.0)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("transient")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_retry_exhausted(self):
        call_count = 0

        @retry(max_attempts=2, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise OSError("always fails")

        with pytest.raises(IOError):
            always_fails()
        assert call_count == 2


class TestMetrics:
    def test_yolo_txt_to_boxes_labels_missing(self, tmp_path):
        missing = tmp_path / "nonexistent.txt"
        boxes, labels = yolo_txt_to_boxes_labels(missing, 640, 480)
        assert len(boxes) == 0
        assert len(labels) == 0

    def test_yolo_txt_to_boxes_labels_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        boxes, labels = yolo_txt_to_boxes_labels(f, 640, 480)
        assert len(boxes) == 0

    def test_yolo_txt_to_boxes_labels_valid(self, tmp_path):
        f = tmp_path / "valid.txt"
        f.write_text("0 0.5 0.5 0.4 0.4\n1 0.3 0.3 0.1 0.1\n")
        boxes, labels = yolo_txt_to_boxes_labels(f, 640, 480)
        assert len(boxes) == 2
        assert labels[0].item() == 0
        assert labels[1].item() == 1
        assert boxes[0, 0] < boxes[0, 2]

    def test_yolo_txt_to_boxes_labels_bad_line(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("0 0.5 0.5 0.4\n")  # only 4 fields
        boxes, labels = yolo_txt_to_boxes_labels(f, 640, 480)
        assert len(boxes) == 0

    def test_box_iou(self):
        b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        iou = box_iou(b1, b2)
        assert iou.shape == (1, 1)
        assert iou[0, 0] > 0
        assert iou[0, 0] < 1.0

    def test_box_iou_no_overlap(self):
        b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        iou = box_iou(b1, b2)
        assert iou[0, 0] == 0.0

    def test_compute_precision_recall(self):
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        label = torch.tensor([0])
        score = torch.tensor([0.9])
        preds = [{"boxes": box, "labels": label, "scores": score}]
        gts = [{"boxes": box, "labels": label}]
        prec, rec = compute_precision_recall(preds, gts)
        assert prec == 1.0
        assert rec == 1.0

    def test_compute_precision_recall_empty(self):
        empty = torch.empty((0, 4))
        el = torch.empty((0,), dtype=torch.int64)
        es = torch.empty((0,), dtype=torch.float32)
        preds = [{"boxes": empty, "labels": el, "scores": es}]
        gts = [{"boxes": empty, "labels": el}]
        prec, rec = compute_precision_recall(preds, gts)
        assert prec == 0.0
        assert rec == 0.0


class TestIOUtils:
    def test_ensure_dir(self, tmp_path):
        d = tmp_path / "new" / "deep" / "dir"
        assert not d.exists()
        ensure_dir(d)
        assert d.exists()
        assert d.is_dir()

    def test_ensure_dir_exists(self, tmp_path):
        ensure_dir(tmp_path)
        assert tmp_path.exists()
