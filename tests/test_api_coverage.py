from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class _CpuTensor:
    """Mock torch tensor that wraps numpy and supports .cpu().numpy() chain."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class TestLoadModel:
    def test_load_model_not_found(self):
        from src.models.api_server import load_model

        with pytest.raises(FileNotFoundError, match="not found"):
            load_model("/nonexistent/path.pt")

    @patch("src.models.api_server.YOLO")
    def test_load_model_success(self, mock_yolo):
        import tempfile

        from src.models.api_server import load_model

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            f.write(b"dummy")
            f.flush()
            model = load_model(f.name, device="cpu")
            assert model is not None
            mock_yolo.assert_called_once()


class TestResolveWeightsPath:
    def test_explicit_weights_exists(self, tmp_path):
        from src.models.api_server import resolve_weights_path

        w = tmp_path / "model.pt"
        w.write_text("dummy")
        result = resolve_weights_path(str(w), str(tmp_path))
        assert result == w

    def test_weights_not_found_raises(self, tmp_path):
        from src.models.api_server import resolve_weights_path

        with pytest.raises(FileNotFoundError):
            resolve_weights_path("nonexistent.pt", str(tmp_path))

    def test_weights_dir_not_found_raises(self, tmp_path):
        from src.models.api_server import resolve_weights_path

        missing_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_weights_path(None, str(missing_dir))


class TestExtractRawDetections:
    def test_empty_result(self):
        from src.models.api_server import extract_raw_detections

        class NoBoxes:
            pass

        dets = extract_raw_detections(NoBoxes(), {0: "test"})
        assert dets == []

    def test_with_boxes(self):
        from src.models.api_server import extract_raw_detections

        class MockBoxes:
            def __init__(self):
                self.xyxy = _CpuTensor(np.array([[10, 20, 100, 200]]))
                self.conf = _CpuTensor(np.array([0.9]))
                self.cls = _CpuTensor(np.array([0]))

            def __len__(self):
                return 1

        class MockResult:
            boxes = MockBoxes()

        dets = extract_raw_detections(MockResult(), {0: "Vehicle"})
        assert len(dets) == 1
        assert dets[0]["class_name"] == "Vehicle"
        assert dets[0]["confidence"] == 0.9

    def test_unknown_class(self):
        from src.models.api_server import extract_raw_detections

        class MockBoxes:
            def __init__(self):
                self.xyxy = _CpuTensor(np.array([[0, 0, 10, 10]]))
                self.conf = _CpuTensor(np.array([0.5]))
                self.cls = _CpuTensor(np.array([99]))

            def __len__(self):
                return 1

        class MockResult:
            boxes = MockBoxes()

        dets = extract_raw_detections(MockResult(), {})
        assert dets[0]["class_name"] == "class_99"


class TestDrawBoxesFromDetections:
    def test_empty_detections(self):
        from src.models.api_server import draw_boxes_from_detections

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_boxes_from_detections(img, [])
        assert np.array_equal(img, result)

    def test_with_detection(self):
        from src.models.api_server import draw_boxes_from_detections

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [{"bbox": [10, 20, 100, 120], "class_name": "Vehicle", "confidence": 0.9, "track_id": 1}]
        result = draw_boxes_from_detections(img, detections)
        assert not np.array_equal(img, result)
        assert result[20, 10].tolist() != [0, 0, 0]

    def test_with_track_id(self):
        from src.models.api_server import draw_boxes_from_detections

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = [{"bbox": [10, 10, 50, 50], "class_name": "Pedestrian", "confidence": 0.8, "track_id": 5}]
        result = draw_boxes_from_detections(img, detections)
        assert not np.array_equal(img, result)


class TestDrawBoxes:
    def test_empty_boxes(self):
        from src.models.api_server import draw_boxes

        img = np.zeros((100, 100, 3), dtype=np.uint8)

        class NoBoxes:
            pass

        result = draw_boxes(img, NoBoxes(), {0: "test"})
        assert np.array_equal(img, result)

    def test_with_boxes(self):
        from src.models.api_server import draw_boxes

        img = np.zeros((200, 200, 3), dtype=np.uint8)

        class MockBoxes:
            def __init__(self):
                self.xyxy = _CpuTensor(np.array([[10, 20, 100, 120]]))
                self.conf = _CpuTensor(np.array([0.85]))
                self.cls = _CpuTensor(np.array([0]))

            def __len__(self):
                return 1

        class MockResult:
            boxes = MockBoxes()

        result = draw_boxes(img, MockResult(), {0: "Vehicle"})
        assert not np.array_equal(img, result)


class TestEncodeImage:
    def test_encode(self):
        from src.models.api_server import encode_image_to_base64

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        encoded = encode_image_to_base64(img)
        assert encoded.startswith("data:image/jpeg;base64,")
        assert len(encoded) > 100


class TestCleanupTrackers:
    def test_cleanup_stale(self):
        import time

        from src.models.api_server import cleanup_trackers

        stale = MagicMock()
        stale.last_update = time.monotonic() - 300

        fresh = MagicMock()
        fresh.last_update = time.monotonic()

        trackers = {"stale": stale, "fresh": fresh}
        cleanup_trackers(trackers, max_idle_seconds=60)
        assert "stale" not in trackers
        assert "fresh" in trackers

    def test_no_stale(self):
        import time

        from src.models.api_server import cleanup_trackers

        fresh = MagicMock()
        fresh.last_update = time.monotonic()

        trackers = {"fresh": fresh}
        cleanup_trackers(trackers, max_idle_seconds=60)
        assert len(trackers) == 1
