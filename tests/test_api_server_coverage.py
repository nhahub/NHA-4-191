import numpy as np
import pytest


class TestSessionTracker:
    def test_tracker_init(self):
        from src.models.api_server import SessionTracker

        tracker = SessionTracker(iou_threshold=0.3, max_missed=5)
        assert tracker.iou_threshold == 0.3
        assert tracker.max_missed == 5

    def test_iou_calculation(self):
        from src.models.api_server import SessionTracker

        iou = SessionTracker._iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10]))
        assert iou == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        from src.models.api_server import SessionTracker

        iou = SessionTracker._iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]))
        assert iou == 0.0

    def test_iou_zero_area(self):
        from src.models.api_server import SessionTracker

        iou = SessionTracker._iou(np.array([0, 0, 0, 0]), np.array([0, 0, 10, 10]))
        assert iou == 0.0

    def test_tracker_update_new_detection(self):
        from src.models.api_server import SessionTracker

        tracker = SessionTracker()
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}]
        result = tracker.update(dets)
        assert len(result) == 1
        assert result[0]["track_id"] == 1

    def test_tracker_update_same_detection(self):
        from src.models.api_server import SessionTracker

        tracker = SessionTracker(iou_threshold=0.5)
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}]
        result1 = tracker.update(dets)
        result2 = tracker.update(dets)
        assert result2[0]["track_id"] == result1[0]["track_id"]

    def test_tracker_missed_frames(self):
        from src.models.api_server import SessionTracker

        tracker = SessionTracker(max_missed=2)
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}]
        tracker.update(dets)
        result = tracker.update([])
        assert len(result) == 1
        assert result[0]["confidence"] < 0.9

    def test_tracker_max_missed_drops(self):
        from src.models.api_server import SessionTracker

        tracker = SessionTracker(max_missed=2)
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}]
        tracker.update(dets)
        tracker.update([])
        tracker.update([])
        tracker.update([])
        result = tracker.update([])
        assert len(result) == 0

    def test_tracker_different_class(self):
        from src.models.api_server import SessionTracker

        tracker = SessionTracker()
        tracker.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}])
        result = tracker.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Pedestrian"}])
        assert len(result) == 2


class TestDetectEndpointValidation:
    def test_detect_response_model(self):
        from src.models.api_server import Detection, DetectionResponse

        det = Detection(class_name="Vehicle", confidence=0.9, bbox=[0, 0, 10, 10], track_id=1)
        assert det.class_name == "Vehicle"
        resp = DetectionResponse(
            success=True, detections=[det], annotated_image=None, inference_time_ms=15.0, message="OK"
        )
        assert resp.success is True
        assert len(resp.detections) == 1


class TestTrackState:
    def test_track_state_defaults(self):
        import numpy as np

        from src.models.api_server import TrackState

        t = TrackState(track_id=1, class_name="Vehicle", bbox=np.array([0, 0, 10, 10]), confidence=0.9)
        assert t.missed == 0
        assert t.hits == 1

    def test_track_state_custom(self):
        import numpy as np

        from src.models.api_server import TrackState

        t = TrackState(
            track_id=2, class_name="Pedestrian", bbox=np.array([5, 5, 15, 15]), confidence=0.8, missed=3, hits=5
        )
        assert t.missed == 3
        assert t.hits == 5
