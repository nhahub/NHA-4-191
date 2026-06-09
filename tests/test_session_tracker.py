class TestSessionTrackerDetailed:
    def test_bbox_smoothing(self):
        from src.models.api_server import SessionTracker

        t = SessionTracker(bbox_alpha=0.5, conf_alpha=0.5)
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 1.0, "class_name": "Vehicle"}]
        t.update(dets)
        dets2 = [{"bbox": [2, 2, 12, 12], "confidence": 0.8, "class_name": "Vehicle"}]
        r = t.update(dets2)
        assert r[0]["bbox"] != [2, 2, 12, 12]  # smoothed

    def test_track_confidence_decay_by_miss(self):
        from src.models.api_server import SessionTracker

        t = SessionTracker(max_missed=5)
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}]
        r1 = t.update(dets)
        r2 = t.update([])
        assert r2[0]["confidence"] < r1[0]["confidence"]

    def test_tracker_clears_after_max_missed(self):
        from src.models.api_server import SessionTracker

        t = SessionTracker(max_missed=1)
        t.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}])
        t.update([])
        r = t.update([])
        assert len(r) == 0


class TestSessionTrackerNoMatch:
    def test_different_class_no_match(self):
        from src.models.api_server import SessionTracker

        t = SessionTracker()
        t.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}])
        r = t.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Pedestrian"}])
        assert len(r) == 2  # new track created

    def test_iou_below_threshold(self):
        from src.models.api_server import SessionTracker

        t = SessionTracker(iou_threshold=0.9)
        t.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}])
        r = t.update([{"bbox": [5, 5, 15, 15], "confidence": 0.9, "class_name": "Vehicle"}])
        assert len(r) == 2
