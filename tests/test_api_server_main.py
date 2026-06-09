import sys
from unittest.mock import MagicMock, patch


class TestMain:
    def test_main_parse_only(self):
        from src.models.api_server import main

        test_args = ["prog", "--port", "9000", "--host", "127.0.0.1", "--weights", "/fake.pt", "--device", "cpu"]
        with patch.object(sys, "argv", test_args):
            with patch("src.models.api_server.uvicorn.run") as mock_uvicorn:
                result = main()
                assert result == 0
                mock_uvicorn.assert_called_once()

    def test_main_failure(self):
        from src.models.api_server import main

        test_args = ["prog", "--port", "9000", "--host", "127.0.0.1", "--weights", "/fake.pt"]
        with patch.object(sys, "argv", test_args):
            with patch("src.models.api_server.uvicorn.run", side_effect=RuntimeError("fail")):
                result = main()
                assert result == 1


class TestDrawBoxesFromDetections:
    def test_empty_detections(self):
        import numpy as np

        from src.models.api_server import draw_boxes_from_detections

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = draw_boxes_from_detections(img, [])
        assert np.array_equal(img, result)


class TestCleanupTrackers:
    def test_cleanup(self):
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

    def test_no_idle(self):
        import time

        from src.models.api_server import cleanup_trackers

        t = MagicMock()
        t.last_update = time.monotonic()
        trackers = {"t1": t}
        cleanup_trackers(trackers, max_idle_seconds=60)
        assert len(trackers) == 1


class TestTrackState:
    def test_defaults(self):
        import numpy as np

        from src.models.api_server import TrackState

        ts = TrackState(track_id=1, class_name="V", bbox=np.array([0, 0, 1, 1]), confidence=0.9)
        assert ts.missed == 0
        assert ts.hits == 1
