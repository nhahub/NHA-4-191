import sys
from unittest.mock import MagicMock, patch


class TestMain:
    def test_main_starts_uvicorn(self):
        from src.models.api_server import main

        test_args = ["prog", "--port", "9000", "--weights", "/nonexistent.pt"]
        with patch.object(sys, "argv", test_args):
            with patch("src.models.api_server.uvicorn.run"):
                result = main()
                assert result == 0  # uvicorn.run mocked, returns immediately

    def test_main_uvicorn_exception(self):
        from src.models.api_server import main

        test_args = ["prog", "--port", "9000", "--weights", "/nonexistent.pt"]
        with patch.object(sys, "argv", test_args):
            with patch("src.models.api_server.uvicorn.run", side_effect=RuntimeError("fail")):
                result = main()
                assert result == 1


class TestLoadModel:
    def test_file_not_found(self):
        import pytest

        from src.models.api_server import load_model

        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent.pt")

    def test_load_with_device(self, tmp_path):
        from src.models.api_server import load_model

        w = tmp_path / "m.pt"
        w.write_text("x")
        with patch("src.models.api_server.YOLO") as mock_yolo:
            m = MagicMock()
            mock_yolo.return_value = m
            result = load_model(str(w), device="cpu")
            assert result is not None
            m.to.assert_called_once_with("cpu")


class TestSessionTrackerEdge:
    def test_multiple_tracks(self):
        from src.models.api_server import SessionTracker

        t = SessionTracker()
        dets = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"},
            {"bbox": [20, 20, 30, 30], "confidence": 0.8, "class_name": "Pedestrian"},
        ]
        result = t.update(dets)
        assert len(result) == 2
        assert result[0]["track_id"] != result[1]["track_id"]

    def test_confidence_decay(self):
        from src.models.api_server import SessionTracker

        t = SessionTracker(max_missed=5)
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "Vehicle"}]
        t.update(dets)
        r = t.update([])
        assert r[0]["confidence"] < 0.9
