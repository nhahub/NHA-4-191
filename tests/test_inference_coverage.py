import sys
from unittest.mock import MagicMock, patch

import pytest


class TestInferenceSetupLogging:
    def test_setup_logging_runs(self):
        from src.models.inference import setup_logging

        with patch("src.utils.setup_logging") as mock:
            setup_logging(verbose=True, quiet=False)
            assert mock.called


class TestInferenceLoadModel:
    def test_load_model_not_found(self):
        from src.models.inference import load_model

        with pytest.raises(FileNotFoundError, match="not found"):
            load_model("/nonexistent.pt")

    def test_load_model_success(self, tmp_path):
        from src.models.inference import load_model

        w = tmp_path / "model.pt"
        w.write_text("dummy")
        with patch("src.models.inference.YOLO") as mock_yolo:
            m = MagicMock()
            m.names = {0: "test"}
            mock_yolo.return_value = m
            model = load_model(str(w), device="cpu")
            assert model is not None


class TestInferencePredict:
    def test_predict_source_not_found(self):
        from src.models.inference import predict

        with pytest.raises(FileNotFoundError):
            predict(MagicMock(), "/nonexistent")

    def test_predict_source_found(self, tmp_path):
        from src.models.inference import predict

        src = tmp_path / "img.jpg"
        src.write_text("fake")
        model = MagicMock()
        model.predict.return_value = []
        results = predict(model, str(src), conf=0.5, iou=0.6, imgsz=320)
        assert results == []


class TestInferenceMain:
    def test_main_runs(self):
        from src.models.inference import main

        test_args = ["prog", "--weights", "/nonexistent.pt", "--source", "/nonexistent.jpg"]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1  # error exit
