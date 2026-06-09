import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_yolo():
    with patch("src.models.export.YOLO") as mock:
        model = MagicMock()
        mock.return_value = model
        yield model


class TestExportMain:
    def test_main_success(self, mock_yolo, tmp_path):
        from src.models.export import main

        weights = tmp_path / "model.pt"
        weights.write_text("dummy")
        test_args = ["prog", "--weights", str(weights), "--format", "onnx"]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 0

    def test_main_export_failure(self, mock_yolo, tmp_path):
        from src.models.export import main

        mock_yolo.export.side_effect = RuntimeError("fail")
        weights = tmp_path / "model.pt"
        weights.write_text("dummy")
        test_args = ["prog", "--weights", str(weights), "--format", "onnx"]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1


class TestExportModel:
    def test_export_onnx(self, mock_yolo, tmp_path):
        from src.models.export import export_model

        mock_yolo.export.return_value = "/tmp/model.onnx"
        result = export_model(mock_yolo, formats=["onnx"], output_dir=tmp_path)
        assert "onnx" in result

    def test_export_tflite(self, mock_yolo, tmp_path):
        from src.models.export import export_model

        mock_yolo.export.return_value = "/tmp/model.tflite"
        result = export_model(mock_yolo, formats=["tflite"], output_dir=tmp_path)
        assert "tflite" in result

    def test_export_multiple(self, mock_yolo, tmp_path):
        from src.models.export import export_model

        mock_yolo.export.side_effect = ["/tmp/a.onnx", "/tmp/b.ts"]
        result = export_model(mock_yolo, formats=["onnx", "torchscript"], output_dir=tmp_path)
        assert len(result) == 2

    def test_export_failure_continues(self, mock_yolo, tmp_path):
        from src.models.export import export_model

        mock_yolo.export.side_effect = [RuntimeError("onnx fail"), "/tmp/b.ts"]
        result = export_model(mock_yolo, formats=["onnx", "torchscript"], output_dir=tmp_path)
        assert "onnx" not in result
        assert "torchscript" in result


class TestSetupLogging:
    def test_setup_verbose(self):
        from src.models.export import setup_logging

        result = setup_logging(verbose=True)
        assert result is None

    def test_setup_quiet(self):
        from src.models.export import setup_logging

        result = setup_logging(quiet=True)
        assert result is None
