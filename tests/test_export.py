from unittest.mock import patch


@patch("ultralytics.YOLO")
def test_export_parse_args(mock_yolo):
    import sys

    from src.models.export import parse_args

    test_args = ["train.py", "--weights", "/fake.pt", "--format", "onnx"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
    assert "onnx" in args.format
    assert args.weights == "/fake.pt"
    assert args.output == "models/exports"


@patch("ultralytics.YOLO")
def test_export_parse_args_defaults(mock_yolo):
    import sys

    from src.models.export import parse_args

    with patch.object(sys, "argv", ["train.py", "--weights", "/fake.pt", "--format", "onnx"]):
        args = parse_args()
    assert args.imgsz == 640
    assert args.half is False
