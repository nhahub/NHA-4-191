import sys
from unittest.mock import patch


class TestInferenceParseArgs:
    def test_defaults(self):
        from src.models.inference import parse_args

        with patch.object(sys, "argv", ["prog", "--weights", "/fake.pt", "--source", "/fake.jpg"]):
            args = parse_args()
        assert args.weights == "/fake.pt"
        assert args.source == "/fake.jpg"
        assert args.output is None
        assert args.conf == 0.25
        assert args.device == ""

    def test_custom(self):
        from src.models.inference import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--weights",
                "/w.pt",
                "--source",
                "/s.jpg",
                "--output",
                "/out",
                "--conf",
                "0.5",
                "--device",
                "cpu",
                "--verbose",
            ],
        ):
            args = parse_args()
        assert args.output == "/out"
        assert args.conf == 0.5
        assert args.device == "cpu"
        assert args.verbose is True

    def test_iou_threshold(self):
        from src.models.inference import parse_args

        with patch.object(sys, "argv", ["prog", "--weights", "/w.pt", "--source", "/s.jpg", "--iou", "0.6"]):
            args = parse_args()
        assert args.iou == 0.6

    def test_imgsz(self):
        from src.models.inference import parse_args

        with patch.object(sys, "argv", ["prog", "--weights", "/w.pt", "--source", "/s.jpg", "--imgsz", "1280"]):
            args = parse_args()
        assert args.imgsz == 1280


class TestRealtimeParseArgs:
    def test_defaults(self):
        from src.models.realtime import parse_args

        with patch.object(sys, "argv", ["prog", "--weights", "/fake.pt"]):
            args = parse_args()
        assert args.weights == "/fake.pt"
        assert args.source == "0"
        assert args.output is None
        assert args.conf == 0.25

    def test_custom_source(self):
        from src.models.realtime import parse_args

        with patch.object(sys, "argv", ["prog", "--weights", "/w.pt", "--source", "2"]):
            args = parse_args()
        assert args.source == "2"

    def test_output(self):
        from src.models.realtime import parse_args

        with patch.object(sys, "argv", ["prog", "--weights", "/w.pt", "--output", "out.avi"]):
            args = parse_args()
        assert args.output == "out.avi"

    def test_no_view_flag(self):
        from src.models.realtime import parse_args

        with patch.object(sys, "argv", ["prog", "--weights", "/w.pt", "--no-view"]):
            args = parse_args()
        assert args.no_view is True

    def test_full_options(self):
        from src.models.realtime import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--weights",
                "/w.pt",
                "--source",
                "video.mp4",
                "--output",
                "out.avi",
                "--conf",
                "0.3",
                "--device",
                "cpu",
                "--no-view",
                "--verbose",
            ],
        ):
            args = parse_args()
        assert args.source == "video.mp4"
        assert args.output == "out.avi"
        assert args.conf == 0.3
        assert args.device == "cpu"
        assert args.no_view is True
        assert args.verbose is True
