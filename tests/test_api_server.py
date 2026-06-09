from unittest.mock import patch

from src.mlops.performance_monitor import PerformanceMonitor


def test_performance_monitor():
    pm = PerformanceMonitor(window_size=10)
    stats = pm.get_stats()
    assert stats["total_requests"] == 0
    assert stats["error_rate"] == 0.0

    pm.record_request()
    pm.record_latency(15.0)
    pm.record_latency(25.0)
    pm.record_latency(35.0)
    pm.record_error()

    stats = pm.get_stats()
    assert stats["total_requests"] == 1
    assert stats["error_count"] == 1
    assert stats["error_rate"] == 1.0
    assert stats["latency_ms_avg"] == 25.0
    assert stats["latency_ms_p50"] == 25.0
    assert stats["uptime_seconds"] >= 0


def test_performance_monitor_empty():
    pm = PerformanceMonitor()
    stats = pm.get_stats()
    assert "latency_ms_avg" not in stats


def test_performance_monitor_log(caplog):
    import logging

    pm = PerformanceMonitor()
    pm.record_request()
    pm.record_latency(10.0)
    with caplog.at_level(logging.INFO):
        pm.log_stats()
        assert "Performance stats" in caplog.text


def test_parse_args():
    import sys

    from src.models.api_server import parse_args

    test_args = [
        "prog",
        "--port",
        "9000",
        "--host",
        "127.0.0.1",
        "--weights",
        "/fake.pt",
        "--device",
        "cpu",
        "--verbose",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
    assert args.port == 9000
    assert args.host == "127.0.0.1"
    assert args.weights == "/fake.pt"
    assert args.device == "cpu"
    assert args.verbose is True


def test_parse_args_defaults():
    import sys

    from src.models.api_server import parse_args

    with patch.object(sys, "argv", ["prog"]):
        args = parse_args()
    assert args.port == 8000
    assert args.host == "0.0.0.0"
    assert args.conf == 0.25
    assert args.disable_tracking is False


def test_encode_image_to_base64():
    import numpy as np

    from src.models.api_server import encode_image_to_base64

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    b64 = encode_image_to_base64(img)
    assert b64.startswith("data:image/jpeg;base64,")
    assert len(b64) > 100


def test_extract_raw_detections_empty():
    from src.models.api_server import extract_raw_detections

    class NoBoxes:
        pass

    result = NoBoxes()
    dets = extract_raw_detections(result, {0: "test"})
    assert dets == []
