import pytest


class TestMetricsEdge:
    def test_compute_precision_recall_no_match(self):
        import torch

        from src.utils.metrics import compute_precision_recall

        pb = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pl = torch.tensor([0])
        ps = torch.tensor([0.9])
        gb = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
        gl = torch.tensor([0])
        p, r = compute_precision_recall(
            [{"boxes": pb, "labels": pl, "scores": ps}],
            [{"boxes": gb, "labels": gl}],
        )
        assert p == 0.0
        assert r == 0.0

    def test_box_iou_multiple(self):
        import torch

        from src.utils.metrics import box_iou

        b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        b2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        iou = box_iou(b1, b2)
        assert iou.shape == (2, 1)


class TestLoggerEdge:
    def test_error_formatting(self):
        import logging

        from src.utils.logger import JSONFormatter

        fmt = JSONFormatter()
        record = logging.LogRecord("t", logging.ERROR, "", 0, "err", (), None)
        try:
            raise ValueError("test error")
        except ValueError:
            record.exc_info = (ValueError, ValueError("test error"), None)
        output = fmt.format(record)
        assert "exception" in output


class TestExceptionsEdge:
    def test_retry_rejects_wrong_exception(self):
        from src.utils.exceptions import retry

        call_count = 0

        @retry(max_attempts=2, delay=0.01, exceptions=(KeyError,))
        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("wrong type")

        with pytest.raises(ValueError):
            fn()
        assert call_count == 1
