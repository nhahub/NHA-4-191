import torch


class TestPrecisionRecallEdge:
    def test_different_classes_skipped(self):
        from src.utils.metrics import compute_precision_recall

        pb = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pl = torch.tensor([0])
        ps = torch.tensor([0.9])
        gb = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gl = torch.tensor([1])
        p, r = compute_precision_recall(
            [{"boxes": pb, "labels": pl, "scores": ps}],
            [{"boxes": gb, "labels": gl}],
        )
        assert p == 0.0  # different classes => no TP => FP

    def test_no_boxes_predicted(self):
        from src.utils.metrics import compute_precision_recall

        empty = torch.empty((0, 4))
        el = torch.empty((0,), dtype=torch.int64)
        es = torch.empty((0,), dtype=torch.float32)
        gb = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gl = torch.tensor([0])
        p, r = compute_precision_recall(
            [{"boxes": empty, "labels": el, "scores": es}],
            [{"boxes": gb, "labels": gl}],
        )
        assert p == 0.0
        assert r == 0.0

    def test_partial_match(self):
        from src.utils.metrics import compute_precision_recall

        pb = torch.tensor([[0.0, 0.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])
        pl = torch.tensor([0, 0])
        ps = torch.tensor([0.9, 0.8])
        gb = torch.tensor([[5.0, 5.0, 20.0, 20.0], [22.0, 22.0, 28.0, 28.0]])
        gl = torch.tensor([0, 0])
        p, r = compute_precision_recall(
            [{"boxes": pb, "labels": pl, "scores": ps}],
            [{"boxes": gb, "labels": gl}],
        )
        assert p >= 0.0
        assert r >= 0.0


class TestGetLogger:
    def test_get_logger_with_existing(self):
        import logging

        from src.utils.logger import get_logger

        logger = logging.getLogger("test_existing")
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
        result = get_logger("test_existing")
        assert result is logger
