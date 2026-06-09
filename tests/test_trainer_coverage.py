from pathlib import Path
from unittest.mock import MagicMock


class TestExtractMetrics:
    def test_from_results_attr(self):
        from src.models.trainer import YOLOTrainer

        class R:
            results = {"mAP50": 0.9}

        m = YOLOTrainer._extract_metrics(MagicMock(), R())
        assert m["mAP50"] == 0.9

    def test_from_box(self):
        from src.models.trainer import YOLOTrainer

        class B:
            map50 = 0.85
            map = 0.65

        class R:
            box = B()

        m = YOLOTrainer._extract_metrics(MagicMock(), R())
        assert m["mAP50"] == 0.85
        assert m["mAP50-95"] == 0.65

    def test_empty(self):
        from src.models.trainer import YOLOTrainer

        m = YOLOTrainer._extract_metrics(MagicMock(), {})
        assert m == {}

    def test_from_train_results(self):
        from src.models.trainer import YOLOTrainer

        class MockTrainResults:
            results = {"mAP50": 0.88, "mAP50-95": 0.70}

        m = YOLOTrainer._extract_metrics(MagicMock(), MockTrainResults())
        assert "mAP50" in m


class TestSaveResults:
    def test_creates_json(self, tmp_path):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}}, project_root=tmp_path)
        t.save_dir = tmp_path
        t.results = {"metrics": {"mAP50": 0.9}, "save_dir": str(tmp_path), "epochs_trained": 10}
        t._save_results()
        assert (tmp_path / "results.json").exists()


class TestInit:
    def test_defaults(self):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "yolo11n"}})
        assert t.config["model"]["name"] == "yolo11n"
        assert t.model is None

    def test_with_root(self):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}}, project_root=Path("/tmp"))
        assert t.project_root == Path("/tmp")


class TestHelpers:
    def test_should_auto_preprocess_false(self):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}})
        assert t._should_auto_preprocess(Path("/nope.yaml")) is False


class TestSectionBuilders:
    def test_all_builders(self):
        from src.models.trainer import YOLOTrainer

        a = YOLOTrainer._build_data_args({}, "/d.yaml")
        assert a["imgsz"] == 640
        a = YOLOTrainer._build_train_hyperparams({})
        assert a["epochs"] == 100
        a = YOLOTrainer._build_augmentation_args({})
        assert a["hsv_h"] == 0.015
        a = YOLOTrainer._build_regularization_args({})
        assert a["dropout"] == 0.0
        a = YOLOTrainer._build_validation_args({})
        assert a["val"] is True  # Default: val_interval=1 => val=True
        a = YOLOTrainer._build_device_args({})
        assert a["device"] == "0"
        a = YOLOTrainer._build_advanced_args({})
        assert a["seed"] == 42
        a = YOLOTrainer._build_logging_args({})
        assert a["verbose"] is True


class TestTrainFunction:
    def test_signature(self):
        import inspect

        from src.models.trainer import train

        sig = inspect.signature(train)
        assert "config_path" in sig.parameters
