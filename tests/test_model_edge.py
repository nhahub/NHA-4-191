from unittest.mock import MagicMock

import pytest

from src.models.model_factory import AVAILABLE_MODELS, get_model_info, list_available_models


def test_available_models():
    assert "yolo11n" in AVAILABLE_MODELS
    assert "yolo11m" in AVAILABLE_MODELS
    assert "yolov8n" in AVAILABLE_MODELS


def test_list_models():
    models = list_available_models()
    assert len(models) > 0
    assert "name" in models[0]
    assert "params_m" in models[0]


def test_get_model_info():
    from unittest.mock import MagicMock

    model = MagicMock()
    model.names = {0: "Vehicle", 1: "Pedestrian"}
    param = MagicMock()
    param.numel.return_value = 1000000
    model.parameters.return_value = [param]

    info = get_model_info(model)
    assert info["num_classes"] == 2
    assert info["size_mb"] > 0


def test_load_config_not_found(tmp_path):
    from src.models.trainer import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_callbacks(tmp_path):
    from src.models.callbacks import ModelCheckpoint, TrainingLogger

    logger = TrainingLogger()
    assert logger.train_start_time is None

    config = {
        "model": {"name": "yolo11m"},
        "training": {"epochs": 10},
        "data": {"yaml_path": "data.yaml", "batch_size": 16, "imgsz": 640},
        "device": {"device": "0"},
    }
    logger.on_train_start(config)
    assert logger.train_start_time is not None

    logger.on_epoch_end(1, {"loss": 0.5})
    logger.on_train_end({"metrics": {"mAP50": 0.9}})

    ckpt = ModelCheckpoint(tmp_path, save_best=True, save_last=True)
    assert ckpt.get_best_model_path() is None
    ckpt.on_epoch_end(1, {"mAP50-95": 0.5})
    summary = ckpt.get_checkpoint_summary()
    assert summary["best_metric"] is None or summary["best_metric"] > 0


@pytest.fixture
def sample_trainer(tmp_path):
    from src.models.trainer import YOLOTrainer

    config = {
        "model": {"name": "yolo11n", "pretrained": True},
        "data": {"yaml_path": "tests/data.yaml", "imgsz": 640, "batch_size": 2},
        "training": {"epochs": 1, "optimizer": "auto"},
        "augmentation": {},
        "regularization": {},
        "validation": {"val_interval": 1},
        "device": {"device": "cpu"},
        "advanced": {"seed": 42},
        "logging": {"verbose": True},
        "checkpoint": {"save_period": -1},
    }
    return YOLOTrainer(config=config, project_root=tmp_path)


def test_trainer_init(sample_trainer):
    assert sample_trainer.config["model"]["name"] == "yolo11n"
    assert sample_trainer.project_root is not None


def test_trainer_section_builders(tmp_path):
    from src.models.trainer import YOLOTrainer

    args = YOLOTrainer._build_data_args({"imgsz": 640, "batch_size": 4}, str(tmp_path / "data.yaml"))
    assert args["imgsz"] == 640
    assert args["batch"] == 4

    args = YOLOTrainer._build_train_hyperparams({"epochs": 50})
    assert args["epochs"] == 50

    args = YOLOTrainer._build_validation_args({"val_interval": 2})
    assert args["val"] is True
    args = YOLOTrainer._build_validation_args({"val_interval": 0})
    assert args["val"] is False

    args = YOLOTrainer._build_device_args({"device": "cpu"})
    assert args["device"] == "cpu"

    args = YOLOTrainer._build_regularization_args({"dropout": 0.2, "box": 5.0})
    assert args["dropout"] == 0.2
    assert args["box"] == 5.0

    args = YOLOTrainer._build_advanced_args({"seed": 99, "cache": True})
    assert args["seed"] == 99
    assert args["cache"] is True

    args = YOLOTrainer._build_logging_args({"verbose": False, "exist_ok": True})
    assert args["verbose"] is False
    assert args["exist_ok"] is True


def test_trainer_section_builders_empty_configs(tmp_path):
    from src.models.trainer import YOLOTrainer

    args = YOLOTrainer._build_data_args({}, str(tmp_path / "data.yaml"))
    assert args["imgsz"] == 640

    args = YOLOTrainer._build_train_hyperparams({})
    assert args["epochs"] == 100

    args = YOLOTrainer._build_device_args({})
    assert args["device"] == "0"

    args = YOLOTrainer._build_augmentation_args({})
    assert args["hsv_h"] == 0.015


def test_trainer_build_checkpoint_args(tmp_path):
    from src.models.trainer import YOLOTrainer

    trainer = MagicMock()
    trainer.save_dir = tmp_path
    args = YOLOTrainer._build_checkpoint_args(trainer, {"save_period": 5})
    assert args["save_period"] == 5
    assert args["save"] is True


class TestPredictSource:
    def test_video_file_path(self):
        from src.models.realtime import parse_source

        result = parse_source("video.mp4")
        assert result == ("video.mp4", -1)

    def test_camera_index(self):
        from src.models.realtime import parse_source

        result = parse_source("0")
        assert result[0] is None
        assert result[1] == 0

    def test_camera_index_str(self):
        from src.models.realtime import parse_source

        result = parse_source("2")
        assert result[0] is None
        assert result[1] == 2


def test_trainer_build_data_args(tmp_path):
    from src.models.trainer import YOLOTrainer

    args = YOLOTrainer._build_data_args({"imgsz": 640, "batch_size": 4}, str(tmp_path / "data.yaml"))
    assert args["imgsz"] == 640
    assert args["batch"] == 4
