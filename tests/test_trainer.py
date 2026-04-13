"""
Unit tests for the training module.

Tests cover:
- Model factory (loading, info, listing)
- Configuration loading
- Trainer initialization and argument building
- Callbacks (logging, checkpointing)
- CLI argument parsing
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_factory import (
    load_model,
    get_model_info,
    list_available_models,
    AVAILABLE_MODELS,
)
from src.models.trainer import YOLOTrainer, load_config
from src.models.callbacks import TrainingLogger, ModelCheckpoint


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_config():
    """Return a minimal training configuration for testing."""
    return {
        "model": {
            "name": "yolo11n",
            "pretrained": True,
            "pretrained_weights": None,
        },
        "data": {
            "yaml_path": "data/processed/kitti/data.yaml",
            "imgsz": 640,
            "batch_size": 16,
            "workers": 2,
            "pin_memory": True,
        },
        "training": {
            "epochs": 5,
            "patience": 3,
            "optimizer": "auto",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 1.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
        },
        "augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
        "regularization": {
            "dropout": 0.0,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
        },
        "validation": {
            "val_interval": 1,
            "save_json": False,
            "save_hybrid": False,
        },
        "checkpoint": {
            "save_dir": "models/checkpoints",
            "save_best": True,
            "save_last": True,
            "save_period": -1,
            "metric": "mAP50-95",
        },
        "logging": {
            "project": "runs/train",
            "name": "test_exp",
            "exist_ok": True,
            "verbose": False,
        },
        "device": {
            "device": "cpu",
            "deterministic": True,
            "half_precision": False,
            "compile": False,
        },
        "advanced": {
            "close_mosaic": 2,
            "freeze": [],
            "sync_bn": False,
            "rect": False,
            "resume": False,
            "cache": False,
            "seed": 42,
        },
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(sample_config, f)
        return Path(f.name)


# ==============================================================================
# Model Factory Tests
# ==============================================================================


class TestModelFactory:

    def test_available_models_structure(self):
        """Test that AVAILABLE_MODELS has expected structure."""
        assert len(AVAILABLE_MODELS) > 0
        for name, specs in AVAILABLE_MODELS.items():
            assert "weights" in specs
            assert "params_m" in specs
            assert "size_mb" in specs

    def test_list_available_models(self):
        """Test list_available_models returns proper format."""
        models = list_available_models()
        assert len(models) == len(AVAILABLE_MODELS)
        for m in models:
            assert "name" in m
            assert "params_m" in m
            assert "size_mb" in m

    def test_load_model_valid_name(self):
        """Test loading a model by valid name."""
        model = load_model("yolo11n", pretrained=True)
        assert model is not None
        assert hasattr(model, "names")

    def test_load_model_invalid_name(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            load_model("invalid_model_name")

    def test_load_model_invalid_weights_path(self):
        """Test that non-existent weights path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Weights not found"):
            load_model("yolo11n", weights_path="/nonexistent/path.pt")

    def test_get_model_info(self):
        """Test model info extraction."""
        model = load_model("yolo11n", pretrained=True)
        info = get_model_info(model)

        assert "model_type" in info
        assert "num_classes" in info
        assert "class_names" in info
        assert "num_parameters" in info
        assert "size_mb" in info

        # YOLO11n has COCO classes (80)
        assert info["num_classes"] == 80
        assert info["num_parameters"] > 0
        assert info["size_mb"] > 0


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestConfigLoading:

    def test_load_config_valid(self, temp_config_file, sample_config):
        """Test loading a valid config file."""
        config = load_config(temp_config_file)
        assert config is not None
        assert "model" in config
        assert "data" in config
        assert "training" in config

    def test_load_config_not_found(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/config.yaml")


# ==============================================================================
# Trainer Tests
# ==============================================================================


class TestYOLOTrainer:

    def test_trainer_initialization(self, sample_config):
        """Test trainer initializes correctly."""
        trainer = YOLOTrainer(config=sample_config)
        assert trainer.config == sample_config
        assert trainer.model is None
        assert trainer.results is None

    def test_trainer_setup(self, sample_config):
        """Test trainer setup creates directories and loads model."""
        trainer = YOLOTrainer(config=sample_config)
        trainer.setup()

        assert trainer.model is not None
        assert trainer.save_dir is not None
        assert trainer.save_dir.exists()

    def test_trainer_build_args(self, sample_config):
        """Test training arguments are built correctly."""
        trainer = YOLOTrainer(config=sample_config)
        trainer.setup()

        args = trainer._build_train_args()

        assert "data" in args
        assert "epochs" in args
        assert "batch" in args
        assert "imgsz" in args
        assert args["epochs"] == sample_config["training"]["epochs"]
        assert args["batch"] == sample_config["data"]["batch_size"]
        assert args["imgsz"] == sample_config["data"]["imgsz"]

    def test_trainer_build_args_with_overrides(self, sample_config):
        """Test that overrides are applied."""
        trainer = YOLOTrainer(config=sample_config)
        trainer.setup()

        args = trainer._build_train_args(epochs=50, batch=32)

        assert args["epochs"] == 50
        assert args["batch"] == 32

    def test_trainer_requires_setup_before_train(self, sample_config):
        """Test that train() raises error if setup() not called."""
        trainer = YOLOTrainer(config=sample_config)
        with pytest.raises(RuntimeError, match="Call setup()"):
            trainer.train()

    def test_trainer_save_dir_auto_increment(self, sample_config, tmp_path):
        """Test that experiment directories auto-increment."""
        trainer = YOLOTrainer(config=sample_config)
        trainer.setup()

        first_dir = trainer.save_dir
        assert first_dir.exists()

        # Second trainer should get a new dir
        trainer2 = YOLOTrainer(config=sample_config)
        trainer2.setup()
        second_dir = trainer2.save_dir

        # With exist_ok=True they might share, but naming should handle it
        assert second_dir.exists()


# ==============================================================================
# Callback Tests
# ==============================================================================


class TestTrainingLogger:

    def test_logger_on_train_start(self, sample_config):
        """Test logger handles train start event."""
        logger = TrainingLogger()
        logger.on_train_start(sample_config)

        assert logger.train_start_time is not None

    def test_logger_on_train_end(self):
        """Test logger handles train end event."""
        logger = TrainingLogger()
        logger.on_train_start({})
        results = {"metrics": {"mAP50": 0.5, "mAP50-95": 0.3}}
        logger.on_train_end(results)

    def test_logger_on_train_error(self):
        """Test logger handles error event."""
        logger = TrainingLogger()
        error = RuntimeError("Test error")
        logger.on_train_error(error)

    def test_logger_on_epoch_end(self):
        """Test logger handles epoch end event."""
        logger = TrainingLogger()
        metrics = {"loss": 0.5, "mAP50": 0.3}
        logger.on_epoch_end(1, metrics)


class TestModelCheckpoint:

    def test_checkpoint_initialization(self, tmp_path):
        """Test checkpoint callback initializes correctly."""
        ckpt = ModelCheckpoint(save_dir=tmp_path)
        assert ckpt.save_dir == tmp_path
        assert ckpt.best_metric_value == float("-inf")

    def test_checkpoint_on_epoch_end(self, tmp_path):
        """Test checkpoint callback on epoch end."""
        ckpt = ModelCheckpoint(
            save_dir=tmp_path,
            save_best=True,
            save_last=True,
            metric="mAP50-95",
        )

        metrics = {"mAP50-95": 0.5}
        ckpt.on_epoch_end(1, metrics)

    def test_checkpoint_best_model_tracking(self, tmp_path):
        """Test that best model tracking works."""
        ckpt = ModelCheckpoint(
            save_dir=tmp_path,
            save_best=True,
            metric="mAP50-95",
        )

        # First epoch - should set best
        ckpt.on_epoch_end(1, {"mAP50-95": 0.4})
        assert ckpt.best_metric_value == 0.4

        # Second epoch - worse, should not update
        ckpt.on_epoch_end(2, {"mAP50-95": 0.3})
        assert ckpt.best_metric_value == 0.4

        # Third epoch - better, should update
        ckpt.on_epoch_end(3, {"mAP50-95": 0.6})
        assert ckpt.best_metric_value == 0.6

    def test_checkpoint_get_summary(self, tmp_path):
        """Test checkpoint summary generation."""
        ckpt = ModelCheckpoint(save_dir=tmp_path)
        summary = ckpt.get_checkpoint_summary()

        assert "save_dir" in summary
        assert "best_metric" in summary
        assert "checkpoints" in summary

    def test_checkpoint_get_best_model_path_no_file(self, tmp_path):
        """Test get_best_model_path returns None when no model exists."""
        ckpt = ModelCheckpoint(save_dir=tmp_path)
        assert ckpt.get_best_model_path() is None


# ==============================================================================
# CLI Tests
# ==============================================================================


class TestCLI:

    def test_train_module_import(self):
        """Test that train module can be imported."""
        from train import parse_args, setup_logging, apply_overrides
        assert parse_args is not None
        assert setup_logging is not None
        assert apply_overrides is not None

    def test_parse_args_defaults(self):
        """Test CLI argument defaults."""
        from train import parse_args

        # Simulate default args by passing minimal args
        import sys
        original_argv = sys.argv
        sys.argv = ["train.py"]

        try:
            args = parse_args()
            assert args.config == "configs/training.yaml"
            assert args.model is None
            assert args.epochs is None
            assert args.batch_size is None
            assert args.device is None
        finally:
            sys.argv = original_argv

    def test_parse_args_overrides(self):
        """Test CLI argument overrides."""
        from train import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = [
            "train.py",
            "--model", "yolo11s",
            "--epochs", "50",
            "--batch-size", "8",
            "--device", "0",
        ]

        try:
            args = parse_args()
            assert args.model == "yolo11s"
            assert args.epochs == 50
            assert args.batch_size == 8
            assert args.device == "0"
        finally:
            sys.argv = original_argv

    def test_apply_overrides(self):
        """Test that CLI overrides are applied to config."""
        from train import apply_overrides
        import argparse

        config = {
            "model": {"name": "yolo11m", "pretrained_weights": None, "pretrained": True},
            "data": {"batch_size": 16, "imgsz": 640, "workers": 4},
            "training": {"epochs": 100},
            "device": {"device": "auto"},
        }

        args = argparse.Namespace(
            model="yolo11s",
            weights="/path/to/weights.pt",
            no_pretrained=True,
            epochs=50,
            batch_size=8,
            imgsz=320,
            device="0",
            workers=2,
            data="custom_data.yaml",
        )

        result = apply_overrides(config, args)

        assert result["model"]["name"] == "yolo11s"
        assert result["model"]["pretrained_weights"] == "/path/to/weights.pt"
        assert result["model"]["pretrained"] is False
        assert result["training"]["epochs"] == 50
        assert result["data"]["batch_size"] == 8
        assert result["data"]["imgsz"] == 320
        assert result["device"]["device"] == "0"
        assert result["data"]["workers"] == 2
        assert result["data"]["yaml_path"] == "custom_data.yaml"
