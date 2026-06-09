from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSetupEdgeCases:
    def test_setup_no_save_dir(self):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}}, project_root=Path("/tmp"))
        with patch("src.models.trainer.load_model") as mock:
            mock.return_value = MagicMock()
            t.setup()

    def test_setup_with_experiment_name(self):
        from src.models.trainer import YOLOTrainer

        config = {"model": {"name": "test"}, "logging": {"experiment_name": "my_exp", "exist_ok": True}}
        t = YOLOTrainer(config, project_root=Path("/tmp"))
        with patch("src.models.trainer.load_model") as mock:
            mock.return_value = MagicMock()
            t.setup()


class TestTrainMethods:
    def test_train_no_model_raises(self):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}})
        t.model = None
        with pytest.raises(RuntimeError, match="not initialized"):
            t.train()

    def test_validate_no_model_raises(self):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}})
        t.model = None
        with pytest.raises(RuntimeError, match="not initialized"):
            t.validate()


class TestExportMethods:
    def test_export_with_model(self):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}})
        t.model = MagicMock()
        t.model.export.return_value = "/tmp/model.onnx"
        t.save_dir = Path("/tmp")
        result = t.export(format="onnx")
        assert result == "/tmp/model.onnx"


class TestInitCallbacks:
    def test_init_callbacks_sets_checkpointer(self, tmp_path):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer(
            {
                "model": {"name": "test"},
                "checkpoint": {"save_dir": "ckpts", "save_period": 5},
                "training": {"epochs": 10},
            },
            project_root=tmp_path,
        )
        t.save_dir = tmp_path
        t._init_callbacks()
        assert t.checkpointer is not None
