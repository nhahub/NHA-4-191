class TestTrainerMethods:
    def test_build_checkpoint_args(self, tmp_path):
        from unittest.mock import MagicMock

        from src.models.trainer import YOLOTrainer

        mock = MagicMock()
        mock.save_dir = tmp_path
        a = YOLOTrainer._build_checkpoint_args(mock, {"save_period": 10})
        assert a["save_period"] == 10
        assert a["save"] is True

    def test_build_checkpoint_args_no_save_dir(self):
        from unittest.mock import MagicMock

        from src.models.trainer import YOLOTrainer

        mock = MagicMock()
        mock.save_dir = None
        a = YOLOTrainer._build_checkpoint_args(mock, {})
        assert a["project"] == "runs/train"
        assert a["name"] == "exp"

    def test_config_snapshot(self, tmp_path):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}, "training": {"epochs": 1}}, project_root=tmp_path)
        t.save_dir = tmp_path
        t._save_config_snapshot()
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_init_callbacks(self, tmp_path):
        from src.models.trainer import YOLOTrainer

        t = YOLOTrainer({"model": {"name": "test"}, "training": {"epochs": 1}}, project_root=tmp_path)
        t.save_dir = tmp_path
        t._init_callbacks()
        assert t.checkpointer is not None
