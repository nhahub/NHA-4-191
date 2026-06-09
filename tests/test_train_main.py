import sys
from unittest.mock import MagicMock, patch


class TestMain:
    def test_dry_run(self, tmp_path):
        from train import main

        test_args = ["prog", "--config", "nonexistent.yaml", "--dry-run"]
        with patch.object(sys, "argv", test_args):
            with patch("train.load_config") as mock_load:
                mock_load.return_value = {"model": {"name": "test"}, "training": {}}
                result = main()
                assert result == 0

    def test_list_models(self):
        from train import main

        test_args = ["prog", "--list-models"]
        with patch.object(sys, "argv", test_args):
            with patch("train.list_models") as mock_list:  # noqa: F841
                result = main()
                assert result == 0


class TestHelpers:
    def test_parse_args(self):
        from train import parse_args

        test_args = [
            "prog",
            "--config",
            "custom.yaml",
            "--project-root",
            "/tmp",
            "--data",
            "/data.yaml",
            "--epochs",
            "50",
            "--batch-size",
            "8",
            "--resume",
            "/ckpt.pt",
            "--device",
            "1",
            "--log-file",
            "/tmp/log.txt",
            "--verbose",
            "--dry-run",
            "--list-models",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.config == "custom.yaml"
        assert args.epochs == 50
        assert args.verbose is True

    def test_parse_args_defaults(self):
        from train import parse_args

        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert args.config == "configs/training.yaml"
        assert args.epochs is None
        assert args.verbose is False

    def test_apply_overrides(self):
        from train import apply_overrides

        config = {
            "training": {"epochs": 10},
            "data": {"batch_size": 4, "imgsz": 640, "workers": 2},
            "device": {"device": "0"},
            "model": {"name": "test"},
        }
        args = MagicMock()
        args.epochs = 50
        args.batch_size = 8
        args.imgsz = None
        args.workers = None
        args.device = None
        args.data = None
        args.model = None
        args.weights = None
        args.no_pretrained = None
        result = apply_overrides(config, args)
        assert result["training"]["epochs"] == 50
        assert result["data"]["batch_size"] == 8

    def test_apply_overrides_data(self):
        from train import apply_overrides

        config = {"training": {}, "data": {"imgsz": 640, "workers": 2}, "device": {"device": "0"}, "model": {}}
        args = MagicMock()
        args.epochs = None
        args.batch_size = None
        args.imgsz = None
        args.workers = None
        args.device = None
        args.data = "/new/data.yaml"
        args.model = None
        args.weights = None
        args.no_pretrained = None
        result = apply_overrides(config, args)
        assert result["data"]["yaml_path"] == "/new/data.yaml"

    def test_setup_log_file(self, tmp_path):
        from train import setup_log_file

        f, stream = setup_log_file(str(tmp_path / "test.log"), tmp_path)
        assert f == tmp_path / "test.log"
        assert stream is not None
        stream.close()

    def test_setup_log_file_none(self, tmp_path):
        from train import setup_log_file

        f, stream = setup_log_file(None, tmp_path)
        assert f is not None
        assert f.name.startswith("train_")
        stream.close()
