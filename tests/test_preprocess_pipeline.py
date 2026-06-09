import cv2
import numpy as np
import pytest


class TestProcessImageLabelPair:
    def test_missing_image(self, tmp_path):
        from src.data.preprocess_dataset import process_image_label_pair

        config = {
            "image_processing": {"target_size": [640, 480], "save_format": "png"},
            "label_conversion": {"class_mapping": {"Car": 0}, "exclude_classes": []},
        }
        result = process_image_label_pair(
            tmp_path / "nonexistent.png",
            tmp_path / "nonexistent.txt",
            tmp_path / "out.png",
            tmp_path / "out.txt",
            config,
        )
        assert result["success"] is False
        assert "Failed to load image" in result["error"]

    def test_successful_processing(self, tmp_path):
        from src.data.preprocess_dataset import process_image_label_pair

        img_dir = tmp_path / "img"
        lbl_dir = tmp_path / "lbl"
        out_img = tmp_path / "out" / "images"
        out_lbl = tmp_path / "out" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        out_img.mkdir(parents=True)
        out_lbl.mkdir(parents=True)

        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.png"), img)
        (lbl_dir / "test.txt").write_text("Car 0.0 0 0.0 10 20 50 80 0 0 0 0 0 0 0 0\n")

        config = {
            "image_processing": {"target_size": [320, 240], "save_format": "png"},
            "label_conversion": {"class_mapping": {"Car": 0}, "exclude_classes": [], "min_bbox_size": 0.0},
            "output": {"create_yolo_yaml": False},
        }
        result = process_image_label_pair(
            img_dir / "test.png", lbl_dir / "test.txt", out_img / "test.png", out_lbl / "test.txt", config
        )
        assert result["success"] is True
        assert result["num_objects"] == 1

    def test_with_normalize(self, tmp_path):
        from src.data.preprocess_dataset import process_image_label_pair

        img_dir = tmp_path / "img"
        lbl_dir = tmp_path / "lbl"
        out_img = tmp_path / "out" / "images"
        out_lbl = tmp_path / "out" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        out_img.mkdir(parents=True)
        out_lbl.mkdir(parents=True)

        cv2.imwrite(str(img_dir / "t.png"), np.full((100, 200, 3), 128, dtype=np.uint8))
        (lbl_dir / "t.txt").write_text("Car 0.0 0 0.0 10 20 50 80 0 0 0 0 0 0 0 0\n")

        config = {
            "image_processing": {"target_size": [320, 240], "save_format": "png", "normalize_pixels": True},
            "label_conversion": {"class_mapping": {"Car": 0}, "exclude_classes": [], "min_bbox_size": 0.0},
            "output": {"create_yolo_yaml": False},
        }
        result = process_image_label_pair(
            img_dir / "t.png", lbl_dir / "t.txt", out_img / "t.png", out_lbl / "t.txt", config
        )
        assert result["success"] is True

    def test_exclude_class(self, tmp_path):
        from src.data.preprocess_dataset import process_image_label_pair

        img_dir = tmp_path / "img"
        lbl_dir = tmp_path / "lbl"
        out_img = tmp_path / "out" / "images"
        out_lbl = tmp_path / "out" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        out_img.mkdir(parents=True)
        out_lbl.mkdir(parents=True)

        cv2.imwrite(str(img_dir / "t.png"), np.zeros((100, 200, 3), dtype=np.uint8))
        (lbl_dir / "t.txt").write_text("DontCare 0.0 0 0.0 10 20 50 80 0 0 0 0 0 0 0 0\n")

        config = {
            "image_processing": {"target_size": [320, 240], "save_format": "png"},
            "label_conversion": {"class_mapping": {"Car": 0}, "exclude_classes": ["DontCare"], "min_bbox_size": 0.0},
            "output": {"create_yolo_yaml": False},
        }
        result = process_image_label_pair(
            img_dir / "t.png", lbl_dir / "t.txt", out_img / "t.png", out_lbl / "t.txt", config
        )
        assert result["success"] is True
        assert result["num_objects"] == 0


class TestPreprocessDataset:
    def test_missing_img_dir(self, tmp_path):
        from src.data.preprocess_dataset import preprocess_dataset

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "input: {raw_data_dir: raw, image_subdir: img, label_subdir: lbl}\n"
            "output: {processed_dir: proc, create_yolo_yaml: false}\n"
            "split: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1, "
            "random_seed: 42, shuffle: true}\n"
            "image_processing: {target_size: [640, 480], save_format: png}\n"
            "label_conversion: {class_mapping: {Car: 0}, exclude_classes: []}\n"
            "processing: {show_progress: false, skip_on_error: true}\n"
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            preprocess_dataset(config_path=str(cfg), project_root=str(tmp_path))

    def test_no_images(self, tmp_path):
        from src.data.preprocess_dataset import preprocess_dataset

        (tmp_path / "raw" / "img").mkdir(parents=True)
        (tmp_path / "raw" / "lbl").mkdir(parents=True)
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "input: {raw_data_dir: raw, image_subdir: img, label_subdir: lbl}\n"
            "output: {processed_dir: proc, create_yolo_yaml: false}\n"
            "split: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1, "
            "random_seed: 42, shuffle: true}\n"
            "image_processing: {target_size: [640, 480], save_format: png}\n"
            "label_conversion: {class_mapping: {Car: 0}, exclude_classes: []}\n"
            "processing: {show_progress: false, skip_on_error: true}\n"
        )
        with pytest.raises(ValueError, match="No images found"):
            preprocess_dataset(config_path=str(cfg), project_root=str(tmp_path))


class TestCreateYoloConfig:
    def test_creates_yaml(self, tmp_path):
        from src.data.preprocess_dataset import create_yolo_config

        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        cfg = {"label_conversion": {"class_mapping": {"Car": 0}}, "yolo_config": {"names": {"0": "Car"}}, "output": {}}
        create_yolo_config(proc_dir, cfg, tmp_path)
        yaml_file = proc_dir / "data.yaml"
        assert yaml_file.exists()
        content = yaml_file.read_text()
        assert "Car" in content

    def test_absolute_path_fallback(self, tmp_path):
        from src.data.preprocess_dataset import create_yolo_config

        proc_dir = tmp_path / "proc"
        proc_dir.mkdir()
        cfg = {"label_conversion": {"class_mapping": {"Car": 0}}, "yolo_config": {"names": {"0": "Car"}}, "output": {}}
        create_yolo_config(proc_dir, cfg, tmp_path)
        assert (proc_dir / "data.yaml").exists()


class TestMain:
    def test_main_parse(self):
        import sys
        from unittest.mock import patch

        from src.data.preprocess_dataset import main

        with patch.object(sys, "argv", ["prog", "--config", "/nonexistent.yaml"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0 or exc.value.code == 1

    def test_main_missing_config(self):
        import sys
        from unittest.mock import patch

        from src.data.preprocess_dataset import main

        with patch.object(sys, "argv", ["prog", "--config", "/definitely/missing.yaml"]):
            with pytest.raises(SystemExit):
                main()
