import cv2
import numpy as np
import pytest


class TestProcessImageLabelPair:
    def test_missing_label_file(self, tmp_path):
        from src.data.preprocess_dataset import process_image_label_pair

        img_dir = tmp_path / "img"
        img_dir.mkdir()
        cv2.imwrite(str(img_dir / "t.png"), np.zeros((100, 200, 3), dtype=np.uint8))
        cfg = {
            "image_processing": {"target_size": [320, 240], "save_format": "png"},
            "label_conversion": {"class_mapping": {}, "exclude_classes": [], "min_bbox_size": 0.0},
            "output": {"create_yolo_yaml": False},
        }
        result = process_image_label_pair(
            img_dir / "t.png",
            tmp_path / "nonexistent.txt",
            tmp_path / "out.png",
            tmp_path / "out.txt",
            cfg,
        )
        assert result["success"] is True  # process succeeds, just 0 objects
        assert result["num_objects"] == 0

    def test_exception_handled(self, tmp_path):
        from src.data.preprocess_dataset import process_image_label_pair

        cfg = {
            "image_processing": {"target_size": [320, 240], "save_format": "png"},
            "label_conversion": {"class_mapping": {"Car": 0}, "exclude_classes": [], "min_bbox_size": 0.0},
            "output": {"create_yolo_yaml": False},
        }
        result = process_image_label_pair(
            tmp_path / "t.png",
            tmp_path / "l.txt",
            tmp_path / "out.png",
            tmp_path / "out.txt",
            cfg,
        )
        assert result["success"] is False


class TestYoloConfigEdgeCases:
    def test_yaml_with_invalid_class_ids(self, tmp_path):
        from src.data.preprocess_dataset import create_yolo_config

        d = tmp_path / "p"
        d.mkdir()
        cfg = {
            "label_conversion": {"class_mapping": {"Car": 0, "Ped": 2}},
            "yolo_config": {"names": {"0": "Car", "2": "Ped"}},
        }
        with pytest.raises(ValueError, match="contiguous"):
            create_yolo_config(d, cfg, tmp_path)

    def test_yaml_no_names_in_config(self, tmp_path):
        from src.data.preprocess_dataset import create_yolo_config

        d = tmp_path / "p"
        d.mkdir()
        cfg = {"label_conversion": {"class_mapping": {}}, "yolo_config": {}}
        with pytest.raises(ValueError, match="No classes"):
            create_yolo_config(d, cfg, tmp_path)


class TestPreprocessDatasetMore:
    def test_no_label_dir(self, tmp_path):
        from src.data.preprocess_dataset import preprocess_dataset

        (tmp_path / "raw" / "img").mkdir(parents=True)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "input: {raw_data_dir: raw, image_subdir: img, label_subdir: lbl}\n"
            "output: {processed_dir: proc, create_yolo_yaml: false}\n"
            "split: {train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1, "
            "random_seed: 42, shuffle: true}\n"
            "image_processing: {target_size: [640, 480], save_format: png}\n"
            "label_conversion: {class_mapping: {Car: 0}, exclude_classes: [], min_bbox_size: 0.01}\n"
            "processing: {show_progress: false, skip_on_error: true}\n"
        )
        with pytest.raises(FileNotFoundError):
            preprocess_dataset(config_path=str(cfg), project_root=str(tmp_path))
