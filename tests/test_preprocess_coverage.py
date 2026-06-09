from pathlib import Path

import numpy as np
import pytest

from src.data.preprocess_dataset import (
    filter_classes,
    filter_small_boxes,
    get_class_mapping,
    get_yolo_class_names,
    load_config,
    normalize_image,
    resize_image,
    setup_output_directories,
    split_dataset,
)


class TestLoadConfig:
    def test_load_config_valid(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("key: value\nnum: 42\n")
        result = load_config(str(cfg))
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_load_config_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "missing.yaml"))


class TestGetClassMapping:
    def test_basic(self):
        config = {"label_conversion": {"class_mapping": {"Car": 0, "Pedestrian": 1}}}
        assert get_class_mapping(config) == {"Car": 0, "Pedestrian": 1}

    def test_empty(self):
        config = {"label_conversion": {"class_mapping": {}}}
        assert get_class_mapping(config) == {}


class TestGetYoloClassNames:
    def test_from_yolo_config(self):
        config = {"yolo_config": {"names": {"0": "Car", "1": "Pedestrian"}}}
        assert get_yolo_class_names(config) == ["Car", "Pedestrian"]

    def test_from_class_mapping_fallback(self):
        config = {
            "yolo_config": {},
            "label_conversion": {"class_mapping": {"Car": 0, "Pedestrian": 1}},
        }
        assert get_yolo_class_names(config) == ["Car", "Pedestrian"]

    def test_from_class_mapping_merge(self):
        config = {
            "yolo_config": {},
            "label_conversion": {"class_mapping": {"Car": 0, "Pedestrian": 1, "Cyclist": 0}},
        }
        names = get_yolo_class_names(config)
        assert len(names) == 2  # Car (id=0) and Pedestrian (id=1)
        assert names[0] == "Car"  # First name mapped to id 0
        assert names[1] == "Pedestrian"

    def test_empty_raises(self):
        config = {"yolo_config": {}, "label_conversion": {"class_mapping": {}}}
        with pytest.raises(ValueError, match="No classes found"):
            get_yolo_class_names(config)

    def test_non_contiguous_ids_raises(self):
        config = {"yolo_config": {"names": {"0": "Car", "2": "Pedestrian"}}}
        with pytest.raises(ValueError, match="contiguous"):
            get_yolo_class_names(config)

    def test_invalid_class_id_raises(self):
        config = {"yolo_config": {"names": {"bad": "Car"}}}
        with pytest.raises(ValueError, match="Invalid yolo_config"):
            get_yolo_class_names(config)


class TestFilterClasses:
    def test_basic_filter(self):
        bboxes = [[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]]
        result_boxes, result_labels = filter_classes(["Car", "DontCare"], [0, 1], bboxes, {"Car": 0}, ["DontCare"])
        assert len(result_boxes) == 1
        assert result_labels == [0]

    def test_all_excluded(self):
        result_boxes, result_labels = filter_classes(
            ["DontCare"], [0], [[0.5, 0.5, 0.4, 0.4]], {"Car": 0}, ["DontCare"]
        )
        assert len(result_boxes) == 0

    def test_unmapped_class_skipped(self):
        result_boxes, result_labels = filter_classes(["Unknown"], [0], [[0.5, 0.5, 0.4, 0.4]], {"Car": 0}, [])
        assert len(result_boxes) == 0


class TestFilterSmallBoxes:
    def test_filter_small(self):
        bboxes = [[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.01, 0.01]]
        result, labels = filter_small_boxes(bboxes, [0, 1], min_size=0.02)
        assert len(result) == 1
        assert labels == [0]

    def test_all_above_threshold(self):
        bboxes = [[0.5, 0.5, 0.4, 0.4]]
        result, labels = filter_small_boxes(bboxes, [0], 0.01)
        assert len(result) == 1

    def test_all_below_threshold(self):
        bboxes = [[0.5, 0.5, 0.01, 0.01]]
        result, labels = filter_small_boxes(bboxes, [0], 0.1)
        assert len(result) == 0


class TestNormalizeImage:
    def test_normalize(self):
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        result = normalize_image(img)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_black_image(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = normalize_image(img)
        assert result.sum() == 0.0

    def test_white_image(self):
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = normalize_image(img)
        assert result.max() == 1.0


class TestResizeImage:
    def test_resize(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img, (640, 480))
        assert result.shape == (480, 640, 3)

    def test_resize_same_size(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = resize_image(img, (100, 100))
        assert result.shape == (100, 100, 3)


class TestSplitDataset:
    def test_basic_split(self):
        files = [Path(f"img_{i}.jpg") for i in range(100)]
        splits = split_dataset(files, 0.7, 0.15, 0.15, shuffle=False)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15

    def test_no_test_set(self):
        files = [Path(f"img_{i}.jpg") for i in range(10)]
        splits = split_dataset(files, 0.8, 0.2, 0.0, shuffle=False)
        assert len(splits["train"]) == 8
        assert len(splits["val"]) == 2
        assert len(splits["test"]) == 0

    def test_ratios_must_sum_to_one(self):
        files = [Path("test.jpg")]
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            split_dataset(files, 0.5, 0.5, 0.5)

    def test_shuffle_respects_seed(self):
        files = [Path(f"img_{i}.jpg") for i in range(20)]
        s1 = split_dataset(files, 0.5, 0.25, 0.25, random_seed=42)["train"]
        s2 = split_dataset(files, 0.5, 0.25, 0.25, random_seed=42)["train"]
        assert s1 == s2

    def test_empty_list(self):
        splits = split_dataset([], 0.7, 0.15, 0.15, shuffle=False)
        assert len(splits["train"]) == 0
        assert len(splits["val"]) == 0
        assert len(splits["test"]) == 0


class TestSetupOutputDirs:
    def test_creates_dirs(self, tmp_path):
        out = str(tmp_path / "out")
        setup_output_directories(out, ["train", "val"])
        assert (Path(out) / "images" / "train").is_dir()
        assert (Path(out) / "labels" / "val").is_dir()
