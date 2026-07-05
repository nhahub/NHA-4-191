import os

import cv2
import numpy as np
import pytest
from preprocess_dataset import (
    convert_labels,
    filter_classes,
    resize_images,
    save_preprocessed_dataset,
    split_dataset,
)

# Fixtures


@pytest.fixture
def sample_image(tmp_path):
    img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), img)
    return path, img


@pytest.fixture
def sample_kitti_lines():
    return [
        "Car 0.0 0 -1.57 100 150 300 250 1.5 1.8 4.0 1.0 1.5 10.0 0.0",
        "Pedestrian 0.0 0 -1.57 400 100 480 300 1.7 0.6 0.8 2.0 1.5 8.0 0.1",
        "Cyclist 0.0 0 -1.57 600 200 700 350 1.7 0.6 1.8 3.0 1.5 12.0 0.0",
        "DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10",
        "Van 0.0 0 -1.57 50 100 200 300 1.5 1.8 4.5 0.5 1.5 15.0 0.0",
    ]


# 1. resize_images  (3 tests)


class TestResizeImages:
    def test_output_shape_matches_target(self, sample_image):
        path, _ = sample_image
        result = resize_images([path], target_size=(640, 640))
        assert result[0].shape == (640, 640, 3)

    def test_custom_target_size(self, sample_image):
        path, _ = sample_image
        result = resize_images([path], target_size=(320, 320))
        assert result[0].shape == (320, 320, 3)

    def test_missing_image_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resize_images([tmp_path / "ghost.jpg"])

    def test_multiple_images(self, tmp_path):
        paths = []
        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            p = tmp_path / f"img_{i}.jpg"
            cv2.imwrite(str(p), img)
            paths.append(p)
        result = resize_images(paths, target_size=(64, 64))
        assert len(result) == 3


# 2. convert_labels  (4 tests)


class TestConvertLabels:
    def test_known_classes_are_converted(self, sample_kitti_lines):
        result = convert_labels(sample_kitti_lines, img_w=1242, img_h=375)
        assert len(result) == 3  # DontCare and Van filtered out

    def test_output_format_is_correct(self, sample_kitti_lines):
        result = convert_labels(sample_kitti_lines, img_w=1242, img_h=375)
        for label in result:
            parts = label.split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            assert cls_id in (0, 1, 2)
            cx, cy, w, h = map(float, parts[1:])
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0

    def test_unknown_classes_are_skipped(self):
        lines = [
            "DontCare -1 -1 -10 0 0 50 50 -1 -1 -1 -1000 -1000 -1000 -10",
            "Van 0 0 0 100 100 200 200 1 1 1 1 1 1 0",
            "Truck 0 0 0 100 100 200 200 1 1 1 1 1 1 0",
        ]
        result = convert_labels(lines, img_w=1242, img_h=375)
        assert result == []

    def test_empty_input_returns_empty(self):
        result = convert_labels([], img_w=1242, img_h=375)
        assert result == []


# 3. filter_classes  (3 tests)


class TestFilterClasses:
    def test_filters_correct_classes(self, sample_kitti_lines):
        result = filter_classes(sample_kitti_lines, ["Car", "Pedestrian"])
        types = [line.split()[0] for line in result]
        assert set(types) == {"Car", "Pedestrian"}

    def test_empty_allowed_returns_empty(self, sample_kitti_lines):
        result = filter_classes(sample_kitti_lines, [])
        assert result == []

    def test_empty_input_returns_empty(self):
        result = filter_classes([], ["Car"])
        assert result == []


# 4. split_dataset  (3 tests)


class TestSplitDataset:
    def test_split_ratio_is_correct(self):
        files = [f"img_{i}.jpg" for i in range(100)]
        train, val = split_dataset(files, val_ratio=0.2, seed=42)
        assert len(train) + len(val) == 100
        assert abs(len(val) - 20) <= 1  # allow ±1 due to rounding

    def test_no_overlap_between_splits(self):
        files = [f"img_{i}.jpg" for i in range(50)]
        train, val = split_dataset(files, val_ratio=0.2)
        assert set(train).isdisjoint(set(val))

    def test_invalid_ratio_raises_error(self):
        with pytest.raises(ValueError):
            split_dataset(["a.jpg"], val_ratio=1.5)

    def test_empty_list_returns_empty_splits(self):
        train, val = split_dataset([])
        assert train == [] and val == []


# 5. save_preprocessed_dataset  (3 tests)


class TestSavePreprocessedDataset:
    def _make_data(self, n=3):
        images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(n)]
        labels = [["0 0.5 0.5 0.3 0.4"] for _ in range(n)]
        filenames = [f"sample_{i}" for i in range(n)]
        return images, labels, filenames

    def test_files_are_saved(self, tmp_path):
        images, labels, filenames = self._make_data()
        save_preprocessed_dataset(images, labels, filenames, str(tmp_path))
        for name in filenames:
            assert os.path.exists(tmp_path / "images" / "train" / f"{name}.jpg")
            assert os.path.exists(tmp_path / "labels" / "train" / f"{name}.txt")

    def test_saved_count_is_correct(self, tmp_path):
        images, labels, filenames = self._make_data(5)
        result = save_preprocessed_dataset(images, labels, filenames, str(tmp_path))
        assert result["saved_count"] == 5

    def test_val_split_creates_correct_dirs(self, tmp_path):
        images, labels, filenames = self._make_data(2)
        result = save_preprocessed_dataset(images, labels, filenames, str(tmp_path), split="val")
        assert "val" in result["images_dir"]
        assert os.path.isdir(result["images_dir"])
        assert os.path.isdir(result["labels_dir"])
