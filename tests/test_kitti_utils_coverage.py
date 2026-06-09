import cv2
import numpy as np
import pytest


class TestLoadKittiImage:
    def test_load_success(self, tmp_path):
        from src.data.kitti_utils import load_kitti_image

        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        result = load_kitti_image(str(img_path))
        assert result.shape == (100, 100, 3)
        assert result is not None

    def test_load_failure(self):
        from src.data.kitti_utils import load_kitti_image

        with pytest.raises(ValueError, match="Failed to load"):
            load_kitti_image("/nonexistent/image.png")


class TestLoadKittiLabels:
    def test_missing_file(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        result = load_kitti_labels(str(tmp_path / "missing.txt"), 100, 100)
        assert result == ([], [], [])

    def test_valid_label(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "test.txt"
        lbl.write_text("Car 0.0 0 0.0 100 200 300 400 0 0 0 0 0 0 0 0\n")
        bboxes, labels, names = load_kitti_labels(str(lbl), 800, 600)
        assert len(bboxes) == 1
        assert names == ["Car"]
        assert 0.1 <= bboxes[0][0] <= 0.5

    def test_skip_dontcare(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "test.txt"
        lbl.write_text("DontCare 0.0 0 0.0 100 200 300 400 0 0 0 0 0 0 0 0\n")
        bboxes, labels, names = load_kitti_labels(str(lbl), 800, 600)
        assert len(bboxes) == 0

    def test_keep_dontcare(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "test.txt"
        lbl.write_text("DontCare 0.0 0 0.0 100 200 300 400 0 0 0 0 0 0 0 0\n")
        bboxes, labels, names = load_kitti_labels(str(lbl), 800, 600, skip_dontcare=False)
        assert len(bboxes) == 1

    def test_too_few_fields(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "test.txt"
        lbl.write_text("Car 0.0 0\n")
        bboxes, labels, names = load_kitti_labels(str(lbl), 800, 600)
        assert len(bboxes) == 0

    def test_invalid_bbox(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "test.txt"
        lbl.write_text("Car 0.0 0 0.0 a b c d 0 0 0 0 0 0 0 0\n")
        bboxes, labels, names = load_kitti_labels(str(lbl), 800, 600)
        assert len(bboxes) == 0

    def test_zero_width_skipped(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "test.txt"
        lbl.write_text("Car 0.0 0 0.0 100 200 100 200 0 0 0 0 0 0 0 0\n")
        bboxes, labels, names = load_kitti_labels(str(lbl), 800, 600)
        assert len(bboxes) == 0

    def test_unknown_class_defaults_to_misc(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "test.txt"
        lbl.write_text("Unicorn 0.0 0 0.0 100 200 300 400 0 0 0 0 0 0 0 0\n")
        bboxes, labels, names = load_kitti_labels(str(lbl), 800, 600)
        assert len(bboxes) == 1
        assert labels[0] == 7


class TestYoloToPixel:
    def test_conversion(self):
        from src.data.kitti_utils import yolo_to_pixel

        bboxes = [[0.5, 0.5, 0.4, 0.4]]
        result = yolo_to_pixel(bboxes, 1000, 800)
        assert len(result) == 1
        x_min, y_min, x_max, y_max = result[0]
        assert x_min < x_max
        assert y_min < y_max
        assert x_min >= 0
        assert x_max <= 1000

    def test_edge_bbox(self):
        from src.data.kitti_utils import yolo_to_pixel

        bboxes = [[0.0, 0.0, 0.0, 0.0]]
        result = yolo_to_pixel(bboxes, 100, 100)
        assert result[0] == [0, 0, 0, 0]

    def test_full_image(self):
        from src.data.kitti_utils import yolo_to_pixel

        bboxes = [[0.5, 0.5, 1.0, 1.0]]
        result = yolo_to_pixel(bboxes, 100, 100)
        assert result[0] == [0, 0, 100, 100]

    def test_empty_input(self):
        from src.data.kitti_utils import yolo_to_pixel

        result = yolo_to_pixel([], 100, 100)
        assert result == []


class TestVisualizeBboxes:
    def test_returns_image(self):
        from src.data.kitti_utils import visualize_bboxes

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = visualize_bboxes(img, [[0.5, 0.5, 0.4, 0.4]], ["Vehicle"], show=False)
        assert result.shape == (100, 100, 3)
        assert not np.array_equal(result, img)

    def test_empty_bboxes(self):
        from src.data.kitti_utils import visualize_bboxes

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = visualize_bboxes(img, [], [], show=False)
        assert np.array_equal(result, img)


class TestGetDatasetStatistics:
    def test_no_images(self, tmp_path):
        from src.data.kitti_utils import get_dataset_statistics

        stats = get_dataset_statistics(str(tmp_path), str(tmp_path))
        assert stats["total_images"] == 0
        assert stats["total_objects"] == 0

    def test_with_image_no_label(self, tmp_path):
        from src.data.kitti_utils import get_dataset_statistics

        img_dir = tmp_path / "image_2"
        img_dir.mkdir()
        cv2.imwrite(str(img_dir / "000000.png"), np.zeros((100, 100, 3), dtype=np.uint8))
        stats = get_dataset_statistics(str(img_dir), str(tmp_path / "label_2"))
        assert stats["total_images"] == 1
        assert stats["images_with_no_labels"] == 1

    def test_with_image_and_label(self, tmp_path):
        from src.data.kitti_utils import get_dataset_statistics

        img_dir = tmp_path / "image_2"
        lbl_dir = tmp_path / "label_2"
        img_dir.mkdir()
        lbl_dir.mkdir()
        cv2.imwrite(str(img_dir / "000000.png"), np.zeros((100, 100, 3), dtype=np.uint8))
        (lbl_dir / "000000.txt").write_text("Car 0.0 0 0.0 10 20 50 80 0 0 0 0 0 0 0 0\n")
        stats = get_dataset_statistics(str(img_dir), str(lbl_dir))
        assert stats["total_images"] == 1
        assert stats["total_objects"] == 1
        assert stats["class_counts"]["Car"] == 1
        assert stats["avg_objects_per_image"] == 1.0
