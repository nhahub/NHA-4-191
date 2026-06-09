class TestPreprocessSimpleEdgeCases:
    def test_get_yolo_class_names_from_list(self):
        from src.data.preprocess_dataset import get_yolo_class_names

        config = {"yolo_config": {"names": {"0": "Vehicle", "1": "Pedestrian"}}}
        names = get_yolo_class_names(config)
        assert names == ["Vehicle", "Pedestrian"]

    def test_get_yolo_class_names_from_mapping_no_names(self):
        from src.data.preprocess_dataset import get_yolo_class_names

        config = {"yolo_config": {}, "label_conversion": {"class_mapping": {"Car": 0, "Pedestrian": 1}}}
        names = get_yolo_class_names(config)
        assert len(names) == 2


class TestYoloToPixelExtra:
    def test_yolo_to_pixel_corner(self):
        from src.data.kitti_utils import yolo_to_pixel

        bboxes = [[0.5, 0.5, 1.0, 1.0]]
        result = yolo_to_pixel(bboxes, 100, 100)
        assert result == [[0, 0, 100, 100]]

    def test_yolo_to_pixel_negative(self):
        from src.data.kitti_utils import yolo_to_pixel

        bboxes = [[-0.5, -0.5, 0.1, 0.1]]
        result = yolo_to_pixel(bboxes, 100, 100)
        x_min, y_min, x_max, y_max = result[0]
        assert x_min < 0  # negative coordinates


class TestLoadKittiLabelsMore:
    def test_empty_line_skipped(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "t.txt"
        lbl.write_text("\n\n")
        result = load_kitti_labels(str(lbl), 100, 100)
        assert result == ([], [], [])

    def test_label_line_too_short(self, tmp_path):
        from src.data.kitti_utils import load_kitti_labels

        lbl = tmp_path / "t.txt"
        lbl.write_text("short\n")
        result = load_kitti_labels(str(lbl), 100, 100)
        assert len(result[0]) == 0
