def test_kitti_classes():
    from src.data.kitti_utils import CLASS_ID_TO_NAME, KITTI_CLASSES

    assert len(KITTI_CLASSES) > 0
    assert "Car" in KITTI_CLASSES
    assert "Pedestrian" in KITTI_CLASSES
    assert 0 in CLASS_ID_TO_NAME


def test_kitti_utils_print_stats(capsys):
    from src.data.kitti_utils import print_dataset_statistics

    stats = {
        "total_images": 10,
        "total_objects": 20,
        "avg_objects_per_image": 2.0,
        "images_with_no_labels": 0,
        "image_sizes": [(100, 100)],
        "class_counts": {"Car": 10},
        "bbox_areas": [100],
        "occlusion_stats": {0: 10},
        "truncation_stats": {0.0: 10},
    }
    print_dataset_statistics(stats)
    captured = capsys.readouterr()
    assert "Car" in captured.out


def test_yolo_to_pixel():
    from src.data.kitti_utils import yolo_to_pixel

    bboxes = [[0.5, 0.5, 0.4, 0.4]]
    pixel = yolo_to_pixel(bboxes, 1000, 800)
    assert len(pixel) == 1
    x_min, y_min, x_max, y_max = pixel[0]
    assert x_min < x_max
    assert y_min < y_max
    assert x_min >= 0
    assert y_min >= 0
    assert x_max <= 1000
    assert y_max <= 800


def test_yolo_label_io(tmp_path):
    from src.data.kitti_utils import load_yolo_labels, save_yolo_labels

    bboxes = [[0.5, 0.5, 0.4, 0.4]]
    labels = [0]
    lbl_file = tmp_path / "test.txt"
    save_yolo_labels(str(lbl_file), bboxes, labels)
    assert lbl_file.exists()
    loaded_boxes, loaded_labels = load_yolo_labels(str(lbl_file))
    assert len(loaded_boxes) == 1
    assert loaded_labels == [0]


def test_load_yolo_labels_missing():
    from src.data.kitti_utils import load_yolo_labels

    boxes, labels = load_yolo_labels("/nonexistent/file.txt")
    assert len(boxes) == 0
    assert len(labels) == 0


def test_save_empty_labels(tmp_path):
    from src.data.kitti_utils import load_yolo_labels, save_yolo_labels

    lbl_file = tmp_path / "empty.txt"
    save_yolo_labels(str(lbl_file), [], [])
    boxes, labels = load_yolo_labels(str(lbl_file))
    assert len(boxes) == 0
    assert len(labels) == 0
