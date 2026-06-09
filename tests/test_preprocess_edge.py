def test_filter_classes():
    from src.data.preprocess_dataset import filter_classes

    bboxes = [[0.5, 0.5, 0.4, 0.4]]
    labels = ["Car", "DontCare"]
    filtered_bboxes, filtered_labels = filter_classes(labels, [0, 1], bboxes, {"Car": 0, "Pedestrian": 1}, ["DontCare"])
    assert len(filtered_bboxes) == 1
    assert filtered_labels == [0]


def test_filter_small_boxes():
    from src.data.preprocess_dataset import filter_small_boxes

    bboxes = [[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.01, 0.01]]
    filtered, labels = filter_small_boxes(bboxes, [0, 1], min_size=0.02)
    assert len(filtered) == 1
    assert labels == [0]


def test_normalize_image():
    import numpy as np

    from src.data.preprocess_dataset import normalize_image

    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    normalized = normalize_image(img)
    assert normalized.dtype == np.float32
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0


def test_get_yolo_class_names_valid():
    from src.data.preprocess_dataset import get_yolo_class_names

    config = {"yolo_config": {"names": {"0": "Vehicle", "1": "Pedestrian"}}}
    names = get_yolo_class_names(config)
    assert names == ["Vehicle", "Pedestrian"]


def test_get_class_mapping():
    from src.data.preprocess_dataset import get_class_mapping

    config = {"label_conversion": {"class_mapping": {"Car": 0, "Pedestrian": 1}}}
    mapping = get_class_mapping(config)
    assert mapping["Car"] == 0
    assert mapping["Pedestrian"] == 1
