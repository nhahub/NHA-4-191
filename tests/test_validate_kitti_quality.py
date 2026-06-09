from pathlib import Path

import cv2
import numpy as np


def _make_label_file(path: Path, lines: list[str]):
    path.write_text("\n".join(lines))


def _make_test_image(path: Path, w: int = 100, h: int = 100):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_validate_kitti_quality_missing_folder(tmp_path):
    from src.data.validate_kitti_quality import validate_kitti_quality

    result = validate_kitti_quality(str(tmp_path / "nonexistent"))
    assert result is None


def test_validate_kitti_quality_clean(tmp_path):
    img_dir = tmp_path / "image_2"
    lbl_dir = tmp_path / "label_2"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir()

    _make_test_image(img_dir / "000000.png")
    _make_label_file(lbl_dir / "000000.txt", ["Car 0.0 0 0.0 50 50 100 100 0 0 0 0 0 0 0 0"])

    from src.data.validate_kitti_quality import validate_kitti_quality

    result = validate_kitti_quality(str(tmp_path))
    assert result is not None
    assert len(result["corrupted"]) == 0
    assert len(result["missing"]) == 0
    assert len(result["invalid"]) == 0


def test_validate_kitti_quality_corrupted(tmp_path):
    img_dir = tmp_path / "image_2"
    lbl_dir = tmp_path / "label_2"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir()

    bad_file = img_dir / "corrupt.png"
    bad_file.write_bytes(b"not an image")

    _make_label_file(lbl_dir / "corrupt.txt", ["Car 0.0 0 0.0 0 0 0 0 50 50 100 100 0 0 0 0"])

    from src.data.validate_kitti_quality import validate_kitti_quality

    result = validate_kitti_quality(str(tmp_path))
    assert result is not None
    assert len(result["corrupted"]) == 1


def test_validate_kitti_quality_missing_label(tmp_path):
    img_dir = tmp_path / "image_2"
    lbl_dir = tmp_path / "label_2"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir()

    _make_test_image(img_dir / "000000.png")

    from src.data.validate_kitti_quality import validate_kitti_quality

    result = validate_kitti_quality(str(tmp_path))
    assert result is not None
    assert len(result["missing"]) == 1


def test_validate_kitti_quality_invalid_bbox(tmp_path):
    img_dir = tmp_path / "image_2"
    lbl_dir = tmp_path / "label_2"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir()

    _make_test_image(img_dir / "000000.png", w=100, h=100)
    _make_label_file(lbl_dir / "000000.txt", ["Car 0.0 0 0.0 0 0 0 0 -10 -10 200 200 0 0 0 0"])

    from src.data.validate_kitti_quality import validate_kitti_quality

    result = validate_kitti_quality(str(tmp_path))
    assert result is not None
    assert len(result["invalid"]) == 1
