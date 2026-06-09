from pathlib import Path

import pytest


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")


def test_verify_dataset_files_match(tmp_path, capsys):
    img_dir = tmp_path / "image_2"
    lbl_dir = tmp_path / "label_2"
    _touch(img_dir / "000000.png")
    _touch(img_dir / "000001.png")
    _touch(lbl_dir / "000000.txt")
    _touch(lbl_dir / "000001.txt")

    from src.data.verify_dataset import verify_dataset

    verify_dataset(str(tmp_path))
    captured = capsys.readouterr()
    assert "Image and label counts match" in captured.out
    assert "All files are readable" in captured.out


def test_verify_dataset_mismatch(tmp_path, capsys):
    img_dir = tmp_path / "image_2"
    lbl_dir = tmp_path / "label_2"
    _touch(img_dir / "000000.png")
    _touch(lbl_dir / "000000.txt")
    _touch(lbl_dir / "000001.txt")

    from src.data.verify_dataset import verify_dataset

    verify_dataset(str(tmp_path))
    captured = capsys.readouterr()
    assert "do not match" in captured.out


def test_verify_dataset_missing_folder(tmp_path):
    from src.data.verify_dataset import verify_dataset

    with pytest.raises(FileNotFoundError):
        verify_dataset(str(tmp_path / "nonexistent"))
