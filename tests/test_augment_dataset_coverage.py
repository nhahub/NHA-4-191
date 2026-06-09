import cv2
import numpy as np
import pytest


class TestAugmentDataset:
    def test_no_images_in_dir(self, tmp_path):
        from src.data.augment_dataset import augment_dataset

        (tmp_path / "empty_img").mkdir(parents=True)
        (tmp_path / "empty_lbl").mkdir(parents=True)
        stats = augment_dataset(
            img_dir=str(tmp_path / "empty_img"),
            label_dir=str(tmp_path / "empty_lbl"),
            output_img_dir=str(tmp_path / "out_img"),
            output_label_dir=str(tmp_path / "out_lbl"),
        )
        assert stats["total"] == 0
        assert stats["successful"] == 0

    def test_skip_missing_labels(self, tmp_path):
        from src.data.augment_dataset import augment_dataset

        img_dir = tmp_path / "img"
        lbl_dir = tmp_path / "lbl"
        out_img = tmp_path / "out_img"
        out_lbl = tmp_path / "out_lbl"
        for d in [img_dir, lbl_dir, out_img, out_lbl]:
            d.mkdir(parents=True)
        cv2.imwrite(str(img_dir / "test.png"), np.zeros((100, 100, 3), dtype=np.uint8))
        stats = augment_dataset(
            img_dir=str(img_dir),
            label_dir=str(lbl_dir),
            output_img_dir=str(out_img),
            output_label_dir=str(out_lbl),
        )
        assert stats["skipped"] == 1

    def test_successful_augmentation(self, tmp_path):
        from src.data.augment_dataset import augment_dataset

        img_dir = tmp_path / "img"
        lbl_dir = tmp_path / "lbl"
        out_img = tmp_path / "out_img"
        out_lbl = tmp_path / "out_lbl"
        for d in [img_dir, lbl_dir, out_img, out_lbl]:
            d.mkdir(parents=True)
        cv2.imwrite(str(img_dir / "test.png"), np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        (lbl_dir / "test.txt").write_text("Car 0.0 0 0.0 10 20 50 80 0 0 0 0 0 0 0 0\n")
        stats = augment_dataset(
            img_dir=str(img_dir),
            label_dir=str(lbl_dir),
            output_img_dir=str(out_img),
            output_label_dir=str(out_lbl),
            augmentations_per_image=1,
            preset="light",
        )
        assert stats["successful"] == 1

    def test_num_images_limit(self, tmp_path):
        from src.data.augment_dataset import augment_dataset

        img_dir = tmp_path / "img"
        lbl_dir = tmp_path / "lbl"
        out_img = tmp_path / "out_img"
        out_lbl = tmp_path / "out_lbl"
        for d in [img_dir, lbl_dir, out_img, out_lbl]:
            d.mkdir(parents=True)
        for i in range(5):
            cv2.imwrite(str(img_dir / f"{i:04d}.png"), np.zeros((100, 100, 3), dtype=np.uint8))
        stats = augment_dataset(
            img_dir=str(img_dir),
            label_dir=str(lbl_dir),
            output_img_dir=str(out_img),
            output_label_dir=str(out_lbl),
            num_images=2,
        )
        assert stats["total"] == 2

    def test_image_size_option(self, tmp_path):
        from src.data.augment_dataset import augment_dataset

        img_dir = tmp_path / "img"
        lbl_dir = tmp_path / "lbl"
        out_img = tmp_path / "out_img"
        out_lbl = tmp_path / "out_lbl"
        for d in [img_dir, lbl_dir, out_img, out_lbl]:
            d.mkdir(parents=True)
        cv2.imwrite(str(img_dir / "test.png"), np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8))
        (lbl_dir / "test.txt").write_text("Car 0.0 0 0.0 10 20 50 80 0 0 0 0 0 0 0 0\n")
        stats = augment_dataset(
            img_dir=str(img_dir),
            label_dir=str(lbl_dir),
            output_img_dir=str(out_img),
            output_label_dir=str(out_lbl),
            augmentations_per_image=1,
            image_size=(64, 128),
        )
        assert stats["total"] == 1


class TestMain:
    def test_main_missing_args(self):
        import sys
        from unittest.mock import patch

        from src.data.augment_dataset import main

        with patch.object(sys, "argv", ["prog"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_invalid_image_size(self):
        import sys
        from unittest.mock import patch

        from src.data.augment_dataset import main

        test_args = [
            "prog",
            "--img-dir",
            "/x",
            "--label-dir",
            "/x",
            "--output-img-dir",
            "/x",
            "--output-label-dir",
            "/x",
            "--image-size",
            "invalid",
        ]
        with patch.object(sys, "argv", test_args):
            main()  # should print error and return gracefully
