import cv2
import numpy as np
import pytest


def _make_test_image(path, w=100, h=100):
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_label(path, lines):
    path.write_text("\n".join(lines))


class TestKITTIDataset:
    @pytest.fixture
    def kitti_data(self, tmp_path):
        img_dir = tmp_path / "image_2"
        lbl_dir = tmp_path / "label_2"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir()
        _make_test_image(img_dir / "000000.png")
        _make_label(lbl_dir / "000000.txt", ["Car 0.0 0 0.0 10 20 50 80 0 0 0 0 0 0 0 0"])
        return img_dir, lbl_dir

    def test_init(self, kitti_data):
        from src.data.kitti_dataset import KITTIDataset

        img_dir, lbl_dir = kitti_data
        ds = KITTIDataset(str(img_dir), str(lbl_dir))
        assert len(ds) == 1
        sample = ds[0]
        assert "image" in sample
        assert "bboxes" in sample
        assert "labels" in sample

    def test_init_no_transform(self, kitti_data):
        from src.data.kitti_dataset import KITTIDataset

        img_dir, lbl_dir = kitti_data
        ds = KITTIDataset(str(img_dir), str(lbl_dir), transform=None)
        assert len(ds) == 1

    def test_image_path_returned(self, kitti_data):
        from src.data.kitti_dataset import KITTIDataset

        img_dir, lbl_dir = kitti_data
        ds = KITTIDataset(str(img_dir), str(lbl_dir), return_image_path=True)
        sample = ds[0]
        assert "image_path" in sample

    def test_empty_dataset(self, tmp_path):
        from src.data.kitti_dataset import KITTIDataset

        img_dir = tmp_path / "empty_img"
        lbl_dir = tmp_path / "empty_lbl"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir()
        ds = KITTIDataset(str(img_dir), str(lbl_dir))
        assert len(ds) == 0

    def test_no_labels_file(self, tmp_path):
        from src.data.kitti_dataset import KITTIDataset

        img_dir = tmp_path / "img"
        img_dir.mkdir(parents=True)
        _make_test_image(img_dir / "000000.png")
        lbl_dir = tmp_path / "lbl"
        lbl_dir.mkdir()
        ds = KITTIDataset(str(img_dir), str(lbl_dir))
        sample = ds[0]
        assert len(sample["bboxes"]) == 0

    def test_torch_dataset(self, kitti_data):
        import torch

        from src.data.kitti_dataset import KITTIDatasetTorch

        img_dir, lbl_dir = kitti_data
        ds = KITTIDatasetTorch(str(img_dir), str(lbl_dir), normalize=False)
        assert len(ds) == 1
        sample = ds[0]
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)

    def test_torch_dataset_normalize(self, kitti_data):
        import torch

        from src.data.kitti_dataset import KITTIDatasetTorch

        img_dir, lbl_dir = kitti_data
        ds = KITTIDatasetTorch(str(img_dir), str(lbl_dir), normalize=True)
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].dtype == torch.float32

    def test_collate_fn(self):
        import torch

        from src.data.kitti_dataset import collate_fn

        batch = [
            {
                "image": torch.zeros(3, 100, 100),
                "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
                "labels": torch.tensor([0]),
            },
            {
                "image": torch.zeros(3, 100, 100),
                "bboxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.long),
            },
        ]
        result = collate_fn(batch)
        assert "images" in result
        assert "bboxes" in result
        assert "labels" in result

    def test_create_data_loaders(self, kitti_data):
        from src.data.kitti_dataset import create_data_loaders

        img_dir, lbl_dir = kitti_data
        train_loader, val_loader = create_data_loaders(
            train_img_dir=str(img_dir),
            train_label_dir=str(lbl_dir),
            batch_size=4,
        )
        assert train_loader is not None
        assert val_loader is None
