from pathlib import Path


def verify_dataset(dataset_root: str = "data/raw/KITTI/training") -> None:
    data_dir = Path(dataset_root)
    image_folder = data_dir / "image_2"
    label_folder = data_dir / "label_2"
    image_extensions = (".png",)
    label_extensions = (".txt",)

    for folder in [image_folder, label_folder]:
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

    image_files = [f for f in image_folder.iterdir() if f.suffix in image_extensions]
    label_files = [f for f in label_folder.iterdir() if f.suffix in label_extensions]

    num_images = len(image_files)
    num_labels = len(label_files)

    print(f"Number of images: {num_images}")
    print(f"Number of labels: {num_labels}")

    if num_images != num_labels:
        print("Warning: Number of images and labels do not match!")
    else:
        print("Image and label counts match.")

    corrupted_files = []
    for folder, files in [(image_folder, image_files), (label_folder, label_files)]:
        for file_path in files:
            full_path = folder / file_path
            try:
                with open(full_path, "rb") as f:
                    f.read(1024)
            except Exception:
                corrupted_files.append(str(full_path))

    if corrupted_files:
        print("Corrupted/unreadable files found:")
        for f in corrupted_files:
            print(f" - {f}")
    else:
        print("All files are readable and not corrupted.")


if __name__ == "__main__":
    verify_dataset()
