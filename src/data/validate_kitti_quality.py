import hashlib
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm


def validate_kitti_quality(
    dataset_root: str = "data/raw/KITTI/training",
) -> dict[str, list] | None:
    corrupted_images: list[str] = []
    missing_labels: list[str] = []
    invalid_labels: list[str] = []
    duplicates: list[tuple[str, str]] = []
    hashes: dict[str, str] = {}

    data_dir = Path(dataset_root)
    image_folder = data_dir / "image_2"
    label_folder = data_dir / "label_2"
    image_extensions = (".png",)

    print("--- Starting KITTI Data Quality Audit ---")

    if not image_folder.is_dir():
        print(f"Error: Image folder not found at {image_folder}")
        return None

    images = [f for f in image_folder.iterdir() if f.suffix.lower() in image_extensions]

    for img_path in tqdm(images):
        label_path = label_folder / f"{img_path.stem}.txt"

        try:
            with Image.open(img_path) as img:
                img.verify()
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                raise ValueError("cv2 could not read image")
            img_h, img_w = cv_img.shape[:2]
        except Exception:
            corrupted_images.append(img_path.name)
            continue

        with open(img_path, "rb") as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        if img_hash in hashes:
            duplicates.append((img_path.name, hashes[img_hash]))
            continue
        hashes[img_hash] = img_path.name

        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue

        try:
            with open(label_path) as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        invalid_labels.append(f"{img_path.name} (Missing Columns)")
                        break
                    left, top, right, bottom = map(float, parts[4:8])
                    if left < 0 or top < 0 or right > img_w or bottom > img_h or left >= right or top >= bottom:
                        invalid_labels.append(img_path.name)
                        break
        except Exception:
            invalid_labels.append(img_path.name)

    clean_samples = [
        img.name
        for img in images
        if img.name not in corrupted_images and img.name not in missing_labels and img.name not in invalid_labels
    ]
    with open("clean_index.txt", "w") as f:
        for img in clean_samples:
            f.write(Path(img).stem + "\n")

    print(f"Clean dataset index saved with {len(clean_samples)} samples.")

    print("\n" + "=" * 40)
    print("      KITTI QUALITY ASSESSMENT REPORT")
    print("=" * 40)
    print(f"Total PNG Images Found:     {len(images)}")
    print(f"Corrupted Images:           {len(corrupted_images)}")
    print(f"Missing Label Files:        {len(missing_labels)}")
    print(f"Invalid/Out-of-bounds Bboxes: {len(invalid_labels)}")
    print(f"Exact Duplicates Found:     {len(duplicates)}")
    print(f"Final Clean Images:         {len(clean_samples)}")
    print("=" * 40)

    if not any([corrupted_images, missing_labels, invalid_labels, duplicates]):
        print("Result: Dataset is CLEAN and ready for preprocessing!")
    else:
        print("Warning: Issues found. Please check the lists.")

    return {
        "corrupted": corrupted_images,
        "missing": missing_labels,
        "invalid": invalid_labels,
        "duplicates": duplicates,
    }


if __name__ == "__main__":
    validate_kitti_quality()
