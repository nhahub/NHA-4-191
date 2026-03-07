"""
Dataset Preprocessing Module

This module handles the complete preprocessing pipeline for KITTI dataset:
- Resizes images to target model input size
- Converts KITTI annotations to YOLO format
- Splits dataset into train/val/test sets
- Organizes data in structured folders (data/processed/)
"""

import os
import cv2
import yaml
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import argparse
import sys

try:
    from .kitti_utils import load_kitti_labels, save_yolo_labels
except ImportError:
    # If relative import fails, we're running as a script
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from kitti_utils import load_kitti_labels, save_yolo_labels


def load_config(config_path: str = "configs/preprocessing.yaml") -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_directories(output_dir: str, splits: List[str]) -> None:
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    print(f"[OK] Created output directories in {output_dir}")


def get_class_mapping(config: Dict) -> Dict[str, int]:
    return config['label_conversion']['class_mapping']


def filter_classes(
    class_names: List[str],
    class_labels: List[int],
    bboxes: List[List[float]],
    class_mapping: Dict[str, int],
    exclude_classes: List[str]
) -> Tuple[List[List[float]], List[int]]:
    filtered_bboxes = []
    filtered_labels = []
    
    for bbox, class_name in zip(bboxes, class_names):
        # Skip excluded classes
        if class_name in exclude_classes:
            continue
        
        # Only keep classes in our mapping
        if class_name in class_mapping:
            filtered_bboxes.append(bbox)
            filtered_labels.append(class_mapping[class_name])
    
    return filtered_bboxes, filtered_labels


def filter_small_boxes(
    bboxes: List[List[float]],
    class_labels: List[int],
    min_size: float
) -> Tuple[List[List[float]], List[int]]:
    filtered_bboxes = []
    filtered_labels = []
    
    for bbox, label in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        bbox_area = width * height
        
        if bbox_area >= min_size:
            filtered_bboxes.append(bbox)
            filtered_labels.append(label)
    
    return filtered_bboxes, filtered_labels


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def split_dataset(
    image_files: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 42,
    shuffle: bool = True
) -> Dict[str, List[Path]]:
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle if requested
    if shuffle:
        random.seed(random_seed)
        image_files = image_files.copy()
        random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Split
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }
    
    print(f"\n Dataset Split:")
    print(f"  Train: {len(splits['train'])} images ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(splits['val'])} images ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(splits['test'])} images ({test_ratio*100:.0f}%)")
    
    return splits


def process_image_label_pair(
    image_path: Path,
    label_path: Path,
    output_img_path: Path,
    output_lbl_path: Path,
    config: Dict
) -> Dict:
    stats = {'success': False, 'num_objects': 0, 'error': None}
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            stats['error'] = "Failed to load image"
            return stats
        
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        target_size = tuple(config['image_processing']['target_size'])  # (width, height)
        resized_img = resize_image(image, target_size)
        
        # Optional: Normalize pixels (usually done in dataloader)
        if config['image_processing'].get('normalize_pixels', False):
            resized_img = normalize_image(resized_img)
            # Convert back to uint8 for saving
            resized_img = (resized_img * 255).astype(np.uint8)
        
        # Save image
        save_format = config['image_processing']['save_format']
        output_img_path = output_img_path.with_suffix(f'.{save_format}')
        
        if save_format == 'jpg':
            quality = config['image_processing'].get('jpeg_quality', 95)
            cv2.imwrite(str(output_img_path), resized_img, 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(str(output_img_path), resized_img)
        
        # Load and convert labels
        bboxes, class_labels, class_names = load_kitti_labels(
            str(label_path),
            orig_w,
            orig_h,
            skip_dontcare=False  # We'll filter manually
        )
        
        # Filter classes
        class_mapping = get_class_mapping(config)
        exclude_classes = config['label_conversion'].get('exclude_classes', [])
        filtered_bboxes, filtered_labels = filter_classes(
            class_names, class_labels, bboxes, class_mapping, exclude_classes
        )
        
        # Filter small boxes
        min_size = config['label_conversion'].get('min_bbox_size', 0.01)
        final_bboxes, final_labels = filter_small_boxes(
            filtered_bboxes, filtered_labels, min_size
        )
        
        # Save YOLO labels
        save_yolo_labels(str(output_lbl_path), final_bboxes, final_labels)
        
        stats['success'] = True
        stats['num_objects'] = len(final_labels)
        
    except Exception as e:
        stats['error'] = str(e)
    
    return stats


def preprocess_dataset(
    config_path: str = "configs/preprocessing.yaml",
    project_root: Optional[str] = None
) -> Dict:
    # Determine project root
    if project_root is None:
        # Assume script is in src/data/ and project root is two levels up
        project_root = Path(__file__).parent.parent.parent
    project_root = Path(project_root)
    
    # Load configuration
    config_path = project_root / config_path
    print(f"Loading config from {config_path}")
    config = load_config(str(config_path))
    
    # Setup paths
    input_dir = project_root / config['input']['raw_data_dir']
    img_dir = input_dir / config['input']['image_subdir']
    label_dir = input_dir / config['input']['label_subdir']
    output_dir = project_root / config['output']['processed_dir']
    
    print(f"Input:  {img_dir}")
    print(f"Output: {output_dir}")
    
    # Verify input directories exist
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    # Create output directories
    splits = ['train', 'val', 'test']
    setup_output_directories(str(output_dir), splits)
    
    # Get all images
    image_extensions = ['.png', '.jpg', '.jpeg']
    all_images = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in {img_dir}")
    
    print(f"\nFound {len(all_images)} images")
    
    # Split dataset
    split_config = config['split']
    dataset_splits = split_dataset(
        all_images,
        split_config['train_ratio'],
        split_config['val_ratio'],
        split_config['test_ratio'],
        split_config['random_seed'],
        split_config['shuffle']
    )
    
    # Processing statistics
    stats = {
        'total': len(all_images),
        'successful': 0,
        'failed': 0,
        'total_objects': 0,
        'errors': []
    }
    
    # Process each split
    show_progress = config['processing'].get('show_progress', True)
    skip_on_error = config['processing'].get('skip_on_error', True)
    
    print("\nProcessing images...")
    
    for split_name, image_list in dataset_splits.items():
        print(f"\n  Processing {split_name} set...")
        
        iterator = tqdm(image_list) if show_progress else image_list
        
        for img_path in iterator:
            # Paths
            label_name = img_path.stem + '.txt'
            label_path = label_dir / label_name
            
            output_img_path = output_dir / "images" / split_name / img_path.name
            output_lbl_path = output_dir / "labels" / split_name / label_name
            
            # Process
            result = process_image_label_pair(
                img_path, label_path,
                output_img_path, output_lbl_path,
                config
            )
            
            if result['success']:
                stats['successful'] += 1
                stats['total_objects'] += result['num_objects']
            else:
                stats['failed'] += 1
                error_msg = f"{img_path.name}: {result['error']}"
                stats['errors'].append(error_msg)
                
                if not skip_on_error:
                    raise RuntimeError(f"Processing failed: {error_msg}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total images:      {stats['total']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Failed:            {stats['failed']}")
    print(f"Total objects:     {stats['total_objects']}")
    
    if stats['errors']:
        print(f"\n{len(stats['errors'])} errors occurred:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")
    
    print(f"\nOutput saved to: {output_dir}")
    
    # Create YOLO data.yaml if requested
    if config['output'].get('create_yolo_yaml', True):
        create_yolo_config(output_dir, config, project_root)
    
    return stats


def create_yolo_config(
    processed_dir: Path,
    config: Dict,
    project_root: Path
) -> None:
    data_yaml_path = processed_dir / "data.yaml"
    
    class_mapping = get_class_mapping(config)
    # Sort by class ID
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    class_names = [name for name, _ in sorted_classes]
    
    # Create relative paths from project root
    train_path = processed_dir / "images" / "train"
    val_path = processed_dir / "images" / "val"
    test_path = processed_dir / "images" / "test"
    
    # Make paths relative to project root for portability
    try:
        train_rel = train_path.relative_to(project_root)
        val_rel = val_path.relative_to(project_root)
        test_rel = test_path.relative_to(project_root)
    except ValueError:
        # If relative path fails, use absolute
        train_rel = train_path
        val_rel = val_path
        test_rel = test_path
    
    yaml_content = {
        'path': str(project_root),  # Dataset root
        'train': str(train_rel),
        'val': str(val_rel),
        'test': str(test_rel),
        'nc': len(class_names),  # Number of classes
        'names': class_names  # Class names
    }
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created YOLO config: {data_yaml_path}")


def main():
    """Command-line interface for dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess KITTI dataset for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python preprocess_dataset.py

  # Use custom config
  python preprocess_dataset.py --config configs/custom_preprocessing.yaml

  # Specify project root
  python preprocess_dataset.py --project-root /path/to/Road-Sense
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/preprocessing.yaml',
        help='Path to preprocessing configuration YAML'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Project root directory (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    try:
        stats = preprocess_dataset(
            config_path=args.config,
            project_root=args.project_root
        )
        
        # Exit with error code if processing failed
        if stats['failed'] > 0:
            exit(1)
            
    except Exception as e:
        print(f"\n[fail] Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
