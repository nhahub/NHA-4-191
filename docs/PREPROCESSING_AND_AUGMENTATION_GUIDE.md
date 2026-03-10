# Preprocessing and Augmentation Guide
**Project:** Road-Sense - Real-Time Object Detection for Autonomous Vehicles  
**Documentation for:** Milestone 1 Deliverables  
**Last Updated:** March 2026

---

## Table of Contents
1. [Overview](#overview)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Augmentation Strategy](#augmentation-strategy)
4. [Configuration Management](#configuration-management)
5. [Execution Guide](#execution-guide)
6. [Validation and Quality Checks](#validation-and-quality-checks)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide documents the complete preprocessing and augmentation pipeline for the Road-Sense project. The pipeline converts raw KITTI format data into YOLO-ready format with comprehensive quality validation and augmentation strategies.

### Pipeline Goals
✅ Convert KITTI format → YOLO format  
✅ Standardize image sizes (640×640 for YOLO)  
✅ Merge and filter object classes  
✅ Create reproducible train/val/test splits  
✅ Apply data augmentation for robustness  
✅ Validate data quality at every step  

### Key Outputs
- **Preprocessed Dataset:** `data/processed/kitti/`
- **Augmented Dataset:** `data/augmented/` (optional)
- **YOLO Config:** `data.yaml` (auto-generated)
- **Statistics:** Class distribution, object counts

---

## Preprocessing Pipeline

### Step 1: Input Data Validation

**Script:** `src/data/validate_kitti_quality.py`

```bash
# Run validation before preprocessing
python src/data/validate_kitti_quality.py
```

**Checks Performed:**
- ✅ Image file integrity (no corruption)
- ✅ Label file existence (every image has label)
- ✅ Bounding box validity (coordinates in range [0, image_size])
- ✅ Duplicate detection (exact image hashes)
- ✅ Format compliance (KITTI .txt format)

**Input Requirements:**
```
data/raw/KITTI/training/
├── image_2/           # PNG images (7,481 files)
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── label_2/           # KITTI annotations (7,481 files)
    ├── 000000.txt
    ├── 000001.txt
    └── ...
```

**KITTI Label Format:**
```
<class> <truncation> <occlusion> <alpha> <x1> <y1> <x2> <y2> <h> <w> <l> <x> <y> <z> <rotation_y>
│       │            │           │       └──────┬──────┘  └─────────┬─────────┘  │
│       │            │           │         2D bbox (pixels)    3D dimensions    rotation
│       │            │           │
│       │            │           └─ Observation angle (radians)
│       │            └─ Occlusion level (0=visible, 3=heavily occluded)
│       └─ Truncation percentage (0.0=not truncated, 1.0=fully truncated)
└─ Object class (Car, Pedestrian, Cyclist, etc.)

Example:
Car 0.00 0 -1.58 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.65 1.87 8.41 -1.56
```

**Validation Output:**
```
========================================
      KITTI QUALITY ASSESSMENT REPORT
========================================
Total PNG Images Found:     7481
Corrupted Images:           0
Missing Label Files:        0
Invalid/Out-of-bounds Bboxes: 0
Exact Duplicates Found:     0
Final Clean Images:         7481
========================================
Result: Dataset is CLEAN and ready for preprocessing!
```

---

### Step 2: Image Preprocessing

**Configuration:** `configs/preprocessing.yaml`

```yaml
image_processing:
  target_size: [640, 640]    # [width, height] for YOLO
  save_format: "jpg"         # jpg or png
  jpeg_quality: 95           # 85-100 (higher = better quality)
  normalize_pixels: false    # Done during training (dataloader)
```

**Operations:**
1. **Load Image:** Read PNG from `data/raw/KITTI/training/image_2/`
2. **Resize:** Scale to 640×640 (letterbox with padding to maintain aspect ratio)
3. **Format Convert:** PNG → JPG (reduce file size by ~60% with minimal quality loss)
4. **Save:** Write to `data/processed/kitti/images/{split}/`

**Resize Strategy (Letterbox):**
```python
# Maintain aspect ratio, add padding
Original: 1242×375 pixels
Step 1:   Scale to fit 640×640 → 640×193 (aspect ratio preserved)
Step 2:   Add padding (top/bottom) → 640×640 (centered)
Result:   No distortion, realistic appearance
```

**Why JPG instead of PNG?**
- **File Size:** JPG @ 95% quality ≈ 40-60 KB vs PNG ≈ 150-200 KB
- **Training Speed:** Faster I/O (smaller files)
- **Quality Loss:** Minimal at 95% quality (imperceptible)
- **Disk Space:** 7,481 images: ~400 MB (JPG) vs ~1.2 GB (PNG)

---

### Step 3: Label Conversion (KITTI → YOLO)

**Script:** `src/data/kitti_utils.py` (function: `kitti_to_yolo()`)

**YOLO Format:**
```
<class_id> <x_center> <y_center> <width> <height>
│          └────────────┬────────────────┘
│                Normalized coordinates (0.0 to 1.0)
└─ Integer class ID (0, 1, 2, ...)

Example:
0 0.538 0.623 0.091 0.275
│ └───────┬──────────┘
│    Normalized bbox (center + size)
└─ Class 0 (Vehicle)
```

**Conversion Formula:**
```python
# KITTI bounding box (pixels, absolute coordinates)
x1, y1, x2, y2 = 614.24, 181.78, 727.31, 284.77
image_width, image_height = 1242, 375

# Convert to YOLO format (normalized, center-based)
x_center = ((x1 + x2) / 2) / image_width   = 0.538
y_center = ((y1 + y2) / 2) / image_height  = 0.623
width    = (x2 - x1) / image_width         = 0.091
height   = (y2 - y1) / image_height        = 0.275

# Output: 0 0.538 0.623 0.091 0.275
```

**Class Mapping:**
```yaml
label_conversion:
  class_mapping:
    # Merge vehicle types → Class 0
    Car: 0
    Van: 0
    Truck: 0
    
    # Merge pedestrian types → Class 1
    Pedestrian: 1
    Person_sitting: 1
    
    # Cyclists → Class 2
    Cyclist: 2
  
  # Classes to skip (not written to output)
  exclude_classes:
    - "DontCare"     # Ambiguous regions
    - "Misc"         # Unspecified objects
    - "Tram"         # Not relevant for Road-Sense
```

**Filtering:**
```yaml
label_conversion:
  min_bbox_size: 0.005  # Minimum area (0.5% of image)
  
# Explanation:
# At 640×640 pixels:
# 0.5% of image = 0.005 × 640 × 640 = 2,048 pixels²
# Min bbox ≈ 45×45 pixels (very small, but detectable)
# 
# Filters out:
# - Very distant objects (too small to detect reliably)
# - Heavily occluded objects (tiny visible portion)
# - Annotation errors (spurious tiny boxes)
```

---

### Step 4: Dataset Splitting

**Configuration:** `configs/preprocessing.yaml`

```yaml
split:
  train_ratio: 0.7      # 70% → 5,237 images
  val_ratio: 0.2        # 20% → 1,496 images
  test_ratio: 0.1       # 10% → 748 images
  random_seed: 42       # For reproducibility
  shuffle: true         # Randomize before splitting
```

**Process:**
```python
# Pseudocode for splitting
all_images = sorted(glob("data/raw/KITTI/training/image_2/*.png"))

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Shuffle
random.shuffle(all_images)

# Split indices
total = len(all_images)  # 7,481
train_end = int(total * 0.7)  # 5,237
val_end = train_end + int(total * 0.2)  # 6,733

train_images = all_images[:train_end]          # 5,237
val_images = all_images[train_end:val_end]     # 1,496
test_images = all_images[val_end:]             # 748
```

**Output Structure:**
```
data/processed/kitti/
├── images/
│   ├── train/        # 5,237 images
│   ├── val/          # 1,496 images
│   └── test/         # 748 images
└── labels/
    ├── train/        # 5,237 .txt files
    ├── val/          # 1,496 .txt files
    └── test/         # 748 .txt files
```

**Why These Ratios?**
- **70% Training:** Standard for deep learning (sufficient for fine-tuning)
- **20% Validation:** Monitor overfitting, hyperparameter tuning
- **10% Test:** Final evaluation (held out, never seen during training)
- **No Stratification:** Images contain multiple classes, hard to stratify

---

### Step 5: YOLO Configuration File

**Auto-generated:** `data/processed/kitti/data.yaml`

```yaml
# YOLOv8/v11 Dataset Configuration
path: /absolute/path/to/data/processed/kitti  # Dataset root
train: images/train  # Relative to 'path'
val: images/val
test: images/test

# Number of classes
nc: 3

# Class names (must match class_id in labels)
names:
  0: Vehicle      # Car, Van, Truck merged
  1: Pedestrian   # Pedestrian, Person_sitting merged
  2: Cyclist
```

**Usage in YOLO Training:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pre-trained model
results = model.train(
    data='data/processed/kitti/data.yaml',  # ← Uses this file
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

### Step 6: Processing Execution

**Script:** `src/data/preprocess_dataset.py`

```bash
# Method 1: Run as module (recommended)
python -m src.data.preprocess_dataset

# Method 2: Run directly
python src/data/preprocess_dataset.py

# Method 3: With custom config
python -m src.data.preprocess_dataset --config configs/custom_preprocessing.yaml
```

**Output (Example):**
```
[1] Validating input data...
 Found 7,481 images
 Found 7,481 label files
 No corrupted images

[2] Preprocessing KITTI dataset...
Processing: 100%|██████████| 7481/7481 [05:23<00:00, 23.14it/s]

 Successfully processed: 7,481 images
 Total objects: 38,186
   ├─ Vehicle: 32,750 (85.73%)
   ├─ Pedestrian: 4,709 (11.06%)
   └─ Cyclist: 1,627 (4.01%)

[3] Dataset split:
   ├─ Train: 5,237 images (70.00%)
   ├─ Val: 1,496 images (20.01%)
   └─ Test: 748 images (9.99%)

 Output saved to: data/processed/kitti/
 Created data.yaml configuration file

[OK] Preprocessing complete!
```

**Processing Time:**
- **Hardware:** Modern CPU (Intel i7 or AMD Ryzen 7)
- **Time:** ~5-8 minutes for 7,481 images
- **Speed:** ~20-25 images/second

---

## Augmentation Strategy

### Overview

Data augmentation is applied **during training time** (not during preprocessing) using YOLO's built-in augmentation pipeline. This approach:
- ✅ Saves disk space (no duplicate augmented images stored)
- ✅ Provides infinite variations (augmented on-the-fly)
- ✅ Automatically handled by YOLO trainer

### Augmentation Types

#### 1. Geometric Augmentations

**Mosaic Augmentation** (YOLO-specific)
```yaml
mosaic: 1.0  # Probability of mosaic augmentation

# Process:
# 1. Select 4 random training images
# 2. Randomly scale and place in 2×2 grid
# 3. Crop to final size (640×640)
# 
# Benefits:
# - More objects per training image (better for small objects)
# - Diverse context (forces model to handle multiple scenes)
# - Reduces overfitting
```

**Horizontal Flip**
```yaml
fliplr: 0.5  # 50% probability

# Purpose:
# - Handle left-hand vs right-hand traffic
# - Improve generalization to mirrored scenes
# - Double effective dataset size
```

**Scaling and Translation**
```yaml
scale: 0.5   # Scale image by 0.5x to 1.5x
translate: 0.1  # Translate by ±10% of image size

# Purpose:
# - Simulate different camera distances
# - Handle camera vibration/movement
# - Improve detection at various scales
```

**Rotation**
```yaml
degrees: 5.0  # Rotate by ±5 degrees

# Purpose:
# - Handle camera tilt
# - Minor slope variations
# Note: Small rotation to preserve driving realism (roads are mostly flat)
```

**Shear and Perspective**
```yaml
shear: 2.0       # Shear by ±2 degrees
perspective: 0.0001  # Slight perspective distortion

# Purpose:
# - Simulate different camera angles
# - Handle lens distortion
```

#### 2. Photometric Augmentations

**Brightness and Contrast**
```yaml
hsv_h: 0.015  # Hue shift ±1.5%
hsv_s: 0.7    # Saturation adjustment (0.3x to 1.7x)
hsv_v: 0.4    # Value/brightness adjustment (0.6x to 1.4x)

# Purpose:
# - Simulate different lighting conditions
# - Morning/afternoon sun variations
# - Shadow/highlight variations
# - Weather changes (cloudy vs sunny)
```

**Blur and Noise** (Custom, if needed)
```python
# Applied via Albumentations (optional)
import albumentations as A

augmentations = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(blur_limit=7, p=0.2),
])

# Purpose:
# - Simulate camera motion
# - Sensor noise (low light)
# - Out-of-focus regions
```

#### 3. Mixup and CutMix

**Mixup**
```yaml
mixup: 0.1  # 10% probability

# Process:
# 1. Select two images
# 2. Blend: mixed_image = alpha * img1 + (1-alpha) * img2
# 3. Combine labels from both images
# 
# Benefits:
# - Reduces overfitting
# - Improves generalization
# - Creates smoother decision boundaries
```

**CopyPaste** (YOLO-specific)
```yaml
copy_paste: 0.1  # 10% probability

# Process:
# 1. Select instance from another image
# 2. Paste onto current image
# 3. Update labels
# 
# Benefits:
# - Increases object count per image
# - Better for minority classes (Pedestrian, Cyclist)
```

### Augmentation Configuration

**YOLO Training with Augmentation:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data/processed/kitti/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    
    # Augmentation parameters
    mosaic=1.0,         # Enable mosaic
    mixup=0.1,          # 10% mixup
    copy_paste=0.1,     # 10% copy-paste
    degrees=5.0,        # ±5° rotation
    translate=0.1,      # ±10% translation
    scale=0.5,          # 0.5x to 1.5x scaling
    shear=2.0,          # ±2° shear
    perspective=0.0001, # Slight perspective
    flipud=0.0,         # No vertical flip (unrealistic for driving)
    fliplr=0.5,         # 50% horizontal flip
    hsv_h=0.015,        # HSV augmentation
    hsv_s=0.7,
    hsv_v=0.4,
)
```

### Custom Augmentation Pipeline (Optional)

**Script:** `src/data/augment_dataset.py`

If you want to pre-generate augmented images (e.g., for exploration):

```bash
# Generate augmented samples
python src/data/augment_dataset.py \
    --input data/processed/kitti/images/train \
    --output data/augmented/images \
    --num_augmentations 2 \
    --visualize
```

**Augmentation Pipeline (Albumentations):**
```python
import albumentations as A

augmentation_pipeline = A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=0.5),
    
    # Photometric
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    
    # Weather simulation
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
    A.RandomRain(slant_lower=-10, slant_upper=10, p=0.1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## Configuration Management

### Main Configuration File

**Location:** `configs/preprocessing.yaml`

```yaml
# ====================================
# KITTI Dataset Preprocessing Config
# ====================================

# Input paths (raw KITTI data)
input:
  raw_data_dir: "data/raw/KITTI/training"
  image_subdir: "image_2"
  label_subdir: "label_2"

# Output paths (processed YOLO-ready data)
output:
  processed_dir: "data/processed/kitti"
  create_yolo_yaml: true

# Image preprocessing
image_processing:
  target_size: [640, 640]
  save_format: "jpg"
  jpeg_quality: 95
  normalize_pixels: false  # Done in dataloader

# Label conversion (KITTI → YOLO)
label_conversion:
  format: "yolo"
  
  class_mapping:
    Car: 0
    Van: 0
    Truck: 0
    Pedestrian: 1
    Person_sitting: 1
    Cyclist: 2
  
  exclude_classes:
    - "DontCare"
    - "Misc"
    - "Tram"
  
  min_bbox_size: 0.005

# Dataset splitting
split:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  random_seed: 42
  shuffle: true

# Processing options
processing:
  skip_on_error: true
  show_progress: true
  verify_output: true

# YOLO data.yaml configuration
yolo_config:
  names:
    0: "Vehicle"
    1: "Pedestrian"
    2: "Cyclist"
  nc: 3
```

### Multi-Dataset Configuration (Future)

**Location:** `configs/multi_dataset_preprocessing.yaml`

For Stage 2 (adding traffic signs from GTSDB):

```yaml
# Multi-dataset configuration (KITTI + GTSDB)
datasets:
  - name: "kitti"
    type: "kitti"
    path: "data/raw/KITTI/training"
    classes: [0, 1, 2]  # Vehicle, Pedestrian, Cyclist
    
  - name: "gtsdb"
    type: "gtsdb"
    path: "data/raw/GTSDB"
    classes: [3, 4, 5, 6, 7]  # Traffic signs (5 categories)

output:
  processed_dir: "data/processed/multi_dataset"
  nc: 8  # 3 (KITTI) + 5 (GTSDB)
```

---

## Execution Guide

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# - opencv-python
# - numpy
# - pyyaml
# - tqdm
# - pillow
# - ultralytics (YOLO)
```

### Step-by-Step Execution

#### 1. Validate Raw Data

```bash
# Check data quality before preprocessing
python src/data/validate_kitti_quality.py

# Expected output:
# ✅ Total PNG Images Found: 7481
# ✅ Corrupted Images: 0
# ✅ Missing Label Files: 0
# ✅ Invalid/Out-of-bounds Bboxes: 0
# ✅ Exact Duplicates Found: 0
```

#### 2. Run Preprocessing

```bash
# Run with default configuration
python -m src.data.preprocess_dataset

# Or with custom config
python -m src.data.preprocess_dataset --config configs/custom_preprocessing.yaml
```

#### 3. Verify Output

```bash
# Check output structure
ls -lh data/processed/kitti/

# Expected:
# data.yaml
# images/train/ (5,237 files)
# images/val/ (1,496 files)
# images/test/ (748 files)
# labels/train/ (5,237 files)
# labels/val/ (1,496 files)
# labels/test/ (748 files)
```

#### 4. Verify YOLO Labels

```bash
# Run verification script
python src/data/verify_dataset.py

# Checks:
# ✅ All images have corresponding labels
# ✅ All labels have valid format
# ✅ Bounding boxes in range [0, 1]
# ✅ Class IDs match data.yaml
```

#### 5. Visualize Samples

```python
# Quick visualization
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load a sample image and label
img_path = "data/processed/kitti/images/train/000000.jpg"
label_path = "data/processed/kitti/labels/train/000000.txt"

img = Image.open(img_path)
with open(label_path, 'r') as f:
    labels = f.readlines()

# Parse and visualize (use YOLO's built-in plotting)
model = YOLO('yolov8n.pt')

# Visualize with labels
results = model(img_path, save=True)
# Output saved to runs/detect/predict/
```

---

## Validation and Quality Checks

### Pre-Preprocessing Validation

**Script:** `src/data/validate_kitti_quality.py`

```python
# What it checks:
1. Image integrity (no corrupted files)
2. Label file existence (1:1 mapping)
3. Bounding box validity (in range [0, img_width], [0, img_height])
4. Duplicate detection (SHA256 hash comparison)
5. Format compliance (KITTI format parsing)

# Usage:
python src/data/validate_kitti_quality.py

# Output:
# - Console report (pass/fail)
# - Log file: logs/data_quality.log
```

### Post-Preprocessing Validation

**Script:** `src/data/verify_dataset.py`

```python
# What it checks:
1. YOLO format compliance (class_id, x_center, y_center, w, h)
2. Coordinate range validation (all values in [0.0, 1.0])
3. Class ID consistency (match data.yaml)
4. Image-label pairing (every image has a label, vice versa)
5. Dataset statistics (object counts, class distribution)

# Usage:
python src/data/verify_dataset.py --data data/processed/kitti/data.yaml

# Output:
# - Pass/fail report
# - Statistics (objects per class, per split)
```

### Statistical Validation

**Script:** `scripts/quick_stats.py`

```bash
# Generate statistics
python scripts/quick_stats.py

# Outputs:
# - experiments/visualization/dataset_analysis/dataset_statistics.csv
# - Class distribution plots
# - Bounding box size histograms
# - Objects per image distribution
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Corrupted image" error
```bash
# Symptom:
# Error: Cannot read image at data/raw/KITTI/training/image_2/000123.png

# Solution:
# 1. Run validation script
python src/data/validate_kitti_quality.py

# 2. If corrupted images found, re-download KITTI dataset
# 3. Verify MD5 checksums (see KITTI website)
```

#### Issue 2: "Missing label file" error
```bash
# Symptom:
# Warning: Label file not found for 000456.png

# Solution:
# 1. Check that image and label counts match:
ls data/raw/KITTI/training/image_2/ | wc -l
ls data/raw/KITTI/training/label_2/ | wc -l

# 2. Set skip_on_error: true in config to skip missing files
```

#### Issue 3: "Invalid bounding box" warning
```bash
# Symptom:
# Warning: Bounding box out of range for 000789.txt

# Cause:
# - Label file has coordinates outside image dimensions
# - Annotation error in KITTI dataset

# Solution:
# - Set skip_on_error: true (script will skip invalid bbox)
# - Or manually fix the label file
```

#### Issue 4: "Out of memory" error
```bash
# Symptom:
# MemoryError during preprocessing

# Solution:
# 1. Reduce batch size (process fewer images at once)
# 2. Close other applications
# 3. Use smaller image size (e.g., 416×416 instead of 640×640)
```

#### Issue 5: YOLO training fails to find data
```bash
# Symptom:
# FileNotFoundError: data.yaml not found

# Solution:
# 1. Use absolute path in data.yaml:
path: /home/abdallah/Coding/DEPI-Project/Road-Sense/data/processed/kitti

# 2. Or train from project root directory:
cd /home/abdallah/Coding/DEPI-Project/Road-Sense
python train.py
```

### Debugging Tips

```bash
# 1. Check preprocessing output
python -m src.data.preprocess_dataset 2>&1 | tee preprocessing.log

# 2. Validate one image at a time
python -c "
from src.data import kitti_utils
img_path = 'data/raw/KITTI/training/image_2/000000.png'
label_path = 'data/raw/KITTI/training/label_2/000000.txt'
kitti_utils.validate_single_image(img_path, label_path)
"

# 3. Check YOLO format
head -5 data/processed/kitti/labels/train/000000.txt
# Should output:
# 0 0.538 0.623 0.091 0.275
# 1 0.321 0.456 0.032 0.078
# ...
```

---

## Appendix: File Locations

### Scripts
- **Preprocessing:** `src/data/preprocess_dataset.py`
- **Validation:** `src/data/validate_kitti_quality.py`
- **Verification:** `src/data/verify_dataset.py`
- **Augmentation:** `src/data/augment_dataset.py`
- **Utilities:** `src/data/kitti_utils.py`

### Configuration
- **Main Config:** `configs/preprocessing.yaml`
- **Multi-Dataset Config:** `configs/multi_dataset_preprocessing.yaml`

### Data Directories
- **Raw Data:** `data/raw/KITTI/training/`
- **Processed Data:** `data/processed/kitti/`
- **Augmented Data:** `data/augmented/` (optional)

### Documentation
- **This Guide:** `docs/PREPROCESSING_AND_AUGMENTATION_GUIDE.md`
- **Dataset Exploration:** `docs/DATASET_EXPLORATION_REPORT.md`
- **Training Strategy:** `docs/MULTI_DATASET_TRAINING_STRATEGY.md`

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Maintained By:** Road-Sense Team
