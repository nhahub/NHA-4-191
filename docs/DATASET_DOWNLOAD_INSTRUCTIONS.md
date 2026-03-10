# Dataset Download Instructions
**Project:** Road-Sense - Real-Time Object Detection for Autonomous Vehicles  
**Last Updated:** March 2026

---

## Overview

This document provides step-by-step instructions for downloading and setting up the datasets required for the Road-Sense project.

---

## KITTI Vision Benchmark Suite (Primary Dataset)

### 1. Register and Access

1. **Visit the KITTI website:**  
   http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

2. **Create an account:**
   - Click "Register" in the top right
   - Fill in your information
   - Verify your email
   - Login to access downloads

### 2. Download Required Files

Download the following files from the **2D Object Detection** section:

| File | Size | Description |
|------|------|-------------|
| `data_object_image_2.zip` | 12 GB | Left color images (7,481 training + 7,518 testing) |
| `data_object_label_2.zip` | 5 MB | Training labels (7,481 annotation files) |

**Direct Links (after login):**
- Images: http://www.cvlibs.net/download.php?file=data_object_image_2.zip
- Labels: http://www.cvlibs.net/download.php?file=data_object_label_2.zip

### 3. Extract and Organize

```bash
# Navigate to project root
cd /path/to/Road-Sense

# Create directory structure
mkdir -p data/raw/KITTI/training

# Extract images
unzip data_object_image_2.zip -d data/raw/KITTI/training/

# Extract labels
unzip data_object_label_2.zip -d data/raw/KITTI/training/

# Verify structure
ls data/raw/KITTI/training/
# Expected output:
# image_2/  label_2/
```

### 4. Verify Dataset Integrity

```bash
# Check file counts
echo "Images: $(ls data/raw/KITTI/training/image_2/*.png | wc -l)"
echo "Labels: $(ls data/raw/KITTI/training/label_2/*.txt | wc -l)"

# Expected output:
# Images: 7481
# Labels: 7481
```

### 5. Run Quality Validation

```bash
# Run validation script
python src/data/validate_kitti_quality.py

# Expected output:
# ========================================
#       KITTI QUALITY ASSESSMENT REPORT
# ========================================
# Total PNG Images Found:     7481
# Corrupted Images:           0
# Missing Label Files:        0
# Invalid/Out-of-bounds Bboxes: 0
# Exact Duplicates Found:     0
# Final Clean Images:         7481
# ========================================
# Result: Dataset is CLEAN and ready for preprocessing!
```

---

## COCO Dataset (Pre-trained Weights Only)

**No manual download required.**

Pre-trained YOLOv8/v11 models trained on COCO are automatically downloaded by the Ultralytics library:

```python
from ultralytics import YOLO

# Downloads pre-trained COCO weights automatically (~6 MB)
model = YOLO('yolov8n.pt')   # Nano model
model = YOLO('yolov8s.pt')   # Small model
model = YOLO('yolov8m.pt')   # Medium model
```

**Model sizes:**
- YOLOv8n: 6 MB (fastest, least accurate)
- YOLOv8s: 22 MB (balanced)
- YOLOv8m: 52 MB (slower, more accurate)

**Where models are stored:**
- Linux/Mac: `~/.cache/torch/hub/ultralytics/`
- Windows: `C:\Users\YourName\.cache\torch\hub\ultralytics\`

---

## German Traffic Sign Detection Benchmark (GTSDB) - Stage 2

**⚠️ Not needed for Milestone 1. Required for Stage 2 (traffic sign detection).**

### 1. Access the Dataset

1. **Visit:** https://benchmark.ini.rub.de/gtsdb_dataset.html

2. **Request Access:**
   - Fill out the registration form
   - Agree to terms of use
   - Wait for email confirmation

3. **Download:**
   - Training images + annotations: `FullIJCNN2013.zip` (~300 MB)
   - Test images: `TestIJCNN2013.zip` (~200 MB)

### 2. Extract and Organize

```bash
# Create directory structure
mkdir -p data/raw/GTSDB

# Extract training data
unzip FullIJCNN2013.zip -d data/raw/GTSDB/training/

# Extract test data
unzip TestIJCNN2013.zip -d data/raw/GTSDB/testing/

# Verify structure
ls data/raw/GTSDB/
# Expected output:
# training/  testing/
```

**Dataset size:**
- Training: ~900 images with bounding boxes
- Test: ~300 images

---

## Alternative: Use Sample Data for Testing

If you just want to test the preprocessing pipeline without downloading the full dataset:

### 1. Use Provided Samples

The repository includes 5 sample images and labels:

```bash
# Sample files are already in the repository
ls data/samples/kitti/
# image_2/  label_2/

# Test preprocessing on samples
python -m src.data.preprocess_dataset --config configs/sample_preprocessing.yaml
```

### 2. Expected Output

```
data/processed/kitti_samples/
├── data.yaml
├── images/
│   ├── train/   (3 images)
│   ├── val/     (1 image)
│   └── test/    (1 image)
└── labels/
    ├── train/   (3 labels)
    ├── val/     (1 label)
    └── test/    (1 label)
```

---

## Troubleshooting

### Issue: Download is very slow

**Solution:**
- Use a download manager (e.g., `wget`, `curl`, `aria2c`)
- Resume interrupted downloads

```bash
# Download with wget (supports resume)
wget -c http://www.cvlibs.net/download.php?file=data_object_image_2.zip

# Download with aria2c (multi-connection, faster)
aria2c -x 16 -s 16 http://www.cvlibs.net/download.php?file=data_object_image_2.zip
```

### Issue: Extraction fails (corrupted archive)

**Solution:**
1. Verify file integrity:
```bash
# Check file size
ls -lh data_object_image_2.zip
# Expected: ~12 GB

# Calculate MD5 checksum (if provided by KITTI)
md5sum data_object_image_2.zip
```

2. If corrupted, re-download the file

### Issue: Not enough disk space

**Requirements:**
- KITTI raw data: 12 GB
- KITTI processed data: 500 MB
- Total required: ~15 GB free space

**Solution:**
- Free up disk space
- Use external hard drive
- Process dataset on a different machine

### Issue: "Permission denied" when extracting

**Solution:**
```bash
# Check file permissions
ls -l data_object_image_2.zip

# Add read permission if needed
chmod +r data_object_image_2.zip

# Extract with sudo (if necessary)
sudo unzip data_object_image_2.zip -d data/raw/KITTI/training/
```

---

## Dataset Directory Structure (Final)

After downloading and organizing all datasets, your structure should look like:

```
Road-Sense/
├── data/
│   ├── raw/
│   │   ├── KITTI/
│   │   │   ├── training/
│   │   │   │   ├── image_2/          # 7,481 PNG images
│   │   │   │   └── label_2/          # 7,481 TXT labels
│   │   │   └── testing/
│   │   │       └── image_2/          # 7,518 PNG images (no labels)
│   │   └── GTSDB/                    # (Stage 2 only)
│   │       ├── training/
│   │       └── testing/
│   ├── processed/
│   │   └── kitti/                    # Generated by preprocessing
│   │       ├── data.yaml
│   │       ├── images/
│   │       │   ├── train/
│   │       │   ├── val/
│   │       │   └── test/
│   │       └── labels/
│   │           ├── train/
│   │           ├── val/
│   │           └── test/
│   └── samples/
│       └── kitti/                    # Included in repository
│           ├── image_2/              # 5 sample images
│           └── label_2/              # 5 sample labels
├── src/
├── configs/
├── docs/
└── ...
```

---

## Verification Checklist

After downloading, verify that:

- [ ] KITTI training images: 7,481 files in `data/raw/KITTI/training/image_2/`
- [ ] KITTI training labels: 7,481 files in `data/raw/KITTI/training/label_2/`
- [ ] File count matches: `ls image_2/*.png | wc -l` = `ls label_2/*.txt | wc -l`
- [ ] No corrupted images: Run `python src/data/validate_kitti_quality.py`
- [ ] Pre-trained YOLO weights: Downloaded automatically on first use
- [ ] (Optional) GTSDB data: Only for Stage 2

---

## Next Steps

After downloading the dataset:

1. **Run quality validation:**
   ```bash
   python src/data/validate_kitti_quality.py
   ```

2. **Run preprocessing:**
   ```bash
   python -m src.data.preprocess_dataset
   ```

3. **Verify output:**
   ```bash
   python src/data/verify_dataset.py --data data/processed/kitti/data.yaml
   ```

4. **Proceed to model training (Milestone 2):**
   ```bash
   python train.py
   ```

---

## References

- **KITTI Dataset:** http://www.cvlibs.net/datasets/kitti/
- **KITTI Paper:** Geiger, A., Lenz, P., & Urtasun, R. (2012). "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite." CVPR 2012.
- **Ultralytics YOLO:** https://docs.ultralytics.com/
- **GTSDB:** https://benchmark.ini.rub.de/gtsdb_dataset.html

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Maintained By:** Road-Sense Team

**Related Documents:**
- [Dataset Exploration Report](DATASET_EXPLORATION_REPORT.md)
- [Preprocessing and Augmentation Guide](PREPROCESSING_AND_AUGMENTATION_GUIDE.md)
- [Dataset Upload Guidelines](DATASET_UPLOAD_GUIDELINES.md)
