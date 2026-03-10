# Dataset Upload Guidelines
**Project:** Road-Sense - Real-Time Object Detection for Autonomous Vehicles  
**Purpose:** Git repository best practices for dataset management  
**Last Updated:** March 2026

---

## Overview

This document explains what to upload to Git for reproducibility and what to exclude to avoid repository bloat. Large datasets should **NOT** be uploaded to Git. Instead, we provide dataset metadata, download instructions, and sample files for validation.

### ⚠️ Critical Rule: DO NOT Upload Raw Datasets to Git

**Why?**
- **Size:** KITTI dataset is ~12 GB (training images + labels)
- **Git Limit:** GitHub has a 100 MB single file limit, 1 GB repository soft limit
- **Performance:** Large files slow down clone/push/pull operations
- **Redundancy:** Raw datasets are publicly available (KITTI, COCO, etc.)

---

## What to Upload to Git ✅

### 1. Dataset Metadata and Configuration

**Upload these files:**
```
configs/
├── preprocessing.yaml                    # Preprocessing configuration
└── multi_dataset_preprocessing.yaml      # Multi-dataset config

docs/
├── DATASET_EXPLORATION_REPORT.md         # Complete dataset analysis
├── PREPROCESSING_AND_AUGMENTATION_GUIDE.md  # Pipeline documentation
├── DATASET_UPLOAD_GUIDELINES.md          # This document
└── data_quality_report.md                # Validation report

experiments/visualization/dataset_analysis/
└── dataset_statistics.csv                # Class distribution stats (5 KB)

reports/research/
├── Abdallah_dataset_analysis.md          # Dataset comparison
└── AyaAhmed_dataset_analysis.md          # Additional analysis
```

**Why?**
- Small text files (< 100 KB each)
- Essential for reproducing the preprocessing pipeline
- Documents dataset characteristics and decisions

---

### 2. Dataset Download Instructions

**File:** `docs/DATASET_DOWNLOAD_INSTRUCTIONS.md` (see below)

```markdown
# Dataset Download Instructions

## KITTI Vision Benchmark Suite

### 1. Register and Download
1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
2. Create an account (free)
3. Download the following files:
   - **Left color images of object data set (12 GB)** → `data_object_image_2.zip`
   - **Training labels of object data set (5 MB)** → `data_object_label_2.zip`

### 2. Extract Files
```bash
# Create directory structure
mkdir -p data/raw/KITTI/training

# Extract images
unzip data_object_image_2.zip -d data/raw/KITTI/training/

# Extract labels
unzip data_object_label_2.zip -d data/raw/KITTI/training/

# Verify structure
ls data/raw/KITTI/training/
# Expected output:
# image_2/  (7,481 PNG files)
# label_2/  (7,481 TXT files)
```

### 3. Verify Download (Optional)
```bash
# Run validation script
python src/data/validate_kitti_quality.py

# Expected output:
# ✅ Total PNG Images Found: 7481
# ✅ Corrupted Images: 0
# ✅ Missing Label Files: 0
```

## COCO Dataset (for pre-trained weights only)

No manual download needed. Pre-trained YOLOv8/v11 models are automatically downloaded:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads from Ultralytics automatically
```

## Future: Traffic Sign Dataset (Stage 2)

### German Traffic Sign Detection Benchmark (GTSDB)
1. Visit: https://benchmark.ini.rub.de/gtsdb_dataset.html
2. Download: Training images + annotations
3. Extract to: `data/raw/GTSDB/`


**Include this file in Git:** ✅ Yes (it's documentation)

---

### 3. Sample Dataset Files

**Purpose:** Allow users to test preprocessing without downloading full dataset

**Create sample directory:**
```
data/samples/
├── kitti/
│   ├── image_2/
│   │   ├── 000000.png      # Sample image 1
│   │   ├── 000001.png      # Sample image 2
│   │   ├── 000002.png      # Sample image 3
│   │   ├── 000003.png      # Sample image 4
│   │   └── 000004.png      # Sample image 5
│   └── label_2/
│       ├── 000000.txt      # Sample label 1
│       ├── 000001.txt      # Sample label 2
│       ├── 000002.txt      # Sample label 3
│       ├── 000003.txt      # Sample label 4
│       └── 000004.txt      # Sample label 5
└── README.md               # Explanation of samples
```

**Sample README.md:**
```markdown
# Sample Dataset Files

This directory contains 5 sample images and labels from the KITTI dataset for testing the preprocessing pipeline.

## Usage

Test preprocessing on samples:
```bash
# Modify config to use samples
python -m src.data.preprocess_dataset --config configs/sample_preprocessing.yaml
```

## Note
These are *samples only*. Download the full KITTI dataset (see docs/DATASET_DOWNLOAD_INSTRUCTIONS.md) for model training.
```

**Include in Git:**
- ✅ 5-10 sample images (~2-5 MB total)
- ✅ Corresponding labels (< 5 KB total)
- ✅ README.md

**Why?**
- Test preprocessing without full download
- Verify environment setup works
- Small enough for Git (< 5 MB total)

---

### 4. Preprocessing Scripts and Utilities
src/data/
├── __init__.py
├── preprocess_dataset.py              # Main preprocessing script
├── kitti_utils.py                     # KITTI format utilities
├── augmentations.py                   # Augmentation pipeline
├── validate_kitti_quality.py          # Data quality validation
├── verify_dataset.py                  # Post-processing verification
├── augment_dataset.py                 # Augmentation script
└── PREPROCESSING.md                   # Documentation

scripts/
├── dataset_exploration_analysis.py    # EDA script
├── quick_stats.py                     # Statistics generation
└── quick_visualization.py             # Visualization script

tests/
├── test_kitti_utils.py                # Unit tests
└── test_augmentations.py              # Augmentation tests

```

**Why?**
- Essential for reproducibility
- Users can preprocess data themselves
- Version-controlled code changes

---

### 5. Data Statistics and Analysis Results

**Upload these outputs:**
```
experiments/visualization/dataset_analysis/
├── dataset_statistics.csv             # Class distribution (5 KB)
├── class_distribution_plot.png        # Visualization (50 KB)
├── bbox_size_distribution.png         # Visualization (60 KB)
└── objects_per_image_histogram.png    # Visualization (45 KB)

docs/
└── data_quality_report.md             # Validation results (1 KB)
```

**Why?**
- Small files (< 100 KB each)
- Useful for understanding dataset without downloading
- Documents baseline assumptions

---

### 6. Processed Dataset Structure (Empty Directories)

**Create placeholder directories with .gitkeep:**
```bash
# Create structure
mkdir -p data/{raw,processed,augmented}/{kitti,gtsdb}
mkdir -p data/processed/kitti/{images,labels}/{train,val,test}

# Add .gitkeep to preserve structure
find data -type d -exec touch {}/.gitkeep \;
```

**Git will track:**
```
data/
├── raw/
│   ├── KITTI/
│   │   └── .gitkeep
│   └── GTSDB/
│       └── .gitkeep
├── processed/
│   └── kitti/
│       ├── images/
│       │   ├── train/.gitkeep
│       │   ├── val/.gitkeep
│       │   └── test/.gitkeep
│       └── labels/
│           ├── train/.gitkeep
│           ├── val/.gitkeep
│           └── test/.gitkeep
├── augmented/
│   └── .gitkeep
└── samples/
    └── kitti/
        ├── image_2/ (5 sample images)
        └── label_2/ (5 sample labels)
```

**Why?**
- Preserves directory structure for new users
- `.gitkeep` files are tracked (empty dirs are not)
- Users know where to place downloaded data

---

## What NOT to Upload to Git ❌

### 1. Raw Dataset Files

**DO NOT upload:**
```
❌ data/raw/KITTI/training/image_2/*.png     (12 GB)
❌ data/raw/KITTI/training/label_2/*.txt     (5 MB, but 7,481 files)
❌ data/raw/KITTI/testing/image_2/*.png      (7 GB)
❌ data/raw/GTSDB/ (entire directory)
```

**Exception:** 5-10 sample files (see Section 3)

**Why?**
- Too large for Git (12+ GB)
- Publicly available (users can download)
- Violates GitHub usage policies

---

### 2. Processed Dataset Files

**DO NOT upload:**
```
❌ data/processed/kitti/images/train/*.jpg   (400 MB)
❌ data/processed/kitti/images/val/*.jpg     (120 MB)
❌ data/processed/kitti/images/test/*.jpg    (60 MB)
❌ data/processed/kitti/labels/**/*.txt      (2 MB, but 7,481 files)
```

**Why?**
- Users can generate these by running preprocessing
- Adds 500+ MB to repository
- Changes with every preprocessing config update

---

### 3. Augmented Dataset Files

**DO NOT upload:**
```
❌ data/augmented/images/*.jpg               (Potentially 1+ GB)
❌ data/augmented/labels/*.txt               (Potentially 10+ MB)
```

**Why?**
- Generated on-the-fly during training (YOLO augmentation)
- If pre-augmented, can be regenerated with script
- Extremely large (multiple GB)

---

### 4. Model Checkpoints (Partially Upload)

**DO NOT upload:**
```
❌ models/checkpoints/yolov8_kitti_epoch*.pt  (Each file ~6 MB, 100+ epochs = 600+ MB)
❌ runs/detect/train/weights/*.pt             (Training run outputs)
```

**DO upload:**
```
✅ models/checkpoints/best.pt                 (Final best model, ~6 MB) - Optional
✅ models/checkpoints/final.pt                (Final model, ~6 MB) - Optional
```

**Why NOT upload all checkpoints?**
- 100 epochs × 6 MB/epoch = 600 MB
- Users only need best model
- Training outputs can be regenerated

**Alternative: Use Git LFS**
```bash
# If you must track large models, use Git Large File Storage
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/checkpoints/best.pt
git commit -m "Add best model"
```

---

### 5. Training Run Artifacts

**DO NOT upload:**
```
❌ runs/detect/train/                        (1+ GB, contains images/plots)
❌ runs/detect/val/
❌ wandb/                                     (Weights & Biases local cache)
❌ lightning_logs/                            (PyTorch Lightning logs)
```

**Alternative:**
- Use `.gitignore` to exclude these directories
- Push metrics to remote logging (Weights & Biases, TensorBoard, MLflow)

---

## .gitignore Configuration

**Create `.gitignore` at project root:**

```gitignore
# ====================================
# Road-Sense .gitignore
# ====================================

# =====================================
# Raw Datasets (DO NOT COMMIT)
# =====================================
data/raw/KITTI/training/image_2/*.png
data/raw/KITTI/training/label_2/*.txt
data/raw/KITTI/testing/
data/raw/GTSDB/
data/raw/COCO/

# =====================================
# Processed Datasets (DO NOT COMMIT)
# =====================================
data/processed/*/images/**/*.jpg
data/processed/*/images/**/*.png
data/processed/*/labels/**/*.txt

# =====================================
# Augmented Data (DO NOT COMMIT)
# =====================================
data/augmented/

# =====================================
# Model Checkpoints (Except Best/Final)
# =====================================
models/checkpoints/*
!models/checkpoints/best.pt
!models/checkpoints/final.pt
!models/checkpoints/.gitkeep

# =====================================
# Training Artifacts (DO NOT COMMIT)
# =====================================
runs/
wandb/
lightning_logs/
mlruns/
.tensorboard/

# =====================================
# Temporary Files
# =====================================
*.log
*.tmp
*.cache
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.ipynb_checkpoints/

# =====================================
# Environment Files
# =====================================
.env
.venv/
venv/
env/
ENV/
*.egg-info/

# =====================================
# IDE Files
# =====================================
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# =====================================
# Jupyter Notebook Outputs (Keep Code)
# =====================================
notebooks/.ipynb_checkpoints/

# ====================================
# ALLOWED: Sample Files (5-10 samples)
# ====================================
!data/samples/kitti/image_2/*.png
!data/samples/kitti/label_2/*.txt
```

**Add to Git:**
```bash
git add .gitignore
git commit -m "Add .gitignore for dataset management"
```

---

## Repository Size Management

### Before First Commit

**Check repository size:**
```bash
# Check size of files to be committed
git add --dry-run -A && git diff --cached --stat

# Check total directory size
du -sh .git/
du -sh data/
```

**Expected sizes:**
| Directory | Size | Include in Git? |
|-----------|------|-----------------|
| `src/` | < 1 MB | ✅ Yes |
| `configs/` | < 100 KB | ✅ Yes |
| `docs/` | < 500 KB | ✅ Yes |
| `tests/` | < 500 KB | ✅ Yes |
| `scripts/` | < 200 KB | ✅ Yes |
| `data/samples/` | < 5 MB | ✅ Yes (5-10 files) |
| `data/raw/` | 12+ GB | ❌ **NO** (add to .gitignore) |
| `data/processed/` | 500+ MB | ❌ **NO** (add to .gitignore) |
| `experiments/visualization/` | < 500 KB | ✅ Yes (plots/CSV only) |

**Total repository size (with .gitignore):** < 10 MB ✅

---

### After Accidentally Committing Large Files

**If you committed large files by mistake:**

```bash
# Remove large files from Git history
git filter-branch --tree-filter 'rm -rf data/raw/KITTI/training/image_2' HEAD

# Or use BFG Repo-Cleaner (faster)
# Download BFG: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-folders data/raw/KITTI --no-blob-protection .git

# Force push (WARNING: Rewrites history)
git push origin main --force
```

**Prevent future mistakes:**
```bash
# Add pre-commit hook to check file sizes
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Check for files larger than 10 MB
MAX_SIZE=10485760  # 10 MB in bytes

large_files=$(git diff --cached --name-only | while read file; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [ "$size" -gt "$MAX_SIZE" ]; then
            echo "$file ($(($size / 1048576)) MB)"
        fi
    fi
done)

if [ -n "$large_files" ]; then
    echo "❌ ERROR: The following files are larger than 10 MB:"
    echo "$large_files"
    echo ""
    echo "Add them to .gitignore or use Git LFS."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

---

## Alternative: Using Git LFS (Advanced)

If you *must* track large files (e.g., best model checkpoint):

### Install Git LFS

```bash
# Install Git LFS
# Ubuntu/Debian:
sudo apt-get install git-lfs

# macOS:
brew install git-lfs

# Windows: Download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install
```

### Track Large Files

```bash
# Track model checkpoints
git lfs track "*.pt"
git lfs track "models/checkpoints/*.pt"

# Track sample images (optional)
git lfs track "data/samples/**/*.png"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Verify LFS Tracking

```bash
# Check tracked files
git lfs ls-files

# Check file size
git lfs status
```

**Note:** Git LFS has storage limits on GitHub (1 GB free, then paid tiers)

---

## Reproducibility Checklist

To ensure your dataset pipeline is reproducible:

### ✅ Include in Git:
- [ ] `configs/preprocessing.yaml` (preprocessing configuration)
- [ ] `src/data/*.py` (all preprocessing scripts)
- [ ] `docs/DATASET_EXPLORATION_REPORT.md` (dataset analysis)
- [ ] `docs/PREPROCESSING_AND_AUGMENTATION_GUIDE.md` (step-by-step guide)
- [ ] `docs/DATASET_DOWNLOAD_INSTRUCTIONS.md` (where to download datasets)
- [ ] `.gitignore` (exclude large files)
- [ ] `requirements.txt` (dependencies)
- [ ] `data/samples/` (5-10 sample files for testing)
- [ ] `experiments/visualization/dataset_analysis/dataset_statistics.csv` (stats)

### ✅ Document in README.md:
- [ ] Dataset sources (KITTI, COCO links)
- [ ] Download instructions (step-by-step)
- [ ] Preprocessing commands (how to run scripts)
- [ ] Expected directory structure
- [ ] Sample outputs (what users should see)

### ❌ Exclude from Git:
- [ ] `data/raw/KITTI/` (12+ GB raw data)
- [ ] `data/processed/` (500+ MB processed data)
- [ ] `data/augmented/` (1+ GB augmented data)
- [ ] `runs/` (training artifacts)
- [ ] `wandb/`, `lightning_logs/` (experiment logs)

---

## Sharing Large Files (Alternatives to Git)

If you need to share preprocessed data with team members:

### Option 1: Cloud Storage
**Upload to:**
- Google Drive (15 GB free)
- Dropbox (2 GB free)
- OneDrive (5 GB free)
- AWS S3 (pay-as-you-go)

**Share link in README:**
```markdown
## Preprocessed Dataset (Optional)

If you don't want to preprocess yourself, download the preprocessed dataset:
- **Google Drive:** https://drive.google.com/file/d/YOUR_FILE_ID
- **Size:** 500 MB (ZIP)
- **Contents:** data/processed/kitti/ (ready for YOLO training)

Extract to project root:
```bash
unzip kitti_processed.zip -d data/processed/
```
```

### Option 2: Academic Data Repositories
**Zenodo (free, unlimited for research):**
1. Create account: https://zenodo.org/
2. Upload dataset (~500 MB)
3. Get DOI (permanent link)
4. Add DOI to README.md

### Option 3: DVC (Data Version Control)
**For Teams:**
```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Track data directory
dvc add data/processed/kitti

# Commit to Git (only tracks metadata, not files)
git add data/processed/kitti.dvc .dvc/config
git commit -m "Track processed dataset with DVC"

# Push data to remote storage (e.g., AWS S3)
dvc remote add -d storage s3://mybucket/road-sense
dvc push
```

**Users pull data:**
```bash
# Clone repo
git clone https://github.com/your-team/Road-Sense.git

# Pull data
dvc pull
```

---

## Summary: Quick Reference

### What to Upload ✅
1. **Code:** All `.py` scripts
2. **Configs:** `.yaml` files
3. **Docs:** `.md` files
4. **Stats:** `.csv` files, plots (< 100 KB each)
5. **Samples:** 5-10 example images/labels
6. **.gitignore:** Exclude large files
7. **Structure:** Empty directories with `.gitkeep`

### What NOT to Upload ❌
1. **Raw Data:** `data/raw/KITTI/` (12+ GB)
2. **Processed Data:** `data/processed/` (500+ MB)
3. **Augmented Data:** `data/augmented/` (1+ GB)
4. **Training Runs:** `runs/`, `wandb/` (1+ GB)
5. **All Checkpoints:** Keep only `best.pt` (6 MB)

### Repository Size Target
- **Total:** < 10 MB (without large files)
- **With best model:** < 20 MB (if including `best.pt`)

### Reproducibility Strategy
1. **Document:** How to download datasets
2. **Provide:** Scripts to preprocess data
3. **Share:** Configuration files
4. **Validate:** Sample files for testing
5. **Version Control:** Code, not data

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Maintained By:** Road-Sense Team

**Related Documents:**
- [Dataset Exploration Report](DATASET_EXPLORATION_REPORT.md)
- [Preprocessing and Augmentation Guide](PREPROCESSING_AND_AUGMENTATION_GUIDE.md)
- [Dataset Download Instructions](#) (create separately)
