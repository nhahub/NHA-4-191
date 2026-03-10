# Quick Setup Guide
**Project:** Road-Sense - Real-Time Object Detection for Autonomous Vehicles  
**Purpose:** Get started in 5 minutes  
**Last Updated:** March 2026

---

## 🚀 Quick Setup (5 Minutes)

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Road-Sense.git
cd Road-Sense
```

### 2. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Test with Sample Data
```bash
# Use included sample files (5 images)
python -m src.data.preprocess_dataset --config configs/sample_preprocessing.yaml

# Expected output:
# ✅ Successfully processed: 5 images
```

### 4. Verify Installation
```bash
# Check YOLO installation
python -c "from ultralytics import YOLO; print('✅ YOLO installed')"

# Check PyTorch
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
```

**✅ Setup complete!** You can now test preprocessing with sample data.

---

## 📦 Full Dataset Setup (1 Hour)

### 1. Download KITTI Dataset

**Register:**
1. Visit http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
2. Create account and login

**Download:**
- `data_object_image_2.zip` (12 GB)
- `data_object_label_2.zip` (5 MB)

### 2. Extract Data
```bash
# Create directory
mkdir -p data/raw/KITTI/training

# Extract
unzip data_object_image_2.zip -d data/raw/KITTI/training/
unzip data_object_label_2.zip -d data/raw/KITTI/training/

# Verify
ls data/raw/KITTI/training/
# Expected: image_2/  label_2/
```

### 3. Validate Dataset
```bash
python src/data/validate_kitti_quality.py

# Expected:
# ✅ Total PNG Images Found: 7481
# ✅ Corrupted Images: 0
# ✅ Missing Label Files: 0
```

### 4. Run Preprocessing
```bash
python -m src.data.preprocess_dataset

# Takes ~5-8 minutes
# Output: data/processed/kitti/ (YOLO-ready)
```

### 5. Verify Output
```bash
python src/data/verify_dataset.py --data data/processed/kitti/data.yaml

# Expected:
# ✅ Train: 5,237 images
# ✅ Val: 1,496 images
# ✅ Test: 748 images
```

**✅ Ready for model training!**

---

## 🎯 Quick Commands

### Preprocessing
```bash
# Default config
python -m src.data.preprocess_dataset

# Custom config
python -m src.data.preprocess_dataset --config configs/custom_preprocessing.yaml

# From Python
python -c "from src.data import preprocess_dataset; preprocess_dataset()"
```

### Validation
```bash
# Validate raw data
python src/data/validate_kitti_quality.py

# Verify processed data
python src/data/verify_dataset.py --data data/processed/kitti/data.yaml
```

### Statistics
```bash
# Generate stats
python scripts/quick_stats.py

# Visualize samples
python scripts/quick_visualization.py --split train --num_samples 5
```

### Model Training (Milestone 2)
```bash
# Coming soon
# python train.py --data data/processed/kitti/data.yaml --epochs 100
```

---

## 📚 Documentation

- **[README.md](../README.md)** - Full project documentation
- **[Dataset Exploration Report](../docs/DATASET_EXPLORATION_REPORT.md)** - Dataset analysis
- **[Preprocessing Guide](../docs/PREPROCESSING_AND_AUGMENTATION_GUIDE.md)** - Step-by-step pipeline
- **[Download Instructions](../docs/DATASET_DOWNLOAD_INSTRUCTIONS.md)** - How to download datasets

---

## 🆘 Troubleshooting

### Import Errors
```bash
# Ensure you're in project root
cd /path/to/Road-Sense

# Activate virtual environment
source venv/bin/activate
```

### Dataset Not Found
```bash
# Check directory structure
ls data/raw/KITTI/training/
# Should show: image_2/  label_2/

# If missing, re-extract
unzip data_object_image_2.zip -d data/raw/KITTI/training/
```

### CUDA Not Available
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, PyTorch will use CPU (slower but works)
```

---

**⭐ Quick Start Complete! Ready to build your autonomous vehicle detection system! ⭐**
