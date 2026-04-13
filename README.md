# Road-Sense
Real-time object detection for autonomous driving scenes using the YOLO pipeline.

## Published Deliverables
- Reports: included under `reports/`
- Documentation: included under `docs/`
- Visualization website: included under `presentation/`
- Live deployed website (GitHub Pages): https://abdallah4z.github.io/Road-Sense/index.html

## What This Project Does
Road-Sense focuses on 3-class road-object detection:
- Vehicle (Car, Van, Truck)
- Pedestrian (Pedestrian, Person_sitting)
- Cyclist

Current baseline is trained on KITTI with a complete preprocessing and validation workflow.

## Current Status
- Milestone 1: Complete (dataset preparation and preprocessing)
- Milestone 2: Complete (baseline model training)
- Latest baseline run: exp34332
- Best mAP50-95 (B): 0.76786
- Best mAP50 (B): 0.94159

For full metrics and artifact tracking, see the detailed reports section below.

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Validate raw KITTI data:
```bash
python src/data/validate_kitti_quality.py
```

3. Run preprocessing:
```bash
python -m src.data.preprocess_dataset
```

4. Verify output dataset:
```bash
python src/data/verify_dataset.py --data data/processed/kitti/data.yaml
```

For complete setup and troubleshooting, see docs/QUICK_SETUP_GUIDE.md.

## Project Layout
- configs/: preprocessing and training configuration
- src/data/: dataset conversion, validation, augmentation scripts
- docs/: implementation guides and technical documentation
- reports/: milestone summaries and technical reports
- models/checkpoints/: model checkpoints

## Docs and Reports

Core docs:
- docs/QUICK_SETUP_GUIDE.md
- docs/DATASET_DOWNLOAD_INSTRUCTIONS.md
- docs/PREPROCESSING_AND_AUGMENTATION_GUIDE.md
- docs/TRAINING_REPORT_EXP34332.md
- docs/PROJECT_DETAILS.md

Key reports:
- reports/REPORTS_INDEX.md
- reports/MILESTONE_2_EXECUTIVE_SUMMARY_EXP34332.md
- reports/MILESTONE_2_TECHNICAL_REPORT_EXP34332.md
- reports/PROJECT_STATUS_REPORT.md

## Team Members
- Abdallah Zain - https://github.com/Abdallah4Z
- Ahmed Elkady - https://github.com/ahmed9194
- Aya Ahmed - https://github.com/aya335
- FatmaElzahraa Wahby - https://github.com/fatmawahby
- Menna Tuallah Farghaly - https://github.com/fa290
- Mohamed Abd El Mawgoud - https://github.com/MohamedAbdelMawjoud

## Advisor
- Aya Abdallah

## License
MIT License. Dataset licenses remain with their original sources (KITTI, COCO, GTSDB).
