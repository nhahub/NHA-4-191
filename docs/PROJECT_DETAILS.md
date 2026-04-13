# Project Details

This document stores detailed project information moved out of README to keep the repository entry point concise.

## Project Scope
Road-Sense is an end-to-end machine learning project for real-time road-scene object detection in autonomous driving workflows.

Target object groups:
- Vehicle (Car, Van, Truck)
- Pedestrian (Pedestrian, Person_sitting)
- Cyclist

Stage 2 target addition:
- Traffic signs

## Dataset Summary
Primary dataset: KITTI Vision Benchmark Suite

Key properties:
- Total training images: 7,481
- Total usable annotations: 38,186
- Source format: KITTI labels converted to YOLO format
- Typical image size range: 1242x375 to 1392x512 (resized to 640x640 for training)
- Environment coverage: urban, highway, and rural roads

Class distribution after class merging:
- Vehicle: 32,750 (85.73%)
- Pedestrian: 4,709 (11.06%)
- Cyclist: 1,627 (4.01%)

Dataset split:
- Train: 5,237 images (70%)
- Validation: 1,496 images (20%)
- Test: 748 images (10%)

Quality validation snapshot:
- Corrupted images: 0
- Missing label files: 0
- Invalid or out-of-bounds boxes: 0
- Exact duplicates: 0

## Technical Workflow
1. Validate raw KITTI files
2. Convert KITTI annotations to YOLO format
3. Merge classes into 3-class setup
4. Split dataset with fixed reproducible strategy
5. Verify processed dataset integrity
6. Train YOLO baseline and archive run artifacts

## Repository Structure (Detailed)
- configs/: YAML configs for preprocessing, multi-dataset strategy, and training
- data/raw/: original datasets (not committed)
- data/processed/: YOLO-ready datasets (not committed)
- data/augmented/: optional augmented outputs
- src/data/: data conversion, validation, and augmentation modules
- scripts/: quick analysis and visualization helpers
- experiments/: benchmark and visualization outputs
- runs/: training runs and training artifacts
- models/checkpoints/: exported baseline checkpoint copies
- reports/: milestone and technical reporting

## Milestone Progress
- Milestone 1 (Data Collection and Preprocessing): complete
- Milestone 2 (Model Development and Training): complete
- Milestone 3 (Deployment and Testing): planned
- Milestone 4 (MLOps and Monitoring): planned
- Milestone 5 (Final Documentation and Presentation): planned

## Related Documents
- docs/QUICK_SETUP_GUIDE.md
- docs/DATASET_EXPLORATION_REPORT.md
- docs/PREPROCESSING_AND_AUGMENTATION_GUIDE.md
- docs/DATASET_DOWNLOAD_INSTRUCTIONS.md
- docs/TRAINING_REPORT_EXP34332.md
- reports/PROJECT_STATUS_REPORT.md
