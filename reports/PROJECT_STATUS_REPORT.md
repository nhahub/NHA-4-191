# Project Status Report

Project: Road-Sense
Track: DEPI AI and Data Science Track
Last Updated: April 2026

## Executive Status
- Overall progress: 2 of 5 milestones completed
- Data pipeline: production-ready for KITTI preprocessing and verification
- Baseline model: trained and archived (run exp34332)

## Milestone Snapshot
- Milestone 1: Complete
- Milestone 2: Complete
- Milestone 3: Planned
- Milestone 4: Planned
- Milestone 5: Planned

## Baseline Training Highlights (exp34332)
Best validation scores observed during training:
- mAP50-95 (B): 0.76786 (epoch 95)
- mAP50 (B): 0.94159 (epoch 87)
- Precision (B): 0.92421 (epoch 40)
- Recall (B): 0.91550 (epoch 71)

Final epoch scores (epoch 100):
- mAP50-95 (B): 0.76517
- mAP50 (B): 0.93534
- Precision (B): 0.90017
- Recall (B): 0.90167

## Artifact Inventory
Primary artifacts:
- runs/train/exp34332/results.csv
- runs/train/exp34332/results.png
- runs/train/exp34332/confusion_matrix.png
- runs/train/exp34332/weights/best.pt
- runs/train/exp34332/weights/last.pt

Archived visual summaries:
- experiments/visualization/runs/exp34332/results.png
- experiments/visualization/runs/exp34332/confusion_matrix.png
- experiments/visualization/runs/exp34332/confusion_matrix_normalized.png
- experiments/visualization/runs/exp34332/BoxPR_curve.png
- experiments/visualization/runs/exp34332/BoxF1_curve.png
- experiments/visualization/runs/exp34332/BoxP_curve.png
- experiments/visualization/runs/exp34332/BoxR_curve.png

Exported checkpoint copy:
- models/checkpoints/best-3classes-exp34332.pt

## Next Phase Priorities
1. Build inference and deployment path for Milestone 3.
2. Add traffic sign data integration for multi-dataset training strategy.
3. Define monitoring and retraining plan for Milestone 4.

## Related Reports
- reports/MILESTONE_2_EXECUTIVE_SUMMARY_EXP34332.md
- reports/MILESTONE_2_TECHNICAL_REPORT_EXP34332.md
- docs/TRAINING_REPORT_EXP34332.md
- docs/PROJECT_DETAILS.md
