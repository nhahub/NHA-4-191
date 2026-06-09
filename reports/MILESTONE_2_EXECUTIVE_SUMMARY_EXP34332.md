# Milestone 2 Executive Summary (exp34332)

Project: Road-Sense
Milestone: 2 - Model Training and Validation
Run ID: exp34332
Date: April 2026

## Objective

Train a reliable YOLO-based detector on the 3-class KITTI setup used in Road-Sense:

- Vehicle
- Pedestrian
- Cyclist

## Final Outcome

Milestone 2 training completed successfully with strong detection quality and stable convergence.

Best validation performance from the run:

- mAP50-95 (B): 0.76786 (epoch 95)
- mAP50 (B): 0.94159 (epoch 87)
- Precision (B): 0.92421 (epoch 40)
- Recall (B): 0.91550 (epoch 71)

Final epoch (epoch 100):

- mAP50-95 (B): 0.76517
- mAP50 (B): 0.93534
- Precision (B): 0.90017
- Recall (B): 0.90167

## Why This Matters

- The model achieved high localization and classification quality for road-scene objects.
- Final metrics stayed close to the run best, indicating stable training behavior.
- Artifacts are organized for reproducibility and handoff.

## Artifacts Ready

- Full run directory: `runs/train/exp34332`
- Curated visualizations: `experiments/visualization/runs/exp34332`
- Model checkpoint copy: `models/checkpoints/best-3classes-exp34332.pt`
- Detailed technical report: `reports/MILESTONE_2_TECHNICAL_REPORT_EXP34332.md`

## Milestone 2 Status

Status: COMPLETE

This run can be used as the baseline checkpoint for next-stage evaluation and deployment work.
