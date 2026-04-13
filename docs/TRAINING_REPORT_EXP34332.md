# Training Report - exp34332

## Summary
This document records the finalized YOLO11m training run used as the baseline Road-Sense model checkpoint for Milestone 2.

- Run directory: runs/train/exp34332
- Task: Object detection (3 merged classes)
- Classes: Vehicle, Pedestrian, Cyclist
- Model: yolo11m.pt (Ultralytics)
- Epochs: 100
- hardware: Nvidia A6000 48GB

## Training Configuration
From runs/train/exp34332/args.yaml:

- Batch size: 32
- Workers: 16
- Device: 1
- AMP: true
- Cache: ram
- Deterministic: false
- Optimizer: auto
- Patience: 30
- Image size: 640

## Best Validation Metrics
Computed from runs/train/exp34332/results.csv:

| Metric | Best Value | Epoch |
|--------|------------|-------|
| mAP50-95 (B) | 0.76786 | 95 |
| mAP50 (B) | 0.94159 | 87 |
| Precision (B) | 0.92421 | 40 |
| Recall (B) | 0.91550 | 71 |

## Final Epoch Metrics (Epoch 100)

| Metric | Value |
|--------|-------|
| Precision (B) | 0.90017 |
| Recall (B) | 0.90167 |
| mAP50 (B) | 0.93534 |
| mAP50-95 (B) | 0.76517 |

## Artifact Inventory
Run outputs generated in runs/train/exp34332:

- results.csv
- results.png
- confusion_matrix.png
- confusion_matrix_normalized.png
- BoxPR_curve.png
- BoxF1_curve.png
- weights/best.pt
- weights/last.pt

Mirrored release artifacts:

- experiments/visualization/runs/exp34332/results.png
- experiments/visualization/runs/exp34332/confusion_matrix.png
- experiments/visualization/runs/exp34332/confusion_matrix_normalized.png
- experiments/visualization/runs/exp34332/BoxPR_curve.png
- experiments/visualization/runs/exp34332/BoxF1_curve.png
- experiments/visualization/runs/exp34332/BoxP_curve.png
- experiments/visualization/runs/exp34332/BoxR_curve.png
- experiments/visualization/runs/exp34332/val_batch0_pred.jpg
- experiments/visualization/runs/exp34332/val_batch1_pred.jpg
- experiments/visualization/runs/exp34332/val_batch2_pred.jpg
- models/checkpoints/best-3classes-exp34332.pt

## Reproducibility
To reproduce a run with current repository defaults:

1. Ensure KITTI data has been preprocessed to YOLO format.
2. Run: python train.py --config configs/training.yaml
3. Optionally override runtime options, for example:
   python train.py --epochs 100 --batch-size 32 --workers 16 --device 1 

## Notes
- The runs directory is ignored by default in .gitignore.
- The copied run checkpoint is stored at models/checkpoints/best-3classes-exp34332.pt.
- Copied run visualizations are stored at experiments/visualization/runs/exp34332.
- By default, .gitignore excludes *.pt files unless explicitly whitelisted.
