# Milestone 2 Technical Report (exp34332)

Project: Road-Sense
Milestone: 2 - Training and Validation
Run ID: exp34332

## 1. Run Context

- Task: Object Detection
- Dataset classes: Vehicle, Pedestrian, Cyclist
- Model: yolo11m.pt (Ultralytics)
- Epochs: 100
- Run directory: `runs/train/exp34332`

## 2. Training Configuration Snapshot

From `runs/train/exp34332/args.yaml`:

- Batch size: 32
- Workers: 16
- Device: 1
- AMP: true
- Cache: ram
- Deterministic: false
- Optimizer: auto
- Patience: 30
- Image size: 640
- Save dir: `/home/skyvision/Sorry/Road-Sense/runs/train/exp34332`

## 3. Best Validation Metrics

Computed from `runs/train/exp34332/results.csv`:

| Metric | Best Value | Epoch |
|---|---:|---:|
| mAP50-95 (B) | 0.76786 | 95 |
| mAP50 (B) | 0.94159 | 87 |
| Precision (B) | 0.92421 | 40 |
| Recall (B) | 0.91550 | 71 |

## 4. Final Epoch Metrics

Epoch 100:

| Metric | Value |
|---|---:|
| Precision (B) | 0.90017 |
| Recall (B) | 0.90167 |
| mAP50 (B) | 0.93534 |
| mAP50-95 (B) | 0.76517 |

## 5. Artifact Inventory

Primary run outputs:

- `runs/train/exp34332/results.csv`
- `runs/train/exp34332/results.png`
- `runs/train/exp34332/confusion_matrix.png`
- `runs/train/exp34332/confusion_matrix_normalized.png`
- `runs/train/exp34332/BoxPR_curve.png`
- `runs/train/exp34332/BoxF1_curve.png`
- `runs/train/exp34332/weights/best.pt`
- `runs/train/exp34332/weights/last.pt`

Curated visualization mirror:

- `experiments/visualization/runs/exp34332/results.png`
- `experiments/visualization/runs/exp34332/confusion_matrix.png`
- `experiments/visualization/runs/exp34332/confusion_matrix_normalized.png`
- `experiments/visualization/runs/exp34332/BoxPR_curve.png`
- `experiments/visualization/runs/exp34332/BoxF1_curve.png`
- `experiments/visualization/runs/exp34332/BoxP_curve.png`
- `experiments/visualization/runs/exp34332/BoxR_curve.png`
- `experiments/visualization/runs/exp34332/val_batch0_pred.jpg`
- `experiments/visualization/runs/exp34332/val_batch1_pred.jpg`
- `experiments/visualization/runs/exp34332/val_batch2_pred.jpg`

Checkpoint mirror:

- `models/checkpoints/best-3classes-exp34332.pt`

## 6. Reproducibility Procedure

1. Ensure preprocessed KITTI dataset exists at `data/processed/kitti/data.yaml`.
2. Run training with repository defaults:

```bash
python train.py --config configs/training.yaml
```

3. Optional run override example:

```bash
python train.py --epochs 100 --batch-size 32 --workers 16 --device 1
```

## 7. Notes and Risks

- `runs/` content is typically excluded from version control.
- `*.pt` files are often ignored by default; confirm `.gitignore` policy if checkpoint tracking is required.
- This report documents run artifacts present locally at reporting time.
