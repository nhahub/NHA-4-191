# Model Development Summary

**Project:** Road-Sense - Real-Time Object Detection for Autonomous Vehicles
**Date:** July 2026
**Final Model:** YOLO11m (HPO-optimized)

---

## 1. Model Selection

The final model was selected after evaluating multiple architectures on KITTI (3 classes):

| Criteria | YOLO11m (HPO) | Faster R-CNN |
|----------|:-------------:|:------------:|
| mAP@50 | **0.935** | N/A |
| mAP@50:95 | **0.725** | N/A |
| FPS (GPU) | **36.4** | <10 (est.) |
| Model Size | **38.8 MB** | ~167 MB |

**Result:** YOLO11m selected — best balance of accuracy, speed, and size.

## 2. Optimization & Export

| Format | Size | mAP@50:95 Drop | FPS (GPU) |
|--------|:----:|:--------------:|:---------:|
| PyTorch (.pt) | 38.8 MB | — (baseline) | 33.6 |
| ONNX FP16 | 38.3 MB | 0.0 | **44.1** |
| TorchScript | 77.0 MB | 0.0 | 33.2 |
| TFLite INT8 | **19.7 MB** | **-0.0000** | 36.4† |

## 3. Deployment Artifacts

- `models/checkpoints/HPO_run/weights/best.pt` — PyTorch model
- `models/checkpoints/HPO_run/weights/best.onnx` — ONNX export
- `models/checkpoints/HPO_run/weights/best.torchscript` — TorchScript export
- `models/checkpoints/HPO_run/weights/best_saved_model/` — TFLite exports

## 4. Documentation Delivered

| Document | Location |
|----------|----------|
| Model Evaluation Report | `reports/MODEL_EVALUATION_REPORT.md` |
| Model Comparison Report | `docs/models/MODEL_COMPARISON_REPORT.md` |
| Training Comparison Report | `reports/TRAINING_COMPARISON_REPORT.md` |
| Training Report (exp34332) | `reports/MILESTONE_2_TECHNICAL_REPORT_EXP34332.md` |
| HPO Report | `reports/HPO_REPORT.md` |
| Final Project Report | `docs/FINAL_PROJECT_REPORT.md` |
| API Documentation | `docs/API_DOCUMENTATION.md` |
| Deployment Guide | `docs/DEPLOYMENT_GUIDE.md` |

## 5. Performance KPIs

| KPI | Target | Achieved | Status |
|-----|:------:|:--------:|:------:|
| mAP@50 | ≥0.90 | **0.935** | ✅ |
| FPS (GPU) | ≥30 | **36.4** | ✅ |
| Memory (RAM) | <2 GB | **1.66 GB** | ✅ |
| Model Size | <50 MB | **38.8 MB** | ✅ |
