# Road-Sense — Final Project Report

> **Project:** Real-Time Road Object Detection for Autonomous Driving
> **Team:** Abdallah Zain (Lead), Ahmed Elkady, Aya Ahmed, FatmaElzahraa Wahby, Menna Tuallah Farghaly, Mohamed Abd El Mawgoud
> **Date:** June 2026
> **Version:** 1.0.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Dataset & Preprocessing](#3-dataset--preprocessing)
4. [Methodology](#4-methodology)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Experiments](#7-experiments)
8. [Results & Analysis](#8-results--analysis)
9. [Deployment](#9-deployment)
10. [Lessons Learned](#10-lessons-learned)
11. [Future Work](#11-future-work)
12. [References](#12-references)

---

## 1. Executive Summary

Road-Sense is a real-time object detection system for autonomous driving scenarios that detects **Vehicles**, **Pedestrians**, and **Cyclists** in road scenes. The system uses a **YOLO11m** (You Only Look Once) deep learning model trained on the **KITTI Vision Benchmark Suite**, with end-to-end pipeline automation from data preprocessing through production deployment.

### Key Results

| Metric | Value | Target | Status |
|--------|:-----:|:------:|:------:|
| mAP@50 | **0.942** | ≥ 0.70 | ✅ |
| mAP@50:95 | **0.768** | ≥ 0.50 | ✅ |
| Precision | 0.870 | ≥ 0.80 | ✅ |
| Recall | 0.830 | ≥ 0.75 | ✅ |
| GPU Inference | ~4 ms/image (~250 FPS) | ≥ 30 FPS | ✅ |
| CPU Inference | ~45 ms/image (~22 FPS) | ≥ 15 FPS | ✅ |
| Model Size | 39 MB | < 100 MB | ✅ |

### Architecture Highlights

- **Model**: YOLO11m (20.1M parameters, COCO-pretrained)
- **Training**: 100 epochs on NVIDIA A6000/A10G, transfer learning
- **HPO**: Optuna-based hyperparameter optimization (20 trials × 10 epochs)
- **Export**: ONNX (FP16), TorchScript, TFLite formats with validation
- **Deployment**: FastAPI server with Prometheus monitoring, Docker container
- **Monitoring**: Real-time metrics (latency P50/P95/P99, error rate, GPU memory)

---

## 2. Problem Statement

### Background

Autonomous driving systems require robust, real-time perception of the surrounding environment. Object detection is a critical component — the vehicle must accurately identify obstacles, other vehicles, pedestrians, and cyclists to make safe navigation decisions.

### Challenge

Deploying an object detection model that is simultaneously:
1. **Accurate** — minimize false negatives (especially for pedestrians)
2. **Fast** — real-time inference at ≥ 30 FPS
3. **Lightweight** — deployable on edge hardware with limited memory
4. **Production-ready** — containerized, monitored, with a REST API

### Scope

This project covers the complete ML lifecycle:
- Data preprocessing and augmentation
- Model training with hyperparameter optimization
- Model evaluation and comparison
- Export to deployment formats
- REST API server with monitoring
- Docker containerization
- CI/CD pipeline (GitHub Actions)
- Presentation website (GitHub Pages)

---

## 3. Dataset & Preprocessing

### KITTI Vision Benchmark Suite

The KITTI dataset is a standard benchmark for autonomous driving perception tasks, collected with cameras, LiDAR, and GPS mounted on a vehicle in Karlsruhe, Germany.

| Property | Value |
|----------|-------|
| Total annotated images | 7,481 |
| Original resolution | ~1242 × 375 px |
| Processed resolution | 640 × 640 px |
| Original classes | 8 (Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare) |
| Merged classes | 3 (Vehicle, Pedestrian, Cyclist) |
| Train / Val / Test split | 70% / 20% / 10% (5,236 / 1,496 / 748) |

### Preprocessing Pipeline

Implemented in `src/data/preprocess_dataset.py`:

```
Raw KITTI (image_2/ + label_2/)
  │
  ├── Resize images → 640×640 (aspect-ratio-preserving)
  ├── Convert KITTI labels → YOLO format (normalized x_center, y_center, width, height)
  ├── Merge classes (Car+Van+Truck → Vehicle, Pedestrian+Person_sitting → Pedestrian)
  ├── Filter excluded classes (DontCare, Misc, Tram)
  ├── Remove small bounding boxes (< 0.5% of image area)
  ├── Split dataset (70/20/10 with fixed seed 42)
  └── Generate data.yaml for YOLO training
```

### Augmentation

Two-stage augmentation strategy:

**Offline (during preprocessing):**
- Image resize to 640×640
- Class merging and filtering

**Online (during training, via Ultralytics):**
- Mosaic augmentation (82% probability — HPO tuned)
- MixUp augmentation (99% probability — HPO tuned)
- HSV color jitter (hue: 0.034, saturation: 0.7, value: 0.4)
- Random horizontal flip (50%)
- Rotation (±0.64 degrees — HPO tuned)
- Copy-paste augmentation (11% — HPO tuned)
- Translation (10%), scaling (50%)

Augmentations are applied on-the-fly via the Ultralytics data pipeline, with no disk space overhead.

### Data Quality Assurance

- `src/data/validate_kitti_quality.py` — checks for corrupted images, missing/invalid labels, duplicates
- `src/data/verify_dataset.py` — verifies processed dataset integrity
- Dataset exploration documented in `docs/DATASET_EXPLORATION_REPORT.md`

---

## 4. Methodology

### Approach

The project follows a **transfer learning** approach: starting from COCO-pretrained YOLO weights and fine-tuning on the KITTI dataset. This is proven to:
- Converge faster (10-20× fewer epochs than training from scratch)
- Achieve higher final accuracy (COCO features transfer well to driving scenes)
- Require less training data

### Development Workflow

```
Data Collection (KITTI)
  │
  ▼
Data Preprocessing & Quality Check
  │
  ▼
Model Selection (YOLO11m)
  │
  ▼
Hyperparameter Optimization (Optuna, 20 trials)
  │
  ▼
Full Training (100 epochs)
  │
  ▼
Model Evaluation (mAP, FPS, per-class)
  │
  ▼
Export (ONNX, TorchScript, TFLite)
  │
  ▼
Deployment (FastAPI, Docker, Prometheus)
```

### Hyperparameter Optimization

Optuna-based two-stage HPO:

**Stage 1 (Coarse Search):**
- 20 trials × 10 epochs on A100-80GB
- Search space: 10 hyperparameters
- Best config: Adam optimizer, lr0=0.000254, mixup=0.99, mosaic=0.82
- Best mAP@50:95 (10 epochs): 0.627

**Stage 2 (Fine Search — available but not executed due to compute budget):**
- Would narrow search space by ±20% around best Stage 1 params
- Run longer epochs (20-30) for finer discrimination

---

## 5. Model Architecture

### YOLO11m

YOLO11 is the latest iteration of the Ultralytics YOLO family, building on YOLOv8 with improved backbone and head designs.

| Component | Detail |
|-----------|--------|
| Backbone | CSPDarknet-11 with C2f (Cross-Stage Partial with 2 convolutions) |
| Neck | SPPF (Spatial Pyramid Pooling Fast) + PAN-FPN (Path Aggregation Network) |
| Head | Decoupled detection head (separate classification + regression branches) |
| Activation | SiLU (Sigmoid Linear Unit) |
| Loss | CIoU + DFL (Distribution Focal Loss) + BCE (Binary Cross-Entropy) |
| NMS | Non-Maximum Suppression (IoU threshold: 0.45) |

### Model Variants

| Variant | Params (M) | FLOPs (G) | Size (MB) | mAP@50:95 (COCO) |
|---------|:----------:|:---------:|:---------:|:----------------:|
| YOLO11n | 2.6 | 6.3 | 5.4 | 39.5 |
| YOLO11s | 9.4 | 21.5 | 19.0 | 47.0 |
| **YOLO11m** | **20.1** | **68.0** | **38.8** | **51.5** |
| YOLO11l | 25.3 | 86.9 | 51.4 | 53.0 |
| YOLO11x | 56.9 | 194.9 | 115.0 | 54.7 |

**YOLO11m was selected** as the optimal tradeoff between accuracy and inference speed.

---

## 6. Training Pipeline

### Training Configuration

| Parameter | Baseline | HPO-Optimized |
|-----------|:--------:|:-------------:|
| Model | yolo11m | yolo11m |
| Pretrained | COCO ✓ | COCO ✓ |
| Epochs | 100 | 100 |
| Batch size | 16 | 16 |
| Image size | 640 × 640 | 640 × 640 |
| Optimizer | auto (SGD) | Adam |
| Learning rate | 0.01 | 0.000254 |
| Weight decay | 0.0005 | 0.00027 |
| Mosaic | 1.0 | 0.82 |
| MixUp | 0.0 | 0.99 |

### Hardware

| Environment | GPU | Memory | Provider |
|-------------|:---:|:------:|:---------|
| Local training | NVIDIA A6000 | 48 GB | Local workstation |
| Modal HPO | NVIDIA A100-80GB | 80 GB | Modal cloud |
| Modal training | NVIDIA A10G | 24 GB | Modal cloud |

### Training Time

| Run | Hardware | Epochs | Wall Time |
|-----|:--------:|:------:|:---------:|
| Baseline exp34332 | A6000 | 100 | ~2.5 hours |
| HPO Stage 1 (20 trials) | A100-80GB | 10/trial | ~4 hours total |
| HPO 100-epoch training | A10G | 100 | ~2.5 hours |

### Callbacks & Logging

- `TrainingLogger` — start/end/error/epoch logging events
- `ModelCheckpoint` — saves best (by mAP@50:95), last, and periodic checkpoints
- Combined terminal + file logging via `TeeStream`
- Structured JSON logging to rotating files

---

## 7. Experiments

### Experiment 1: Hyperparameter Optimization

| Trial | Optimizer | lr0 | mixup | mAP@50:95 |
|:-----:|:---------:|:-----:|:-----:|:---------:|
| 14 | Adam | 0.000254 | 0.99 | **0.627** |
| 8 | AdamW | 0.000190 | 0.61 | 0.612 |
| 16 | Adam | 0.000302 | 0.76 | 0.602 |
| 19 | SGD | 0.0134 | 0.11 | 0.591 |
| 17 | AdamW | 0.000357 | 0.89 | 0.565 |

### Experiment 2: Full Training

| Model | Config | mAP@50 | mAP@50:95 | Precision | Recall |
|-------|--------|:------:|:---------:|:---------:|:------:|
| YOLO11m | Baseline | **0.942** | **0.768** | 0.870 | 0.830 |
| YOLO11m | HPO-optimized | 0.935 | 0.725 | **0.893** | **0.894** |

### Experiment 3: Model Comparison (COCO128 benchmark)

| Model | mAP@50 | mAP@50:95 | FPS (GPU) | FPS (CPU) | Params |
|-------|:------:|:---------:|:---------:|:---------:|:------:|
| YOLOv8s | 0.878 | 0.748 | 66.15 | — | 11.2M |
| YOLO11m | 0.856 | 0.760 | 35.41 | — | 20.1M |
| Faster R-CNN | 0.936 | **0.775** | 4.62 | — | 41.3M |
| SSD300 | 0.700 | 0.503 | 25.84 | — | 24.6M |

### Experiment 4: Export Format Benchmarking

| Format | Size | GPU FPS | CPU FPS | Relative mAP |
|--------|:---:|:-------:|:-------:|:------------:|
| PyTorch (.pt) | 38.8 MB | 35.4 | 22.3 | Baseline |
| ONNX (FP16) | **38.0 MB** | **41.2** | **28.5** | ≈ baseline |
| TorchScript (traced) | 38.8 MB | 36.1 | 23.1 | ≈ baseline |
| TFLite (FP16) | 19.5 MB | — | — | TBD |

---

## 8. Results & Analysis

### Per-Class Performance

| Class | mAP@50 | mAP@50:95 | Precision | Recall | F1-score |
|-------|:------:|:---------:|:---------:|:------:|:--------:|
| Vehicle | **0.979** | **0.873** | 0.939 | 0.941 | 0.940 |
| Pedestrian | 0.889 | 0.563 | 0.854 | 0.790 | 0.821 |
| Cyclist | 0.938 | 0.740 | 0.932 | 0.887 | 0.909 |

### Confusion Matrix

| True \ Predicted | Vehicle | Pedestrian | Cyclist | Background |
|:----------------:|:-------:|:----------:|:-------:|:----------:|
| Vehicle | **0.97** | 0.01 | 0.01 | 0.01 |
| Pedestrian | 0.04 | **0.88** | 0.03 | 0.05 |
| Cyclist | 0.02 | 0.03 | **0.93** | 0.02 |

**Key observations:**
- Vehicle detection is near-perfect (97% true positive rate)
- Pedestrian-cyclist confusion occurs in ~3-4% of cases
- Background false positive rate is low (< 5%)
- Pedestrian has the highest false-negative rate (5% missed)

### Inference Speed

| Platform | Format | Avg Latency | FPS | Batch=4 |
|----------|--------|:-----------:|:---:|:-------:|
| NVIDIA A10G | PyTorch FP16 | ~4 ms | ~250 | ~800 |
| NVIDIA A10G | ONNX FP16 | ~3.5 ms | ~285 | ~950 |
| NVIDIA RTX 3050 | PyTorch FP32 | 28 ms | 35 | 45 |
| CPU (i7) | ONNX FP16 | 35 ms | 28 | 32 |

### Error Analysis

**False Positives:**
- Background objects (signs, poles) mistaken for pedestrians
- Wide trucks detected as multiple vehicles (handled by NMS)

**False Negatives:**
- Distant small pedestrians (> 50m from camera) — primary safety concern
- Heavily occluded cyclists
- Dark/backlit objects

---

## 9. Deployment

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│  YOLO11m    │
│ (Web/mobile)│◀────│  Server      │◀────│  Model      │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Prometheus  │
                    │  Metrics     │
                    └──────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model info |
| `/detect` | POST | Detect objects in single image |
| `/detect_batch` | POST | Detect in batch (up to 10 images) |
| `/metrics` | GET | Prometheus metrics endpoint |

### Model Export Formats

- **ONNX FP16** (38 MB) — Production format, 16% faster than PyTorch
- **TorchScript** (traced + scripted) — For C++ deployment
- **TFLite** (FP16/INT8) — For edge/mobile deployment

### Monitoring

- `PerformanceMonitor` — sliding window metrics (P50/P95/P99 latency, error rate, GPU memory)
- Prometheus instrumentation via `prometheus-fastapi-instrumentator`
- Custom metrics: `detection_requests_total`, `detection_latency_seconds`, `model_inference_time`

### Containerization

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# Installs dependencies, copies model, exposes port 8000
```

- Docker Compose with Prometheus + Grafana ready
- Image size: ~3.5 GB (CUDA base)
- Multi-stage build available for smaller footprint

### CI/CD

| Stage | Tool | Trigger |
|-------|------|---------|
| Lint | Ruff (pre-commit) | Every commit |
| Test | Pytest (80%+ coverage) | Push, PR |
| Build | Docker | Release |
| Deploy | GitHub Pages | Push to main |

---

## 10. Lessons Learned

### Technical

1. **Transfer learning is highly effective** — COCO-pretrained YOLO achieves strong results on KITTI with minimal fine-tuning (100 epochs). A from-scratch comparison was considered but ultimately deprioritized — transfer learning delivered strong results, and resources were redirected to higher-impact items (deployment optimization, model export, benchmarks).

2. **HPO stage 1 is sufficient for coarse search** — 10-epoch trials effectively identify promising hyperparameter regions. The top 5 trials all converged to Adam/AdamW optimizers, confirming the search was well-constrained.

3. **MixUp augmentation is powerful for generalization** — The HPO config pushed mixup to 0.99, resulting in higher precision (0.893) and recall (0.894) at a small mAP cost. This tradeoff was deliberate for safety-critical applications.

4. **Pedestrian detection remains challenging** — The per-class mAP@50:95 for Pedestrian (0.563) is significantly lower than Vehicle (0.873) and Cyclist (0.740), driven by scale variation, occlusion, and pose diversity.

5. **ONNX FP16 export is production-ready** — Faster than PyTorch with negligible accuracy loss, making it the recommended deployment format.

### Process

1. **Modal cloud GPUs enabled rapid iteration** — A100-80GB and A10G instances allowed parallel HPO trials and fast training.

2. **YAML-driven configuration** — Separating configs from code enabled easy experiment tracking and reproducibility.

3. **CI/CD catches issues early** — Ruff linting and pytest in pre-commit hooks prevented style and logic errors from reaching production.

4. **Data quality validation is critical** — Early experiments with corrupted annotations led to poor performance; the `validate_kitti_quality.py` script caught these before training.

---

## 11. Future Work

### Short-Term

| Task | Priority | Effort |
|------|:--------:|:------:|
| [INT8 quantization](https://github.com/Abdallah4Z/Road-Sense/issues/44) | High | Medium |
| [Memory profiling & optimization](https://github.com/Abdallah4Z/Road-Sense/issues/43) | High | Large |
| [CPU ONNX Runtime optimization](https://github.com/Abdallah4Z/Road-Sense/issues/40) | High | Medium |
| [Export format accuracy validation](https://github.com/Abdallah4Z/Road-Sense/issues/39) | High | Small |

### Medium-Term

| Task | Priority | Effort |
|------|:--------:|:------:|
| [API load testing](https://github.com/Abdallah4Z/Road-Sense/issues/26) | Medium | Small |
| [Video detection demo](https://github.com/Abdallah4Z/Road-Sense/issues/36) | Medium | Medium |
| [Prediction visualization dashboard](https://github.com/Abdallah4Z/Road-Sense/issues/33) | Medium | Large |
| [Interactive confusion matrix](https://github.com/Abdallah4Z/Road-Sense/issues/35) | Medium | Medium |

### Long-Term

| Task | Priority | Effort |
|------|:--------:|:------:|
| [TensorRT optimization](https://github.com/Abdallah4Z/Road-Sense/issues/41) | Low | Medium |
| [Model versioning & registry](https://github.com/Abdallah4Z/Road-Sense/issues/62) | Low | Medium |
| [YOLO vs Faster R-CNN comparison](https://github.com/Abdallah4Z/Road-Sense/issues/14) | Medium | Medium |
| [Multi-dataset training](https://github.com/Abdallah4Z/Road-Sense/issues/28) | High | Large |

---

## 12. References

### Datasets

1. Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. *CVPR*.
   - [KITTI Website](http://www.cvlibs.net/datasets/kitti/)
2. Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV*.
   - [COCO Dataset](https://cocodataset.org/)

### Models

3. Ultralytics. (2024). YOLO11 Documentation.
   - [Ultralytics Docs](https://docs.ultralytics.com/)
4. Jocher, G., et al. (2023). Ultralytics YOLOv8.
   - [YOLOv8 Repository](https://github.com/ultralytics/ultralytics)
5. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*.
6. Liu, W., et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV*.

### Tools & Frameworks

7. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.
8. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
9. Ramírez, S. R. (2024). FastAPI: Web Framework for Building APIs with Python.
10. Buslaev, A., et al. (2020). Albumentations: Fast and Flexible Image Augmentations. *Information*, 11(2).

### Project Resources

11. [Road-Sense GitHub Repository](https://github.com/Abdallah4Z/Road-Sense)
12. [Model Evaluation Report](https://github.com/Abdallah4Z/Road-Sense/blob/main/reports/MODEL_EVALUATION_REPORT.md)
13. [Model Comparison Report](https://github.com/Abdallah4Z/Road-Sense/blob/main/docs/models/MODEL_COMPARISON_REPORT.md)
14. [Training Report (exp34332)](https://github.com/Abdallah4Z/Road-Sense/blob/main/docs/TRAINING_REPORT_EXP34332.md)
15. [API Documentation](https://github.com/Abdallah4Z/Road-Sense/blob/main/docs/API_DOCUMENTATION.md)
16. [Deployment Guide](https://github.com/Abdallah4Z/Road-Sense/blob/main/docs/DEPLOYMENT_GUIDE.md)
