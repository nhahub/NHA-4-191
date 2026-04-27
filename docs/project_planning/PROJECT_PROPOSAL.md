# Project Proposal: Road-Sense

## 1. Executive Summary

**Road-Sense** is an end-to-end deep learning system for real-time road-scene object detection in autonomous driving contexts. The project leverages the YOLO (You Only Look Once) family of object detectors — specifically YOLO11m — to identify and localize three critical road-user classes (Vehicles, Pedestrians, Cyclists) from dashboard camera footage. Built on the KITTI Vision Benchmark Suite, the system is capable of real-time inference at 35+ FPS on consumer-grade GPUs, with model export support for edge deployment via ONNX, TensorFlow Lite, and TorchScript formats. A FastAPI-based inference server provides a RESTful interface for integration into larger autonomous driving stacks, while a presentation website deployed via GitHub Pages demonstrates detection capabilities.

The project is developed by a team of 6 members under the Digital Egypt Pioneers Initiative (DEPI), supervised by advisor Aya Abdallah.

---

## 2. Project Objectives

| Objective | Description | Success Criterion |
|-----------|-------------|-------------------|
| **O1** | Build a robust data preprocessing pipeline converting raw KITTI annotations (9 classes) into a YOLO-compatible format with 3 consolidated classes | Pipeline produces verified train/val/test splits with no data corruption or label errors |
| **O2** | Train a high-performance object detection model achieving real-time inference speeds | mAP@0.5:0.95 ≥ 0.70, inference ≥ 15 FPS on target hardware |
| **O3** | Develop a production-grade inference server for model deployment | API endpoint responds in < 100ms per frame under moderate load |
| **O4** | Provide multiple model export formats for cross-platform deployment | ONNX, TFLite, and TorchScript exports all pass validation benchmarks |
| **O5** | Create a public-facing presentation website with live detection demo | Website deployed via GitHub Pages with working detection demo |
| **O6** | Deliver comprehensive documentation covering all project phases | All milestone documentation complete, peer-reviewed, and submitted |

---

## 3. Project Scope

### 3.1 In-Scope

- **Data Processing**: Conversion of KITTI raw data (label files + images) to YOLO-format labels; class remapping (9 → 3 target classes: Vehicle, Pedestrian, Cyclist); train/val/test stratified splitting (70/20/10); data quality validation (corruption detection, duplicate removal, out-of-bounds box fixing)
- **Data Augmentation**: Albumentations-based augmentation pipelines at three severity levels (light, medium, heavy) including geometric transforms, color adjustments, and weather-effect simulation (rain, fog, sun flare)
- **Model Training**: YOLOv11m baseline training; hyperparameter configuration via YAML config files; checkpoint management and early stopping; comprehensive metrics logging (mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, confusion matrices)
- **Model Benchmarking**: Comparative evaluation against YOLOv8s, SSD300 VGG16, and Faster R-CNN ResNet50 FPN v2 on accuracy (mAP), speed (FPS, latency), and resource usage (model size, parameter count, memory footprint)
- **Model Export**: ONNX, TensorFlow Lite, and TorchScript exports with FP16 half-precision support
- **Inference Server**: FastAPI-based REST API with POST `/detect` endpoint; per-session temporal object tracking; health-check endpoint; CORS support
- **Real-time Detection**: Webcam/video-file detection with live FPS overlay and video recording
- **Presentation Website**: GitHub Pages-deployed static site with documentation browser, metrics dashboard, visualization gallery, and live detection demo

### 3.2 Out-of-Scope

- Real autonomous vehicle integration or hardware-in-the-loop testing
- Multi-camera fusion or LiDAR integration
- End-to-end autonomous driving decision-making (planning/control layers)
- Real-time video streaming infrastructure (RTSP/RTMP ingest)
- Mobile app development
- Large-scale distributed training across multiple nodes
- Traffic sign recognition (planned as future enhancement via GTSDB integration)
- Semantic segmentation or instance segmentation
- Object tracking across multiple camera feeds

---

## 4. Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|----------|
| **Advisor: Aya Abdallah** | Project supervisor | Technical direction, milestone evaluation, grading |
| **DEPI Program** | Funding & oversight body | Program outcomes, skill development, graduation requirements |
| **Team Members (6)** | Developers | Learning outcomes, project completion, portfolio building |
| **End Users (Autonomous Driving Researchers)** | Indirect beneficiaries | Model availability, documentation quality, reproducibility |
| **GitHub Community** | Open-source consumers | Code quality, documentation, ease of use |

---

## 5. Technical Approach

### 5.1 Methodology

The project follows a structured **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology adapted for deep learning:

1. **Business Understanding** — Define project objectives, success criteria, and scope for road-scene object detection
2. **Data Understanding** — Explore KITTI dataset statistics, class distributions, image properties, and annotation quality
3. **Data Preparation** — Preprocess, validate, and augment KITTI data into YOLO-compatible format
4. **Modeling** — Train, evaluate, and benchmark YOLO variants; select optimal model
5. **Deployment** — Package model into inference server with REST API; export to production formats
6. **Documentation & Evaluation** — Comprehensive reporting, GitHub Pages site, final presentation

### 5.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YOLO11m as primary model** | Best accuracy-speed tradeoff (mAP@0.5:0.95=0.768, 35 FPS on RTX 3050) |
| **KITTI dataset** | De facto standard for autonomous driving perception research; well-annotated, real-world driving scenes |
| **Albumentations for augmentation** | Bounding-box-aware transformations; production-proven in computer vision pipelines |
| **FastAPI for inference server** | Async-first Python framework; automatic OpenAPI docs; high throughput |
| **GitHub Pages for website** | Zero-cost hosting; automatic CI/CD deployment; good for static content |
| **PyTorch + Ultralytics** | Industry-standard deep learning framework; seamless YOLO integration; active maintenance |

### 5.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├──────────────┬──────────────────┬───────────────────────────────┤
│  Raw KITTI   │  Preprocessing   │  YOLO-Formatted Dataset       │
│  (images +   │ ───────────────► │  (images + labels,            │
│   labels)    │  class remapping │   train/val/test splits)      │
│  12 GB       │  split generation│   ~400 MB                     │
└──────────────┴──────────────────┴───────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                        │
├──────────────┬──────────────────┬───────────────────────────────┤
│  YOLO11m     │  Augmentation    │  Trained Model                │
│  pretrained  │ ───────────────► │  (best.pt)                    │
│  weights     │  Albumentations  │  ~38.8 MB                     │
└──────────────┴──────────────────┴───────────────────────────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                   ┌──────────┐  ┌──────────┐  ┌───────────┐
                   │  ONNX    │  │  TFLite  │  │TorchScript│
                   │  Export  │  │  Export  │  │  Export   │
                   └──────────┘  └──────────┘  └───────────┘
                         │              │              │
                         ▼              ▼              ▼
                   ┌──────────────────────────────────────────────┐
                   │              DEPLOYMENT                      │
                   ├────────────────┬─────────────────────────────┤
                   │  FastAPI       │  Real-time Detection CLI    │
                   │  Inference     │  (webcam/video)             │
                   │  Server        │                             │
                   └────────────────┴─────────────────────────────┘
```

---

## 6. Expected Outcomes

### 6.1 Deliverables

| Phase | Deliverable | Format |
|-------|-------------|--------|
| M1 | Preprocessed KITTI dataset (YOLO format) | Directory of images + labels + data.yaml |
| M1 | Data quality report | Markdown document |
| M1 | Augmentation pipeline code | Python module |
| M2 | Trained YOLO11m model weights | `.pt` file |
| M2 | Training report with metrics | Markdown + JSON |
| M2 | Model comparison report | Markdown + CSV |
| M3 | FastAPI inference server | Python module |
| M3 | Exported models (ONNX, TFLite, TorchScript) | Model files |
| M3 | Model export benchmarks | Markdown + JSON |
| M4 | Docker container image | Dockerfile + compose |
| M4 | CI/CD pipeline | GitHub Actions workflow |
| M5 | Presentation website | GitHub Pages (HTML/CSS/JS) |
| M5 | Final project documentation | Markdown documents |
| M5 | Final presentation slides | PDF / PowerPoint |

### 6.2 Success Metrics

- **Model Accuracy**: mAP@0.5:0.95 ≥ 0.75, mAP@0.5 ≥ 0.92
- **Inference Speed**: ≥ 15 FPS on CPU, ≥ 30 FPS on GPU
- **API Latency**: p95 response time < 150ms per detection request
- **Code Quality**: Ruff linting passes with zero errors; test coverage ≥ 80%
- **Documentation**: All 5 milestone docs completed and reviewed
- **Deployment**: Website live on GitHub Pages with functioning detection demo

---

## 7. Timeline & Milestones

| Milestone | Duration | Key Activities |
|-----------|----------|----------------|
| M1: Data Collection & Preprocessing | Weeks 1-2 | KITTI download, preprocessing, quality validation, augmentation |
| M2: Model Development & Training | Weeks 3-5 | YOLO training, hyperparameter tuning, model comparison |
| M3: Deployment & API Development | Weeks 6-7 | FastAPI server, model export, Docker setup |
| M4: MLOps & Monitoring | Weeks 8-9 | CI/CD, testing, performance monitoring |
| M5: Final Documentation & Presentation | Weeks 10-12 | Website finalization, documentation, presentation |

---

## 8. Resource Requirements

### 8.1 Hardware

| Resource | Specification | Purpose |
|----------|--------------|---------|
| Training GPU | NVIDIA A6000 (cloud) or RTX 3050 (local) | YOLO model training |
| Development Machines | 6 laptops/desktops with Python 3.7+ | Code development, testing |
| Storage | ~20 GB free space | Dataset + models + artifacts |

### 8.2 Software

| Resource | Version | Purpose |
|----------|---------|---------|
| Python | 3.7+ | Primary programming language |
| PyTorch | 2.0+ | Deep learning framework |
| Ultralytics | 8.0+ | YOLO training & inference |
| FastAPI | 0.100+ | REST API server |
| OpenCV | 4.5+ | Image processing |
| Albumentations | 1.3+ | Data augmentation |
| Docker | 24.0+ | Containerization |

### 8.3 Team

| Role | Count | Skills Required |
|------|-------|-----------------|
| Data Engineer | All | Python, data processing, OpenCV |
| ML Engineer | All | PyTorch, YOLO, model optimization |
| Backend Developer | 1-2 | FastAPI, Docker, REST APIs |
| Frontend/Docs Developer | 1 | HTML/CSS/JS, technical writing |

---

## 9. Budget

| Item | Cost (EGP) | Notes |
|------|------------|-------|
| Cloud GPU (Google Colab Pro) | 0 | Free tier used for training |
| Domain Name | 0 | GitHub Pages subdomain |
| Software Licenses | 0 | All tools are open-source |
| Storage | 0 | Local + GitHub + Google Drive |
| **Total** | **0 EGP** | Fully open-source toolchain |

---

## 10. Project Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Advisor | Aya Abdallah | | |
| Team Lead | Abdallah Zain | | |
| DEPI Coordinator | | | |
