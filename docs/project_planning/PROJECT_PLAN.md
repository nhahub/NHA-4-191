# Project Plan: Road-Sense

## 1. Overview

| Field | Value |
|-------|-------|
| **Project Name** | Road-Sense: Real-Time Road Scene Object Detection |
| **Duration** | 12 Weeks |
| **Start Date** | Week 1 |
| **End Date** | Week 12 |
| **Team Size** | 6 Members |
| **Advisor** | Aya Abdallah |

---

## 2. Project Timeline (Gantt Chart)

### 2.1 Phase Breakdown

```
Phase                    | W1  | W2  | W3  | W4  | W5  | W6  | W7  | W8  | W9  | W10 | W11 | W12 |
-------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
M1: DATA PREPROCESSING   |█████|█████|     |     |     |     |     |     |     |     |     |     |
  - KITTI Download       |█████|     |     |     |     |     |     |     |     |     |     |     |
  - Preprocessing Script |█████|█████|     |     |     |     |     |     |     |     |     |     |
  - Quality Validation   |     |█████|     |     |     |     |     |     |     |     |     |     |
  - Augmentation Pipeline|     |█████|     |     |     |     |     |     |     |     |     |     |
  - EDA & Data Report    |     |█████|     |     |     |     |     |     |     |     |     |     |

M2: MODEL DEVELOPMENT    |     |     |█████|█████|█████|     |     |     |     |     |     |     |
  - YOLO Model Factory   |     |     |█████|     |     |     |     |     |     |     |     |     |
  - YOLO Baseline Train  |     |     |█████|█████|     |     |     |     |     |     |     |     |
  - YOLO Hyperparameter  |     |     |     |█████|     |     |     |     |     |     |     |     |
  - Custom CNN Develop   |     |     |█████|█████|     |     |     |     |     |     |     |     |
  - Custom CNN Training  |     |     |     |█████|█████|     |     |     |     |     |     |     |
  - Model Comparison     |     |     |     |     |█████|     |     |     |     |     |     |     |
  - Training Reports     |     |     |     |     |█████|     |     |     |     |     |     |     |
  - HPO Report           |     |     |     |     |█████|     |     |     |     |     |     |     |

M3: DEPLOYMENT & API     |     |     |     |     |     |█████|█████|     |     |     |     |     |
  - Inference Server     |     |     |     |     |     |█████|█████|     |     |     |     |     |
  - Model Export         |     |     |     |     |     |█████|     |     |     |     |     |     |
  - Real-time Detection  |     |     |     |     |     |     |█████|     |     |     |     |     |
  - Docker Container     |     |     |     |     |     |     |█████|     |     |     |     |     |

M4: MLOPS & TESTING      |     |     |     |     |     |     |     |█████|█████|     |     |     |
  - Unit Tests           |     |     |     |     |     |     |     |█████|     |     |     |     |
  - CI/CD Pipeline       |     |     |     |     |     |     |     |█████|     |     |     |     |
  - Benchmark Export     |     |     |     |     |     |     |     |     |█████|     |     |     |
  - Performance Monitor  |     |     |     |     |     |     |     |     |█████|     |     |     |

M5: DOCS & PRESENTATION  |     |     |     |     |     |     |     |     |     |█████|█████|█████|
  - Presentation Website |     |     |     |     |     |     |     |     |     |█████|█████|     |
  - Final Documentation  |     |     |     |     |     |     |     |     |     |█████|█████|█████|
  - Final Presentation   |     |     |     |     |     |     |     |     |     |     |     |█████|
  - Project Submission   |     |     |     |     |     |     |     |     |     |     |     |█████|
```

### 2.2 Detailed Task Breakdown

#### M1: Data Collection & Preprocessing (Weeks 1-2)

| Task ID | Task | Duration | Dependencies | Assigned To |
|---------|------|----------|--------------|-------------|
| M1.1 | Download KITTI raw dataset (images + labels, ~12 GB) | 3 days | None | All members |
| M1.2 | Implement KITTI-to-YOLO label conversion script | 4 days | M1.1 | Abdallah Zain |
| M1.3 | Implement class remapping (9 KITTI classes → 3 target classes) | 2 days | M1.2 | Abdallah Zain |
| M1.4 | Implement train/val/test stratified split (70/20/10) | 2 days | M1.3 | Menna Tuallah Farghaly |
| M1.5 | Implement data quality validation (corruption, duplicates, OOB boxes) | 4 days | M1.2 | FatmaElzahraa Wahby |
| M1.6 | Write data quality report | 2 days | M1.5 | Aya Ahmed |
| M1.7 | Implement Albumentations augmentation pipeline (3 severity levels) | 4 days | M1.4 | FatmaElzahraa Wahby |
| M1.8 | Run EDA and write dataset exploration report | 3 days | M1.4 | Ahmed Elkady |
| M1.9 | Verify preprocessed dataset integrity | 2 days | M1.8 | All members |

#### M2: Model Development & Training (Weeks 3-5)

| Task ID | Task | Duration | Dependencies | Assigned To |
|---------|------|----------|--------------|-------------|
| M2.1 | Set up YOLO model factory (support 10 YOLO variants) | 3 days | M1.9 | Abdallah Zain |
| M2.2 | Implement training lifecycle manager (YOLOTrainer) | 5 days | M2.1 | Abdallah Zain and FatmaElzahraa Wahby |
| M2.3 | Create training configuration YAML (hyperparameters) | 2 days | M2.2 | FatmaElzahraa Wahby |
| M2.4 | Build custom CNN model from scratch (architecture design, layers, forward pass) | 5 days | M1.9 | Mohamed Abd El Mawgoud |
| M2.5 | Train custom CNN model (hyperparameter tuning, evaluation) | 5 days | M2.4 | Ahmed Elkady |
| M2.6 | Run YOLO11m baseline training (100 epochs) | 5 days | M2.3 | Abdallah Zain |
| M2.7 | Implement training callbacks (logging, checkpointing) | 3 days | M2.2 | Aya Ahmed |
| M2.8 | Run YOLO hyperparameter tuning experiments | 5 days | M2.6 | Abdallah Zain |
| M2.9 | Benchmark custom CNN vs YOLO vs SSD vs Faster R-CNN | 5 days | M2.5, M2.6 | All members |
| M2.10 | Write training report (exp34332 baseline) | 3 days | M2.6 | Menna Tuallah Farghaly |
| M2.11 | Write model comparison report | 3 days | M2.9 | Menna Tuallah Farghaly |
| M2.12 | Write HPO report (hyperparameter optimization results) | 3 days | M2.5, M2.8 | Menna Tuallah Farghaly |

#### M3: Deployment & API Development (Weeks 6-7)

| Task ID | Task | Duration | Dependencies | Assigned To |
|---------|------|----------|--------------|-------------|
| M3.1 | Implement FastAPI inference server with /detect endpoint | 5 days | M2.10 | Abdallah Zain |
| M3.2 | Implement temporal object tracking for API (IoU + EMA) | 3 days | M3.1 | Ahmed Elkady |
| M3.3 | Implement model export to ONNX, TFLite, TorchScript | 4 days | M2.10 | Abdallah Zain |
| M3.4 | Validate exported model formats | 2 days | M3.3 | Ahmed Elkady |
| M3.5 | Implement real-time webcam detection with FPS overlay | 4 days | M2.10 | Aya Ahmed |
| M3.6 | Write inference CLI for images/video batch processing | 3 days | M2.10 | Menna Tuallah Farghaly |
| M3.7 | Create Dockerfile and containerize application | 3 days | M3.1 | Mohamed Abd El Mawgoud |
| M3.8 | Write API documentation and usage examples | 2 days | M3.1 | FatmaElzahraa Wahby |

#### M4: MLOps & Testing (Weeks 8-9)

| Task ID | Task | Duration | Dependencies | Assigned To |
|---------|------|----------|--------------|-------------|
| M4.1 | Write unit tests for augmentations module (21+ tests) | 4 days | M1.7 | Aya Ahmed |
| M4.2 | Write unit tests for KITTI utilities (12+ tests) | 3 days | M1.2 | Menna Tuallah Farghaly |
| M4.3 | Write unit tests for trainer, callbacks, CLI (22+ tests) | 5 days | M2.5 | Abdallah Zain and Menna Tuallah Farghaly |
| M4.4 | Set up GitHub Actions CI/CD for tests + linting | 3 days | M4.1 | Ahmed Elkady |
| M4.5 | Run export format benchmarks | 3 days | M3.3 | Mohamed Abd El Mawgoud |
| M4.6 | Implement performance monitoring (memory, latency) | 4 days | M3.1 | FatmaElzahraa Wahby |
| M4.7 | Code review and refactoring | 5 days | M4.1-M4.3 | All members |

#### M5: Final Documentation & Presentation (Weeks 10-12)

| Task ID | Task | Duration | Dependencies | Assigned To |
|---------|------|----------|--------------|-------------|
| M5.1 | Build presentation website (GitHub Pages) | 10 days | M3.8 | Abdallah Zain |
| M5.2 | Implement live detection demo page | 5 days | M3.1 | Abdallah Zain |
| M5.3 | Write project planning documentation | 5 days | All | Menna Tuallah Farghaly |
| M5.4 | Write final project report | 7 days | All | FatmaElzahraa Wahby |
| M5.5 | Create final presentation slides | 5 days | M5.4 | All members |
| M5.6 | Conduct dry-run presentation + peer feedback | 3 days | M5.5 | All members |
| M5.7 | Final submission and deliverable packaging | 2 days | M5.6 | Abdallah Zain |

---

## 3. Milestones & Deliverables

| Milestone | Target Date | Deliverables | Acceptance Criteria |
|-----------|-------------|--------------|---------------------|
| **M1: Data Collection & Preprocessing** | End of Week 2 | - Preprocessed KITTI dataset (YOLO format) <br/> - Data quality report <br/> - Augmentation pipeline code <br/> - EDA report | - Data splits verified (70/20/10) <br/> - Zero corrupted/missing files <br/> - Augmentation pipelines produce valid outputs <br/> - All 3 class distributions documented |
| **M2: Model Development & Training** | End of Week 5 | - Trained YOLO11m weights (best.pt) <br/> - Training config YAML <br/> - Training report (exp34332) <br/> - Model comparison report | - mAP@0.5:0.95 ≥ 0.70 <br/> - Training converges within 100 epochs <br/> - YOLO11m outperforms SSD & Faster R-CNN <br/> - All metrics logged and reproducible |
| **M3: Deployment & API Development** | End of Week 7 | - FastAPI inference server <br/> - Exported models (ONNX, TFLite, TorchScript) <br/> - Real-time detection CLI <br/> - Docker container | - API responds in < 100ms <br/> - All export formats produce consistent predictions <br/> - Real-time detection runs at ≥ 15 FPS <br/> - Docker image builds successfully |
| **M4: MLOps & Testing** | End of Week 9 | - Unit test suite (55+ tests) <br/> - GitHub Actions workflow <br/> - Export format benchmarks <br/> - Performance monitoring reports | - All tests pass <br/> - Code coverage ≥ 75% <br/> - CI pipeline runs on every push <br/> - Ruff linting passes with zero errors |
| **M5: Final Documentation & Presentation** | End of Week 12 | - Presentation website (GitHub Pages) <br/> - Final project documentation <br/> - Presentation slides <br/> - All deliverables packaged | - Website deployed and accessible <br/> - All docs complete and reviewed <br/> - Presentation rehearsed and recorded <br/> - All code pushed to main branch |

---

## 4. Resource Allocation

### 4.1 Time Allocation (Person-Hours per Phase)

| Team Member | M1 | M2 | M3 | M4 | M5 | **Total** |
|-------------|----|----|----|----|----|-----------|
| Abdallah Zain | 25 | 35 | 35 | 20 | 40 | **155** |
| Ahmed Elkady | 25 | 20 | 20 | 20 | 15 | **100** |
| Aya Ahmed | 20 | 15 | 20 | 20 | 15 | **90** |
| FatmaElzahraa Wahby | 25 | 10 | 15 | 25 | 15 | **90** |
| Menna Tuallah Farghaly | 15 | 30 | 10 | 20 | 15 | **90** |
| Mohamed Abd El Mawgoud | 10 | 25 | 20 | 20 | 15 | **90** |
| **Total Per Phase** | **120** | **130** | **120** | **125** | **115** | **610** |

### 4.2 Hardware Allocation

| Resource | Allocation | Used By |
|----------|------------|---------|
| GPU (RTX 3050 / A6000) | Training + benchmarking | Abdallah Zain |
| Development Laptops (×6) | Code development, testing, docs | All members |
| Google Drive / GitHub | Dataset storage, code hosting | All members |
| GitHub Pages | Website hosting | Abdallah Zain |

### 4.3 Software/Tool Allocation

| Tool | Purpose | Primary User(s) |
|------|---------|-----------------|
| VS Code + Python | IDE | All members |
| Git + GitHub | Version control | All members |
| Google Colab (Pro free) | Cloud GPU training | Abdallah Zain, Ahmed Elkady |
| Ruff | Python linting | All members |
| Pytest | Unit testing | Aya Ahmed, Abdallah Zain |
| Draw.io / Mermaid | Diagrams | Menna Tuallah Farghaly |
| Canva / PowerPoint | Presentation slides | All members |

---

## 5. Communication Plan

| Channel | Frequency | Participants | Purpose |
|---------|-----------|--------------|---------|
| WhatsApp Group | Daily | All members | Quick updates, questions, blockers |
| Weekly Standup (Google Meet) | Weekly (30 min) | All members | Progress review, next steps |
| GitHub Issues | As needed | All members | Bug tracking, feature requests |
| GitHub Pull Requests | Per task | Reviewer + Author | Code review |
| Advisor Meeting (Google Meet) | Bi-weekly | All members + Advisor | Milestone review, feedback |

---

## 6. Quality Management Plan

| Quality Dimension | Measurement | Target | Monitoring Method |
|-------------------|-------------|--------|-------------------|
| Code Quality | Ruff linting score | Zero errors | Pre-commit hook + CI |
| Test Coverage | pytest-cov percentage | ≥ 75% | CI pipeline report |
| Model Accuracy | mAP@0.5:0.95 | ≥ 0.70 | Training logs |
| API Performance | p95 response time | < 150ms | Load testing |
| Documentation | Completion checklist | 100% items checked | Milestone reviews |
