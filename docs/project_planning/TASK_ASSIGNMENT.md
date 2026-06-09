# Task Assignment & Roles

## 1. Team Structure

```
                    ┌──────────────────────────┐
                    │   Project Advisor        │
                    │   Aya Abdallah           │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   Team Lead              │
                    │   Abdallah Zain          │
                    │   (YOLO training, API,   │
                    │    export, preproc,      │
                    │    website, unit tests)  │
                    └───────────┬──────────────┘
                                │
        ┌───────────────┬───────┼───────┬───────────────┐
        │               │       │       │               │
  ┌─────▼──────┐ ┌─────▼───────┐ │ ┌─────▼───────┐ ┌─────▼────────┐
  │ Ahmed      │ │ Aya         │ │ │ Fatma       │ │ Menna        │
  │ Elkady     │ │ Ahmed       │ │ │ Elzahraa    │ │ Tuallah      │
  │ ML + CI/CD │ │ Data/Backend│ │ │ Wahby       │ │ Farghaly     │
  │ (CNN train,│ │ (data report│ │ │ Dataset Eng │ │ Docs + CLI   │
  │  temporal  │ │  callbacks, │ │ │ (quality,   │ │ (splits,     │
  │  tracking, │ │  realtime,  │ │ │  augments,  │ │  reports,    │
  │  CI/CD,    │ │  aug tests) │ │ │  config,    │ │  HPO, CLI,   │
  │  EDA)      │ │             │ │ │  monitoring,│ │  tests)      │
  └────────────┘ └─────────────┘ │ │  final rpt) │ └──────────────┘
                                 │ └─────────────┘
                         ┌───────▼──────────┐
                         │ Mohamed Abd El   │
                         │ Mawgoud          │
                         │ Custom CNN Dev   │
                         │ + Docker +       │
                         │ Export Benchmarks│
                         └──────────────────┘
```

---

## 2. Roles & Responsibilities

### 2.1 Abdallah Zain — Team Lead & Primary ML Engineer

| Responsibility | Description | Tasks |
|----------------|-------------|-------|
| **Team Coordination** | Lead weekly standups, assign tasks, track progress, resolve blockers | — |
| **Architecture Design** | Design system architecture, define technical stack, make key design decisions | — |
| **Data Preprocessing** | Implement KITTI-to-YOLO label conversion and class remapping | M1.2, M1.3 |
| **YOLO Model Factory** | Set up model factory supporting 10 YOLO variants | M2.1 |
| **Training Lifecycle Manager** | Build YOLOTrainer with Fatma | M2.2 (shared) |
| **YOLO Baseline Training** | Run YOLO11m baseline training (100 epochs) | M2.6 |
| **YOLO Hyperparameter Tuning** | Run YOLO hyperparameter tuning experiments | M2.8 |
| **Inference Server** | Build FastAPI inference server with /detect endpoint | M3.1 |
| **Model Export** | Export to ONNX, TFLite, TorchScript | M3.3 |
| **Unit Testing** | Write tests for trainer, callbacks, CLI with Menna | M4.3 (shared) |
| **Presentation Website** | Build GitHub Pages site and live demo page | M5.1, M5.2 |
| **Final Submission** | Package and submit final deliverables | M5.7 |
| **Code Review** | Review all pull requests for code quality and correctness | — |

### 2.2 Ahmed Elkady — ML & Backend Engineer

| Responsibility | Description | Tasks |
|----------------|-------------|-------|
| **EDA & Data Report** | Run EDA and write dataset exploration report | M1.8 |
| **Temporal Object Tracking** | Implement IoU + EMA tracking for API | M3.2 |
| **Export Validation** | Validate exported model formats | M3.4 |
| **CI/CD Pipeline** | Set up GitHub Actions for tests + linting | M4.4 |

### 2.3 Aya Ahmed — Data & Backend Developer

| Responsibility | Description | Tasks |
|----------------|-------------|-------|
| **Data Quality Report** | Write data quality report | M1.6 |
| **Training Callbacks** | Implement logging and checkpointing callbacks | M2.7 |
| **Real-time Detection** | Build webcam/video detection with FPS overlay | M3.5 |
| **Augmentation Testing** | Write unit tests for augmentation module | M4.1 |

### 2.4 FatmaElzahraa Wahby — Dataset Engineer

| Responsibility | Description | Tasks |
|----------------|-------------|-------|
| **Data Quality Validation** | Implement corruption detection, duplicate removal, OOB box fixing | M1.5 |
| **Augmentation Pipeline** | Implement Albumentations pipelines (3 severity levels) | M1.7 |
| **Training Lifecycle Manager** | Co-build YOLOTrainer with Abdallah | M2.2 (shared) |
| **Training Config** | Create training configuration YAML | M2.3 |
| **API Documentation** | Write API documentation and usage examples | M3.8 |
| **Performance Monitoring** | Implement memory/latency monitoring | M4.6 |
| **Final Report** | Write final project report | M5.4 |

### 2.5 Menna Tuallah Farghaly — Documentation & Testing

| Responsibility | Description | Tasks |
|----------------|-------------|-------|
| **Data Split** | Implement train/val/test stratified split | M1.4 |
| **Training Reports** | Write training report (exp34332 baseline) | M2.10 |
| **Model Comparison Report** | Write model comparison report | M2.11 |
| **HPO Report** | Write hyperparameter optimization results report | M2.12 |
| **Inference CLI** | Write image/video batch inference CLI | M3.6 |
| **KITTI Utility Tests** | Write unit tests for KITTI utilities | M4.2 |
| **Trainer Unit Tests** | Co-write tests for trainer, callbacks, CLI with Abdallah | M4.3 (shared) |
| **Project Documentation** | Write project planning docs (proposal, plan, risk, KPIs) | M5.3 |

### 2.6 Mohamed Abd El Mawgoud — ML Engineer & DevOps

| Responsibility | Description | Tasks |
|----------------|-------------|-------|
| **Docker Setup** | Create Dockerfile and containerize application | M3.7 |
| **Export Benchmarks** | Run export format benchmarks | M4.5 |

---

## 3. Task Ownership Matrix (by Plan Tasks)

| Task ID | Task | Assigned To |
|---------|------|-------------|
| M1.1 | KITTI raw dataset download | All members |
| M1.2 | KITTI-to-YOLO label conversion | Abdallah Zain |
| M1.3 | Class remapping (9 → 3) | Abdallah Zain |
| M1.4 | Train/val/test stratified split | Menna Tuallah Farghaly |
| M1.5 | Data quality validation | FatmaElzahraa Wahby |
| M1.6 | Data quality report | Aya Ahmed |
| M1.7 | Augmentation pipeline (3 levels) | FatmaElzahraa Wahby |
| M1.8 | EDA & dataset exploration report | Ahmed Elkady |
| M1.9 | Dataset integrity verification | All members |
| M2.1 | YOLO model factory (10 variants) | Abdallah Zain |
| M2.2 | Training lifecycle manager (YOLOTrainer) | Abdallah Zain + FatmaElzahraa Wahby |
| M2.3 | Training configuration YAML | FatmaElzahraa Wahby |
| M2.4 | YOLO11m baseline training | Abdallah Zain |
| M2.5 | Training callbacks (logging, checkpointing) | Aya Ahmed |
| M2.6 | YOLO hyperparameter tuning | Abdallah Zain |
| M2.7 | Model benchmarking (YOLO vs SSD vs Faster R-CNN) | All members |
| M2.8 | Training report (exp34332) | Menna Tuallah Farghaly |
| M2.9 | Model comparison report | Menna Tuallah Farghaly |
| M2.10 | HPO report | Menna Tuallah Farghaly |
| M3.1 | FastAPI inference server | Abdallah Zain |
| M3.2 | Temporal object tracking (IoU + EMA) | Ahmed Elkady |
| M3.3 | Model export (ONNX, TFLite, TorchScript) | Abdallah Zain |
| M3.4 | Export format validation | Ahmed Elkady |
| M3.5 | Real-time webcam detection | Aya Ahmed |
| M3.6 | Inference CLI | Menna Tuallah Farghaly |
| M3.7 | Docker containerization | Mohamed Abd El Mawgoud |
| M3.8 | API documentation | FatmaElzahraa Wahby |
| M4.1 | Augmentation unit tests | Aya Ahmed |
| M4.2 | KITTI utility unit tests | Menna Tuallah Farghaly |
| M4.3 | Trainer/callbacks/CLI unit tests | Abdallah Zain + Menna Tuallah Farghaly |
| M4.4 | GitHub Actions CI/CD | Ahmed Elkady |
| M4.5 | Export format benchmarks | Mohamed Abd El Mawgoud |
| M4.6 | Performance monitoring (memory, latency) | FatmaElzahraa Wahby |
| M4.7 | Code review & refactoring | All members |
| M5.1 | Presentation website (GitHub Pages) | Abdallah Zain |
| M5.2 | Live detection demo page | Abdallah Zain |
| M5.3 | Project planning documentation | Menna Tuallah Farghaly |
| M5.4 | Final project report | FatmaElzahraa Wahby |
| M5.5 | Final presentation slides | All members |
| M5.6 | Dry-run presentation + peer feedback | All members |
| M5.7 | Final submission packaging | Abdallah Zain |

---

## 4. Module/Component Ownership

| Module/Component | Primary Owner | Secondary | Reviewer |
|------------------|---------------|-----------|----------|
| `src/data/kitti_utils.py` | Abdallah Zain | Menna Tuallah | Ahmed Elkady |
| `src/data/kitti_dataset.py` | Abdallah Zain | FatmaElzahraa | Mohamed |
| `src/data/augmentations.py` | FatmaElzahraa Wahby | Aya Ahmed | Abdallah Zain |
| `src/data/augment_dataset.py` | FatmaElzahraa Wahby | Abdallah Zain | Menna Tuallah |
| `src/data/preprocess_dataset.py` | Abdallah Zain | Mohamed | Ahmed Elkady |
| `src/data/validate_kitti_quality.py` | FatmaElzahraa Wahby | Menna Tuallah | Aya Ahmed |
| `src/data/verify_dataset.py` | Ahmed Elkady | FatmaElzahraa | Abdallah Zain |
| `src/models/model_factory.py` | Abdallah Zain | Ahmed Elkady | Mohamed |
| `src/models/trainer.py` | Abdallah Zain | FatmaElzahraa Wahby | Mohamed |
| `src/models/inference.py` | Menna Tuallah Farghaly | Ahmed Elkady | Abdallah Zain |
| `src/models/realtime.py` | Aya Ahmed | Mohamed | Abdallah Zain |
| `src/models/api_server.py` | Abdallah Zain | FatmaElzahraa | Aya Ahmed |
| `src/models/export.py` | Abdallah Zain | Mohamed | Ahmed Elkady |
| `src/models/callbacks.py` | Aya Ahmed | Abdallah Zain | Ahmed Elkady |
| `configs/training.yaml` | FatmaElzahraa Wahby | Abdallah Zain | Menna Tuallah |
| `configs/preprocessing.yaml` | Abdallah Zain | FatmaElzahraa | Menna Tuallah |
| `tests/test_augmentations.py` | Aya Ahmed | Menna Tuallah | Abdallah Zain |
| `tests/test_kitti_utils.py` | Menna Tuallah | FatmaElzahraa | Abdallah Zain |
| `tests/test_trainer.py` | Abdallah Zain | Menna Tuallah Farghaly | Mohamed |
| `scripts/benchmark_models.py` | Abdallah Zain | Ahmed Elkady | Mohamed |
| `scripts/compare_models.py` | Abdallah Zain | Menna Tuallah | Ahmed Elkady |
| `scripts/export_and_benchmark_formats.py` | Mohamed | Abdallah Zain | Ahmed Elkady |
| `scripts/dataset_exploration_analysis.py` | Ahmed Elkady | Menna Tuallah | FatmaElzahraa |
| `presentation/` (all files) | Abdallah Zain | FatmaElzahraa | All members |
| `Dockerfile` | Mohamed | Abdallah Zain | Aya Ahmed |
| `.github/workflows/deploy.yml` | Ahmed Elkady | Mohamed | Abdallah Zain |
| `train.py` | Abdallah Zain | Ahmed Elkady | Mohamed |
| `docs/project_planning/*` | Menna Tuallah | All members | Aya Ahmed |
| `docs/literature_review/*` | Menna Tuallah | All members | FatmaElzahraa |

---

## 5. Communication Responsibilities

| Role | Person | Communication Duties |
|------|--------|---------------------|
| **Advisor Liaison** | Abdallah Zain | Schedule advisor meetings, share progress updates, relay feedback |
| **GitHub Admin** | Abdallah Zain | Manage repository settings, branch protection, access control |
| **Meeting Minutes** | Menna Tuallah Farghaly | Document meeting notes, action items, decisions |
| **CI/CD Steward** | Ahmed Elkady | Monitor CI pipeline health, fix build issues |
| **Documentation Coordinator** | Menna Tuallah Farghaly | Ensure all docs are written, formatted, and submitted on time |
| **Quality Champion** | Aya Ahmed | Track test coverage, enforce coding standards |

---

## 6. Escalation Path

```
Issue Identified
       │
       ▼
Team Member → Team Lead (Abdallah Zain)
       │                       │
       │                   Can resolve?
       │                   ├── Yes → Implement solution
       │                   └── No  → Escalate to Advisor (Aya Abdallah)
       │
       └── Document in GitHub Issue
```

---

## 7. Task Tracking

| Tool | Purpose | URL / Location |
|------|---------|----------------|
| **GitHub Issues** | Bug tracking, feature requests, task management | `https://github.com/{org}/Road-Sense/issues` |
| **GitHub Projects** | Kanban board for sprint tracking | GitHub Projects tab |
| **Meeting Notes** | Action items from weekly standups | `docs/meeting_notes/` [in progress] |
| **Progress Dashboard** | Milestone completion tracking | `docs/project_planning/PROJECT_PLAN.md` (Section 3) |
