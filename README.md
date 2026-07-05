# Road-Sense

Real-time road object detection for autonomous driving using YOLO models on the KITTI dataset.

[![CI](https://github.com/Abdallah4Z/Road-Sense/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdallah4Z/Road-Sense/actions)
[![GitHub Pages](https://github.com/Abdallah4Z/Road-Sense/actions/workflows/deploy.yml/badge.svg)](https://github.com/Abdallah4Z/Road-Sense/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Overview

Road-Sense detects **vehicles**, **pedestrians**, and **cyclists** in driving scenes using YOLO11m. It includes:

- **Training pipeline** — transfer learning with YOLO, HPO via Optuna, custom CNN support
- **Model export** — ONNX (FP16), TorchScript, TFLite
- **Inference API** — FastAPI server with real-time tracking, batching, and Prometheus metrics
- **CLI tools** — training, inference, dataset preprocessing/validation, benchmarking
- **Presentation website** — live demo with GitHub Pages deployment

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 16 GB |
| GPU | — | NVIDIA with 8 GB+ (CUDA 12) |
| Disk | 10 GB | 30 GB (with KITTI dataset) |
| OS | Linux, macOS, WSL2 | Linux (Ubuntu 22.04+) |

---

## Quick Start

```bash
# 1. Install
git clone https://github.com/Abdallah4Z/Road-Sense.git
cd Road-Sense
pip install -r requirements.txt

# 2. Validate raw KITTI data (optional, requires dataset)
python src/data/validate_kitti_quality.py

# 3. Run inference on a sample image
python src/models/inference.py \
    --weights models/checkpoints/best-3classes-exp34332.pt \
    --source data/sample.jpg \
    --output predictions/

# 4. Start the API server
python src/models/api_server.py --port 8000
# POST an image: curl -X POST -F "image=@test.jpg" http://localhost:8000/detect
```

> **New user setup time:** ≤ 30 minutes (KPI DC-05)

---

## Model Performance

### Best Model (Transfer Learning — exp34332)

| Metric | Value |
|--------|-------|
| Model | YOLO11m (COCO pretrained) |
| mAP@50 | **0.942** |
| mAP@50:95 | **0.768** |
| Precision | 0.870 |
| Recall | 0.830 |
| Inference (GPU) | ~4 ms/image |
| Inference (CPU) | ~45 ms/image |
| Model size | 39 MB (FP16 ONNX: 38 MB) |

### Per-Class Breakdown

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| Vehicle | 0.979 | 0.873 |
| Pedestrian | 0.889 | 0.563 |
| Cyclist | 0.938 | 0.740 |

---

## Usage

### CLI: Train

```bash
# Train with default config
python train.py

# Train with custom config + overrides
python train.py --config configs/training_hpo.yaml --epochs 100 --batch-size 16

# List available models
python train.py --list-models

# Dry run (print config, don't train)
python train.py --dry-run

# Resume from checkpoint
python train.py --resume models/checkpoints/last.pt
```

### CLI: Inference

```bash
# Single image
python src/models/inference.py \
    --weights models/checkpoints/best-3classes-exp34332.pt \
    --source data/sample.jpg

# Video file
python src/models/inference.py \
    --weights models/checkpoints/best-3classes-exp34332.pt \
    --source video.mp4 --output result.mp4

# Directory of images
python src/models/inference.py \
    --weights models/checkpoints/best-3classes-exp34332.pt \
    --source data/samples/ --output predictions/ --save-txt
```

### API: FastAPI Server

```bash
# Start server
python src/models/api_server.py --port 8000 --device 0
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + model status |
| POST | `/detect` | Detect objects in single image |
| POST | `/detect_batch` | Detect objects in multiple images |
| GET | `/metrics` | Prometheus metrics |

**Example requests:**

```bash
# Single image
curl -X POST http://localhost:8000/detect \
  -F "image=@test.jpg" \
  -F "conf=0.4"

# Batch
curl -X POST http://localhost:8000/detect_batch \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg"

# Health check
curl http://localhost:8000/health
```

### Python API

```python
from ultralytics import YOLO

model = YOLO("models/checkpoints/best-3classes-exp34332.pt")
results = model.predict("data/sample.jpg", conf=0.25)

for r in results:
    print(f"Detected {len(r.boxes)} objects")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        print(f"  {model.names[cls_id]}: {conf:.2f} at {bbox}")
```

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│   CLI / API  │────▶│   Inference  │────▶│   Detections  │
│   (FastAPI)  │     │  Pipeline    │     │  + Annotations│
└──────────────┘     └──────┬───────┘     └───────────────┘
                            │
                    ┌───────▼───────┐
                    │  YOLO Model   │
                    │  (ONNX/PT)    │
                    └───────────────┘
                            ▲
                    ┌───────┴───────┐
                    │   Trainer     │
                    │   + HPO       │
                    └───────────────┘
```

---

## Project Structure

```
Road-Sense/
├── configs/              # YAML configs (training, preprocessing, HPO)
├── data/                 # Dataset: raw/ → processed/ → augmented/
│   ├── raw/KITTI/
│   ├── processed/kitti/
│   └── augmented/
├── docs/                 # Documentation
├── experiments/          # Experiment results (HPO, benchmarks, comparisons)
├── models/
│   ├── checkpoints/      # Trained weights (.pt, .onnx)
│   └── exports/          # Export format variants
├── notebooks/            # Jupyter notebooks (EDA, training, augmentation)
├── presentation/         # GitHub Pages website
├── reports/              # Milestone and technical reports
├── scripts/              # Utility scripts (benchmarks, HPO, exports)
├── src/
│   ├── data/             # Dataset loading, preprocessing, validation, augmentation
│   ├── deployment/       # Model exporter
│   ├── mlops/            # Performance monitoring, logging
│   ├── models/           # Trainer, inference, API server, model factory
│   └── utils/            # Constants, logging, metrics, exceptions
├── tests/                # Pytest unit tests (47+ tests)
├── train.py              # Training entry point
├── Dockerfile            # Production container
├── docker-compose.yml    # Orchestration
├── pyproject.toml        # Project metadata, ruff, pytest config
└── requirements.txt      # Python dependencies
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Setup Guide](docs/QUICK_SETUP_GUIDE.md) | Step-by-step environment setup |
| [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) | Local, Docker, and cloud deployment |
| [Docker Usage Guide](docs/DOCKER_USAGE.md) | Docker build, run, and troubleshooting |
| [API Documentation](docs/API_DOCUMENTATION.md) | Full API reference |
| [Final Project Report](docs/FINAL_PROJECT_REPORT.md) | Comprehensive project summary |
| [Model Evaluation Report](reports/MODEL_EVALUATION_REPORT.md) | Performance metrics and analysis |
| [Training Report](docs/TRAINING_REPORT_EXP34332.md) | Baseline training results |
| [HPO Report](reports/HPO_REPORT.md) | Hyperparameter optimization |
| [Model Comparison](docs/models/MODEL_COMPARISON_REPORT.md) | YOLO vs SSD vs Faster R-CNN |
| [Project Details](docs/PROJECT_DETAILS.md) | Methodology and technical details |

**Presentation website:** https://abdallah4z.github.io/Road-Sense/

---

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api_server.py -v
```

**Current coverage:** ~80%+ across core modules. Pre-commit hooks enforce ruff linting on every commit.

---

## Docker

```bash
# Build
docker compose build

# Run
docker compose up -d

# API available at http://localhost:8000
```

See [Docker Usage Guide](docs/DOCKER_USAGE.md) for detailed instructions.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. Make changes and ensure pre-commit passes: `pre-commit run --all-files`
6. Write/update tests: `pytest tests/`
7. Submit a pull request

---

## Team

| Member | Role |
|--------|------|
| [Abdallah Zain](https://github.com/Abdallah4Z) | Lead, ML Engineering |
| [Ahmed Elkady](https://github.com/ahmed9194) | Backend, API |
| [Aya Ahmed](https://github.com/aya335) | Data Pipeline |
| [FatmaElzahraa Wahby](https://github.com/fatmawahby) | Model Evaluation |
| [Menna Tuallah Farghaly](https://github.com/fa290) | Training Pipeline |
| [Mohamed Abd El Mawgoud](https://github.com/MohamedAbdelMawjoud) | Custom CNN |

**Advisor:** Aya Abdallah

---

## License

MIT License. Dataset licenses remain with their original sources (KITTI, COCO, GTSDB).
