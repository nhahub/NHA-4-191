# Tests for Road-Sense Project

Tests are run via **pytest** with **coverage** tracking.

## Quick Start

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=term

# With HTML coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Current Coverage: **80%** (320 tests)

| File | Coverage |
|------|----------|
| `src/utils/*` | 97-100% |
| `src/data/augmentations.py` | 96% |
| `src/data/kitti_utils.py` | 92% |
| `src/data/validate_kitti_quality.py` | 89% |
| `src/models/export.py` | 87% |
| `src/data/verify_dataset.py` | 83% |
| `src/models/api_server.py` | 82% |
| `src/models/inference.py` | 82% |
| `src/data/kitti_dataset.py` | 79% |
| `src/data/augment_dataset.py` | 78% |
| `src/data/preprocess_dataset.py` | 74% |
| `src/models/trainer.py` | 75% |
| `src/models/model_factory.py` | 92% |
| `src/models/callbacks.py` | 94% |
| `src/mlops/performance_monitor.py` | 100% |

## Test Files

### Core Data Tests
- `test_augmentations.py` — Data augmentation pipelines (96% coverage)
- `test_kitti_utils.py` — KITTI utility functions (load/save labels, conversion)
- `test_kitti_dataset.py` — KITTI dataset class basics
- `test_kitti_dataset_coverage.py` — Dataset torch, collate, data loaders
- `test_kitti_utils_coverage.py` — Image loading, label parsing edge cases, statistics
- `test_preprocess_pipeline.py` — Preprocessing pipeline, image-label processing
- `test_preprocess_coverage.py` — Class mapping, filtering, splitting, normalization
- `test_preprocess_extra.py` — Edge cases, error paths
- `test_validate_kitti_quality.py` — Dataset quality validation
- `test_verify_dataset.py` — Dataset verification
- `test_augment_dataset_coverage.py` — Batch augmentation pipeline, CLI

### Model Tests
- `test_trainer.py` — Trainer core (config, setup, callbacks)
- `test_model_edge.py` — Model factory, callbacks, trainer section builders
- `test_trainer_coverage.py` — Extract metrics, save results, section builders
- `test_trainer_more.py` — Config snapshots, checkpoint args, init callbacks
- `test_trainer_final.py` — Setup edge cases, train/validate/export error paths
- `test_export.py` — Export CLI parsing
- `test_export_coverage.py` — Export model success/failure, multi-format
- `test_inference_coverage.py` — Model loading, prediction, CLI setup

### API Server Tests
- `test_api_server.py` — Performance monitor, parse args, encoding
- `test_api_server_coverage.py` — SessionTracker, TrackState, Response models
- `test_api_server_run.py` — Main entry point, load model with device
- `test_session_tracker.py` — Tracker smoothing, decay, multi-class matching
- `test_resolve_weights.py` — Weight path resolution edge cases
- `test_api_coverage.py` — Drawing functions, cleanup, detection extraction

### Utils Tests
- `test_utils_module.py` — Constants, exceptions, metrics, I/O utilities
- `test_logger.py` — JSON formatter, file rotation, console handlers
- `test_metrics_edge.py` — Precision/recall edge cases, box IoU
- `test_final_push.py` — Retry decorator, box IoU multi, logger exceptions

### CLI / Main Script Tests
- `test_cli_scripts.py` — Inference + realtime CLI argument parsing
- `test_train_main.py` — Train script parse args, overrides, log file setup
- `test_edge_cases.py` — Yolo-to-pixel edge cases, label parsing

## Writing New Tests

Follow pytest conventions:
- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`
- Fixtures: use `tmp_path` for temp files, `capsys` for stdout capture

## CI

Tests auto-run on push/PR via `.github/workflows/ci.yml` (ruff + pytest + coverage).
