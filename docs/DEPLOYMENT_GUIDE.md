# Deployment Guide — Road-Sense

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores, 2.5 GHz | 8 cores, 3.0 GHz+ |
| RAM | 4 GB | 8 GB |
| GPU | — | NVIDIA A10G / RTX 3080+ (CUDA 12) |
| Disk | 5 GB (model + code) | 30 GB (with dataset) |
| Network | Broadband | Low-latency for API |

---

## 1. Local Deployment (pip + CLI)

### Prerequisites
- Python 3.10+
- Git
- pip

### Steps

```bash
# Clone
git clone https://github.com/Abdallah4Z/Road-Sense.git
cd Road-Sense

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download model weights (Git LFS)
git lfs pull
# or manually download from releases:
# https://github.com/Abdallah4Z/Road-Sense/releases
```

### Run Inference

```bash
# Single image
python src/models/inference.py \
    --weights models/checkpoints/best-3classes-exp34332.pt \
    --source data/sample.jpg

# Video
python src/models/inference.py \
    --weights models/checkpoints/best-3classes-exp34332.pt \
    --source video.mp4 --output result.mp4
```

### Run API Server

```bash
# Default (port 8000, GPU if available)
python src/models/api_server.py

# Custom port and device
python src/models/api_server.py --port 9000 --device cpu

# With custom weights (supports .pt, .onnx, .torchscript)
python src/models/api_server.py \
    --weights models/exports/best-3classes-exp34332.onnx

# With HPO-optimized model
python src/models/api_server.py \
    --weights models/checkpoints/HPO_run/weights/best.pt
```

---

## 2. Docker Deployment

### Prerequisites
- Docker Engine 24+
- Docker Compose v2+
- NVIDIA Container Toolkit (for GPU)

### Quick Start

```bash
# Build and start
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# API is live at http://localhost:8000
curl http://localhost:8000/health

# Run inference
curl -X POST http://localhost:8000/detect \
  -F "image=@test.jpg" \
  -F "conf=0.25"
```

### Manual Docker

```bash
# Build image
docker build -t road-sense:latest .

# Run with CPU
docker run -d \
  --name road-sense \
  -p 8000:8000 \
  -v ./models:/app/models \
  road-sense:latest

# Run with GPU
docker run -d \
  --name road-sense \
  --gpus all \
  -p 8000:8000 \
  -v ./models:/app/models \
  road-sense:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `MODEL_PATH` | `models/checkpoints/best.pt` | Path to model weights |
| `MODEL_CONFIDENCE` | `0.25` | Default confidence threshold |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Create a `.env` file in the project root (see `.env.example`).

---

## 3. Cloud Deployment

### GitHub Pages (Frontend)

The presentation website is automatically deployed via GitHub Actions on push to `main`:

```yaml
# .github/workflows/deploy.yml — already configured
```

Live at: https://abdallah4z.github.io/Road-Sense/

### Modal Cloud (Training)

For GPU-accelerated training:
```bash
# Setup
pip install modal
python3 -m modal setup

# Upload dataset
modal volume create road-sense-data
modal volume put road-sense-data data data
modal volume put road-sense-data configs configs
modal volume put road-sense-data models models

# Run training
modal run scripts/train_modal.py --epochs 100
```

### Custom Cloud VM (API)

For production API serving, deploy the Docker container to any cloud provider:

```bash
# Pull image on cloud VM
docker pull ghcr.io/abdallah4z/road-sense:latest

# Run with reverse proxy (Caddy/Nginx)
docker run -d \
  --name road-sense \
  --gpus all \
  -p 8000:8000 \
  -v /data/models:/app/models \
  --restart unless-stopped \
  ghcr.io/abdallah4z/road-sense:latest

# Set up Nginx reverse proxy with SSL
```

---

## 4. Health Checks

The API exposes a health endpoint:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/checkpoints/best-3classes-exp34332.pt",
  "tracking_enabled": true,
  "stats": {
    "uptime_seconds": 1234.56,
    "total_requests": 42,
    "error_rate": 0.0,
    "latency_ms_p95": 12.3
  }
}
```

Docker Compose includes a health check that runs every 30 seconds.

---

## 5. Monitoring

Prometheus metrics are available at `/metrics` when the server starts:

```bash
curl http://localhost:8000/metrics
```

Pre-configured metrics:
- Request count, latency (p50/p95/p99), error rate
- Model inference time
- Uptime

---

## 6. Optimization Tools

### Validate Exported Model Accuracy

Compare mAP across all export formats to ensure accuracy is preserved:

```bash
# Full validation (requires dataset)
python scripts/validate_exports.py \
    --weights models/checkpoints/HPO_run/weights/best.pt \
    --data data/processed/kitti/data.yaml

# Structure-only check (no dataset needed)
python scripts/validate_exports.py \
    --weights models/checkpoints/HPO_run/weights/best.pt \
    --structure-only
```

### INT8 Quantization

Export and evaluate quantized TFLite models for edge deployment:

```bash
# Export TFLite (FP16 + INT8) with accuracy comparison
python scripts/quantize_model.py \
    --weights models/checkpoints/HPO_run/weights/best.pt \
    --data data/processed/kitti/data.yaml

# Full pipeline with speed benchmark
python scripts/quantize_model.py \
    --weights models/checkpoints/HPO_run/weights/best.pt \
    --data data/processed/kitti/data.yaml \
    --benchmark
```

### CPU Inference Benchmark

Measure and compare ONNX Runtime vs PyTorch CPU inference speed:

```bash
# Basic benchmark
python scripts/benchmark_onnx_cpu.py \
    --weights models/checkpoints/HPO_run/weights/best.pt

# With explicit ONNX path and more images
python scripts/benchmark_onnx_cpu.py \
    --weights models/checkpoints/HPO_run/weights/best.pt \
    --onnx models/exports/best.onnx \
    --num-images 500
```

---

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch size, use `--device cpu`, or export to FP16 ONNX |
| Model not found | Run `git lfs pull` or download from releases |
| Docker GPU not available | Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |
| Port in use | Change port with `--port 9000` or set `API_PORT=9000` |
| Slow CPU inference | Export to ONNX and use ONNX Runtime (`pip install onnxruntime`) |
