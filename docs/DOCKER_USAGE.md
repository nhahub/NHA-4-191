# Docker Usage Guide — Road-Sense

## Prerequisites

- **Docker Engine** 24.0+
- **Docker Compose** v2.0+ (normally bundled with Docker Desktop)
- **NVIDIA Container Toolkit** (optional, for GPU acceleration)

### Verify Installation

```bash
docker --version
docker compose version
```

---

## 1. Building the Image

### Using Docker Compose (Recommended)

```bash
docker compose build
```

### Using Docker Directly

```bash
# Standard build
docker build -t road-sense:latest .

# Build without cache (clean rebuild)
docker build --no-cache -t road-sense:latest .
```

The Dockerfile uses a **multi-stage build**:
- **Builder stage**: Installs dependencies and builds Python packages
- **Runtime stage**: Copies only the runtime artifacts (smaller final image)

**Image size:** ~2.8 GB (includes PyTorch, CUDA runtime, and OpenCV)

---

## 2. Running the Container

### With Docker Compose

```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Restart
docker compose restart
```

### With Docker (Manual)

```bash
# CPU only
docker run -d \
  --name road-sense \
  -p 8000:8000 \
  -v ./models:/app/models \
  -v ./.env:/app/.env:ro \
  road-sense:latest

# With GPU
docker run -d \
  --name road-sense \
  --gpus all \
  -p 8000:8000 \
  -v ./models:/app/models \
  -v ./.env:/app/.env:ro \
  road-sense:latest
```

The container starts the FastAPI inference server on port 8000.

---

## 3. Volume Mounts

| Host Path | Container Path | Required | Purpose |
|-----------|---------------|----------|---------|
| `./models` | `/app/models` | Yes | Model weights and exports |
| `./.env` | `/app/.env:ro` | No | Environment variables (read-only) |

### Model Directory Structure

The container expects the following structure under `/app/models/`:

```
models/
├── checkpoints/
│   ├── best-3classes-exp34332.pt      # Baseline PyTorch weights
│   └── HPO_run/weights/
│       ├── best.pt                    # HPO-optimized weights
│       └── best.onnx                  # HPO-optimized ONNX export
└── exports/
    ├── best-3classes-exp34332.pt      # (fallback)
    └── best-3classes-exp34332.onnx    # ONNX export
```

---

## 4. Verifying the Container is Running

```bash
# Check container status
docker ps

# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_path": "models/checkpoints/best-3classes-exp34332.pt",
#   "tracking_enabled": true,
#   "stats": { ... }
# }
```

---

## 5. Running Inference

```bash
# Single image via API
curl -X POST http://localhost:8000/detect \
  -F "image=@/path/to/image.jpg" \
  -F "conf=0.25"

# Batch inference
curl -X POST http://localhost:8000/detect_batch \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg"
```

---

## 6. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` | Flush stdout/stderr immediately |
| `PYTHONPATH` | `/app` | Python module search path |

These are set in the Dockerfile and don't normally need changing.

For application-level configuration, use a `.env` file:

```bash
# .env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_CONFIDENCE=0.25
LOG_LEVEL=INFO
```

---

## 7. Health Check

Docker Compose includes an auto-configured health check:

```yaml
healthcheck:
  test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 15s
```

The container status will show as `healthy` once the model is loaded and the API is responding.

---

## 8. Troubleshooting

### Container exits immediately

```bash
# Check logs
docker compose logs

# Common causes:
# - Model weights not found (mount ./models)
# - Port already in use (change port or stop other container)
# - Out of memory (check docker stats)
```

### GPU not available in container

```bash
# Install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Then restart Docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi
```

### Slow inference

```bash
# Check if GPU is being used
docker logs road-sense | grep "device"

# Fall back to ONNX Runtime (CPU optimized)
# The container runs on GPU by default; for CPU, use ONNX export
```

### Image size too large

The multi-stage build keeps the image at ~2.8 GB (primarily PyTorch + CUDA). To reduce size:
- Use a CPU-only image (remove CUDA dependencies)
- Use ONNX Runtime instead of PyTorch for inference
- Prune unused Docker images: `docker image prune -a`

---

## 9. Cleanup

```bash
# Stop and remove containers
docker compose down

# Remove image
docker rmi road-sense:latest

# Remove all unused Docker resources
docker system prune -a
```
