# Road-Sense API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /health

Health check endpoint. Returns server status and performance statistics.

**Response**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/exports/best-3classes-exp34332-original.pt",
  "tracking_enabled": true,
  "stats": {
    "uptime_seconds": 120.5,
    "total_requests": 42,
    "error_count": 0,
    "error_rate": 0.0,
    "requests_per_second": 0.35,
    "latency_ms_avg": 85.2,
    "latency_ms_p50": 72.1,
    "latency_ms_p95": 145.3,
    "latency_ms_p99": 210.8
  }
}
```

**Example**

```bash
curl http://localhost:8000/health
```

---

### POST /detect

Run object detection on an uploaded image. Returns detected objects and an annotated image.

**Request**

| Parameter | Type | Location | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `image` | `UploadFile` | File | required | Must be image | JPEG or PNG image to analyze |
| `conf` | `float` | Form | `0.25` | 0.0 to 1.0 | Confidence threshold |
| `session_id` | `string` | Form | `"default"` | — | Session ID for temporal tracking |

**Response**

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether inference succeeded |
| `detections` | `Detection[]` | Array of detected objects |
| `annotated_image` | `string` | Base64-encoded JPEG with bounding boxes |
| `inference_time_ms` | `float` | Inference latency in milliseconds |
| `message` | `string` | Human-readable status message |

**Detection object**

| Field | Type | Description |
|-------|------|-------------|
| `class_name` | `string` | Object class (`Vehicle`, `Pedestrian`, `Cyclist`) |
| `confidence` | `float` | Detection confidence score |
| `bbox` | `float[4]` | Bounding box `[x1, y1, x2, y2]` in pixel coordinates |
| `track_id` | `int | null` | Tracking ID (null if tracking disabled) |

**Example**

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@data/samples/test.jpg" \
  -F "conf=0.3" \
  -F "session_id=demo-1"
```

**Response**

```json
{
  "success": true,
  "detections": [
    {
      "class_name": "Vehicle",
      "confidence": 0.92,
      "bbox": [120, 45, 340, 210],
      "track_id": 1
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "inference_time_ms": 85.3,
  "message": "Detected 1 tracked objects"
}
```

---

### POST /detect_batch

Run object detection on multiple images in a single batched request. Returns a list of detection results, one per input image.

**Request**

| Parameter | Type | Location | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `images` | `UploadFile[]` | Files | required | Must be images | List of JPEG/PNG images to analyze |
| `conf` | `float` | Form | `0.25` | 0.0 to 1.0 | Confidence threshold |

**Response**

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether all inferences succeeded |
| `results` | `DetectionResponse[]` | Array of per-image detection results |
| `total_time_ms` | `float` | Total processing time for all images |
| `message` | `string` | Human-readable status message |

**Example**

```bash
curl -X POST http://localhost:8000/detect_batch \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "conf=0.3"
```

**Response**

```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "detections": [
        {
          "class_name": "Vehicle",
          "confidence": 0.92,
          "bbox": [120, 45, 340, 210]
        }
      ],
      "annotated_image": "data:image/jpeg;base64,...",
      "inference_time_ms": 82.1,
      "message": "Detected 1 objects"
    }
  ],
  "total_time_ms": 175.4,
  "message": "Processed 2 images successfully"
}
```

---

### GET /metrics

Prometheus metrics endpoint. Only available when `prometheus-fastapi-instrumentator` is installed.

**Example**

```bash
curl http://localhost:8000/metrics
```

---

## Deployment

### Docker

```bash
docker build -t road-sense .
docker run -p 8000:8000 road-sense
```

### Docker Compose

```bash
cp .env.example .env
docker compose up -d
```

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `400` | Invalid image file or missing file |
| `422` | Missing required parameters |
| `500` | Internal inference error |
| `503` | Model not loaded (server still starting) |
