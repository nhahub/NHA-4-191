# Road-Sense: Real-Time Object Detection for Autonomous Vehicles

## Slide 1: Title
**Road-Sense**
Real-Time Object Detection for Autonomous Vehicles
Team: Abdallah Zain, Menna Farghaly, Aya Ahmed, FatmaElzahraa Wahby, Mohamed El Mawgoud, Ahmed Elkady
Date: July 2026

---

## Slide 2: Problem Statement
- Autonomous vehicles must detect objects in real-time
- KITTI benchmark: 3 classes (Vehicle, Pedestrian, Cyclist)
- Target: ≥30 FPS, mAP@50 ≥0.90, RAM <2 GB

---

## Slide 3: Dataset & Preprocessing
- **KITTI Vision Benchmark Suite**
  - 7,481 images, 24,128 annotated objects
  - 3 classes: Vehicle (76%), Pedestrian (20%), Cyclist (4%)
- **Preprocessing Pipeline:**
  - KITTI → YOLO format conversion
  - 70/20/10 train/val/test split
  - Augmentation: Mosaic, MixUp, HSV, flips
  - Quality validation (corruption/duplicate detection)

---

## Slide 4: Model Architecture
- **Selected: YOLO11m** (Ultralytics)
  - One-stage detector, anchor-free
  - 20.0M parameters, 67.7 GFLOPs
  - COCO pre-trained → transfer learning to KITTI
- **Alternatives evaluated:**
  - YOLOv8s, SSD300, Faster R-CNN (COCO128 benchmark)
  - Faster R-CNN fine-tuned on KITTI (10 epochs)

---

## Slide 5: Training & HPO
- **Baseline:** YOLO11m, 100 epochs
  - mAP@50: **0.942**, mAP@50:95: **0.768**
- **HPO:** Optuna, 2-stage
  - Stage 1: 10 epochs × 20 trials (coarse search)
  - Stage 2: 100 epochs with best config
  - Key findings: AdamW, lr=0.00042, mixup=0.99
  - mAP@50: **0.935**, Precision: **0.893**, Recall: **0.894**

---

## Slide 6: Results - Detection Performance

| Metric | YOLO11m Baseline | YOLO11m HPO |
|--------|:----------------:|:-----------:|
| mAP@50 | **0.942** | 0.935 |
| mAP@50:95 | **0.768** | 0.725 |
| Precision | 0.870 | **0.893** |
| Recall | 0.830 | **0.894** |

| Class | mAP@50 | mAP@50:95 |
|-------|:------:|:---------:|
| Vehicle | 0.979 | 0.873 |
| Pedestrian | 0.889 | 0.563 |
| Cyclist | 0.938 | 0.740 |

---

## Slide 7: Results - Speed

| Format | Size | GPU FPS | p95 Latency | Peak RAM |
|--------|:----:|:-------:|:-----------:|:--------:|
| PyTorch (.pt) | 38.8 MB | 33.6 | 32.1 ms | 2.85 GB |
| ONNX FP16 | **38.3 MB** | **44.1** | **28.2 ms** | 3.33 GB |
| TorchScript | 77.0 MB | 33.2 | 33.3 ms | 3.34 GB |
| TFLite INT8 | **19.7 MB** | 36.4† | — | — |

**ONNX FP16 is recommended:** 44 FPS, zero accuracy loss.

---

## Slide 8: Deployment Architecture

```
Client → FastAPI Server → ONNX Runtime → Detection Results
                    ↓
           Prometheus Metrics
                    ↓
           Performance Monitor (RAM < 2 GB)
```

- **API:** FastAPI with /detect and /health endpoints
- **Runtime:** ONNX Runtime (CUDA) or ONNXRunner (CPU fallback)
- **Monitoring:** Latency p50/p95/p99, RAM budget check (KPI SP-11)
- **Container:** Docker + Docker Compose

---

## Slide 9: Demo
- Live video detection with bounding boxes
- FPS counter overlay
- Per-class confidence scores
- Pre-recorded demo available

---

## Slide 10: Lessons Learned
1. **Transfer learning is highly effective** — COCO pretrained → strong KITTI results with minimal fine-tuning
2. **HPO stage 1 (10 epochs) sufficient** for coarse hyperparameter search
3. **MixUp augmentation** (0.99) improves generalization for safety-critical applications
4. **ONNX FP16 is production-ready** — faster than PyTorch, zero accuracy loss
5. **TFLite INT8** — 51% smaller model, zero accuracy loss
6. **Pedestrian detection remains challenging** (mAP@50:95=0.563)

---

## Slide 11: Future Work
- TensorRT optimization (2× FPS improvement expected)
- Multi-dataset training (GTSDB, COCO)
- Custom CNN architecture exploration
- Real-time video processing pipeline
- Live metrics dashboard on website

---

## Slide 12: Thank You
**Questions?**
GitHub: https://github.com/Abdallah4Z/Road-Sense
Presentation website: https://abdallah4z.github.io/Road-Sense/
