# Model Comparison Report: Road-Sense Object Detection

## 1. Objective
Evaluate multiple object detection models (YOLOv8, YOLOv11, SSD, Faster R-CNN) using pre-trained COCO weights and select the best fit for Road-Sense deployment.

## 2. Models Evaluated
- `yolov8s.pt` (Ultralytics YOLOv8, small variant)
- `yolo11m.pt` (Ultralytics YOLOv11, medium variant)
- `ssd300_vgg16` (TorchVision SSD)
- `fasterrcnn_resnet50_fpn_v2` (TorchVision Faster R-CNN)

## 3. Architecture Notes
### YOLOv8 (One-stage detector)
- Single-stage, anchor-free detection head.
- Prioritizes low latency and high throughput.
- Strong fit for real-time applications where frame rate is critical.

### YOLOv11 (One-stage detector, newer YOLO generation)
- Single-stage design with architecture and training improvements over earlier YOLO variants.
- Typically provides better accuracy-speed trade-off than smaller YOLO variants, at moderate compute cost.
- Good candidate when near-real-time speed and stronger localization quality are both required.

### SSD300 VGG16 (One-stage detector)
- Uses multi-scale feature maps with default boxes (anchors).
- Historically fast but generally lower modern accuracy than recent YOLO variants.
- Larger backbone (VGG16) increases model size.

### Faster R-CNN ResNet50 FPN v2 (Two-stage detector)
- Region Proposal Network + second-stage classifier/regressor.
- Usually high detection quality, especially for difficult objects.
- Computationally heavy, high latency, and weaker real-time suitability.

## 4. Benchmark Environment
Benchmarked with `bm.py` / `scripts/benchmark_models.py`.

### Runtime setup
- Frameworks: PyTorch, TorchVision, Ultralytics, TorchMetrics
- Weights: COCO pre-trained defaults
- Dataset: `coco128` sample subset
- Images used: 12 labeled images
- Input size: `640`
- Confidence threshold: `0.25`
- Warmup iterations: `2`
- Benchmark repeats: `3`

### Output files
- CUDA results: `artifacts/model_benchmark_results.cuda.csv`
- CPU results: `artifacts/model_benchmark_results.cpu.csv`
- Default active result file remains CUDA: `artifacts/model_benchmark_results.csv`

## 5. Inference Benchmark Results

### GPU (CUDA) nvidia RTX3050 4Gb
| Model | mAP@0.5 | mAP@0.5:0.95 | Latency (ms) | FPS | Size (MB) | Params (M) | CPU Mem Delta (MB) | GPU Peak Mem (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN R50 FPN v2 | 0.9362 | 0.7748 | 216.49 | 4.62 | 167.11 | 43.712 | 237.09 | 772.24 |
| YOLOv11m | 0.8564 | 0.7598 | 28.24 | 35.41 | 38.80 | 20.115 | 41.11 | 227.65 |
| YOLOv8s | 0.8778 | 0.7479 | 15.12 | 66.15 | 21.54 | 11.167 | 735.84 | 78.60 |
| SSD300 VGG16 | 0.6997 | 0.5034 | 38.70 | 25.84 | 135.99 | 35.642 | 176.37 | 362.25 |

### CPU Intel core i5 gen 12
| Model | mAP@0.5 | mAP@0.5:0.95 | Latency (ms) | FPS | Size (MB) | Params (M) | CPU Mem Delta (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN R50 FPN v2 | 0.9362 | 0.7693 | 4527.47 | 0.22 | 167.11 | 43.712 | 1068.42 |
| YOLOv11m | 0.8564 | 0.7598 | 408.65 | 2.45 | 38.80 | 20.115 | 312.98 |
| YOLOv8s | 0.8778 | 0.7479 | 178.18 | 5.61 | 21.54 | 11.167 | 227.45 |
| SSD300 VGG16 | 0.6997 | 0.5034 | 453.80 | 2.20 | 135.99 | 35.642 | 47.98 |

## 6. Analysis
- Best absolute accuracy: Faster R-CNN (`mAP@0.5:0.95 = 0.7748` on GPU), but too slow for real-time Road-Sense use.
- Best GPU speed: YOLOv8s (`66.15 FPS`) with strong accuracy.
- Best balanced GPU trade-off: YOLOv11m (`35.41 FPS`, `mAP@0.5:0.95 = 0.7598`).
- SSD is outperformed by YOLO variants on both accuracy and practical deployment profile.
- CPU-only deployment is not real-time for any tested model; YOLOv8s is fastest on CPU but still limited (`5.61 FPS`).

## 7. Final Selection
**Selected model: `YOLOv11m`**

### Why YOLOv11m for Road-Sense
- Delivers near Faster R-CNN quality (`0.7598` vs `0.7748` mAP@0.5:0.95) at much lower latency.
- Maintains real-time-capable GPU throughput (`35.41 FPS`).
- Model footprint (`38.8 MB`) is deployment-friendly compared with two-stage detectors.
- Better safety-oriented balance than YOLOv8s when prioritizing stronger localization quality over maximum FPS.

## 8. Deployment Recommendation
- Primary runtime: GPU inference (`cuda:0`) for production Road-Sense.
- Keep `YOLOv8s` as fallback profile when ultra-low latency is needed.
- Avoid Faster R-CNN and SSD for real-time edge path unless use case shifts to offline or server-side batch analysis.

## 9. Reproducibility
Use the benchmark runner:

```bash
cd scripts
python3 benchmark_models.py --device cuda:0
python3 benchmark_models.py --device cpu
```

Current benchmark artifacts used in this report:
- `artifacts/model_benchmark_results.cuda.csv`
- `artifacts/model_benchmark_results.cpu.csv`

---

## 10. KITTI Fine-Tuned Results

The tables above represent **COCO128 pre-trained** models. After fine-tuning on KITTI (3 classes: Vehicle, Pedestrian, Cyclist), the models achieved:

### Comparison: KITTI-Trained YOLO11m vs Fine-Tuned Faster R-CNN

| Metric | YOLO11m Baseline | YOLO11m HPO | Faster R-CNN (10 epochs) |
|--------|:----------------:|:-----------:|:-----------------------:|
| **mAP@50** | **0.942** | 0.935 | N/A |
| **mAP@50:95** | **0.768** | 0.725 | N/A |
| **Precision** | 0.870 | **0.893** | 0.843 |
| **Recall** | 0.830 | **0.894** | 0.897 |
| **F1 Score** | ~0.849 | **~0.893** | 0.869 |
| **FPS (RTX 3050)** | 33.6 | **36.4** | N/A |
| **Model Size** | 38.8 MB | 38.8 MB | ~167 MB |
| **Training Data** | 5236 images | 5236 images | 2067 images |
| **Epochs** | 100 | 100 | 10 |

> Full details in `reports/TRAINING_COMPARISON_REPORT.md`

### Key Takeaways from Fine-Tuning
- **YOLO11m HPO** is the best KITTI-trained model: highest precision (0.893), strong recall (0.894), and real-time speed
- **Faster R-CNN** shows competitive recall (0.897) but lower precision (0.843); only 10 epochs of training vs 100 for YOLO
- **ONNX FP16 export** further improves speed to **44.1 FPS** without accuracy loss (see `reports/MODEL_EVALUATION_REPORT.md`)
- **TFLite INT8 quantization** reduces model size to **19.7 MB** (51% smaller) with zero accuracy loss
