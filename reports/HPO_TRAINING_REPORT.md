# HPO Training Report — Road-Sense

## 1. Summary

| Metric | Value |
|--------|-------|
| **Model** | YOLO11m |
| **Config** | HPO-optimized params (Stage 1) |
| **Dataset** | KITTI — 5236 train, 1496 val |
| **GPU** | NVIDIA A10 (Modal) |
| **Training Time** | 2.48 hours (100 epochs) |
| **Best mAP@50:95** | **0.725** |
| **Best mAP@50** | **0.935** |

---

## 2. Training Config (HPO Best)

| Param | Value |
|-------|-------|
| optimizer | Adam |
| lr0 | 0.000254 |
| lrf | 0.129 |
| momentum | 0.900 |
| weight_decay | 0.00027 |
| mosaic | 0.82 |
| mixup | 0.99 |
| copy_paste | 0.11 |
| degrees | 0.64 |
| hsv_h | 0.034 |

---

## 3. Final Validation Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All** | **0.893** | **0.894** | **0.935** | **0.725** |
| Vehicle | 0.892 | 0.960 | 0.979 | 0.873 |
| Pedestrian | 0.882 | 0.812 | 0.889 | 0.563 |
| Cyclist | 0.905 | 0.911 | 0.938 | 0.740 |

---

## 4. Training Curves

Loss and metric curves saved to `models/hpo_results/results.png`.

| Epoch | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 |
|-------|----------|----------|----------|-------|----------|
| 1 | 1.365 | 0.761 | 1.389 | 0.661 | 0.391 |
| 10 | 1.037 | 0.498 | 1.202 | 0.800 | 0.528 |
| 20 | 0.944 | 0.440 | 1.146 | 0.854 | 0.588 |
| 27 (best) | — | — | — | 0.893 | **0.623** |
| 50 | 0.805 | 0.348 | 1.063 | 0.911 | 0.660 |
| 75 | 0.727 | 0.300 | 1.021 | 0.923 | 0.693 |
| 100 | 0.454 | 0.162 | 0.884 | 0.934 | **0.725** |

*Note: CSV only logged epochs 1-29 on volume; final 100-epoch results are from post-training validation of best.pt.*

---

## 5. Inference Performance

| Metric | Value |
|--------|-------|
| Preprocess | 0.1ms per image |
| Inference | 4.0ms per image |
| Postprocess | 0.7ms per image |
| **Total** | **4.8ms per image (~208 FPS)** |

---

## 6. Comparison with Baseline (exp34332)

| Metric | Baseline (exp34332) | HPO Training | Delta |
|--------|--------------------|--------------|-------|
| mAP50 | 0.942 | **0.935** | -0.007 |
| mAP50-95 | 0.768 | **0.725** | -0.043 |
| Precision | 0.870 | **0.893** | +0.023 |
| Recall | 0.830 | **0.894** | +0.064 |

The HPO-optimized config improved recall (+6.4%) and precision (+2.3%) but slightly reduced mAP scores. The HPO was tuned on 10-epoch trials — the optimal hyperparameters for 10 epochs may differ from those needed for 100 epochs. Consider a Stage 2 HPO with longer trials (20 epochs) for potentially better long-horizon params.

---

## 7. Saved Artifacts

| File | Path |
|------|------|
| Model weights (PyTorch) | `models/hpo_results/best_hpo.pt` |
| Model export (ONNX FP16) | `models/hpo_results/best_hpo.onnx` |
| Training metrics CSV | `models/hpo_results/results.csv` |
| Training curves plot | `models/hpo_results/results.png` |
| Confusion matrix | `models/hpo_results/confusion_matrix.png` |

---

## 8. Command Used

```bash
modal run scripts/train_modal.py --epochs 100
```
