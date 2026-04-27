# Risk Assessment & Mitigation Plan

## 1. Risk Matrix Overview

| Risk ID | Risk Description | Category | Probability | Impact | Risk Score | Mitigation Strategy |
|---------|-----------------|----------|-------------|--------|------------|---------------------|
| R-01 | Insufficient GPU compute time for training | Technical | High | High | **Critical** | Contingency: Use free Google Colab tier with background execution; optimize for smaller YOLO variants (YOLO11n/s) if needed |
| R-02 | Dataset download corruption or incomplete data | Data | Medium | High | **High** | Verify MD5 checksums after download; maintain backup source (Google Drive mirror); implement data validation script |
| R-03 | Model fails to meet accuracy targets (mAP < 0.70) | Technical | Medium | High | **High** | Iterative hyperparameter tuning; fallback to YOLO11x (larger model); ensemble multiple checkpoints; transfer learning from better pretrained weights |
| R-04 | Team member unavailable due to illness/emergency | Human | Medium | Medium | **Medium** | Cross-train all members on key modules; maintain documentation so any member can pick up; redistribute tasks weekly |
| R-05 | Version control conflicts or merge issues | Process | Medium | Low | **Medium** | Enforce branch protection rules; require PR reviews; use feature branches; frequent small commits |
| R-06 | Dependency version incompatibility (e.g., PyTorch + CUDA) | Technical | Medium | Medium | **Medium** | Pin all dependency versions in requirements.txt; use Docker for environment consistency; document conda environment export |
| R-07 | Scope creep — feature additions beyond plan | Process | Medium | Low | **Medium** | Strict milestone gating; defer out-of-scope features to future work; change request process requires advisor approval |
| R-08 | GitHub Pages deployment failure or broken website | Technical | Low | Medium | **Medium** | Test site locally before deployment; maintain fallback deployment via Netlify or Vercel; run CI dry-run on PR |
| R-09 | KITTI dataset class imbalance (Vehicle dominant at 86%) | Data | High | Medium | **High** | Use class-weight-aware loss functions; apply targeted augmentation for minority classes (Pedestrian, Cyclist); evaluate per-class mAP separately |
| R-10 | Real-time inference below target FPS on CPU | Technical | Medium | High | **High** | Optimize with ONNX Runtime; apply INT8 quantization for TFLite; fall back to YOLO11n (nano variant); use frame skipping for video |
| R-11 | API server crashes under concurrent load | Technical | Low | Medium | **Medium** | Implement request queuing; add rate limiting; use async FastAPI with proper timeout handling; load test before deployment |
| R-12 | Data preprocessing pipeline produces incorrect labels | Data | Low | High | **Medium** | Implement cross-validation of label conversion (visual inspection of 100+ random samples); write unit tests for all conversion functions |
| R-13 | Export format produces degraded accuracy vs PyTorch | Technical | Medium | Medium | **Medium** | Validate export accuracy on test set; benchmark all export formats; document any accuracy loss; use FP16 as fallback |
| R-14 | Team communication breakdown | Human | Low | High | **Medium** | Weekly standup meetings; shared WhatsApp group for daily updates; clear escalation path; meeting minutes documented |
| R-15 | Advisor feedback requires major rework near deadline | Process | Low | High | **Medium** | Regular bi-weekly advisor check-ins; share interim deliverables early; maintain buffer time in schedule (weeks 10-12) |
| R-16 | Data storage limits exceeded (local + cloud) | Technical | Medium | Low | **Low** | Use git-lfs for large files; store raw dataset in Google Drive; clean up intermediate artifacts; use .gitignore for large directories |

---

## 2. Risk Categorization

### 2.1 Technical Risks (7 items)

| Risk ID | Risk | Probability | Impact | Score | Mitigation |
|---------|------|-------------|--------|-------|------------|
| R-01 | Insufficient GPU compute | High | High | **9** | Colab Pro free tier, fallback to smaller YOLO variants |
| R-03 | Model accuracy below target | Medium | High | **6** | Iterative tuning, fallback to larger model, ensemble |
| R-06 | Dependency conflicts | Medium | Medium | **4** | Docker, pinned requirements, conda export |
| R-08 | Website deployment failure | Low | Medium | **3** | Local testing, fallback hosting |
| R-10 | Inference too slow on CPU | Medium | High | **6** | ONNX Runtime, quantization, nano variant |
| R-11 | API server crashes | Low | Medium | **3** | Async handling, rate limiting, load testing |
| R-16 | Storage limits exceeded | Medium | Low | **2** | git-lfs, Google Drive, cleanup scripts |

### 2.2 Data Risks (3 items)

| Risk ID | Risk | Probability | Impact | Score | Mitigation |
|---------|------|-------------|--------|-------|------------|
| R-02 | Corrupted dataset download | Medium | High | **6** | MD5 verification, backup mirror |
| R-09 | Class imbalance (86% Vehicle) | High | Medium | **6** | Class weighting, targeted augmentation |
| R-12 | Incorrect label conversion | Low | High | **3** | Visual inspection, unit tests |

### 2.3 Human/Process Risks (5 items)

| Risk ID | Risk | Probability | Impact | Score | Mitigation |
|---------|------|-------------|--------|-------|------------|
| R-04 | Team member unavailable | Medium | Medium | **4** | Cross-training, documentation |
| R-05 | Git merge conflicts | Medium | Low | **2** | Branch protection, PR reviews |
| R-07 | Scope creep | Medium | Low | **2** | Milestone gating, change control |
| R-14 | Communication breakdown | Low | High | **3** | Weekly meetings, escalation path |
| R-15 | Major rework late in project | Low | High | **3** | Early interim deliverables, buffer time |

---

## 3. Risk Response Plan

### 3.1 Critical Risks (Score ≥ 8)

**R-01: Insufficient GPU Compute Time**

| Step | Action | Trigger | Owner |
|------|--------|---------|-------|
| 1 | Monitor Colab session time limits and GPU quota usage | Weekly | Ahmed Elkady |
| 2 | Save checkpoints to Google Drive every 10 epochs | Training start | Ahmed Elkady |
| 3 | If GPU time exhausted: resume from latest checkpoint on alternative GPU | Session timeout | Abdallah Zain |
| 4 | If no GPU available: train YOLO11n/s as fallback (lighter, faster) | 24h without GPU | Abdallah Zain |
| 5 | Document any accuracy loss from fallback model | Fallback used | Ahmed Elkady |

### 3.2 High Risks (Score 5-7)

**R-03: Model Fails Accuracy Target**

| Step | Action | Trigger | Owner |
|------|--------|---------|-------|
| 1 | Analyze validation metrics per class to identify weak spots | After epoch 50 | Ahmed Elkady |
| 2 | Adjust hyperparameters (learning rate, augmentation, loss weights) | After analysis | Ahmed Elkady |
| 3 | If mAP@0.5:0.95 < 0.65 by epoch 80: switch to YOLO11x | Epoch 80 checkpoint | Abdallah Zain |
| 4 | If still below target: ensemble top-3 checkpoints | Final evaluation | Abdallah Zain |

**R-09: Class Imbalance Hurts Minority Classes**

| Step | Action | Trigger | Owner |
|------|--------|---------|-------|
| 1 | Monitor per-class mAP in validation logs | Every epoch | Ahmed Elkady |
| 2 | Apply heavy augmentation to minority class samples | After epoch 30 check | Aya Ahmed |
| 3 | Adjust class weights in YOLO loss function | If Pedestrian mAP < 0.50 | Abdallah Zain |
| 4 | Evaluate with class-balanced metrics | Final evaluation | Menna Tuallah |

**R-10: Real-time Inference Below Target FPS**

| Step | Action | Trigger | Owner |
|------|--------|---------|-------|
| 1 | Profile inference latency by component | First test | Mohamed |
| 2 | Apply ONNX Runtime optimizations | Latency > 100ms | Mohamed |
| 3 | Apply INT8 quantization for TFLite export | ONNX still > 80ms | Mohamed |
| 4 | Fall back to YOLO11n variant | Still below 15 FPS | Abdallah Zain |

### 3.3 Medium Risks (Score 3-4)

| Risk ID | Action Plan | Owner |
|---------|-------------|-------|
| R-04 | Maintain skill matrix; rotate documentation writing weekly; create handoff checklist | Abdallah Zain |
| R-06 | Use `pip freeze > requirements.txt` after each environment change; test in clean venv before committing | Mohamed |
| R-11 | Add request timeout middleware; configure uvicorn workers based on CPU cores; run `locust` load test | Abdallah Zain |
| R-15 | Submit draft docs 1 week before advisor review deadline; maintain "buffer week" in schedule | Menna Tuallah |

---

## 4. Contingency Budget

| Item | Time Buffer | Notes |
|------|-------------|-------|
| Training failures/restarts | 5 days | Accounted for in M2 schedule |
| Data re-processing | 2 days | If quality checks fail |
| API integration issues | 3 days | Buffer in M3 schedule |
| Documentation rework | 5 days | Buffer in M5 schedule |
| **Total Buffer** | **15 days** | Spread across all milestones |

---

## 5. Risk Monitoring & Review

| Frequency | Activity | Responsible |
|-----------|----------|-------------|
| **Weekly** | Review risk status at team standup; update probability/impact ratings | Abdallah Zain |
| **Bi-weekly** | Present risk dashboard to advisor during check-in meetings | Abdallah Zain |
| **Per Milestone** | Conduct risk retrospective; document lessons learned | All members |
| **Ad-hoc** | Trigger risk response plan when any risk materializes | Assigned owner |

---

## 6. Risk Register Template

| Field | Description |
|-------|-------------|
| Risk ID | Unique identifier (R-XX) |
| Description | Clear statement of the risk |
| Category | Technical / Data / Human / Process |
| Probability | Low (1) / Medium (2) / High (3) |
| Impact | Low (1) / Medium (2) / High (3) |
| Risk Score | Probability × Impact (1-9) |
| Trigger | Event that indicates risk is materializing |
| Response Plan | Step-by-step actions to mitigate |
| Owner | Person responsible for monitoring |
| Status | Active / Mitigated / Closed |
| Last Reviewed | Date of last status update |
