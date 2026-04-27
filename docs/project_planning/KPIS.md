# Key Performance Indicators (KPIs)

## 1. KPI Framework

The success of Road-Sense is measured across **five dimensions**: Model Performance, System Performance, Code Quality, Documentation, and Project Management. Each KPI has a defined target, measurement method, and owner.

---

## 2. Model Performance KPIs

| KPI ID | KPI | Target | Measurement Method | Frequency | Current Status |
|--------|-----|--------|-------------------|-------|----------------|
| **MP-01** | Mean Average Precision (mAP@0.5:0.95) | ≥ **0.75** | TorchMetrics MeanAveragePrecision on test set | Per training run | **0.768** ✅ |
| **MP-02** | Mean Average Precision (mAP@0.5) | ≥ **0.92** | TorchMetrics MeanAveragePrecision at IoU=0.5 | Per training run |  **0.942** ✅ |
| **MP-03** | Precision (all classes) | ≥ **0.90** | YOLO validation metrics | Per training run |  **0.924** ✅ |
| **MP-04** | Recall (all classes) | ≥ **0.88** | YOLO validation metrics | Per training run |  **0.916** ✅ |
| **MP-05** | F1 Score | ≥ **0.89** | 2 × (Precision × Recall) / (Precision + Recall) | Per training run | **0.920** ✅ |
| **MP-06** | Per-class mAP@0.5 (Vehicle) | ≥ **0.95** | Per-class mAP on test set | Per training run |  - |
| **MP-07** | Per-class mAP@0.5 (Pedestrian) | ≥ **0.80** | Per-class mAP on test set | Per training run | - |
| **MP-08** | Per-class mAP@0.5 (Cyclist) | ≥ **0.85** | Per-class mAP on test set | Per training run | - |
| **MP-09** | Training convergence speed | ≤ **100 epochs** | Epoch count to reach best mAP | Per training run |  **95 epochs** ✅ |
| **MP-10** | Overfitting gap (train vs val mAP) | ≤ **0.10** | Train mAP - Val mAP | Per training run | - |

---

## 3. System Performance KPIs

| KPI ID | KPI | Target | Measurement Method | Frequency |Current Status |
|--------|-----|--------|-------------------|-------|----------------|
| **SP-01** | GPU Inference Speed (RTX 3050) | ≥ **30 FPS** | FPS counter over 1000 frames | Per model version |  **35.4 FPS** ✅ |
| **SP-02** | CPU Inference Speed | ≥ **15 FPS** | FPS counter over 1000 frames | Per model version | - |
| **SP-03** | GPU Inference Latency (p95) | ≤ **35 ms** | Per-frame timing over 1000 frames | Per model version | **~28 ms** ✅ |
| **SP-04** | CPU Inference Latency (p95) | ≤ **70 ms** | Per-frame timing over 1000 frames | Per model version | - |
| **SP-05** | API Response Time (p95) | ≤ **150 ms** | Load testing with locust (100 requests) | Per deployment |  - |
| **SP-06** | API Throughput | ≥ **10 req/s** | Concurrent request handling test | Per deployment | - |
| **SP-07** | API Uptime | ≥ **99.5%** | Health-check endpoint monitoring | Continuous |  - |
| **SP-08** | Model Size (PyTorch) | ≤ **50 MB** | File size of best.pt | Per model | **38.8 MB** ✅ |
| **SP-09** | Model Size (ONNX exported) | ≤ **80 MB** | File size of exported .onnx | Per export |  - |
| **SP-10** | Docker Image Size | ≤ **3 GB** | `docker images` command | Per build |  - |
| **SP-11** | Memory Usage (inference) | ≤ **2 GB RAM** | psutil during inference | Per test |  - |
| **SP-12** | Export Accuracy Drop (ONNX vs PyTorch) | ≤ **0.01 mAP** | Compare mAP on test set | Per export | - |
| **SP-13** | Export Accuracy Drop (TFLite vs PyTorch) | ≤ **0.02 mAP** | Compare mAP on test set | Per export |  - |

---

## 4. Code Quality KPIs

| KPI ID | KPI | Target | Measurement Method | Frequency |  Current Status |
|--------|-----|--------|-------------------|-----------|----------------|
| **CQ-01** | Test Pass Rate | **100%** | pytest execution | Per commit |  **55+ tests pass** ✅ |
| **CQ-02** | Code Coverage | ≥ **75%** | pytest-cov report | Per milestone |  - |
| **CQ-03** | Linting Score (Ruff) | **Zero errors** | ruff check | Per commit | ✅ |
| **CQ-04** | Type Hint Coverage | ≥ **80%** of functions | Manual review + mypy | Per milestone | - |
| **CQ-05** | PR Review Time | ≤ **48 hours** | Time from PR open to merge | Per PR | - |
| **CQ-06** | PR Merge Rate | ≥ **90%** first-approval | Approved PRs / Total PRs | Per milestone | - |
| **CQ-07** | Documentation Coverage of Public APIs | **100%** | Manual audit of docstrings | Per milestone |  - |

---

## 5. Documentation KPIs

| KPI ID | KPI | Target | Measurement Method | Frequency | Current Status |
|--------|-----|--------|-------------------|-----------|----------------|
| **DC-01** | Milestone Documents Complete | **5/5** | Milestone checklist | Per milestone |M1 ✅ M2 ✅ M3 ⬜ M4 ⬜ M5 ⬜ |
| **DC-02** | README Completeness Score | **100%** | Rubric: overview, setup, usage, architecture, team, license | Per milestone | - |
| **DC-03** | Website Pages Deployed | **5/5** (home, docs, reports, visuals, detect) | GitHub Pages deployment | Per milestone |  ✅ |
| **DC-04** | Code Comment Density | ≥ **1 comment per 50 lines** | `rg -c '#' src/` | Per milestone | - |
| **DC-05** | Setup Guide Clarity | **New user sets up in ≤ 30 min** | User testing | Per milestone |  ✅ |

---

## 6. Project Management KPIs

| KPI ID | KPI | Target | Measurement Method | Frequency | Current Status |
|--------|-----|--------|-------------------|-----------|----------------|
| **PM-01** | Milestone On-Time Delivery | **100%** | Actual vs planned delivery date | Per milestone |  - |
| **PM-02** | Task Completion Rate (weekly) | ≥ **80%** | Completed tasks / Planned tasks | Weekly |  - |
| **PM-03** | Team Member Utilization | ≥ **80%** | Hours logged / Hours planned | Weekly |  - |
| **PM-04** | Advisor Meeting Attendance | **100%** | Attendees / Invited | Bi-weekly |  - |
| **PM-05** | Risk Mitigation Effectiveness | ≥ **90%** | Risks successfully mitigated / Total risks realized | Per milestone |  - |
| **PM-06** | Meeting Action Item Closure | ≥ **85%** within 1 week | Closed items / Total action items | Weekly |  - |

---

## 7. User Adoption / Impact KPIs

| KPI ID | KPI | Target | Measurement Method | Frequency | Current Status |
|--------|-----|--------|-------------------|----------- |----------------|
| **UA-01** | GitHub Stars | ≥ **10** | GitHub repository star count | Per milestone | - |
| **UA-02** | GitHub Forks / Clone Count | ≥ **5** forks | GitHub insights | Per milestone |  - |
| **UA-03** | Website Visitors | ≥ **100 unique** | GitHub Pages analytics (or manual) | Per milestone |  - |
| **UA-04** | Demo Page Interactions | ≥ **50 detection runs** | Logged interactions | Per milestone |  - |

---

## 8. KPI Dashboard Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ROAD-SENSE KPI DASHBOARD (END OF PROJECT)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODEL PERFORMANCE    │  SYSTEM PERFORMANCE    │  CODE QUALITY             │
│                        │                        │                           │
│  mAP@0.5:0.95  ██████░│  GPU FPS        ███████│  Test Pass %   ███████   │
│  mAP@0.5       ███████│  CPU FPS        ████░░░│  Coverage      ████░░░   │
│  Precision     ███████│  API p95 (ms)   █████░░│  Ruff Errors   ███████   │
│  Recall        ███████│  Model Size     ███████│  PR Time (h)   ██████░   │
│  F1            ███████│  Docker Size    ████░░░│  Doc Coverage  ██████░   │
│                        │                        │                           │
│  ███████ = On Target   │  ██████░ = Near Target  │  ████░░░ = Needs Work   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. KPI Review Schedule

| Review Point | KPIs Reviewed | Participants |
|-------------|---------------|--------------|
| **Weekly Standup** | PM-02, PM-03, PM-06 | All team members |
| **Bi-weekly Advisor Meeting** | MP-01 to MP-10, SP-01 to SP-04 | Advisor + All members |
| **Milestone Gate Review** | All KPIs | Advisor + All members |
| **Final Evaluation** | All KPIs | Advisor + External evaluators |

---

## 10. Corrective Action Triggers

| Trigger Condition | Action |
|-------------------|--------|
| Any MP KPI below 80% of target | Schedule model improvement sprint | 
| Any SP KPI below 70% of target | Performance optimization sprint | 
| CQ-01 (test pass rate) below 100% | Block PR merges until all tests pass | 
| PM-01 (on-time delivery) below target | Re-plan remaining milestones with advisor | 
| DC-01 (documents missing) | Reallocate team members to documentation | 
