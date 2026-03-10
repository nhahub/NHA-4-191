# Multi-Dataset Training Strategy for Road-Sense Project
**Real-Time Object Detection for Autonomous Vehicles**

---

##  Project Goal
Build a YOLO-based model that detects:
- ✅ **Vehicles** (Car, Van, Truck)
- ✅ **Pedestrians** (including Person_sitting)
- ✅ **Cyclists**
- ✅ **Traffic Signs** (Speed limits, Warning, Mandatory, Priority, Prohibition)

---

##  Available Datasets

### 1. KITTI Dataset
- **Objects**: Vehicles, Pedestrians, Cyclists
- **Format**: Detection (2D bounding boxes)
- **Size**: 7,481 training images
- **Annotation**: KITTI format → convert to YOLO
- **✅ Status**: Ready for processing


### 2. German Traffic Sign Detection Benchmark (GTSDB) - RECOMMENDED
- **Objects**: Traffic signs in driving scenes
- **Format**: ✅ **Detection** (bounding boxes)
- **Size**: ~900 images with ~1,200 signs
- **Annotation**: Real bounding boxes in context
- **Status**: ⚠️ Need to download and process

---

##  Recommended Training Strategy

### **OPTION 1: Two-Stage Training (EASIEST - RECOMMENDED)**

This approach trains the model in two sequential stages, keeping tasks separate.

#### **Stage 1: KITTI Fine-Tuning (Weeks 2-3)**
```yaml
Goal: Detect vehicles, pedestrians, and cyclists
Base Model: YOLOv8/v11 pre-trained on COCO
Dataset: KITTI only
Classes: 3 (Vehicle, Pedestrian, Cyclist)
```

**Steps:**
1. **Preprocess KITTI**
   ```bash
   python scripts/preprocess_data.py
   # Uses configs/preprocessing.yaml
   # Output: data/processed/kitti/
   ```

2. **Fine-tune YOLO**
   ```python
   from ultralytics import YOLO
   
   # Load COCO pre-trained model
   model = YOLO('yolov8n.pt')
   
   # Fine-tune on KITTI
   results = model.train(
       data='data/processed/kitti/data.yaml',
       epochs=100,
       imgsz=640,
       batch=16,
       name='kitti_vehicles_pedestrians',
       patience=20,
       save=True,
       device=0  # GPU
   )
   ```

3. **Evaluate**
   - mAP@0.5
   - mAP@0.5:0.95
   - FPS (frames per second)

**Expected Output:**
- Model detects vehicles and people in driving scenes
- mAP > 0.70 (70%)
- FPS > 30 for real-time performance

---

#### **Stage 2: Traffic Sign Integration (Week 4)**

**Three Sub-Options:**

##### **2A: Use GTSDB (Best - Real Data)** ⭐ RECOMMENDED
```yaml
Goal: Add traffic sign detection
Dataset: GTSDB (German Traffic Sign Detection Benchmark)
Size: ~900 images with bounding boxes
Approach: Continue training from Stage 1 model
```

**Steps:**
1. **Download GTSDB**
   - Source: https://benchmark.ini.rub.de/gtsdb_dataset.html
   - Contains signs in real driving context with bounding boxes

2. **Preprocess GTSDB**
   - Convert to YOLO format
   - Group 43 classes → 5 categories (Speed, Warning, Mandatory, Priority, Prohibition)

3. **Continue Training**
   ```python
   # Load Stage 1 model
   model = YOLO('runs/detect/kitti_vehicles_pedestrians/weights/best.pt')
   
   # Add new classes for traffic signs
   # Fine-tune on GTSDB while preserving vehicle/pedestrian detection
   results = model.train(
       data='data/processed/gtsdb/data.yaml',  # Combined with KITTI
       epochs=50,
       imgsz=640,
       batch=16,
       name='combined_kitti_gtsdb',
       freeze=10,  # Freeze first 10 layers to preserve vehicle detection
       device=0
   )
   ```

**Advantages:**
- ✅ Real traffic signs in driving context
- ✅ Proper bounding boxes
- ✅ No synthetic data artifacts

**Disadvantages:**
- ❌ Small dataset (~900 images)
- ❌ Need to download additional data

---

##### **2B: Synthetic Data Generation (Good - More Data)**
```yaml
Goal: Generate synthetic detection dataset from GTSRB
Method: Paste traffic signs into KITTI images
Size: Generate 5,000-10,000 synthetic images
```

**Steps:**
1. **Create Synthetic Dataset**
   ```python
   # Pseudo-code for synthetic data generation
   for kitti_image in kitti_images:
       # Load KITTI image
       img = load_image(kitti_image)
       
       # Randomly select 1-3 traffic signs from GTSRB
       signs = random.sample(gtsrb_train, k=random.randint(1, 3))
       
       for sign in signs:
           # Resize sign to realistic size (50-150 pixels)
           sign_resized = resize(sign, size=random.randint(50, 150))
           
           # Find valid placement (not overlapping with vehicles)
           position = find_valid_position(img, existing_bboxes)
           
           # Paste sign with random transformations
           img = paste_with_augmentation(img, sign_resized, position)
           
           # Add bounding box annotation
           bboxes.append([position, sign_class])
       
       # Save synthetic image and labels
       save_yolo_format(img, bboxes)
   ```

2. **Combine with KITTI**
   ```
   data/processed/combined/
   ├── images/
   │   ├── train/
   │   │   ├── kitti_000000.jpg     # Original KITTI
   │   │   ├── kitti_000001.jpg
   │   │   ├── synthetic_000000.jpg  # Synthetic with signs
   │   │   └── synthetic_000001.jpg
   ```

3. **Train on Combined Dataset**
   ```python
   model = YOLO('runs/detect/kitti_vehicles_pedestrians/weights/best.pt')
   
   results = model.train(
       data='data/processed/combined/data.yaml',
       epochs=50,
       imgsz=640,
       batch=16,
       name='combined_synthetic',
       device=0
   )
   ```

**Advantages:**
- ✅ Large dataset (controllable size)
- ✅ Uses existing GTSRB data
- ✅ No additional downloads

**Disadvantages:**
- ❌ Synthetic artifacts
- ❌ May not generalize well to real signs in scenes
- ❌ Requires implementation of data generation pipeline

---

##### **2C: Multi-Task Learning (Advanced)**
```yaml
Goal: Separate detection and classification branches
Architecture: YOLO backbone + 2 heads
- Head 1: Detect vehicles, pedestrians, cyclists, sign_region
- Head 2: Classify sign_region into 43 classes
```

**Not Recommended for this project** due to complexity and time constraints.

---

### **OPTION 2: Joint Training (More Complex)**

Train on both datasets simultaneously from the start.

**Approach:**
1. Preprocess both KITTI and GTSDB
2. Merge datasets into single training set
3. Train YOLO from COCO weights on combined dataset

**Advantages:**
- ✅ Single training phase
- ✅ Joint optimization

**Disadvantages:**
- ❌ More complex data preprocessing
- ❌ Harder to debug issues
- ❌ May require careful data balancing (KITTI has more images)

---

## 📋 Step-by-Step Action Plan

### **Week 2-3: KITTI Training**

1. ✅ **Update Current Config** (preprocessing.yaml)
   - Include Van, Truck in Vehicle class
   - Merge Person_sitting with Pedestrian
   - Set target_size to [640, 640]

2. ✅ **Run Preprocessing**
   ```bash
   python scripts/preprocess_data.py
   # Output: data/processed/kitti/
   #   ├── images/train, val, test
   #   ├── labels/train, val, test
   #   └── data.yaml
   ```

3. ✅ **Train YOLO on KITTI**
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolov8n.pt')  # or yolov8s, yolov8m for better accuracy
   
   results = model.train(
       data='data/processed/kitti/data.yaml',
       epochs=100,
       imgsz=640,
       batch=16,
       name='kitti_baseline',
       device=0
   )
   ```

4. ✅ **Evaluate Performance**
   ```python
   metrics = model.val()
   print(f"mAP@0.5: {metrics.box.map50}")
   print(f"mAP@0.5:0.95: {metrics.box.map}")
   ```

5. ✅ **Test Real-Time Inference**
   ```python
   results = model.predict(
       source='path/to/test/video.mp4',
       save=True,
       stream=True
   )
   
   # Measure FPS
   for r in results:
       fps = 1.0 / r.speed['inference']
       print(f"FPS: {fps:.1f}")
   ```

**Deliverable for Week 2-3:**
- Trained YOLO model detecting vehicles, pedestrians, cyclists
- Evaluation report with mAP, IoU, FPS metrics
- Visualizations of predictions on test set

---

### **Week 4: Traffic Sign Integration**

**Decision Point:** Choose between 2A (GTSDB) or 2B (Synthetic)

#### **If choosing GTSDB (2A):**

1. ✅ **Download GTSDB**
   - Website: https://benchmark.ini.rub.de/
   - Download: Full dataset or FullIJCNN2013 (900 images)

2. ✅ **Create GTSDB Preprocessing Script**
   ```python
   # scripts/preprocess_gtsdb.py
   # Convert GTSDB format to YOLO format
   # Group 43 classes into 5 categories
   ```

3. ✅ **Combine KITTI + GTSDB**
   ```python
   # Merge both datasets
   # Update data.yaml with all classes
   ```

4. ✅ **Continue Training**
   ```python
   # Load Stage 1 model
   model = YOLO('runs/detect/kitti_baseline/weights/best.pt')
   
   # Fine-tune on combined dataset
   results = model.train(
       data='data/processed/combined/data.yaml',
       epochs=50,
       imgsz=640,
       batch=16,
       name='final_combined',
       freeze=10,  # Preserve vehicle detection
       device=0
   )
   ```

#### **If choosing Synthetic (2B):**

1. ✅ **Implement Synthetic Data Generator**
   ```python
   # scripts/generate_synthetic_signs.py
   # Paste GTSRB signs into KITTI images
   ```

2. ✅ **Generate Dataset**
   ```bash
   python scripts/generate_synthetic_signs.py \
       --kitti-dir data/processed/kitti/images/train \
       --gtsrb-dir data/raw/German-TS/train.p \
       --output-dir data/processed/synthetic \
       --num-images 5000
   ```

3. ✅ **Train on Combined Dataset**
   - Same as GTSDB approach above

**Deliverable for Week 4:**
- Final YOLO model detecting all object classes
- Comprehensive evaluation report
- Real-time testing on video

---

## 🎓 Technical Decisions

### **Class Grouping Strategy**

**KITTI Classes:**
```yaml
Vehicle: [Car, Van, Truck]           # Class 0
Pedestrian: [Pedestrian, Person_sitting]  # Class 1
Cyclist: [Cyclist]                   # Class 2
```

**Traffic Sign Classes (Grouped):**
```yaml
Speed_Limit: [All speed limit signs]     # Class 3
Warning_Sign: [Caution, curves, animals] # Class 4
Mandatory_Sign: [Turn directions]        # Class 5
Priority_Sign: [Stop, Yield, Priority]   # Class 6
Prohibition_Sign: [No entry, No passing] # Class 7
```

**Total: 8 classes**

**Alternative (More Granular):**
- Keep 43 individual traffic sign classes
- Total: 46 classes (3 KITTI + 43 signs)
- ⚠️ More complex, harder to train, slower inference

**Recommendation:** Use grouped approach (8 classes total)

---

### **Data Balance Strategy**

**Problem:**
- KITTI: 7,481 images
- GTSDB: ~900 images
- Imbalance: 8:1 ratio

**Solutions:**
1. **Oversample GTSDB** (repeat images with augmentation)
2. **Undersample KITTI** (use subset of KITTI)
3. **Weighted Loss** (give more weight to traffic sign errors)
4. **Synthetic Data** (generate more sign-containing images)

**Recommendation:** Option 1 (Oversample) + Option 3 (Weighted Loss)

```python
# During training
results = model.train(
    data='data.yaml',
    epochs=50,
    batch=16,
    cls=0.5,  # Classification loss weight
    box=7.5,  # Box loss weight  
    obj=1.0,  # Objectness loss weight
)
```

---

### **Transfer Learning Strategy**

**Layer Freezing:**
```python
# Option 1: Freeze backbone (faster, preserves COCO features)
model.train(..., freeze=10)  # Freeze first 10 layers

# Option 2: Freeze nothing (slower, full fine-tuning)
model.train(..., freeze=0)

# Option 3: Progressive unfreezing
# Epoch 0-20: freeze=15
# Epoch 21-40: freeze=10
# Epoch 41-50: freeze=0
```

**Recommendation:** Start with freeze=10, then unfreeze if stuck

---

## 📝 Updated Config File

I've already created two config files for you:

1. **`configs/preprocessing.yaml`** (Updated)
   - For KITTI preprocessing only
   - Run this first to complete Week 2-3

2. **`configs/multi_dataset_preprocessing.yaml`** (New)
   - For combined KITTI + Traffic Signs preprocessing
   - Use this for Week 4

---

## 🎯 Final Recommendations

### **For Your Project (Best Balance of Quality & Feasibility):**

1. **Week 2-3:** Fine-tune YOLO on KITTI
   - Use updated `preprocessing.yaml`
   - Train on vehicles, pedestrians, cyclists
   - Achieve mAP > 0.70, FPS > 30

2. **Week 4:** Download GTSDB (Option 2A)
   - ~900 images with real bounding boxes
   - Combine with KITTI
   - Fine-tune for traffic signs
   - Achieve mAP > 0.65 on combined dataset

3. **Fallback (if GTSDB unavailable):** Synthetic data (Option 2B)
   - Generate 3,000-5,000 synthetic images
   - Document as "Synthetic Data Augmentation"
   - Acknowledge limitations in final report

### **Why This Approach?**

✅ **Feasible:** Can complete in 4 weeks (2 weeks per stage)
✅ **Professional:** Uses real detection datasets (KITTI + GTSDB)
✅ **Defensible:** Clear methodology, meets project requirements
✅ **Extensible:** Can add more datasets later

---

## 📚 Additional Resources

- **GTSDB Dataset:** https://benchmark.ini.rub.de/gtsdb_dataset.html
- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **KITTI Dataset:** http://www.cvlibs.net/datasets/kitti/
- **Ultralytics YOLO:** https://github.com/ultralytics/ultralytics

---

## ✅ Next Steps

1. ✅ Run the German Traffic Signs exploration notebook
2. ✅ Review this training strategy
3. ✅ Decide: GTSDB (real) or Synthetic (generated)?
4. ✅ Update preprocessing.yaml with my recommendations
5. ⏳ Run preprocessing: `python scripts/preprocess_data.py`
6. ⏳ Start YOLO training on KITTI
7. ⏳ Move to traffic sign integration (Week 4)

---

**Good luck with your project! 🚗🚦**
