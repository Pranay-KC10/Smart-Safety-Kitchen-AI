# Data Collection Guide

## Overview

We need to collect images for two models:
1. **YOLO Detection** - Find objects in images
2. **Stove Classifier** - Determine if stove is ON or OFF

---

## Part 1: YOLO Object Detection Data

### Classes to Detect
| Class ID | Object | Min Images Needed | Notes |
|----------|--------|-------------------|-------|
| 0 | person | 50+ | Can supplement with COCO |
| 1 | knife | 50+ | Various angles, sizes |
| 2 | scissors | 50+ | Open and closed |
| 3 | pan | 50+ | On stove, on counter, in hand |
| 4 | stove | 50+ | Different stove types |
| 5 | flame | 30+ | Gas stove flames, candles |

### Photo Guidelines

**General Tips:**
- Take photos from different angles (top, side, 45°)
- Vary the lighting (bright, dim, natural, artificial)
- Include cluttered backgrounds (realistic kitchen)
- Mix close-up and wide shots

**Scenarios to Capture:**

```
SAFE scenarios (person present):
├── Person holding knife, cutting vegetables
├── Person stirring pan on stove
├── Person standing near stove with flame on
└── Person using scissors

DANGEROUS scenarios (unattended):
├── Knife on counter, no person in frame
├── Scissors on table, no person nearby
├── Pan on stove with flame, person far away or absent
├── Flame visible on stove, no person
└── Multiple hazards in one frame
```

### Folder Structure
```
data/yolo_detection/
├── train/
│   ├── images/
│   │   ├── kitchen_001.jpg
│   │   ├── kitchen_002.jpg
│   │   └── ...
│   └── labels/
│       ├── kitchen_001.txt    # Same name as image!
│       ├── kitchen_002.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

### Label Format (YOLO format)
Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to 0-1 (divide by image width/height)

**Example:** `kitchen_001.txt`
```
0 0.45 0.60 0.30 0.70    # person at center-left
1 0.80 0.40 0.05 0.10    # knife on right side
4 0.20 0.30 0.25 0.20    # stove on left
```

### Labeling Tools (Pick one)
- **LabelImg** - Simple, offline: `pip install labelImg`
- **Roboflow** - Online, free tier: https://roboflow.com
- **CVAT** - Advanced, free: https://cvat.ai

---

## Part 2: Stove Classifier Data

### Classes
| Folder | Description | Min Images |
|--------|-------------|------------|
| stove_on | Stove with visible flame/heat | 50+ |
| stove_off | Stove turned off | 50+ |

### Photo Guidelines
- Crop to just the stove/burner area
- Include different stove types (gas, electric, induction)
- Gas stove ON = visible blue flame
- Electric ON = red/glowing coils
- Various lighting conditions

### Folder Structure
```
data/classifier/
├── stove_on/
│   ├── stove_on_001.jpg
│   ├── stove_on_002.jpg
│   └── ...
└── stove_off/
    ├── stove_off_001.jpg
    ├── stove_off_002.jpg
    └── ...
```

---

## Existing Datasets to Supplement

### For YOLO (COCO classes)
- Person, knife, scissors already in COCO
- Download subset: https://cocodataset.org

### For Kitchen Objects
- **Open Images Dataset** - Has kitchen items
  https://storage.googleapis.com/openimages/web/index.html
- **Kitchen Dataset on Roboflow**
  https://universe.roboflow.com (search "kitchen")

### For Fire/Flame
- **Fire Detection Dataset**
  https://www.kaggle.com/datasets/phylake1337/fire-dataset

---

## Quick Start Checklist

### Minimum Viable Dataset
- [ ] 30 images with knife (attended + unattended)
- [ ] 30 images with scissors (attended + unattended)
- [ ] 30 images with pan on stove
- [ ] 30 images with stove/flame visible
- [ ] 30 images with person in kitchen
- [ ] 50 stove ON images (cropped)
- [ ] 50 stove OFF images (cropped)

### Split Ratio
- **Train:** 80% of images
- **Val:** 20% of images

---

## Training Commands

### After collecting data:

```bash
# Train YOLO Detection (from project root)
cd data/yolo_detection
yolo detect train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640

# Train Stove Classifier
yolo classify train model=yolov8n-cls.pt data=../classifier epochs=50 imgsz=224
```

### Output locations:
- YOLO: `runs/detect/train/weights/best.pt`
- Classifier: `runs/classify/train/weights/best.pt`
