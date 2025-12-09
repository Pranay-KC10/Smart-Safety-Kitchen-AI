# Kitchen Safety AI - Setup Guide

Quick setup guide to run the Kitchen Safety AI on your computer.

## What it detects
- **Flame** - Fire/flames on stove
- **Pan** - Cooking pans
- **Person** - People in kitchen
- **Stove** - Stove/cooking surface
- **knife** - Kitchen knives

## Quick Start (3 steps)

### Step 1: Install Python
Make sure you have Python 3.9+ installed:
```bash
python --version  # Should show 3.9 or higher
```

Download from https://www.python.org/downloads/ if needed.

### Step 2: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required packages
pip install ultralytics opencv-python
```

### Step 3: Run the Webcam Detection
```bash
python run_webcam.py
```

## Controls
- **q** - Quit
- **s** - Save screenshot
- **p** - Pause/Resume

## Options
```bash
# Use a different camera (if you have multiple)
python run_webcam.py --camera 1

# Lower confidence threshold (detect more, but may have false positives)
python run_webcam.py --conf 0.3

# Higher confidence threshold (detect less, but more accurate)
python run_webcam.py --conf 0.7

# Specify custom model path
python run_webcam.py --model path/to/model.pt
```

## Troubleshooting

### "Model not found"
Make sure the model file exists at `runs/detect/train4/weights/best.pt`

### "Could not open camera"
- Try a different camera ID: `python run_webcam.py --camera 1`
- Check if another app is using the camera
- On Mac, grant camera permission in System Preferences

### "ModuleNotFoundError"
Run: `pip install ultralytics opencv-python`

### Slow detection
The model runs on CPU by default. For faster detection:
- Use a computer with NVIDIA GPU
- Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/

## Files to Share
To share with a friend, give them:
1. `run_webcam.py` - The main script
2. `runs/detect/train4/weights/best.pt` - The trained model (6MB)
3. `SETUP.md` - This setup guide

Or share the entire project folder.
