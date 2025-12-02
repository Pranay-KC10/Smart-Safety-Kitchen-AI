""" Smart Kitchen Safety AI
    Goal: To alert the user(s) of this AI tool of any hazards in the kitchen area, or any utensils
    left unattended.
    E.g. Knife unattended and not used: Alert user that knife is not being used and should be stored
    away.
    E.g. Stove is on while user is not nearby: Alert user that stove is on and to turn off when
    not nearby.
    Etc.

    Will utilize:
        1.) ViT to detect and label objects and hazards. (Knife, stovetop on, stovetop off, pan, etc).
        2.)
    To-do List:
        1.)
"""

# ==================== PROJECT ARCHITECTURE ====================
# 3-Stage ML Pipeline (Jetson Nano + Docker):
#
# STAGE 1: OBJECT DETECTION (YOLO - Docker Container 1)
#   - Input: Camera feed (USB/CSI)
#   - Detects: Person, stove, knife, pan, pot, fire, smoke
#   - Output: Bounding boxes + class labels + coordinates (JSON)
#   - Crops detected objects for next stages
#
# STAGE 2A: CLASSIFICATION (Custom CNN/ResNet - Docker Container 2)
#   - Input: Cropped images from YOLO
#   - Classifies stove status: ON (flames/red coils) vs OFF
#   - Classifies knife status: In-use vs Unattended
#   - Classifies pan/pot: In-use vs Empty
#   - Output: State labels (JSON)
#
# STAGE 2B: REGRESSION/TRACKING (Position Tracker - Docker Container 3)
#   - Input: Person bounding box from YOLO
#   - Tracks person position (x, y coordinates)
#   - Calculates distance between person and stove
#   - Tracks movement over time
#   - Output: Coordinates + distance metrics (JSON)
#
# STAGE 3: SAFETY LOGIC & ALERT SYSTEM (This main.py orchestrator)
#   - Combines outputs from all 3 Docker containers
#   - Applies hazard detection rules:
#       Rule 1: Stove ON + No person nearby → Alert "Stove unattended!"
#       Rule 2: Knife detected + Unattended → Alert "Store knife safely!"
#       Rule 3: Fire/Smoke detected → Alert "FIRE HAZARD!"
#       Rule 4: Stove ON + Empty pan → Alert "Pan overheating!"
#   - Generates alerts: Visual (annotated video) + Audio (buzzer) + Text (console/web)
#   - Logs incidents with timestamps
#
# ==================== DATA FLOW ====================
# Camera → YOLO Detection → JSON Output
#                             │
#                             ├─→ Cropped Image (Stove) → Classifier → "ON/OFF"
#                             ├─→ Cropped Image (Knife) → Classifier → "InUse/Unattended"
#                             └─→ Person BBox → Regression → (x,y) + distance from stove
#                                                               │
#                                                               ▼
#                                      Combine in safety_checker.py → Apply rules → Alerts
#
# ==================== PROJECT STRUCTURE ====================
# Kitchen_AI/
#   ├── docker-compose.yml          # Orchestrates 3 Docker containers
#   ├── models/
#   │   ├── yolo/                   # Docker 1: Object detection
#   │   ├── classifier/             # Docker 2: Stove/knife classification
#   │   └── regression/             # Docker 3: Person tracking
#   ├── src/
#   │   ├── main.py                 # This file - main orchestrator
#   │   ├── camera.py               # Camera input handler
#   │   ├── safety_checker.py       # Hazard detection rules
#   │   └── alert_system.py         # Alert generation
#   ├── data/training/              # Training images for classifier
#   └── outputs/logs/               # Incident logs
#
# ==================== MODEL COMPARISON (For Report) ====================
# Experiments & Results section will compare:
#   - YOLOv5 vs YOLOv8 (detection accuracy + speed on Jetson Nano)
#   - ResNet-18 vs Custom CNN (stove classification accuracy)
# Metrics: Precision, Recall, F1-Score, Inference Speed (FPS), mAP
# ============================================================== 