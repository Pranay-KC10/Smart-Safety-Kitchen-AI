#!/usr/bin/env python3
"""
Kitchen Safety AI - Webcam Detection
=====================================
Real-time object detection for kitchen safety monitoring.

Detects: Flame, Pan, Person, Stove, knife

Usage:
    python run_webcam.py                    # Run with default settings
    python run_webcam.py --camera 1         # Use different camera
    python run_webcam.py --conf 0.3         # Lower confidence threshold

Controls:
    'q' - Quit
    's' - Save screenshot
    'p' - Pause/Resume
"""

import cv2
import argparse
import os
from datetime import datetime
from ultralytics import YOLO

# Class names for our trained model
CLASS_NAMES = {
    0: "Flame",
    1: "Pan",
    2: "Person",
    3: "Stove",
    4: "knife"
}

# Colors for each class (BGR format)
CLASS_COLORS = {
    "Flame": (0, 0, 255),      # Red - danger
    "Pan": (255, 165, 0),      # Orange
    "Person": (0, 255, 0),     # Green
    "Stove": (255, 0, 255),    # Magenta
    "knife": (0, 255, 255)     # Yellow - caution
}


def get_model_path():
    """Find the best model path"""
    # Check common locations
    possible_paths = [
        "runs/detect/train4/weights/best.pt",
        "models/kitchen_safety.pt",
        "best.pt"
    ]

    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for path in possible_paths:
        full_path = os.path.join(script_dir, path)
        if os.path.exists(full_path):
            return full_path

    # If no model found, return default and let YOLO handle error
    return os.path.join(script_dir, "runs/detect/train4/weights/best.pt")


def draw_detections(frame, results, conf_threshold=0.5):
    """Draw bounding boxes and labels on frame"""
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf)
            if conf < conf_threshold:
                continue

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            color = CLASS_COLORS.get(cls_name, (128, 128, 128))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{cls_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return frame, detections


def check_safety_alerts(detections):
    """Check for potential safety hazards"""
    alerts = []

    has_flame = any(d["class"] == "Flame" for d in detections)
    has_person = any(d["class"] == "Person" for d in detections)
    has_stove = any(d["class"] == "Stove" for d in detections)
    has_knife = any(d["class"] == "knife" for d in detections)

    # Alert: Flame detected without person
    if has_flame and not has_person:
        alerts.append("WARNING: Unattended flame detected!")

    # Alert: Stove on without person (if flame visible on stove)
    if has_flame and has_stove and not has_person:
        alerts.append("DANGER: Stove left unattended!")

    return alerts


def main():
    parser = argparse.ArgumentParser(description="Kitchen Safety AI - Webcam Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--save-dir", type=str, default="outputs/screenshots", help="Directory to save screenshots")
    args = parser.parse_args()

    # Load model
    model_path = args.model if args.model else get_model_path()
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease ensure you have trained the model or provide a valid model path.")
        print("Usage: python run_webcam.py --model /path/to/your/model.pt")
        return

    model = YOLO(model_path)
    print("Model loaded successfully!")
    print(f"\nDetecting classes: {', '.join(CLASS_NAMES.values())}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Open webcam
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        print("Try a different camera ID with: python run_webcam.py --camera 1")
        return

    print("\n" + "="*50)
    print("Kitchen Safety AI - Running")
    print("="*50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'p' - Pause/Resume")
    print("="*50 + "\n")

    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1

            # Run detection
            results = model(frame, conf=args.conf, verbose=False)

            # Draw results
            frame, detections = draw_detections(frame, results, args.conf)

            # Check safety alerts
            alerts = check_safety_alerts(detections)

            # Draw alerts on frame
            y_offset = 30
            for alert in alerts:
                cv2.putText(frame, alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30

            # Draw info bar
            info_text = f"Frame: {frame_count} | Objects: {len(detections)} | Press 'q' to quit"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame
        cv2.imshow("Kitchen Safety AI", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(args.save_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
