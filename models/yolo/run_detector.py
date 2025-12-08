"""
YOLO Object Detector for Kitchen Safety AI (Stage 1)

Detects kitchen objects: person, stove, knife, pan, fire, smoke
Outputs bounding boxes, class names, centers, and cropped images for classification
"""

from ultralytics import YOLO
import json
import cv2
import os
from datetime import datetime


class Detector:
    # Kitchen-relevant classes from COCO dataset + custom classes
    # COCO class IDs: person=0, oven=69, knife=43, etc.
    KITCHEN_CLASSES = {
        0: "person",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        49: "orange",
        50: "broccoli",
        69: "oven",        # We'll treat oven as stove
        71: "sink",
        72: "refrigerator",
    }

    # Custom class mapping for kitchen safety (when using custom trained model)
    CUSTOM_CLASSES = {
        0: "person",
        1: "stove",
        2: "knife",
        3: "pan",
        4: "pot",
        5: "fire",
        6: "smoke",
    }

    def __init__(self, model_path="yolo11n.pt", use_custom_classes=False, crop_dir="outputs/crops"):
        self.model = YOLO(model_path)
        self.use_custom_classes = use_custom_classes
        self.crop_dir = crop_dir
        os.makedirs(crop_dir, exist_ok=True)

        # Select class mapping based on model type
        self.class_map = self.CUSTOM_CLASSES if use_custom_classes else self.KITCHEN_CLASSES

    def get_class_name(self, class_id):
        """Convert numeric class ID to string name"""
        if self.use_custom_classes:
            return self.class_map.get(class_id, f"unknown_{class_id}")
        else:
            # For COCO model, map oven -> stove for kitchen context
            if class_id == 69:
                return "stove"
            return self.class_map.get(class_id, f"class_{class_id}")

    def calculate_center(self, bbox):
        """Calculate center point from bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        return [int((x1 + x2) / 2), int((y1 + y2) / 2)]

    def crop_and_save(self, frame, bbox, class_name, detection_id):
        """Crop detected object and save to file"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cropped = frame[y1:y2, x1:x2]

        filename = f"{class_name}_{detection_id:03d}.jpg"
        filepath = os.path.join(self.crop_dir, filename)

        if cropped.size > 0:
            cv2.imwrite(filepath, cropped)
            return filepath
        return None

    def detect(self, frame, confidence_threshold=0.5):
        """
        Run YOLO detection on a frame

        Args:
            frame: Image frame (numpy array or path to image)
            confidence_threshold: Minimum confidence for detections

        Returns:
            dict: Detection results with timestamp, frame info, and detections list
        """
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        detections = []
        detection_count = {}

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = self.get_class_name(class_id)
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()
                center = self.calculate_center(bbox)

                # Track detection count per class for unique filenames
                detection_count[class_name] = detection_count.get(class_name, 0) + 1

                # Crop and save image for classifier
                crop_path = None
                if isinstance(frame, str):
                    img = cv2.imread(frame)
                else:
                    img = frame

                if img is not None:
                    crop_path = self.crop_and_save(
                        img, bbox, class_name, detection_count[class_name]
                    )

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [int(x) for x in bbox],
                    "center": center,
                    "cropped_image_path": crop_path
                })

        return {
            "timestamp": datetime.now().isoformat(),
            "frame_number": 0,  # Will be set by caller for video processing
            "detections": detections
        }

    def detect_from_camera(self, camera_id=0, output_path="outputs/yolo_output.json"):
        """Capture single frame from camera and run detection"""
        cap = cv2.VideoCapture(camera_id)
        ret, frame = cap.read()
        cap.release()

        if ret:
            result = self.detect(frame)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            return result
        return None


def main():
    """Test the detector with a sample image or camera"""
    print("=" * 60)
    print("YOLO DETECTOR TEST - Stage 1")
    print("=" * 60)

    detector = Detector()

    # Test with camera if available, otherwise print usage
    print("\nDetector initialized successfully!")
    print("\nUsage:")
    print("  detector = Detector()")
    print("  results = detector.detect(frame)")
    print("  results = detector.detect_from_camera()")


if __name__ == "__main__":
    main()
