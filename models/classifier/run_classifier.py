"""
Kitchen Object Classifier for Kitchen Safety AI (Stage 2)

Classifies cropped images from YOLO detector:
- Stove: ON (flames visible, red coils) vs OFF
- Knife: in-use (being held) vs unattended (left on counter)
- Pan: in-use vs empty/overheating
"""

from ultralytics import YOLO
import json
import os
from datetime import datetime


class Classifier:
    # Status mappings for different object types
    # These map model class indices to meaningful status strings
    STOVE_STATUS = {
        0: "OFF",
        1: "ON",
    }

    KNIFE_STATUS = {
        0: "in-use",      # Knife being held/used
        1: "unattended",  # Knife left on counter
    }

    PAN_STATUS = {
        0: "in-use",      # Pan with food/being used
        1: "empty",       # Empty pan (potential overheating)
    }

    def __init__(self, stove_model_path="stove_classifier.pt",
                 knife_model_path="knife_classifier.pt",
                 pan_model_path=None):
        """
        Initialize classifiers for different object types

        For now, uses a single general classifier. In production,
        you'd train separate models for each object type.
        """
        self.stove_model = None
        self.knife_model = None
        self.pan_model = None

        # Try to load models if they exist
        if stove_model_path and os.path.exists(stove_model_path):
            self.stove_model = YOLO(stove_model_path)

        if knife_model_path and os.path.exists(knife_model_path):
            self.knife_model = YOLO(knife_model_path)

        if pan_model_path and os.path.exists(pan_model_path):
            self.pan_model = YOLO(pan_model_path)

    def classify_stove(self, crop_path):
        """Classify stove as ON or OFF"""
        if crop_path is None or not os.path.exists(crop_path):
            return None

        if self.stove_model:
            result = self.stove_model(crop_path, verbose=False)[0]
            pred_class = result.probs.top1
            confidence = float(result.probs.top1conf)

            return {
                "status": self.STOVE_STATUS.get(pred_class, "unknown"),
                "confidence": round(confidence, 3),
                "features": {
                    "has_flame": pred_class == 1,
                    "temperature_indicator": "red" if pred_class == 1 else "none"
                }
            }
        else:
            # Placeholder: Return simulated result for testing
            return {
                "status": "ON",  # Default to ON for safety (assume worst case)
                "confidence": 0.85,
                "features": {
                    "has_flame": True,
                    "temperature_indicator": "red"
                }
            }

    def classify_knife(self, crop_path):
        """Classify knife as in-use or unattended"""
        if crop_path is None or not os.path.exists(crop_path):
            return None

        if self.knife_model:
            result = self.knife_model(crop_path, verbose=False)[0]
            pred_class = result.probs.top1
            confidence = float(result.probs.top1conf)

            return {
                "status": self.KNIFE_STATUS.get(pred_class, "unknown"),
                "confidence": round(confidence, 3),
                "features": {
                    "near_person": pred_class == 0,
                    "on_cutting_board": False  # Would need additional detection
                }
            }
        else:
            # Placeholder: Return simulated result for testing
            return {
                "status": "unattended",  # Default for safety
                "confidence": 0.87,
                "features": {
                    "near_person": False,
                    "on_cutting_board": False
                }
            }

    def classify_pan(self, crop_path):
        """Classify pan as in-use or empty"""
        if crop_path is None or not os.path.exists(crop_path):
            return None

        if self.pan_model:
            result = self.pan_model(crop_path, verbose=False)[0]
            pred_class = result.probs.top1
            confidence = float(result.probs.top1conf)

            return {
                "status": self.PAN_STATUS.get(pred_class, "unknown"),
                "confidence": round(confidence, 3),
                "features": {
                    "has_contents": pred_class == 0,
                    "on_heat": True  # Would need stove context
                }
            }
        else:
            return {
                "status": "in-use",
                "confidence": 0.80,
                "features": {
                    "has_contents": True,
                    "on_heat": True
                }
            }

    def classify(self, crop_path, object_type="unknown"):
        """
        General classification method that routes to specific classifiers

        Args:
            crop_path: Path to cropped image
            object_type: Type of object ("stove", "knife", "pan")

        Returns:
            dict: Classification result with status and confidence
        """
        if crop_path is None:
            return None

        object_type = object_type.lower()

        if object_type == "stove" or object_type == "oven":
            return self.classify_stove(crop_path)
        elif object_type == "knife":
            return self.classify_knife(crop_path)
        elif object_type == "pan" or object_type == "pot":
            return self.classify_pan(crop_path)
        else:
            # Unknown object type, return generic result
            return {
                "status": "detected",
                "confidence": 0.5,
                "features": {}
            }

    def classify_all_detections(self, yolo_output):
        """
        Classify all relevant detections from YOLO output

        Args:
            yolo_output: Dictionary with 'detections' list from YOLO

        Returns:
            dict: Classifications keyed by cropped image filename
        """
        classifications = {}

        for detection in yolo_output.get('detections', []):
            crop_path = detection.get('cropped_image_path')
            object_class = detection.get('class', '')

            if crop_path and object_class in ['stove', 'knife', 'pan', 'pot', 'oven']:
                result = self.classify(crop_path, object_class)
                if result:
                    # Key by filename for lookup in safety checker
                    filename = os.path.basename(crop_path)
                    classifications[filename] = result

        return {
            "timestamp": datetime.now().isoformat(),
            "classifications": classifications
        }


def main():
    """Test the classifier"""
    print("=" * 60)
    print("CLASSIFIER TEST - Stage 2")
    print("=" * 60)

    classifier = Classifier()

    print("\nClassifier initialized successfully!")
    print("\nUsage:")
    print("  classifier = Classifier()")
    print("  result = classifier.classify(crop_path, 'stove')")
    print("  result = classifier.classify_stove(crop_path)")
    print("  results = classifier.classify_all_detections(yolo_output)")


if __name__ == "__main__":
    main()
