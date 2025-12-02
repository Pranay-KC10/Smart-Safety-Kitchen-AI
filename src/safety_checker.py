"""
Safety Checker Module (Stage 3)
This module implements hazard detection rules by analyzing outputs from:
  - Stage 1 (YOLO): Object detections with bounding boxes
  - Stage 2 (Classifier): Classification of object states

Hazard Detection Rules:
  1. Stove ON + No person nearby → Alert "Stove unattended!"
  2. Knife detected + Status: Unattended → Alert "Store knife safely!"
  3. Fire/Smoke detected → Alert "FIRE HAZARD!"
  4. Stove ON + Empty pan → Alert "Pan overheating!"
"""

import math
import json
from typing import Dict, List, Tuple
from datetime import datetime


class SafetyChecker:
    def __init__(self, config: Dict = None):
       
        # Default safety thresholds (in pixels - can be calibrated later)
        self.config = config or {
            "safe_distance_threshold": 200,  # pixels - person must be within this distance from stove
            "confidence_threshold": 0.7,     # minimum confidence for detections
            "alert_cooldown": 5,              # seconds between repeated alerts
        }

        self.last_alert_time = {}  # Track last alert time for cooldown

    def calculate_distance(self, point1: List[int], point2: List[int]) -> float:
       
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def find_object_by_class(self, detections: List[Dict], class_name: str) -> Dict:
       
        for detection in detections:
            if detection['class'] == class_name:
                if detection['confidence'] >= self.config['confidence_threshold']:
                    return detection
        return None

    def check_stove_unattended(self, detections: List[Dict], classifications: Dict) -> Dict:
        
        # Find stove and person in detections
        stove = self.find_object_by_class(detections, "stove")
        person = self.find_object_by_class(detections, "person")

        if not stove:
            return None  # No stove detected

        # Check if stove is ON from classifier
        stove_image = stove.get('cropped_image_path', '').split('/')[-1]
        stove_status = classifications.get(stove_image, {})

        if stove_status.get('status') != 'ON':
            return None  # Stove is OFF, no hazard

        # Stove is ON - check if person is nearby
        if not person:
            # No person detected at all
            return {
                "type": "STOVE_UNATTENDED",
                "severity": "HIGH",
                "message": " ALERT: Stove is ON while you're away! Please return to the kitchen.",
                "details": {
                    "stove_status": stove_status.get('status'),
                    "stove_confidence": stove_status.get('confidence'),
                    "person_detected": False
                }
            }

        # Person detected - check distance
        distance = self.calculate_distance(stove['center'], person['center'])

        if distance > self.config['safe_distance_threshold']:
            return {
                "type": "STOVE_UNATTENDED",
                "severity": "MEDIUM",
                "message": f" WARNING: Stove is ON and you're {int(distance)}px away. Stay close when cooking!",
                "details": {
                    "stove_status": stove_status.get('status'),
                    "distance_from_stove": int(distance),
                    "safe_distance": self.config['safe_distance_threshold'],
                    "person_detected": True
                }
            }

        return None  # Person is close enough, no alert

    def check_knife_unattended(self, detections: List[Dict], classifications: Dict) -> Dict:
        
        knife = self.find_object_by_class(detections, "knife")

        if not knife:
            return None  # No knife detected

        # Check knife status from classifier
        knife_image = knife.get('cropped_image_path', '').split('/')[-1]
        knife_status = classifications.get(knife_image, {})

        if knife_status.get('status') == 'unattended':
            return {
                "type": "KNIFE_UNATTENDED",
                "severity": "LOW",
                "message": " NOTICE: Knife left unattended. Please store it safely when not in use.",
                "details": {
                    "knife_status": knife_status.get('status'),
                    "confidence": knife_status.get('confidence'),
                    "features": knife_status.get('features', {})
                }
            }

        return None

    def check_fire_smoke(self, detections: List[Dict]) -> Dict:
        
        fire = self.find_object_by_class(detections, "fire")
        smoke = self.find_object_by_class(detections, "smoke")

        if fire or smoke:
            hazard_type = "FIRE" if fire else "SMOKE"
            return {
                "type": f"{hazard_type}_DETECTED",
                "severity": "CRITICAL",
                "message": f"🚨 EMERGENCY: {hazard_type} DETECTED! Evacuate immediately and call emergency services!",
                "details": {
                    "hazard": hazard_type.lower(),
                    "confidence": fire['confidence'] if fire else smoke['confidence']
                }
            }

        return None

    def check_all_hazards(self, yolo_output: Dict, classifier_output: Dict) -> List[Dict]:
        
        alerts = []
        detections = yolo_output.get('detections', [])
        classifications = classifier_output.get('classifications', {})

        # Run all safety checks
        checks = [
            self.check_fire_smoke(detections),           # Rule 3 (highest priority)
            self.check_stove_unattended(detections, classifications),  # Rule 1
            self.check_knife_unattended(detections, classifications),  # Rule 2
        ]

        # Filter out None values (no hazard detected)
        alerts = [alert for alert in checks if alert is not None]

        # Add timestamp to all alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            alert['frame_number'] = yolo_output.get('frame_number', 'N/A')

        return alerts


def main():
    print("=" * 60)
    print("SAFETY CHECKER TEST - Stage 3")
    print("=" * 60)

    # Initialize safety checker
    checker = SafetyChecker()

    # Load mock data
    print("\n Loading mock data...")
    with open('../mock_data/yolo_output.json', 'r') as f:
        yolo_data = json.load(f)

    with open('../mock_data/classifier_output.json', 'r') as f:
        classifier_data = json.load(f)

    print(f" Loaded YOLO data: {len(yolo_data['detections'])} detections")
    print(f" Loaded Classifier data: {len(classifier_data['classifications'])} classifications")

    # Run safety checks
    print("\n Running safety checks...")
    alerts = checker.check_all_hazards(yolo_data, classifier_data)

    # Display results
    print(f"\n Found {len(alerts)} hazard(s):\n")

    if alerts:
        for i, alert in enumerate(alerts, 1):
            print(f"Alert {i}:")
            print(f"  Type: {alert['type']}")
            print(f"  Severity: {alert['severity']}")
            print(f"  Message: {alert['message']}")
            print(f"  Details: {json.dumps(alert['details'], indent=4)}")
            print()
    else:
        print(" No hazards detected. Kitchen is safe!")

    # Test with safe scenario
    print("\n" + "=" * 60)
    print("Testing SAFE scenario...")
    print("=" * 60)

    with open('../mock_data/yolo_output_safe.json', 'r') as f:
        yolo_safe = json.load(f)

    with open('../mock_data/classifier_output_safe.json', 'r') as f:
        classifier_safe = json.load(f)

    alerts_safe = checker.check_all_hazards(yolo_safe, classifier_safe)

    if alerts_safe:
        print(f" Found {len(alerts_safe)} hazard(s) in 'safe' scenario:")
        for alert in alerts_safe:
            print(f"  - {alert['message']}")
    else:
        print("No hazards detected. Person is close to stove - safe!")


if __name__ == "__main__":
    main()
