"""
Safety Checker Module (Stage 3)
This module implements hazard detection rules by analyzing outputs from:
  - Stage 1 (YOLO): Object detections with bounding boxes
  - Stage 2 (Classifier): Classification of object states

Hazard Detection Rules:
  1. Fire/Smoke detected → CRITICAL "DANGER DANGER DANGER! FIRE!"
  2. Stove ON + No person nearby → HIGH "Fire left unattended - DANGER!"
  3. Knife detected + Unattended → MEDIUM "Knife unattended - could be dangerous!"
  4. Stove ON + Empty pan → HIGH "Pan overheating!"
  5. Person too far from active stove → MEDIUM Warning
"""

import math
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time


class SafetyChecker:
    def __init__(self, config: Dict = None):
        # Default safety thresholds (in pixels - can be calibrated later)
        self.config = config or {
            "safe_distance_threshold": 200,  # pixels - person must be within this distance from stove
            "confidence_threshold": 0.7,     # minimum confidence for detections
            "alert_cooldown": 5,             # seconds between repeated alerts
            "knife_danger_distance": 100,    # if knife is close to edge of counter
        }

        self.last_alert_time = {}  # Track last alert time for cooldown
        self.alert_history = []    # Track alert history for patterns

    def calculate_distance(self, point1: List[int], point2: List[int]) -> float:
        """Calculate Euclidean distance between two points"""
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def find_object_by_class(self, detections: List[Dict], class_name: str) -> Optional[Dict]:
        """Find first detection matching class name with sufficient confidence"""
        for detection in detections:
            if detection['class'] == class_name:
                if detection['confidence'] >= self.config['confidence_threshold']:
                    return detection
        return None

    def find_all_objects_by_class(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """Find all detections matching class name"""
        return [d for d in detections
                if d['class'] == class_name
                and d['confidence'] >= self.config['confidence_threshold']]

    def should_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)

        if current_time - last_time >= self.config['alert_cooldown']:
            self.last_alert_time[alert_type] = current_time
            return True
        return False

    def check_fire_smoke(self, detections: List[Dict]) -> Optional[Dict]:
        """
        RULE 1: Fire or smoke detection - HIGHEST PRIORITY
        This is an emergency situation!
        """
        fire = self.find_object_by_class(detections, "fire")
        smoke = self.find_object_by_class(detections, "smoke")

        if fire:
            return {
                "type": "FIRE_DETECTED",
                "severity": "CRITICAL",
                "message": "DANGER DANGER DANGER! FIRE DETECTED IN KITCHEN! EVACUATE NOW AND CALL 911!",
                "voice_alert": "Fire detected! Danger! Danger! Danger! Evacuate immediately!",
                "details": {
                    "hazard": "fire",
                    "confidence": fire['confidence'],
                    "location": fire.get('center', 'unknown'),
                    "action_required": "EVACUATE AND CALL EMERGENCY SERVICES"
                }
            }

        if smoke:
            return {
                "type": "SMOKE_DETECTED",
                "severity": "CRITICAL",
                "message": "DANGER! SMOKE DETECTED! Possible fire hazard - check kitchen immediately!",
                "voice_alert": "Smoke detected! Danger! Check the kitchen immediately!",
                "details": {
                    "hazard": "smoke",
                    "confidence": smoke['confidence'],
                    "location": smoke.get('center', 'unknown'),
                    "action_required": "CHECK FOR FIRE SOURCE IMMEDIATELY"
                }
            }

        return None

    def check_stove_unattended(self, detections: List[Dict], classifications: Dict) -> Optional[Dict]:
        """
        RULE 2: Stove/fire left unattended
        Active heat source with no one monitoring is dangerous!
        """
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
            # No person detected at all - DANGEROUS!
            return {
                "type": "STOVE_UNATTENDED",
                "severity": "HIGH",
                "message": "ALERT! Stove/fire is ON and left UNATTENDED! This is DANGEROUS! Please return to the kitchen immediately!",
                "voice_alert": "Warning! The stove is on and unattended! This is dangerous! Please return to the kitchen!",
                "details": {
                    "stove_status": stove_status.get('status'),
                    "stove_confidence": stove_status.get('confidence'),
                    "person_detected": False,
                    "risk_level": "HIGH - No supervision",
                    "action_required": "Return to kitchen immediately"
                }
            }

        # Person detected - check distance
        distance = self.calculate_distance(stove['center'], person['center'])

        if distance > self.config['safe_distance_threshold']:
            return {
                "type": "STOVE_TOO_FAR",
                "severity": "MEDIUM",
                "message": f"WARNING: Stove is ON but you're too far away ({int(distance)}px)! Stay close while cooking to prevent accidents!",
                "voice_alert": "Warning! You are too far from the active stove. Please stay close while cooking.",
                "details": {
                    "stove_status": stove_status.get('status'),
                    "distance_from_stove": int(distance),
                    "safe_distance": self.config['safe_distance_threshold'],
                    "person_detected": True,
                    "risk_level": "MEDIUM - Too far from heat source"
                }
            }

        return None  # Person is close enough, safe!

    def check_knife_unattended(self, detections: List[Dict], classifications: Dict) -> Optional[Dict]:
        """
        RULE 3: Knife left unattended
        Sharp objects should be stored safely when not in use!
        """
        knife = self.find_object_by_class(detections, "knife")
        person = self.find_object_by_class(detections, "person")

        if not knife:
            return None  # No knife detected

        # Check knife status from classifier
        knife_image = knife.get('cropped_image_path', '').split('/')[-1]
        knife_status = classifications.get(knife_image, {})

        # Check if knife is unattended
        is_unattended = knife_status.get('status') == 'unattended'

        # Also check if person is nearby (even if classifier says unattended)
        person_near_knife = False
        if person and knife.get('center'):
            knife_person_distance = self.calculate_distance(knife['center'], person['center'])
            person_near_knife = knife_person_distance < self.config['knife_danger_distance']

        if is_unattended and not person_near_knife:
            return {
                "type": "KNIFE_UNATTENDED",
                "severity": "MEDIUM",
                "message": "WARNING: Knife is left unattended! This could be DANGEROUS! Please store the knife safely in a drawer or knife block.",
                "voice_alert": "Warning! A knife is left unattended. This could be dangerous. Please store it safely.",
                "details": {
                    "knife_status": knife_status.get('status'),
                    "confidence": knife_status.get('confidence'),
                    "features": knife_status.get('features', {}),
                    "person_nearby": person_near_knife,
                    "risk_level": "MEDIUM - Sharp object unsecured",
                    "action_required": "Store knife in drawer or knife block"
                }
            }

        return None

    def check_pan_overheating(self, detections: List[Dict], classifications: Dict) -> Optional[Dict]:
        """
        RULE 4: Empty pan on active stove
        Can cause fire or damage!
        """
        pan = self.find_object_by_class(detections, "pan")
        stove = self.find_object_by_class(detections, "stove")

        if not pan or not stove:
            return None

        # Check if stove is ON
        stove_image = stove.get('cropped_image_path', '').split('/')[-1]
        stove_status = classifications.get(stove_image, {})

        if stove_status.get('status') != 'ON':
            return None  # Stove is off

        # Check pan status
        pan_image = pan.get('cropped_image_path', '').split('/')[-1]
        pan_status = classifications.get(pan_image, {})

        # Check if pan is near stove (on the burner)
        if pan.get('center') and stove.get('center'):
            pan_stove_distance = self.calculate_distance(pan['center'], stove['center'])

            if pan_stove_distance < 150 and pan_status.get('status') == 'empty':
                return {
                    "type": "PAN_OVERHEATING",
                    "severity": "HIGH",
                    "message": "DANGER! Empty pan on active stove! This can cause FIRE or damage! Add food/liquid or remove from heat!",
                    "voice_alert": "Danger! Empty pan on hot stove! Risk of fire! Please add contents or remove the pan!",
                    "details": {
                        "pan_status": pan_status.get('status'),
                        "stove_status": stove_status.get('status'),
                        "risk_level": "HIGH - Fire hazard",
                        "action_required": "Add contents to pan or remove from heat"
                    }
                }

        return None

    def check_all_hazards(self, yolo_output: Dict, classifier_output: Dict) -> List[Dict]:
        """
        Run all safety checks and return list of alerts
        Checks are ordered by severity/priority
        """
        alerts = []
        detections = yolo_output.get('detections', [])
        classifications = classifier_output.get('classifications', {})

        # Run all safety checks in priority order
        checks = [
            self.check_fire_smoke(detections),                          # CRITICAL
            self.check_pan_overheating(detections, classifications),    # HIGH
            self.check_stove_unattended(detections, classifications),   # HIGH/MEDIUM
            self.check_knife_unattended(detections, classifications),   # MEDIUM
        ]

        # Filter out None values (no hazard detected)
        alerts = [alert for alert in checks if alert is not None]

        # Add timestamp and frame info to all alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            alert['frame_number'] = yolo_output.get('frame_number', 'N/A')

        # Store in history for pattern detection
        self.alert_history.extend(alerts)

        return alerts

    def get_safety_status(self, yolo_output: Dict, classifier_output: Dict) -> Dict:
        """
        Get overall kitchen safety status
        Returns a summary suitable for display
        """
        alerts = self.check_all_hazards(yolo_output, classifier_output)

        if not alerts:
            return {
                "status": "SAFE",
                "message": "Kitchen is safe! All clear.",
                "alerts": [],
                "color": "green"
            }

        # Find highest severity
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        max_severity = max(alerts, key=lambda a: severity_order.get(a['severity'], 0))

        status_map = {
            "CRITICAL": ("EMERGENCY", "red"),
            "HIGH": ("DANGER", "orange"),
            "MEDIUM": ("WARNING", "yellow"),
            "LOW": ("CAUTION", "blue")
        }

        status, color = status_map.get(max_severity['severity'], ("UNKNOWN", "gray"))

        return {
            "status": status,
            "message": max_severity['message'],
            "alerts": alerts,
            "color": color,
            "alert_count": len(alerts)
        }


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
