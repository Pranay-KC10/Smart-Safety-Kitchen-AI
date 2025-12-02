"""
Smart Kitchen Safety AI - Main Orchestrator (Stage 3)

This is the main entry point that:
1. Receives outputs from Stage 1 (YOLO) and Stage 2 (Classifier)
2. Runs safety checks using SafetyChecker
3. Generates alerts using AlertSystem
4. Coordinates the entire pipeline

For now, this works with JSON files. Later, it will:
- Read from Docker container outputs
- Process real-time camera feed
- Run continuously as a service
"""

import json
import sys
import time
from pathlib import Path

# Import our Stage 3 modules
from safety_checker import SafetyChecker
from alert_system import AlertSystem


class KitchenSafetyOrchestrator:
   
    def __init__(self, config_path: str = None):
        
        # Initialize safety checker with custom config if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "safe_distance_threshold": 200,
                "confidence_threshold": 0.7,
                "alert_cooldown": 5,
            }

        self.safety_checker = SafetyChecker(config)
        self.alert_system = AlertSystem()

        print("[INIT] Kitchen Safety AI Initialized")
        print(f"   - Safe distance: {config['safe_distance_threshold']}px")
        print(f"   - Confidence threshold: {config['confidence_threshold']}")
        print()

    def process_frame(self, yolo_output_path: str, classifier_output_path: str):
        
        # Load data from Stage 1 (YOLO) and Stage 2 (Classifier)
        try:
            with open(yolo_output_path, 'r') as f:
                yolo_data = json.load(f)

            with open(classifier_output_path, 'r') as f:
                classifier_data = json.load(f)

        except FileNotFoundError as e:
            print(f"[ERROR] Could not find input file: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format: {e}")
            return []

        # Run safety checks
        alerts = self.safety_checker.check_all_hazards(yolo_data, classifier_data)

        # Send notifications
        self.alert_system.send_notifications(alerts, enable_audio=True, enable_logging=True)

        return alerts

    def run_continuous_monitoring(self, yolo_dir: str, classifier_dir: str, interval: int = 1):
        
        print(f"[START] Starting continuous monitoring...")
        print(f"   Checking every {interval} second(s)")
        print(f"   Press Ctrl+C to stop\n")

        try:
            while True:
                # In real implementation, this would:
                # 1. Check for new files from YOLO/Classifier
                # 2. Process the latest frame
                # 3. Generate alerts if needed

                # For now, just a placeholder
                print(f"[MONITOR] Monitoring... (This will be implemented with real-time data)")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[STOP] Monitoring stopped by user")
            self.alert_system.print_daily_summary()


def main():
    
    print("=" * 70)
    print("SMART KITCHEN SAFETY AI - STAGE 3")
    print("=" * 70)
    print()

    # Initialize orchestrator
    orchestrator = KitchenSafetyOrchestrator()

    # Test Mode: Process sample data
    print("[TEST] TEST MODE: Processing mock data\n")

    # Test Scenario 1: Hazardous kitchen
    print("[TEST 1] Hazardous Kitchen (stove unattended + knife)")
    print("-" * 70)
    alerts1 = orchestrator.process_frame(
        yolo_output_path="../mock_data/yolo_output.json",
        classifier_output_path="../mock_data/classifier_output.json"
    )

    # Test Scenario 2: Safe kitchen
    print("\n\n[TEST 2] Safe Kitchen (person near stove)")
    print("-" * 70)
    alerts2 = orchestrator.process_frame(
        yolo_output_path="../mock_data/yolo_output_safe.json",
        classifier_output_path="../mock_data/classifier_output_safe.json"
    )

    # Summary
    print("\n\n")
    orchestrator.alert_system.print_daily_summary()

    print("\n" + "=" * 70)
    print("[DONE] Stage 3 Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()