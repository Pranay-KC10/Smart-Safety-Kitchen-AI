"""
Alert System Module (Stage 3)
Handles different types of notifications for kitchen safety alerts:
  - Console notifications (text output)
  - Audio alerts (beep sounds)
  - Logging (save incidents to file)
  - Visual display (terminal colors)
"""

import json
import os
from datetime import datetime
from typing import List, Dict


class AlertSystem:
    def __init__(self, log_dir: str = "../outputs/logs"):
        """
        Initialize AlertSystem

        Args:
            log_dir: Directory to save alert logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # ANSI color codes for terminal
        self.COLORS = {
            "CRITICAL": "\033[91m",  # Red
            "HIGH": "\033[93m",      # Yellow
            "MEDIUM": "\033[93m",    # Yellow
            "LOW": "\033[94m",       # Blue
            "RESET": "\033[0m"       # Reset
        }

    def print_alert(self, alert: Dict):
        """
        Print formatted alert to console with colors

        Args:
            alert: Alert dictionary from SafetyChecker
        """
        severity = alert.get('severity', 'LOW')
        color = self.COLORS.get(severity, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        print(f"\n{color}{'=' * 70}{reset}")
        print(f"{color}[{severity}] {alert['type']}{reset}")
        print(f"{color}{alert['message']}{reset}")
        print(f"{color}Time: {alert.get('timestamp', 'N/A')}{reset}")
        print(f"{color}Frame: {alert.get('frame_number', 'N/A')}{reset}")

        if alert.get('details'):
            print(f"{color}Details:{reset}")
            for key, value in alert['details'].items():
                print(f"  - {key}: {value}")

        print(f"{color}{'=' * 70}{reset}")

    def play_audio_alert(self, severity: str):
        """
        Play audio alert based on severity
        (Uses system beep - works on most systems)

        Args:
            severity: Alert severity level
        """
        try:
            if severity == "CRITICAL":
                # Emergency - 3 rapid beeps
                for _ in range(3):
                    print('\a', end='', flush=True)  # System beep
            elif severity in ["HIGH", "MEDIUM"]:
                # Warning - 2 beeps
                for _ in range(2):
                    print('\a', end='', flush=True)
            else:
                # Low priority - 1 beep
                print('\a', end='', flush=True)
        except Exception as e:
            # Some systems might not support audio
            pass

    def log_alert(self, alert: Dict):
        """
        Save alert to log file

        Args:
            alert: Alert dictionary to log
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"alerts_{timestamp}.json")

        # Read existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # Append new alert
        logs.append(alert)

        # Write back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def send_notifications(self, alerts: List[Dict], enable_audio: bool = True, enable_logging: bool = True):
        """
        Send all types of notifications for a list of alerts

        Args:
            alerts: List of alert dictionaries
            enable_audio: Whether to play audio alerts
            enable_logging: Whether to log alerts to file
        """
        if not alerts:
            print("\n[OK] No hazards detected. Kitchen is safe!")
            return

        print(f"\n[!] SAFETY ALERT: {len(alerts)} hazard(s) detected!\n")

        for alert in alerts:
            # Console notification (always on)
            self.print_alert(alert)

            # Audio notification
            if enable_audio:
                self.play_audio_alert(alert.get('severity', 'LOW'))

            # Log to file
            if enable_logging:
                self.log_alert(alert)

        if enable_logging:
            print(f"\n[LOG] Alerts logged to: {self.log_dir}")

    def get_alert_summary(self) -> Dict:
        """
        Get summary of all alerts from today's log

        Returns:
            Dictionary with alert statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"alerts_{timestamp}.json")

        if not os.path.exists(log_file):
            return {"total": 0, "by_type": {}, "by_severity": {}}

        with open(log_file, 'r') as f:
            logs = json.load(f)

        # Count by type and severity
        by_type = {}
        by_severity = {}

        for alert in logs:
            alert_type = alert.get('type', 'UNKNOWN')
            severity = alert.get('severity', 'UNKNOWN')

            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total": len(logs),
            "by_type": by_type,
            "by_severity": by_severity
        }

    def print_daily_summary(self):
        """Print summary of today's alerts"""
        summary = self.get_alert_summary()

        print("\n" + "=" * 60)
        print("[SUMMARY] DAILY ALERT SUMMARY")
        print("=" * 60)
        print(f"Total Alerts Today: {summary['total']}")

        if summary['total'] > 0:
            print("\nBy Type:")
            for alert_type, count in summary['by_type'].items():
                print(f"  - {alert_type}: {count}")

            print("\nBy Severity:")
            for severity, count in summary['by_severity'].items():
                print(f"  - {severity}: {count}")
        else:
            print("[OK] No alerts today - Kitchen has been safe!")

        print("=" * 60)


def main():
    """Test the AlertSystem"""
    print("=" * 60)
    print("ALERT SYSTEM TEST - Stage 3")
    print("=" * 60)

    # Initialize alert system
    alert_system = AlertSystem()

    # Create sample alerts
    test_alerts = [
        {
            "type": "STOVE_UNATTENDED",
            "severity": "HIGH",
            "message": "[!] ALERT: Stove is ON while you're away!",
            "timestamp": datetime.now().isoformat(),
            "frame_number": 150,
            "details": {
                "stove_status": "ON",
                "person_detected": False
            }
        },
        {
            "type": "KNIFE_UNATTENDED",
            "severity": "LOW",
            "message": "[!] NOTICE: Knife left unattended.",
            "timestamp": datetime.now().isoformat(),
            "frame_number": 150,
            "details": {
                "knife_status": "unattended",
                "confidence": 0.87
            }
        }
    ]

    # Send notifications
    print("\n[INFO] Sending test alerts...\n")
    alert_system.send_notifications(test_alerts, enable_audio=True, enable_logging=True)

    # Print summary
    alert_system.print_daily_summary()


if __name__ == "__main__":
    main()
