from ultralytics import YOLO
import json, cv2

class Detector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, save_crop=True, conf=0.5)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                crop_path = getattr(box, "crop_path", None)

                detections.append({
                    "class": cls,
                    "confidence": conf,
                    "bbox": xyxy,
                    "crop_path": crop_path
                })
        return detections
