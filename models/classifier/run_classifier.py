from ultralytics import YOLO

class Classifier:
    def __init__(self, model_path="yolo11n-cls.pt"):
        self.model = YOLO(model_path)

    def classify(self, crop_path):
        if crop_path is None:
            return None

        result = self.model(crop_path)[0]
        pred_class = result.probs.top1
        confidence = float(result.probs.top1conf)

        return {
            "predicted_class": pred_class,
            "confidence": confidence
        }
