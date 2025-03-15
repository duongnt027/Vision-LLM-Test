from ultralytics import YOLO
import cv2

class Yolov8:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        # COCO class names used by YOLOv8
        self.class_names = self.model.names  # Dictionary {0: 'person', 1: 'bicycle', ...}

    def process(self, image_path, objects):
        results = self.model(image_path)
        detected = {}

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, "Unknown")

                if confidence >= self.conf_threshold:
                    if objects is None or class_name in objects:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if class_name not in detected:
                            detected[class_name] = []
                        detected[class_name].append({
                            "bbox": (x1, y1, x2, y2),
                            "score": confidence
                        })

        return detected
    
    def simplize(self, detected):
        s_detected = {}
        for detected_key, detected_val in detected.items():
            s_detected[detected_key] = []
            for val in detected_val:
                s_detected[detected_key].append(val['bbox'])
        return s_detected