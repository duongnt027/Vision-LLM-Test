import torch
from PIL import Image
from modules.detectors import Base
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from utils import image_640

class OwlVit(Base):

    def __init__(self, model_name: str = 'google/owlvit-base-patch32'):
        super().__init__(model_name)
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)

    def process(self, image_path, objects):
        image = image_640(image_path)
        image = Image.open(image_path)
        inputs = self.processor(text=[objects], images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=0.1
            )
        i = 0
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        detected = {}
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if objects[label] not in detected:
                detected[objects[label]] = []
            detected[objects[label]].append(
                {
                    "bbox": box,
                    "score": score.item(),
                }
            )
        return detected

    def simplize(self, detected):
        s_detected = {}
        for detected_key, detected_val in detected.items():
            s_detected[detected_key] = detected_val['bbox']
        return s_detected