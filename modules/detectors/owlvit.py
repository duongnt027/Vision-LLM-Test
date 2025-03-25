import torch
from PIL import Image
from modules.detectors import Base
from transformers import OwlViTProcessor, OwlViTForObjectDetection, AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from utils import image_640
import numpy as np

class OwlVit(Base):

    def __init__(self, model_name: str = 'google/owlvit-base-patch32'):
        super().__init__(model_name)
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)

    def process(self, image_path, objects, threshold=0.1):
        image = image_640(image_path)
        image = Image.open(image_path)
        inputs = self.processor(text=[objects], images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=threshold
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
    
    def process2(self, image_path, objects, threshold=0.1, nms_threshold=0.3):
        image = image_640(image_path)
        image = Image.open(image_path)
        inputs = self.processor(text=[objects], images=image, return_tensors="pt")
        # outputs = self.model(**inputs)
        with torch.no_grad():
            outputs = self.model.image_guided_detection(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_image_guided_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=threshold,
                nms_threshold = nms_threshold
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
            s_detected[detected_key] = []
            for val in detected_val:
                s_detected[detected_key].append(val['bbox'])
        return s_detected

class OwlVit2(Base):
    def __init__(self, model_name: str = 'google/owlv2-base-patch16'):
        super().__init__(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)

    def get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def process(self, image_path, objects, threshold=0.2):
        image = image_640(image_path)
        image = Image.open(image_path)
        inputs = self.processor(text=[objects], images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        unnormalized_image = self.get_preprocessed_image(inputs.pixel_values)
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes
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
            s_detected[detected_key] = []
            for val in detected_val:
                s_detected[detected_key].append(val['bbox'])
        return s_detected