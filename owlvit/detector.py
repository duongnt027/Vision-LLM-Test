import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class Detector():

    def __init__(self, llm_model='google/owlvit-base-patch32'):

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    def detect(self, image, objects):
        
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
