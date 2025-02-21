import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class Base:

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    def process(self, image, objects):
        raise NotImplementedError("The process method must be implemented by subclasses")