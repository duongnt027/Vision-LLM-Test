import json

from modules.visionllms import BaseVL
from typing import Optional, Any, Dict

class LlamaVL(BaseVL):
    def __init__(self, model_name: str = "", device: str = "cuda", system_prompt: str = "") -> None:
        """Initialize the Llama vision-language model.

        Args:
            model_name (str): Name of the Llama model to use
            device (str): Device to run the model on (e.g., 'cuda', 'cpu')
            system_prompt (str): Default system prompt for the model
        """
        super().__init__(model_name, device, system_prompt)

    def setup_system(self, new_prompt: str) -> None:
        """Update the system prompt and print it.

        Args:
            new_prompt (str): New system prompt to use
        """
        super().setup_system(new_prompt)
        print(f"New system prompt: {new_prompt}")

    def process(self, img_path: Optional[str] = None, prompt: Optional[str] = None) -> None:
        """Process an image with the Llama model.

        Args:
            img_path (str, optional): Path to the input image
            prompt (str, optional): Text prompt to guide the processing
        """
        print(f"Processing image at {img_path} with prompt: {prompt}")