import json
from typing import Optional, Any, Dict

class BaseVL:
    def __init__(self, model_name: str = "", device: str = "cuda", system_prompt: str = "") -> None:
        """Initialize the base vision-language model.

        Args:
            model_name (str): Name of the model to use
            device (str): Device to run the model on (e.g., 'cuda', 'cpu')
            system_prompt (str): Default system prompt for the model
        """
        self.model_name = model_name
        self.device = device
        self.system_prompt = system_prompt

    def setup_system(self, new_prompt: str) -> None:
        """Update the system prompt.

        Args:
            new_prompt (str): New system prompt to use
        """
        self.system_prompt = new_prompt

    def process(self, img_path: Optional[str] = None, prompt: Optional[str] = None) -> Any:
        """Process an image with an optional prompt.

        Args:
            img_path (str, optional): Path to the input image
            prompt (str, optional): Text prompt to guide the processing

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("The process method must be implemented by subclasses")
    
    def to_json(self, json_str: str) -> Dict:
        """Convert a JSON string to a Python dictionary.

        Args:
            json_str (str): Valid JSON string to parse

        Returns:
            Dict: Parsed JSON data as a Python dictionary

        Raises:
            json.JSONDecodeError: If the input string is not valid JSON
        """
        try:
            return json.loads(json_str)  # Changed from load to loads for string input
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON string: {str(e)}", e.doc, e.pos)