import json
import torch
from typing import Optional, Any, Dict
from qwen_vl_utils import process_vision_info
from langchain_core.output_parsers import JsonOutputParser
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from modules.visionllms import BaseVL

class QwenVL(BaseVL):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda", system_prompt: str = "") -> None:
        """Initialize the Llama vision-language model.

        Args:
            model_name (str): Name of the Llama model to use
            device (str): Device to run the model on (e.g., 'cuda', 'cpu')
            system_prompt (str): Default system prompt for the model
        """
        super().__init__(model_name, device, system_prompt)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.system_message = {
                "role": "system",
                "content": system_prompt,
            }
        self.device = device

    def setup_system(self, new_prompt: str) -> None:
        """Update the system prompt and print it.

        Args:
            new_prompt (str): New system prompt to use
        """
        super().setup_system(new_prompt)
        print(f"Set system prompt: {new_prompt}")
        self.system_message = {
                "role": "system",
                "content": new_prompt,
            }

    def process(self, img_path: Optional[str] = None, user_prompt: Optional[str] = None, max_new_tokens: int =128) -> None:
        """Process an image with the Llama model.

        Args:
            img_path (str, optional): Path to the input image
            user_prompt (str, optional): Text user prompt to guide the processing
        """

        messages = []
        messages.append(self.system_message)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        )

        print("Prepared messages:", messages)

        # Generate input text from the processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("Generated text template.")

        # Process vision inputs (images/videos)
        image_inputs, video_inputs = process_vision_info(messages)
        print("Processed vision inputs.")

        # Prepare inputs for the model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        print("Inputs moved to device.")

        # Generate output from the model
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        print("Generated output IDs.")

        # Trim the generated IDs to remove input context
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        print("Trimmed generated IDs.")

        # Decode the output text
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Decoded output text.")

        return output_text

        