from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from langchain_core.output_parsers import JsonOutputParser

class VisionLLM():

    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device='cuda', system_prompt=''):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype="auto", device_map="auto"
                    )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.system_message = [
            {
                'role':'system',
                'content':system_prompt
            }
        ]
        self.device = device

    def filter(self, image_path, user_query, obj_detected, max_new_tokens=128):
        messages = self.system_message
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{image_path}",
                    },
                    {
                        "type": "text",
                        "text": f"""
                        With this given bounding box of objects:{str(obj_detected)}.
                        Extract a bounding box that match: .
                        Answer with JSON type have key 'bbox', value is the box.
                        Answer as short as possible.
                        Given the following list of bounding boxes: 
                        {str(obj_detected)}
                        Extract the bounding boxes whose information matches the following request: 
                        {user_query}
                        Reply as a JSON blob, for example:
                        ```
                        [
                        {'bbox': [40.03, 72.43, 177.76, 115.58], 'key': obj1},
                        {'bbox': [40.03, 72.43, 177.76, 115.58], 'key': obj2}
                        ]
                        ```
                        Keep your answer as short as possible.
                        """
                    },
                ],
            }
        )
        print(messages)
        print(self.system_message)
        print("done message")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("done text")
        image_inputs, video_inputs = process_vision_info(messages)
        print("done inputs")
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        print("done inputs to device")

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        print("done gen id")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        print("done gen id trimmed")
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("done output")

        return output_text