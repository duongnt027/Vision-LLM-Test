import torch
from transformers import pipeline

from modules.llms import BaseLLM

class HuggingFaceLLM(BaseLLM):

    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", system_prompt = "", hf_token=""):
        super().__init__(model_name, system_prompt)
        self.hf_token = hf_token

    def query(self, prompt, temperature=0.7, max_tokens=128000):
        pipe = pipeline(
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=self.hf_token
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
        )

        return outputs[0]["generated_text"][-1]['content']


    

    

