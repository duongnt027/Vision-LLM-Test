
class BaseLLM:
    def __init__(self, model_name: str, system_prompt: str=""):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def query(self, prompt):
        raise NotImplementedError("The query method must be implemented by subclasses")
