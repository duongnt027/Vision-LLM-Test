from modules.llms import OllamaLLM

system_llm = """
You are an expert at analyzing objects from questions. 
Please provide the shortest possible response in JSON format 
with the key 'objects' being a list of the objects found in the question,
in singular form.
For example: 'cats on the couch' -> '{"objects": ["cat", "couch"]}'
"""

llm = OllamaLLM(
        model_name="llama3.2", 
        system_prompt = system_llm
    )

query = "remote between cats"

objects = llm.query(query)
print(objects)