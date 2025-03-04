import requests
import json
import subprocess
import sys
import time
import logging

from modules.llms import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model_name="llama3", system_prompt=""):
        """
        Initialize the OllamaLLM class.
        
        Args:
            model_name (str): Name of the model to use
            system_prompt (str, optional): System prompt to use for all queries
        """
        super().__init__(model_name, system_prompt)
        self.base_url = "http://localhost:11434/api"
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("OllamaLLM")
        
    def ensure_model_exists(self):
        """
        Check if the model exists, and pull it if it doesn't.
        
        Returns:
            bool: True if model exists or was successfully pulled, False otherwise
        """
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/tags")
            models = response.json().get("models", [])
            
            if not any(model["name"] == self.model_name for model in models):
                self.logger.info(f"Model {self.model_name} not found. Pulling model...")
                
                # Use subprocess to pull the model
                process = subprocess.Popen(
                    ["ollama", "pull", self.model_name], 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Display progress
                while process.poll() is None:
                    output = process.stdout.readline()
                    if output:
                        self.logger.info(output.strip())
                    time.sleep(0.1)
                
                # Check result
                if process.returncode != 0:
                    self.logger.error(f"Failed to pull model {self.model_name}")
                    stderr = process.stderr.read()
                    self.logger.error(f"Error: {stderr}")
                    return False
                
                self.logger.info(f"Successfully pulled model {self.model_name}")
            else:
                self.logger.info(f"Model {self.model_name} already exists")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error ensuring model exists: {str(e)}")
            return False
    
    def query(self, prompt, temperature=0.7, max_tokens=2000):
        """
        Send a query to the Ollama API.
        
        Args:
            prompt (str): User prompt to send to the model
            temperature (float): Controls randomness in the response
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: The model's response or an error message
        """
        # First ensure the model exists
        if not self.ensure_model_exists():
            return "Failed to ensure model exists. Check logs for details."
        
        try:
            url = f"{self.base_url}/generate"
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add system prompt if provided
            if self.system_prompt:
                data["system"] = self.system_prompt
            
            self.logger.info(f"Sending query to model {self.model_name}")
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                return response.json().get("response", "No response received")
            else:
                self.logger.error(f"Error from API: {response.status_code} - {response.text}")
                return f"Error: API returned status code {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error: Is Ollama running?")
            return "Error: Could not connect to Ollama. Make sure Ollama is running."
        except Exception as e:
            self.logger.error(f"Unexpected error during query: {str(e)}")
            return f"Error: {str(e)}"
    
    def streaming_query(self, prompt, callback=None, temperature=0.7):
        """
        Send a streaming query to the Ollama API.
        
        Args:
            prompt (str): User prompt to send to the model
            callback (callable): Function to call with each token as it arrives
            temperature (float): Controls randomness in the response
            
        Returns:
            str: The complete response after streaming finishes
        """
        if not self.ensure_model_exists():
            return "Failed to ensure model exists. Check logs for details."
        
        try:
            url = f"{self.base_url}/generate"
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "temperature": temperature
            }
            
            if self.system_prompt:
                data["system"] = self.system_prompt
            
            self.logger.info(f"Starting streaming query to model {self.model_name}")
            
            full_response = ""
            with requests.post(url, json=data, stream=True) as response:
                if response.status_code != 200:
                    return f"Error: API returned status code {response.status_code}"
                
                for line in response.iter_lines():
                    if line:
                        line_json = json.loads(line.decode('utf-8'))
                        token = line_json.get("response", "")
                        full_response += token
                        
                        if callback and callable(callback):
                            callback(token)
            
            return full_response
            
        except Exception as e:
            self.logger.error(f"Error in streaming query: {str(e)}")
            return f"Error: {str(e)}"
    
    def change_model(self, new_model_name):
        """
        Change the model being used.
        
        Args:
            new_model_name (str): Name of the new model to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.model_name = new_model_name
        self.logger.info(f"Model changed to {new_model_name}")
        return self.ensure_model_exists()
    
    def update_system_prompt(self, new_system_prompt):
        """
        Update the system prompt.
        
        Args:
            new_system_prompt (str): New system prompt to use
        """
        self.system_prompt = new_system_prompt
        self.logger.info("System prompt updated")


# Example usage
if __name__ == "__main__":
    # Initialize with a model and system prompt
    llm = OllamaLLM(
        model_name="llama3.2", 
        system_prompt = """
You are an expert at analyzing objects from questions. 
Please provide the shortest possible response in JSON format 
with the key 'objects' being a list of the objects found in the question,
in singular form.
For example: 'cats on the couch' -> '{"objects": ["cat", "couch"]}'
"""
    )
    
    # Basic query
    response = llm.query("Those dogs chases the red ball in the park")
    print("\nResponse:", response)
    

# # Example usage
# if __name__ == "__main__":
    
#     user_prompt = "Those dogs chases the red ball in the park"
    
#     response = ask_ollama(user_prompt, system_prompt=system_prompt)
#     print(response)
