import json
import sys
import time
import logging
import requests
from typing import Optional, Dict, Any, List, Callable

from modules.llms import BaseLLM

class v2LLM(BaseLLM):
    def __init__(self, model_name="llama3", system_prompt="", host="localhost", port=8000):
        """
        Initialize the vLLMLLM class.
        
        Args:
            model_name (str): Name of the model to use
            system_prompt (str, optional): System prompt to use for all queries
            host (str): Host where vLLM server is running
            port (int): Port where vLLM server is running
        """
        super().__init__(model_name, system_prompt)
        self.base_url = f"http://{host}:{port}/v1"
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
        self.logger = logging.getLogger("vLLMLLM")
    
    def check_model_loaded(self) -> bool:
        """
        Check if the model is loaded in vLLM.
        
        Returns:
            bool: True if model exists, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                models = response.json()
                # The exact response format may vary depending on vLLM server implementation
                self.logger.info(f"Model check response: {models}")
                # You might need to adapt this check based on your vLLM server's response format
                return self.model_name in str(models)
            else:
                self.logger.error(f"Failed to check models: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error checking if model is loaded: {str(e)}")
            return False
    
    def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, 
              top_p: float = 1.0, stop: Optional[List[str]] = None) -> str:
        """
        Send a query to the vLLM server.
        
        Args:
            prompt (str): User prompt to send to the model
            temperature (float): Controls randomness in the response
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Nucleus sampling parameter
            stop (list, optional): List of strings that stop generation when encountered
            
        Returns:
            str: The model's response or an error message
        """
        try:
            # vLLM uses OpenAI-compatible API
            url = f"{self.base_url}/completions"
            
            # Prepare the prompt with system prompt if provided
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            if stop:
                payload["stop"] = stop
            
            self.logger.info(f"Sending query to vLLM server for model {self.model_name}")
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                response_json = response.json()
                # Extract text from the response
                # This might need adjustment based on your vLLM server's response format
                return response_json.get("choices", [{}])[0].get("text", "No response received")
            else:
                self.logger.error(f"Error from API: {response.status_code} - {response.text}")
                return f"Error: API returned status code {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error: Is vLLM server running?")
            return "Error: Could not connect to vLLM server. Make sure it's running."
        except Exception as e:
            self.logger.error(f"Unexpected error during query: {str(e)}")
            return f"Error: {str(e)}"
    
    def chat_query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000,
                  top_p: float = 1.0, stop: Optional[List[str]] = None) -> str:
        """
        Send a chat-style query to the vLLM server.
        
        Args:
            prompt (str): User prompt to send to the model
            temperature (float): Controls randomness in the response
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Nucleus sampling parameter
            stop (list, optional): List of strings that stop generation when encountered
            
        Returns:
            str: The model's response or an error message
        """
        try:
            # vLLM uses OpenAI-compatible API
            url = f"{self.base_url}/chat/completions"
            
            messages = []
            
            # Add system message if provided
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
                
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            if stop:
                payload["stop"] = stop
            
            self.logger.info(f"Sending chat query to vLLM server for model {self.model_name}")
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                response_json = response.json()
                # Extract content from the response
                return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response received")
            else:
                self.logger.error(f"Error from API: {response.status_code} - {response.text}")
                return f"Error: API returned status code {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error: Is vLLM server running?")
            return "Error: Could not connect to vLLM server. Make sure it's running."
        except Exception as e:
            self.logger.error(f"Unexpected error during query: {str(e)}")
            return f"Error: {str(e)}"
    
    def streaming_query(self, prompt: str, callback: Optional[Callable[[str], None]] = None, 
                       temperature: float = 0.7, max_tokens: int = 2000,
                       top_p: float = 1.0, stop: Optional[List[str]] = None) -> str:
        """
        Send a streaming query to the vLLM server.
        
        Args:
            prompt (str): User prompt to send to the model
            callback (callable): Function to call with each token as it arrives
            temperature (float): Controls randomness in the response
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Nucleus sampling parameter
            stop (list, optional): List of strings that stop generation when encountered
            
        Returns:
            str: The complete response after streaming finishes
        """
        try:
            # vLLM uses OpenAI-compatible API
            url = f"{self.base_url}/completions"
            
            # Prepare the prompt with system prompt if provided
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True
            }
            
            if stop:
                payload["stop"] = stop
            
            self.logger.info(f"Starting streaming query to vLLM server for model {self.model_name}")
            
            full_response = ""
            with requests.post(url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    return f"Error: API returned status code {response.status_code}"
                
                for line in response.iter_lines():
                    if line:
                        # Remove 'data: ' prefix if present
                        data_str = line.decode('utf-8')
                        if data_str.startswith('data: '):
                            data_str = data_str[6:]
                        
                        # Handle the "[DONE]" marker
                        if data_str == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            token = data.get("choices", [{}])[0].get("text", "")
                            full_response += token
                            
                            if callback and callable(callback):
                                callback(token)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse streaming response: {data_str}")
            
            return full_response
            
        except Exception as e:
            self.logger.error(f"Error in streaming query: {str(e)}")
            return f"Error: {str(e)}"
    
    def streaming_chat_query(self, prompt: str, callback: Optional[Callable[[str], None]] = None, 
                           temperature: float = 0.7, max_tokens: int = 2000,
                           top_p: float = 1.0, stop: Optional[List[str]] = None) -> str:
        """
        Send a streaming chat query to the vLLM server.
        
        Args:
            prompt (str): User prompt to send to the model
            callback (callable): Function to call with each token as it arrives
            temperature (float): Controls randomness in the response
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Nucleus sampling parameter
            stop (list, optional): List of strings that stop generation when encountered
            
        Returns:
            str: The complete response after streaming finishes
        """
        try:
            # vLLM uses OpenAI-compatible API
            url = f"{self.base_url}/chat/completions"
            
            messages = []
            
            # Add system message if provided
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
                
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True
            }
            
            if stop:
                payload["stop"] = stop
            
            self.logger.info(f"Starting streaming chat query to vLLM server for model {self.model_name}")
            
            full_response = ""
            with requests.post(url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    return f"Error: API returned status code {response.status_code}"
                
                for line in response.iter_lines():
                    if line:
                        # Remove 'data: ' prefix if present
                        data_str = line.decode('utf-8')
                        if data_str.startswith('data: '):
                            data_str = data_str[6:]
                        
                        # Handle the "[DONE]" marker
                        if data_str == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            token = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if token:
                                full_response += token
                                
                                if callback and callable(callback):
                                    callback(token)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse streaming response: {data_str}")
            
            return full_response
            
        except Exception as e:
            self.logger.error(f"Error in streaming chat query: {str(e)}")
            return f"Error: {str(e)}"
    
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
    # Example using vLLMLLM
    vllm = v2LLM(
        model_name="llama3", 
        system_prompt="You are a helpful AI assistant that provides concise, accurate information.",
        host="localhost",
        port=8000
    )
    
    print("\nvLLM basic completion response:")
    response = vllm.query("What is machine learning?")
    print(response)
    
    print("\nvLLM chat response:")
    response = vllm.chat_query("What is machine learning?")
    print(response)
    
    print("\nvLLM streaming response:")
    def print_token(token):
        print(token, end="", flush=True)
    
    vllm.streaming_query("Explain neural networks briefly", callback=print_token)