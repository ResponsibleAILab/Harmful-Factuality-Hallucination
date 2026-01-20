import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse

# Available Llama models - simplified to just two options
AVAILABLE_MODELS = {
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"
}

class LlamaClient:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize Llama model
        
        Parameters:
        - model_id: Model ID to load, default is meta-llama/Llama-3.1-8B-Instruct
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {model_id}...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model based on device
        if self.device == "cuda":
            # Use half precision to reduce VRAM usage
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # CPU loading, no half precision
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.to(self.device)
        
        print("Model loading complete!")
    
    def generate(self, prompt, system_message=None, max_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate text using Llama model
        
        Parameters:
        - prompt: User input prompt
        - system_message: System message to guide model behavior
        - max_tokens: Maximum number of tokens to generate
        - temperature: Parameter for generation diversity, higher values make output more random
        - top_p: Nucleus sampling parameter, controls randomness of generation
        
        Returns:
        - Generated text
        """
        # Build conversation messages
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Convert messages to model input format
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.device)
        
        # Record start time
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id  # Set pad_token_id to eos_token_id for open-end generation
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract model response (remove user input part)
        # Find response after the last user message
        if "role=\"user\"" in generated_text:
            response = generated_text.split("role=\"user\"")[-1]
            # Find assistant response content
            if "role=\"assistant\"" in response:
                response = response.split("role=\"assistant\"")[-1]
                if "content=\"" in response:
                    response = response.split("content=\"")[1]
                    # Remove ending quotes and extra markers
                    if "\"" in response:
                        response = response.split("\"")[0]
        else:
            # Simply remove prompt part
            response = generated_text[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):]
        
        # Clean up any remaining role markers and whitespace
        response = response.strip()
        
        # Remove "assistant" prefix if it appears at the beginning
        if response.lower().startswith("assistant"):
            response = response[len("assistant"):].strip()
        
        # Print generation statistics
        elapsed_time = time.time() - start_time
        print(f"Generation time: {elapsed_time:.2f} seconds")
        print(f"Tokens generated: {len(outputs[0]) - len(inputs[0])}")
        
        return response.strip()
    
    def summarize(self, text, max_tokens=256):
        """
        Summarize text using Llama model
        
        Parameters:
        - text: Text to summarize
        - max_tokens: Maximum number of tokens for the summary
        
        Returns:
        - Generated summary
        """
        # Use the same system message and prompt template as OpenAI
        system_message = "Summarize the given text."
        prompt = f"Summarize: {text}"
        
        return self.generate(
            prompt=prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=0.3  # Lower temperature for more deterministic summary
        )

def main():
    """
    Test Llama model functionality
    """
    # Simple command line argument for model selection
    parser = argparse.ArgumentParser(description='Run Llama model')
    parser.add_argument('--model', type=str, choices=['8b', '1b', '3b'], default='3b',
                        help='Model size to use: 8b or 1b or 3b')
    args = parser.parse_args()
    
    # Map argument to model ID
    if args.model == '1b':
        model_id = AVAILABLE_MODELS['llama3.2-1b']
    elif args.model == '3b':
        model_id = AVAILABLE_MODELS['llama3.2-3b']
    else:  # Default to 8b
        model_id = AVAILABLE_MODELS['llama3-8b']
    
    print(f"Using model: {model_id}")
    
    # Initialize Llama client
    try:
        llama_client = LlamaClient(model_id=model_id)
        
        # Test text generation
        prompt = "Explain the basic principles of quantum computers"
        system_message = "You are a quantum physics professor explaining complex concepts in simple terms."
        
        print("\n=== Testing Text Generation ===")
        print(f"Prompt: {prompt}")
        print(f"System message: {system_message}")
        
        response = llama_client.generate(
            prompt=prompt,
            system_message=system_message,
            max_tokens=300
        )
        
        print("\nGenerated response:")
        print(response)
        
        # Test text summarization
        test_text = """
        Artificial Intelligence (AI) is a branch of computer science devoted to creating systems capable of performing tasks that typically require human intelligence.
        These tasks include visual perception, speech recognition, decision-making, and language translation. AI research spans multiple disciplines, including machine learning,
        deep learning, neural networks, natural language processing, and computer vision. In recent years, with increased computing power and data availability,
        AI technology has made significant advances. Large language models like GPT and Llama can generate human-like text, while computer vision systems can
        identify objects in images, sometimes more accurately than humans. While AI brings many benefits, such as improved medical diagnostics and autonomous vehicles,
        it also raises concerns about privacy, bias, and the future of employment. As AI continues to develop, balancing its potential benefits and risks will be an important task for society.
        """
        
        print("\n=== Testing Text Summarization ===")
        print(f"Length of text to summarize: {len(test_text)} characters")
        
        summary = llama_client.summarize(test_text)
        
        print("\nGenerated summary:")
        print(summary)
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 