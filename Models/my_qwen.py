import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse

# Available Qwen text models
AVAILABLE_MODELS = {
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct"
}

class QwenClient:
    def __init__(self, model_id="Qwen/Qwen2-7B-Instruct", use_flash_attention=False):
        """
        Initialize Qwen model
        
        Parameters:
        - model_id: Model ID to load, default is Qwen/Qwen2-7B-Instruct
        - use_flash_attention: Whether to use flash attention for better acceleration
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {model_id}...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto"
        }
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        print("Model loading complete!")
    
    def generate(self, prompt, system_message=None, max_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate text using Qwen model
        
        Parameters:
        - prompt: User input prompt
        - system_message: System message to guide model behavior
        - max_tokens: Maximum number of tokens to generate
        - temperature: Parameter for generation diversity
        - top_p: Nucleus sampling parameter, controls randomness of generation
        
        Returns:
        - Generated text
        """
        # Build conversation
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            # Default system message
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        
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
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract model response (remove user input part)
        response = generated_text[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):]
        
        # Print generation statistics
        elapsed_time = time.time() - start_time
        print(f"Generation time: {elapsed_time:.2f} seconds")
        
        return response.strip()
    
    def summarize(self, text, max_tokens=256):
        """
        Summarize text using Qwen model
        
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
    Test Qwen model functionality
    """
    # Simple command line argument for model selection
    parser = argparse.ArgumentParser(description='Run Qwen model for text generation')
    parser.add_argument('--model', type=str, choices=['7b', '1.5b'], default='7b',
                        help='Model size to use: 7b or 1.5b')
    parser.add_argument('--flash', action='store_true', help='Use flash attention for better performance')
    args = parser.parse_args()
    
    # Map argument to model ID
    if args.model == '1.5b':
        model_id = AVAILABLE_MODELS['qwen2-1.5b']
    else:  # Default to 7b
        model_id = AVAILABLE_MODELS['qwen2-7b']
    
    print(f"Using model: {model_id}")
    
    # Initialize Qwen client
    try:
        qwen_client = QwenClient(model_id=model_id, use_flash_attention=args.flash)
        
        # Test text generation
        prompt = "Explain the concept of quantum computing in simple terms."
        system_message = "You are a helpful assistant that explains complex concepts in simple terms."
        
        print("\n=== Testing Text Generation ===")
        print(f"Prompt: {prompt}")
        
        # Generate response
        response = qwen_client.generate(
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
        AI technology has made significant advances. Large language models can generate human-like text, while computer vision systems can
        identify objects in images, sometimes more accurately than humans. While AI brings many benefits, such as improved medical diagnostics and autonomous vehicles,
        it also raises concerns about privacy, bias, and the future of employment. As AI continues to develop, balancing its potential benefits and risks will be an important task for society.
        """
        
        print("\n=== Testing Text Summarization ===")
        print(f"Length of text to summarize: {len(test_text)} characters")
        
        summary = qwen_client.summarize(test_text)
        
        print("\nGenerated summary:")
        print(summary)
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 