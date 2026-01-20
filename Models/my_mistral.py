from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class MistralClient:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pipeline = None
        
        self.model = None
        self.tokenizer = None
    
    def _load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = pipeline(
                "text-generation", 
                model=self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
        return self.pipeline
    
    def _load_model_and_tokenizer(self):
        if self.model is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        return self.model, self.tokenizer
    
    def mistral_7b(self, prompt: str, system_message: str = "You are a helpful assistant.", 
                   max_tokens: int = 1024, temperature: float = 1.0, top_p: float = 1.0):

        pipeline = self._load_pipeline()
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        response = pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=False
        )
        
        return response[0]['generated_text']
    
    def mistral_7b_advanced(self, prompt: str, system_message: str = "You are a helpful assistant.",
                           max_tokens: int = 1024, temperature: float = 1.0, top_p: float = 1.0):

        model, tokenizer = self._load_model_and_tokenizer()
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        generated_text = response[len(input_text):].strip()
        
        return generated_text
    
    def mistral_function_calling(self, prompt: str, tools: list = None, 
                                system_message: str = "You are a helpful assistant.",
                                max_tokens: int = 1024, temperature: float = 0.0):

        model, tokenizer = self._load_model_and_tokenizer()
        
        conversation = [{"role": "user", "content": prompt}]
        
        if tools:
            inputs = tokenizer.apply_chat_template(
                conversation,
                tools=tools,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        else:
            inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response


def main():
    mistral_client = MistralClient()
    
    prompt = "Write me a bubble sort algorithm in Python."
    
    print("Question:", prompt)
    print("\n" + "="*50)
    
    print("Mistral-7B Response:")
    response = mistral_client.mistral_7b(prompt)
    print(response)
    
    print("\n" + "="*50)
    
    print("Mistral-7B Advanced Control Response:")
    advanced_response = mistral_client.mistral_7b_advanced(prompt, temperature=0.7)
    print(advanced_response)
    
    print("\n" + "="*50)
    
    def get_current_weather(location: str, format: str):
        pass
    
    weather_prompt = "What's the weather like in New York City today?"
    tools = [get_current_weather]
    
    print("Function Calling Test:")
    print("User Question:", weather_prompt)
    function_response = mistral_client.mistral_function_calling(weather_prompt, tools=tools)
    print("Mistral-7B Function Calling Response:")
    print(function_response)


if __name__ == "__main__":
    main() 