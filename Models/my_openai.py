import os
from openai import AzureOpenAI

class OpenAIClient:
    def __init__(self):
        self.endpoint = ""
        self.model_name = "gpt-4o"
        self.deployment = "gpt-4o"
        self.subscription_key = ""
        self.api_version = "2024-12-01-preview"
        
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )
    
    def gpt4o(self, prompt, system_message="You are a helpful assistant.", max_tokens=4096, temperature=1.0, top_p=1.0):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=self.deployment
        )
        
        return response.choices[0].message.content

    def gpt4_1(self, prompt, system_message="You are a helpful assistant.", max_tokens=4096, temperature=1.0, top_p=1.0):
        gpt4_1_client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )
        
        response = gpt4_1_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model="gpt-4.1"  
        )
        
        return response.choices[0].message.content

    def gpt4o_mini(self, prompt, system_message="You are a helpful assistant.", max_tokens=4096, temperature=1.0, top_p=1.0):
        gpt4o_mini_client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )
        
        response = gpt4o_mini_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model="gpt-4o-mini"  
        )
        
        return response.choices[0].message.content

    def gpto1(self, prompt, system_message="You are a helpful assistant.", max_tokens=4096, temperature=None, top_p=1.0):
        o1_client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )
        
        params = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_completion_tokens": max_tokens,
            "top_p": top_p,
            "model": "o1" 
        }
        
        
        response = o1_client.chat.completions.create(**params)
        
        return response.choices[0].message.content

    def o4_mini(self, prompt, system_message="You are a helpful assistant.", max_tokens=4096, temperature=None, top_p=1.0):

        o4_mini_client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )
        
        params = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_completion_tokens": max_tokens,
            "top_p": top_p,
            "model": "o4-mini"  # 使用o4-mini模型部署
        }
        
        response = o4_mini_client.chat.completions.create(**params)
        
        return response.choices[0].message.content

def main():
    openai_helper = OpenAIClient()
    
    prompt = "Write me a bubble sort algorithm in Python."
    response = openai_helper.gpt4o(prompt)
    
    print("Question:", prompt)
    print("\nGPT-4o:")
    print(response)
    
    o1_response = openai_helper.gpto1(prompt)
    print("\nGPT-o1:")
    print(o1_response)
    

    gpt4_1_response = openai_helper.gpt4_1(prompt)
    print("\nGPT-4.1:")
    print(gpt4_1_response)
    
    gpt4o_mini_response = openai_helper.gpt4o_mini(prompt)
    print("\nGPT-4o-mini:")
    print(gpt4o_mini_response)
    
    o4_mini_response = openai_helper.o4_mini(prompt)
    print("\no4-mini:")
    print(o4_mini_response)
    

if __name__ == "__main__":
    main()