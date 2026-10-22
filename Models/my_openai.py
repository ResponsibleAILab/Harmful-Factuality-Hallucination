"""OpenAI/Azure adapter preserving HelpfulHarm's method names.

Credentials are supplied through environment variables. No credentials from the
private development repository are included in this reconstruction.
"""
from __future__ import annotations

import os

MODEL_ALIASES = {
    "gpt4o": "gpt-4o", "gpt4_1": "gpt-4.1", "gpt4.1": "gpt-4.1",
    "gpt4o_mini": "gpt-4o-mini", "gpto1": "o1", "o4_mini": "o4-mini",
}


class OpenAIClient:
    def __init__(self, model="gpt4o", client=None):
        self.model = MODEL_ALIASES.get(model, model)
        if client is not None:
            self.client = client
            return
        from openai import OpenAI, AzureOpenAI
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            key = os.environ.get("AZURE_OPENAI_API_KEY")
            version = os.environ.get("OPENAI_API_VERSION") or os.environ.get("AZURE_OPENAI_API_VERSION")
            if not key or not version:
                raise ValueError("Azure requires AZURE_OPENAI_API_KEY and OPENAI_API_VERSION")
            self.client = AzureOpenAI(api_key=key, azure_endpoint=endpoint,
                                      api_version=version, max_retries=3, timeout=120)
        else:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY, or configure the Azure environment variables")
            self.client = OpenAI(api_key=key, max_retries=3, timeout=120)

    def generate(self, prompt, system_message="You are a helpful assistant.",
                 max_tokens=1024, temperature=0.3, top_p=1.0, model=None):
        model = MODEL_ALIASES.get(model, model) if model else self.model
        env_name = "AZURE_DEPLOYMENT_" + model.upper().replace("-", "_").replace(".", "_")
        deployment = os.environ.get(env_name) or model
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}]
        params = {"model": deployment, "messages": messages}
        if model.startswith(("o1", "o3", "o4")):
            messages[0]["role"] = "developer"
            params["max_completion_tokens"] = max_tokens
        else:
            params.update(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        response = self.client.chat.completions.create(**params)
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Model returned no text (refusal or empty response)")
        return text

    def gpt4o(self, prompt, **kwargs):
        return self.generate(prompt, model="gpt-4o", **kwargs)

    def gpt4_1(self, prompt, **kwargs):
        return self.generate(prompt, model="gpt-4.1", **kwargs)

    def gpt4o_mini(self, prompt, **kwargs):
        return self.generate(prompt, model="gpt-4o-mini", **kwargs)

    def gpto1(self, prompt, **kwargs):
        return self.generate(prompt, model="o1", **kwargs)

    def o4_mini(self, prompt, **kwargs):
        return self.generate(prompt, model="o4-mini", **kwargs)
