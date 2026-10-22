"""Instantiate a provider only when an experiment actually needs generation."""
from __future__ import annotations

import os
from pathlib import Path


def create_generator(model):
    if model.startswith("gemini"):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        key_file = Path(__file__).with_name(".key")
        if not key and key_file.is_file():
            key = key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("Set GEMINI_API_KEY or put the Gemini API key in Models/.key")
        client = genai.Client(api_key=key)

        def generate(prompt, system_message, max_tokens=1024, temperature=0.3, top_p=1.0):
            response = client.models.generate_content(model=model, contents=prompt, config={
                "system_instruction": system_message, "max_output_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p,
            })
            if not response.text or not response.text.strip():
                raise RuntimeError("Gemini returned no text")
            return response.text
        return generate
    if model.startswith("meta-llama/") or model.startswith("llama"):
        from .my_llama import LlamaClient, AVAILABLE_MODELS
        return LlamaClient(AVAILABLE_MODELS.get(model, model)).generate
    if model.startswith("Qwen/") or model.startswith("qwen"):
        from .my_qwen import QwenClient, AVAILABLE_MODELS
        return QwenClient(AVAILABLE_MODELS.get(model, model)).generate
    from .my_openai import OpenAIClient
    return OpenAIClient(model).generate
