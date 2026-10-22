"""Lazy model imports keep CLI help and offline evaluation usable without GPUs."""

__all__ = ["OpenAIClient", "LlamaClient", "QwenClient"]


def __getattr__(name):
    if name == "OpenAIClient":
        from .my_openai import OpenAIClient
        return OpenAIClient
    if name == "LlamaClient":
        from .my_llama import LlamaClient
        return LlamaClient
    if name == "QwenClient":
        from .my_qwen import QwenClient
        return QwenClient
    raise AttributeError(name)
