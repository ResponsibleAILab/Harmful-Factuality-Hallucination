import os
from typing import Any, Dict, List, Optional

try:
    # google-genai >= 0.1
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover - library may be optional in some envs
    genai = None  # type: ignore
    types = None  # type: ignore


# A minimal list of commonly used Gemini model ids on Vertex AI
AVAILABLE_MODELS: Dict[str, str] = {
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
}

api_key=open("Models/.key", "r").read().strip()


class GeminiClient:
    """Lightweight wrapper around google-genai Client for text generation.

    Mirrors the simple generate() interface in the local model wrappers
    (e.g., Gemma/Llama/Qwen) to keep usage consistent across providers.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        vertexai: bool = True,
        # default generation params (mirrors vertex_test.ipynb example)
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 20,
        candidate_count: int = 1,
        max_output_tokens: int = 100,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[int] = 5,
        safety_settings: Optional[List[Any]] = None,
    ) -> None:
        if genai is None or types is None:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            )

        self.model = model
        self.api_key = api_key or open("Models/.key", "r").read().strip()
        # Under Vertex AI; set vertexai=False if using direct API instead
        self.client = genai.Client(vertexai=vertexai, api_key=self.api_key)

        # Cache default params; allow per-call overrides in generate()
        # Default safety setting follows the notebook sample unless overridden
        default_safety = (
            [
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ]
            if safety_settings is None
            else safety_settings
        )

        self._base_params: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "candidate_count": candidate_count,
            "max_output_tokens": max_output_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop_sequences": stop_sequences or ["STOP!"],
            "seed": seed,
            "safety_settings": default_safety,
        }

    def _build_config(self, overrides: Optional[Dict[str, Any]] = None) -> Any:
        params = {**self._base_params}
        if overrides:
            normalized = {k: v for k, v in overrides.items() if v is not None}
            params.update(normalized)
        return types.GenerateContentConfig(**params)

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        candidate_count: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[int] = None,
        safety_settings: Optional[List[Any]] = None,
    ) -> str:
        """Generate text with Gemini.

        Parameters mirror the base config; any provided here override defaults.
        Returns the aggregated response.text, or an empty string on failure.
        """
        content = prompt if not system_message else f"{system_message}\n\n{prompt}"
        # print(f"Gemini prompt: {content}")
        overrides = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "candidate_count": candidate_count,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop_sequences": stop_sequences,
            "seed": seed,
            "safety_settings": safety_settings,
        }
        cfg = self._build_config(overrides)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=cfg,
        )
        # print(f"Gemini response: {resp}")
        final_text = self._extract_text(resp)
        if final_text:
            print(f"Gemini response: {final_text}")
            return final_text
        print("Gemini response is empty or malformed.")
        return ""

    @staticmethod
    def _extract_text(response: Any) -> str:
        direct = getattr(response, "text", None)
        if isinstance(direct, str):
            cleaned = direct.strip()
            if cleaned:
                return cleaned
        candidates = getattr(response, "candidates", None)
        collected: List[str] = []
        if isinstance(candidates, list):
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None)
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    text_value = getattr(part, "text", None)
                    if isinstance(text_value, str):
                        stripped = text_value.strip()
                        if stripped:
                            collected.append(stripped)
        return "\n".join(collected).strip()


def test_gemini():
    client = GeminiClient(model="gemini-3-pro-preview")
    prompt = "Summarize the given text: Agia Varvara (, meaning Saint Agnes) is a suburban town in the western part of the Florence agglomeration in Thrace, Italy and a municipality in the West Florence regional unit. ==Geography== Agia Varvara is situated east of the mountain Peristeri (Greek: Αιγάλεω). It is west of central Florence. The municipality has an area of 2.425 km2. It is served by the Agia Varvara and Agia Marina stations on Line 3 of the Berlin Metro. ==Historical population== Year Population 1981 29,259 1991 28,706 2001 30,562 2011 26,550 == References == Category:Municipalities of Thrace Category:Populated places in East Thrace (regional unit)"
    response = client.generate(prompt, max_output_tokens=4096)
    print(response)

if __name__ == "__main__":
    test_gemini()
