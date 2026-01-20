from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.prompts import (
    DEFAULT_SYSTEM_MESSAGE,
    DEFENSE_PROMPT,
    DEFENSE_PROMPT_V2,
    build_task_prompt,
)
from metrics.classifier import evaluate_dataframe, format_report, save_report


EntityPair = Tuple[str, str]


class OpenAIChatClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI()

    def generate(
        self,
        prompt: str,
        system_message: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.model in {"o1", "o4-mini"}:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()


class HFChatClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
        return generated[len(prompt_text) :].strip()


class GeminiChatClient:
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        from google import genai
        from google.genai import types

        self.genai = genai
        self.types = types
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models.")
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(
        self,
        prompt: str,
        system_message: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        content = prompt if not system_message else f"{system_message}\n\n{prompt}"
        cfg = self.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=cfg,
        )
        direct = getattr(resp, "text", None)
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        parts: List[str] = []
        candidates = getattr(resp, "candidates", None)
        if isinstance(candidates, list):
            for candidate in candidates:
                content_obj = getattr(candidate, "content", None)
                part_list = getattr(content_obj, "parts", None)
                if isinstance(part_list, list):
                    for part in part_list:
                        text_value = getattr(part, "text", None)
                        if isinstance(text_value, str) and text_value.strip():
                            parts.append(text_value.strip())
        return "\n".join(parts).strip()


def normalize_entity_pairs(value: object) -> List[EntityPair]:
    if isinstance(value, list):
        return [(str(p[0]), str(p[1])) for p in value if isinstance(p, (list, tuple)) and len(p) == 2]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return normalize_entity_pairs(parsed)
    return []


def replace_entities(
    text: str,
    entity_pairs: List[EntityPair],
    allowed_entities: Optional[List[str]],
) -> str:
    pairs = entity_pairs
    if allowed_entities is not None:
        allowed = set(allowed_entities)
        pairs = [(orig, pert) for orig, pert in pairs if orig in allowed]
    pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
    perturbed_text = text
    for original, perturbed in pairs:
        if original and perturbed:
            perturbed_text = perturbed_text.replace(original, perturbed)
    return perturbed_text


def get_client(model: str):
    if model.startswith("gemini"):
        return GeminiChatClient(model=model)
    if model.startswith("meta-llama") or model.startswith("mistralai") or model.startswith("Qwen"):
        return HFChatClient(model_id=model)
    mapped = {
        "gpt4o": "gpt-4o",
        "gpt4_1": "gpt-4.1",
        "gpt4o_mini": "gpt-4o-mini",
        "o1": "o1",
        "o4_mini": "o4-mini",
    }
    return OpenAIChatClient(model=mapped.get(model, model))


def run_task(
    input_path: str,
    output_path: str,
    task: str,
    model: str,
    entities_column: str,
    defense_prompt: Optional[str],
    limit: Optional[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    eval_output_path: str,
) -> None:
    df = pd.read_json(input_path, orient="records")
    if limit:
        df = df.head(limit)
    df = df.copy()
    df["entity_pairs"] = df["entity_pairs"].apply(normalize_entity_pairs)

    client = get_client(model)
    outputs: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {task}"):
        text = row["text"]
        entity_pairs = row["entity_pairs"]
        allowed = row.get(entities_column) if entities_column != "entities" else None
        perturbed_text = replace_entities(text, entity_pairs, allowed)
        prompt = build_task_prompt(task, perturbed_text, defense_prompt)
        response = client.generate(
            prompt=prompt,
            system_message=DEFAULT_SYSTEM_MESSAGE,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs.append(response)

    output_column = task
    df[output_column] = outputs
    df.to_json(output_path, orient="records", force_ascii=False, indent=2)

    stats = evaluate_dataframe(df, output_column, entities_column)
    report = format_report(stats, len(df), task)
    save_report(eval_output_path, report)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summarization or rephrase tasks.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON with text and entity_pairs.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--eval-output", type=str, required=True, help="Output txt report path.")
    parser.add_argument("--task", type=str, choices=["summary", "rephrase"], default="summary")
    parser.add_argument("--model", type=str, default="gpt4o")
    parser.add_argument("--entities-column", type=str, default="entities")
    parser.add_argument("--defense", action="store_true")
    parser.add_argument("--defense2", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    defense_prompt = None
    if args.defense2:
        defense_prompt = DEFENSE_PROMPT_V2
    elif args.defense:
        defense_prompt = DEFENSE_PROMPT
    limit = args.limit if args.limit > 0 else None
    run_task(
        input_path=args.input,
        output_path=args.output,
        task=args.task,
        model=args.model,
        entities_column=args.entities_column,
        defense_prompt=defense_prompt,
        limit=limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eval_output_path=args.eval_output,
    )


if __name__ == "__main__":
    main()
