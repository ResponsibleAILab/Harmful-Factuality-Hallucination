from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Models.my_gemini import GeminiClient

MAX_TEXT_LEN: int = 8000


def format_item(item: Dict[str, object], summary: str, in_sum_entities: List[str]) -> Dict[str, object]:
    text = item.get("text", "")
    entities = item.get("entities", [])
    entity_ids = item.get("entityIDs", [])
    spans = item.get("spans", [])
    return {
        "text": text,
        "entities": entities,
        "entityIDs": entity_ids,
        "spans": spans,
        "summary": summary,
        "in_sum_entities": in_sum_entities,
    }


def summarize_text(text: str, client: GeminiClient) -> str:
    trimmed = text if len(text) <= MAX_TEXT_LEN else f"{text[:MAX_TEXT_LEN]}..."
    prompt = f"Summarize: {trimmed}"
    return client.generate(
        prompt=prompt,
        system_message="Summarize the given text.",
        max_output_tokens=1024,
    )


def extract_in_sum_entities(entities: List[object], summary: str) -> List[str]:
    summary_lower = summary.lower()
    results: List[str] = []
    for entity in entities:
        if isinstance(entity, str) and entity.lower() in summary_lower:
            results.append(entity)
    return results


def sample_dataset(dataset_name: str, sample_size: int, client: GeminiClient) -> List[Dict[str, object]]:
    dataset = load_dataset(dataset_name)
    results: List[Dict[str, object]] = []
    for item in dataset["train"]:
        entities = item.get("entities", [])
        if isinstance(entities, list) and len(entities) > 0:
            summary = summarize_text(item.get("text", ""), client)
            in_sum_entities = extract_in_sum_entities(entities, summary)
            results.append(format_item(item, summary, in_sum_entities))
        if len(results) >= sample_size:
            break
    return results


def save_json(data: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WikiEntities and sample records.")
    parser.add_argument("--output", type=str, default="data/sampled_wikientities.json")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="Sayankotor/WikiEntities")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    client = GeminiClient(model="gemini-3-pro-preview")
    records = sample_dataset(args.dataset, args.sample_size, client)
    save_json(records, args.output)


if __name__ == "__main__":
    main()
