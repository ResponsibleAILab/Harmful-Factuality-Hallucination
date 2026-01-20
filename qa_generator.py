from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from evaluation.prompts import CLOSE_QA_GEN_TEMPLATE, OPEN_QA_GEN_TEMPLATE


EntityPair = Tuple[str, str]


def select_answer_entity(item: Dict) -> Optional[str]:
    in_sum = item.get("in_sum_entities")
    if isinstance(in_sum, list) and in_sum:
        return in_sum[0]
    entities = item.get("entities")
    if isinstance(entities, list) and entities:
        return entities[0]
    return None


def find_entity_pair(answer: str, entity_pairs: List[EntityPair]) -> Optional[EntityPair]:
    for original, perturbed in entity_pairs:
        if original == answer:
            return (original, perturbed)
    return None


def call_openai(prompt: str, model: str, system_message: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        top_p=1.0,
    )
    return response.choices[0].message.content.strip()


def generate_questions(
    input_path: str,
    output_path: str,
    model: str,
) -> None:
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    results: List[Dict] = []
    for item in tqdm(data, desc="Generating QA"):
        text = item.get("text", "")
        entity_pairs = item.get("entity_pairs", [])
        if not text or not entity_pairs:
            continue
        answer = select_answer_entity(item)
        if not answer:
            continue
        pair = find_entity_pair(answer, entity_pairs)
        if not pair:
            continue
        open_prompt = OPEN_QA_GEN_TEMPLATE.format(answer=answer, text=text[:3000])
        close_prompt = CLOSE_QA_GEN_TEMPLATE.format(
            text=text[:3000], original=pair[0], perturbed=pair[1]
        )
        open_q = call_openai(open_prompt, model=model)
        close_q = call_openai(close_prompt, model=model)
        record = dict(item)
        record["ori_answer"] = pair[0]
        record["pert_answer"] = pair[1]
        record["open_question"] = open_q
        record["close_question"] = close_q
        record["choice_A"] = pair[0]
        record["choice_B"] = pair[1]
        results.append(record)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA pairs.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4.1")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    generate_questions(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
