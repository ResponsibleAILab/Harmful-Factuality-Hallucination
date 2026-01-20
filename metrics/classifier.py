from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm


EntityPair = Tuple[str, str]


@dataclass(frozen=True)
class EvaluationStats:
    total_pairs: int
    both_count: int
    only_original_count: int
    only_perturbed_count: int
    neither_count: int


def normalize_entity_pairs(value: object) -> List[EntityPair]:
    if isinstance(value, list):
        pairs: List[EntityPair] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                original, perturbed = str(item[0]).strip(), str(item[1]).strip()
                if original and perturbed:
                    pairs.append((original, perturbed))
        return pairs
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return normalize_entity_pairs(parsed)
    return []


def filter_entity_pairs(
    entity_pairs: Iterable[EntityPair],
    row_entities: object,
) -> List[EntityPair]:
    base_pairs = [(orig, pert) for orig, pert in entity_pairs if orig != pert]
    if isinstance(row_entities, list):
        allowed = {str(item).strip() for item in row_entities if str(item).strip()}
        return [(orig, pert) for orig, pert in base_pairs if orig in allowed]
    return base_pairs


def evaluate_dataframe(
    df: pd.DataFrame,
    text_column: str,
    entities_column: str,
) -> EvaluationStats:
    stats = {
        "total_pairs": 0,
        "both_count": 0,
        "only_original_count": 0,
        "only_perturbed_count": 0,
        "neither_count": 0,
    }
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="row"):
        text = row[text_column]
        if not isinstance(text, str) or not text.strip():
            continue
        entity_pairs = normalize_entity_pairs(row["entity_pairs"])
        filtered_pairs = filter_entity_pairs(entity_pairs, row.get(entities_column))
        if not filtered_pairs:
            continue
        text_lower = text.lower()
        for original, perturbed in filtered_pairs:
            stats["total_pairs"] += 1
            original_present = original.lower() in text_lower
            perturbed_present = perturbed.lower() in text_lower
            if original_present and perturbed_present:
                if original.lower() == perturbed.lower():
                    stats["only_perturbed_count"] += 1
                else:
                    stats["both_count"] += 1
            elif original_present:
                stats["only_original_count"] += 1
            elif perturbed_present:
                stats["only_perturbed_count"] += 1
            else:
                stats["neither_count"] += 1
    return EvaluationStats(
        total_pairs=stats["total_pairs"],
        both_count=stats["both_count"],
        only_original_count=stats["only_original_count"],
        only_perturbed_count=stats["only_perturbed_count"],
        neither_count=stats["neither_count"],
    )


def format_report(stats: EvaluationStats, record_count: int, type_label: str) -> str:
    total = stats.total_pairs
    percent = lambda value: (value / total * 100.0) if total else 0.0
    lines = [
        "========== Entity Perturbation Evaluation ==========",
        f"Records: {record_count}",
        f"Total pairs: {total}",
        "",
        f"Entity presence in {type_label}:",
        f"1. Both present: {stats.both_count} ({percent(stats.both_count):.2f}%)",
        f"2. Original only: {stats.only_original_count} ({percent(stats.only_original_count):.2f}%)",
        f"3. Perturbed only: {stats.only_perturbed_count} ({percent(stats.only_perturbed_count):.2f}%)",
        f"4. Neither: {stats.neither_count} ({percent(stats.neither_count):.2f}%)",
        "",
    ]
    return "\n".join(lines)


def save_report(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
