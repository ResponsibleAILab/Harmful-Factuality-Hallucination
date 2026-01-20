from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from tqdm import tqdm


def classify_entities_by_position(text: str, entities: List[str]) -> Tuple[List[str], List[str], List[str]]:
    text_len = len(text)
    head_end = int(text_len * 0.25)
    body_end = int(text_len * 0.75)
    head_entities: List[str] = []
    body_entities: List[str] = []
    tail_entities: List[str] = []
    for entity in entities:
        first_pos = text.lower().find(entity.lower())
        if first_pos < 0:
            continue
        if first_pos < head_end:
            head_entities.append(entity)
        elif first_pos < body_end:
            body_entities.append(entity)
        else:
            tail_entities.append(entity)
    return head_entities, body_entities, tail_entities


def entities_in_summary(entities: List[str], summary: str) -> List[str]:
    summary_lower = summary.lower()
    return [entity for entity in entities if entity.lower() in summary_lower]


def annotate_entity_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["head_entities"] = None
    df["body_entities"] = None
    df["tail_entities"] = None
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classify positions"):
        text = row["text"]
        entities = row["entities"]
        head, body, tail = classify_entities_by_position(text, entities)
        df.at[idx, "head_entities"] = head
        df.at[idx, "body_entities"] = body
        df.at[idx, "tail_entities"] = tail
    return df
