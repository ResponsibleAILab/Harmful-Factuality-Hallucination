from __future__ import annotations

import json
from typing import List, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT: str = """You are an expert text-perturbation assistant. 
        Your job: given an entity and its type (person, location, organization, etc.), 
        produce one substitute that:
        1. Is the same type.
        2. Is NOT a direct synonym, but has slight semantic shift.
        3. Maintains similar length and capitalization.
        4. Matches entity-type rules (e.g. person→similar name, location→similar scale).
        Output MUST be exactly one JSON object, one line, no extra keys, no code fences: {"entity": "original entity", "perturbed": "perturbed entity"}
    """


def build_user_prompt(entity: str) -> str:
    return f"""
                    # Example:
                    # Input: entity="Albert Einstein"
                    # Output: {{"entity":"Albert Einstein","perturbed":"Isaac Newton"}}
                    #
                    # Input: entity="New York City"
                    # Output: {{"entity":"New York City","perturbed":"Los Angeles"}}

                    Now process:
                    Original entity: "{entity}"
                    Return only the JSON object as specified above, no explanation.
                    """


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


def perturb_entities_lier(
    df: pd.DataFrame,
    model: str = "gpt-4o",
    system_prompt: str = SYSTEM_PROMPT,
) -> pd.DataFrame:
    df = df.copy()
    df["entity_pairs"] = None
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LIER perturbation"):
        entities = row["entities"]
        entity_pairs: List[Tuple[str, str]] = []
        for entity in entities:
            prompt = build_user_prompt(entity)
            response = call_openai(prompt, model=model, system_message=system_prompt)
            payload = json.loads(response)
            original = str(payload["entity"]).strip()
            perturbed = str(payload["perturbed"]).strip()
            entity_pairs.append((original, perturbed))
        df.at[idx, "entity_pairs"] = entity_pairs
    return df
