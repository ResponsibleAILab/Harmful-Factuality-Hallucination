"""JSON input and entity replacement shared by the restored entry points."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import tempfile


def as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a list or a serialized list")
    return list(value)


def read_records(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError("Input must be a JSON array of objects (records orientation)")
    return data


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         delete=False) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def check_paths(*paths):
    resolved = [Path(path).resolve() for path in paths if path]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Input, output and report paths must be different")


def entity_pairs(row, entities_column="entities"):
    if "entity_pairs" not in row:
        raise ValueError("Each perturbed record needs entity_pairs")
    selected = None
    if entities_column != "entities":
        if entities_column not in row:
            raise ValueError(f"Missing requested entity column: {entities_column}")
        selected = set(as_list(row[entities_column]))
    result = []
    seen = set()
    for pair in as_list(row["entity_pairs"]):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each entity pair must contain original and replacement")
        original, replacement = pair
        if not isinstance(original, str) or not isinstance(replacement, str):
            raise ValueError("Entity pair entries must be strings")
        if not original or not replacement or original.casefold() == replacement.casefold():
            continue
        if selected is not None and original not in selected:
            continue
        if original in seen:
            if [original, replacement] not in result:
                raise ValueError(f"Conflicting replacements for entity: {original}")
            continue
        result.append([original, replacement])
        seen.add(original)
    return result


def replace_entities(text, pairs):
    """Replace longest matches simultaneously, without cascading replacements."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    mapping = dict(pairs)
    if not mapping:
        return text, []
    pattern = re.compile("|".join(re.escape(s) for s in sorted(mapping, key=len, reverse=True)))
    used = set()

    def substitute(match):
        original = match.group(0)
        used.add(original)
        return mapping[original]

    result = pattern.sub(substitute, text)
    return result, [pair for pair in pairs if pair[0] in used]


def prepare_record(row, entities_column="entities"):
    prepared = dict(row)
    pairs = entity_pairs(row, entities_column)
    if isinstance(row.get("text"), str):
        prepared["perturbed_text"], pairs = replace_entities(row["text"], pairs)
    elif not isinstance(row.get("perturbed_text"), str):
        raise ValueError("Each record needs text or perturbed_text")
    prepared["active_entity_pairs"] = pairs
    return prepared
