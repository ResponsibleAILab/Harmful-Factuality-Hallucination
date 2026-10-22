"""Entity position and theme selection recovered from wikiEntity_exp.py."""
from evaluation.common import as_list


def classify_entities_by_position(text, entities):
    """First occurrence: head 0–25%, body 25–75%, tail 75–100%."""
    groups = {"head_entities": [], "body_entities": [], "tail_entities": []}
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    for entity in as_list(entities):
        if not isinstance(entity, str) or not entity:
            continue
        position = text.casefold().find(entity.casefold())
        if position < 0:
            continue
        key = "head_entities" if position < int(len(text) * .25) else (
            "body_entities" if position < int(len(text) * .75) else "tail_entities")
        groups[key].append(entity)
    return groups


def entities_in_summary(entities, summary):
    """The legacy theme proxy is literal entity presence in the clean summary."""
    return [entity for entity in as_list(entities)
            if isinstance(entity, str) and entity and entity.casefold() in summary.casefold()]


def enrich_record(row):
    result = dict(row)
    result["entities"] = as_list(row.get("entities", []))
    result.update(classify_entities_by_position(row["text"], result["entities"]))
    if isinstance(row.get("summary"), str):
        result["in_sum_entities"] = entities_in_summary(result["entities"], row["summary"])
        result["theme_entities"] = result["in_sum_entities"]
    return result
