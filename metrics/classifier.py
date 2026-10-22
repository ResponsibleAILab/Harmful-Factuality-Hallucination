"""HelpfulHarm's case-insensitive substring classifier and four outcomes.

This is a lexical metric; it does not infer aliases or semantic equivalence.
"""
from evaluation.common import entity_pairs

CATEGORIES = ("both_present", "only_original", "only_perturbed", "neither_present")
INTERPRETATION = {
    "both_present": "Dual Presence", "only_original": "Harmful Factuality",
    "only_perturbed": "Faithful Adherence", "neither_present": "Entity Omission",
}


def classify_pair(output, original, perturbed):
    if not isinstance(output, str):
        raise ValueError("Generation output must be a string")
    if not original or not perturbed or original.casefold() == perturbed.casefold():
        raise ValueError("A metric pair needs distinct non-empty entities")
    original_present = original.casefold() in output.casefold()
    perturbed_present = perturbed.casefold() in output.casefold()
    if original_present and perturbed_present:
        return "both_present"
    if original_present:
        return "only_original"
    if perturbed_present:
        return "only_perturbed"
    return "neither_present"


def evaluate_records(records, output_field="summary", entities_column="entities"):
    counts = dict.fromkeys(CATEGORIES, 0)
    evaluated, failed = 0, 0
    results = []
    for row in records:
        result = dict(row)
        result["entity_evaluation"] = []
        output = row.get(output_field)
        if row.get("generation_error") or not isinstance(output, str) or not output.strip():
            failed += 1
            results.append(result)
            continue
        pairs = row.get("metric_entity_pairs", row.get("active_entity_pairs"))
        if pairs is None:
            pairs = entity_pairs(row, entities_column)
        for original, perturbed in pairs:
            category = classify_pair(output, original, perturbed)
            counts[category] += 1
            result["entity_evaluation"].append({"original": original, "perturbed": perturbed,
                                                 "category": category})
        evaluated += 1
        results.append(result)
    total = sum(counts.values())
    report = {"total_records": len(records), "evaluated_records": evaluated,
              "failed_records": failed, "total_entity_pairs": total,
              "metric": "case-insensitive literal substring presence", "categories": {}}
    for category, count in counts.items():
        report["categories"][category] = {"label": INTERPRETATION[category], "count": count,
                                             "percentage": 100 * count / total if total else 0.0}
    return results, report


def report_text(report):
    lines = ["HFH entity-presence evaluation", f"Metric: {report['metric']}",
             f"Records: {report['total_records']}; failed: {report['failed_records']}",
             f"Evaluated entity pairs: {report['total_entity_pairs']}"]
    lines.extend(f"{value['label']}: {value['count']} ({value['percentage']:.2f}%)"
                 for value in report["categories"].values())
    return "\n".join(lines) + "\n"
