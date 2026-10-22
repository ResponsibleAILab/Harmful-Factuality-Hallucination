"""LLM-Instructed Entity Replacement, ported from perturb_entity_with_openai."""
from __future__ import annotations

import argparse
import json
from evaluation.common import as_list, check_paths, read_records, write_json
from evaluation.prompts import LIER_SYSTEM


def perturb_records(records, generate, model="gpt4o"):
    results = []
    for row in records:
        result = dict(row)
        result["entity_pairs"] = []
        result["perturbation_method"] = "LIER"
        result["perturbation_model"] = model
        for entity in as_list(row.get("entities", [])):
            if not isinstance(entity, str) or not entity:
                raise ValueError("entities must contain non-empty strings")
            prompt = (
                '# Example:\n# Input: entity="Albert Einstein"\n'
                '# Output: {"entity":"Albert Einstein","perturbed":"Isaac Newton"}\n'
                '# Input: entity="New York City"\n'
                '# Output: {"entity":"New York City","perturbed":"Los Angeles"}\n\n'
                f'Now process:\nOriginal entity: "{entity}"\n'
                "Return only the JSON object as specified above, no explanation."
            )
            response = generate(prompt, system_message=LIER_SYSTEM,
                                max_tokens=256, temperature=0.7, top_p=1.0)
            data = json.loads(response)
            replacement = data.get("perturbed")
            if data.get("entity", entity) != entity or not isinstance(replacement, str) or not replacement:
                raise ValueError("LIER response must identify the requested entity and its replacement")
            result["entity_pairs"].append([entity, replacement])
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sampled_wikientities.json")
    parser.add_argument("--output", default="data/perturbed_data.json")
    parser.add_argument("--model", default="gpt4o")
    args = parser.parse_args(argv)
    check_paths(args.input, args.output)
    records = read_records(args.input)
    from Models.client import create_generator
    generate = create_generator(args.model)
    results = []
    for row in records:
        results.extend(perturb_records([row], generate, args.model))
        write_json(args.output, results)
    if not records:
        write_json(args.output, [])
    print(f"Saved {len(results)} perturbed records to {args.output}")


if __name__ == "__main__":
    main()
