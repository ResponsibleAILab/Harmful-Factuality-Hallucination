"""Appendix B question generation recovered from QA_wikiEntity.py."""
from __future__ import annotations

import argparse
from evaluation.common import as_list, check_paths, entity_pairs, prepare_record, read_records, replace_entities, write_json
from evaluation.prompts import OPEN_QUESTION_SYSTEM, CLOSED_QUESTION_SYSTEM


def generate_questions(records, generate):
    results = []
    for row in records:
        pairs = entity_pairs(row)
        eligible = as_list(row.get("in_sum_entities", [])) or as_list(row.get("entities", []))
        selected = next((pair for entity in eligible for pair in pairs if pair[0] == entity), None)
        if selected is None:
            raise ValueError("QA generation needs an answer entity with a distinct replacement")
        original, perturbed = selected
        result = prepare_record(row)
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("QA generation requires the original text")
        open_prompt = (
            f"Based on the following text, create an open-ended question that has the answer: {original}"
            f"\n\nText: {text[:3000]}\n\n"
            "Return format should only include the question itself, without any explanations or prefixes."
        )
        closed_prompt = (
            "Based on the following text, create a question with two options A and B."
            f"\n\nText: {text[:3000]}\n\nOption A should be: {original}\nOption B should be: {perturbed}\n\n"
            "Design a question that can be answered using these two options. The question should relate "
            "to the text content but should not directly indicate which option is the correct answer. "
            "Only return the question content, do not include the options.\n\n"
            "Return format should be:\nQuestion content?"
        )
        for kind, prompt, system in (("open", open_prompt, OPEN_QUESTION_SYSTEM),
                                     ("close", closed_prompt, CLOSED_QUESTION_SYSTEM)):
            question = generate(prompt, system_message=system, max_tokens=512, temperature=0.7, top_p=1.0)
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question generator returned empty output")
            result[kind + "_question"] = question.strip()
            result["perturbed_" + kind + "_question"], _ = replace_entities(question.strip(), pairs)
        result.update(ori_answer=original, pert_answer=perturbed,
                      choice_A=original, choice_B=perturbed, metric_entity_pairs=[selected])
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args(argv)
    check_paths(args.input, args.output)
    records = read_records(args.input)
    from Models.client import create_generator
    generate = create_generator(args.model)
    result = []
    for row in records:
        result.extend(generate_questions([row], generate))
        write_json(args.output, result)
    if not records:
        write_json(args.output, [])
    print(f"Saved {len(result)} question records to {args.output}")


if __name__ == "__main__":
    main()
