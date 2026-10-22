"""Restored summary, rephrase and QA experiment entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from evaluation.common import check_paths, prepare_record, read_records, write_json
from evaluation.prompts import task_prompt
from metrics.classifier import evaluate_records, report_text


def build_parser(defense=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--task", choices=("summary", "rephrase", "qa"), default="summary")
    parser.add_argument("--model", default="gpt4o")
    parser.add_argument("--entities-column", default="entities")
    parser.add_argument("--qa-type", choices=("open", "closed"), default="open")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate existing outputs without loading a model")
    parser.add_argument("--defense", action="store_true", default=defense)
    return parser


def run_pipeline(args, generate=None):
    check_paths(args.input, args.output, args.eval_output)
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    records = [prepare_record(row, args.entities_column) for row in read_records(args.input)]
    if not args.eval_only and records and generate is None:
        from Models.client import create_generator
        generate = create_generator(args.model)
    output_field = args.task if args.task != "qa" else "answer"
    for row in records:
        if args.task == "qa":
            answer = row.get("ori_answer")
            pairs = [pair for pair in row["active_entity_pairs"] if pair[0] == answer]
            if not pairs:
                raise ValueError("QA answer must have a replacement in the selected entity column")
            row["metric_entity_pairs"] = pairs[:1]
        if not args.eval_only:
            row.pop("generation_error", None)
            try:
                prompt, system = task_prompt(row, args.task, args.defense, args.qa_type)
                temperature = args.temperature if args.temperature is not None else (
                    0.0 if args.task == "qa" else 0.3)
                output = generate(prompt, system_message=system, max_tokens=args.max_tokens,
                                  temperature=temperature, top_p=1.0)
                if not isinstance(output, str) or not output.strip():
                    raise ValueError("Empty model output")
                row[output_field] = output.strip()
                row["generation_model"] = args.model
                row["defense_enabled"] = args.defense
            except Exception as error:
                row[output_field] = None
                row["generation_error"] = f"{type(error).__name__}: {error}"
            write_json(args.output, records)
        if args.task == "qa" and args.qa_type == "closed" and not row.get("generation_error"):
            raw = row.get("raw_answer", row.get("answer", ""))
            row["raw_answer"] = raw
            match = re.match(r"^\s*([AB])\b", raw or "", flags=re.I)
            if match:
                row["answer"] = row["choice_" + match.group(1).upper()]
            else:
                row["answer"] = None
                row["generation_error"] = "Invalid multiple-choice answer; expected A or B"
    evaluated, report = evaluate_records(records, output_field, args.entities_column)
    report.update(task=args.task, model=args.model, defense=args.defense,
                  entities_column=args.entities_column, qa_type=args.qa_type if args.task == "qa" else None)
    write_json(args.output, evaluated)
    target = Path(args.eval_output)
    if target.suffix.lower() == ".json":
        write_json(target, report)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_text(report), encoding="utf-8")
    return evaluated, report


def main(argv=None, defense=False):
    parser = build_parser(defense)
    args = parser.parse_args(argv)
    try:
        _, report = run_pipeline(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    print(report_text(report), end="")
    return 1 if report["failed_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
