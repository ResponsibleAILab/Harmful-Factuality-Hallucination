from __future__ import annotations

import argparse

from evaluation.prompts import DEFENSE_PROMPT, DEFENSE_PROMPT_V2
from run_tasks import run_task


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tasks with defense prompts.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON with text and entity_pairs.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--eval-output", type=str, required=True, help="Output txt report path.")
    parser.add_argument("--task", type=str, choices=["summary", "rephrase"], default="summary")
    parser.add_argument("--model", type=str, default="gpt4o")
    parser.add_argument("--entities-column", type=str, default="entities")
    parser.add_argument("--defense2", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    defense_prompt = DEFENSE_PROMPT_V2 if args.defense2 else DEFENSE_PROMPT
    limit = args.limit if args.limit > 0 else None
    run_task(
        input_path=args.input,
        output_path=args.output,
        task=args.task,
        model=args.model,
        entities_column=args.entities_column,
        defense_prompt=defense_prompt,
        limit=limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eval_output_path=args.eval_output,
    )


if __name__ == "__main__":
    main()
