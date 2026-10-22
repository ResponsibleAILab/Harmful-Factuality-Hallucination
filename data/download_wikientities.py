"""Rebuild a WikiEntities sample; the original 1,000-row sample was not recovered."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import check_paths, read_records, write_json
from perturbation.entity_selector import enrich_record
from evaluation.prompts import SUMMARY_SYSTEM


def prepare_records(records, generate=None):
    result = []
    for row in records:
        prepared = enrich_record(row)
        if generate is not None:
            summary = generate(f"Summarize: {row['text']}", system_message=SUMMARY_SYSTEM,
                               max_tokens=1024, temperature=0.3, top_p=1.0)
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Summary generator returned empty output")
            prepared["summary"] = summary
            prepared = enrich_record(prepared)
        result.append(prepared)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Prepare an existing records JSON instead of downloading")
    parser.add_argument("--output", default="data/sampled_wikientities.json")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="Sayankotor/WikiEntities")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"))
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip paid summary generation; only positional groups are created")
    args = parser.parse_args(argv)
    check_paths(args.input, args.output)
    if args.sample_size < 1:
        parser.error("sample-size must be positive")
    if args.input:
        records = read_records(args.input)[:args.sample_size]
    else:
        from datasets import load_dataset
        dataset = load_dataset(args.dataset, split="train")
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.sample_size, len(dataset))))
        records = [dict(row) for row in dataset]
    generate = None
    if not args.skip_enrichment:
        from Models.client import create_generator
        generate = create_generator(args.model)
    result = []
    for row in records:
        result.extend(prepare_records([row], generate))
        write_json(args.output, result)
    if not records:
        write_json(args.output, [])
    print(f"Saved {len(result)} prepared records to {args.output}")


if __name__ == "__main__":
    main()
