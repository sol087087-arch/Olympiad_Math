#!/usr/bin/env python3
"""
Remove records from large dataset that exist in small dataset.
Deduplication is based on the user message content (the problem text).

Usage:
  python dedup_filter.py --large olympiad_final_MATH1.jsonl --small olympiad_merged1.jsonl --output olympiad_new.jsonl
"""

import json
import argparse
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Skipping bad line: {e}")
    return records


def get_problem_text(record: dict) -> str:
    """Extract problem text from record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", required=True, help="Large dataset JSONL")
    parser.add_argument("--small", required=True, help="Small dataset JSONL (to exclude)")
    parser.add_argument("--output", required=True, help="Output JSONL")
    args = parser.parse_args()

    large_path = Path(args.large)
    small_path = Path(args.small)
    output_path = Path(args.output)

    print(f"Loading small dataset: {small_path}")
    small_records = load_jsonl(small_path)
    small_problems = {get_problem_text(r) for r in small_records}
    print(f"  {len(small_records)} records, {len(small_problems)} unique problems")

    print(f"\nLoading large dataset: {large_path}")
    large_records = load_jsonl(large_path)
    print(f"  {len(large_records)} records")

    print(f"\nFiltering...")
    kept = []
    skipped = 0
    for record in large_records:
        problem = get_problem_text(record)
        if problem in small_problems:
            skipped += 1
        else:
            kept.append(record)

    print(f"  Skipped (duplicates): {skipped}")
    print(f"  Kept (new records):   {len(kept)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
