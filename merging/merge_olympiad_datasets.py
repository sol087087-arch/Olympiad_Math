#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge two olympiad JSONL datasets with strict problem matching.

Rules:
- Problems are considered duplicates ONLY if the full problem text matches.
- If duplicates occur, keep the entry with the LONGER solution.
- Output a merged JSONL dataset.

Input format:
{"messages":[
    {"role":"user","content":"problem"},
    {"role":"assistant","content":"solution"}
]}
"""

import json
from pathlib import Path


# ===== INPUT FILES =====

FILE_A = r"C:\lora_training\OLympiad\combined_dataset_clean(2).jsonl"
FILE_B = r"C:\lora_training\OLympiad\olympiad_dataset.jsonl"

OUTPUT = r"C:\lora_training\OLympiad\olympiad_merged1.jsonl"


# ===== HELPERS =====

def load_jsonl(path):
    data = []
    errors = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" in obj and len(obj["messages"]) >= 2:
                    data.append(obj)
                else:
                    errors += 1
            except json.JSONDecodeError:
                errors += 1

    return data, errors


def problem_text(pair):
    return pair["messages"][0]["content"].strip()


def solution_text(pair):
    return pair["messages"][1]["content"].strip()


# ===== MERGE =====

def merge_datasets(data_a, data_b):

    merged = {}

    replaced = 0
    kept = 0
    new = 0

    for pair in data_a + data_b:

        prob = problem_text(pair)

        if prob not in merged:
            merged[prob] = pair
            new += 1
            continue

        # duplicate problem → keep longer solution
        existing = merged[prob]

        if len(solution_text(pair)) > len(solution_text(existing)):
            merged[prob] = pair
            replaced += 1
        else:
            kept += 1

    return list(merged.values()), new, replaced, kept


# ===== MAIN =====

def main():

    print("Loading datasets...")

    data_a, err_a = load_jsonl(FILE_A)
    data_b, err_b = load_jsonl(FILE_B)

    print(f"Dataset A: {len(data_a)} pairs  | errors: {err_a}")
    print(f"Dataset B: {len(data_b)} pairs  | errors: {err_b}")

    merged, new, replaced, kept = merge_datasets(data_a, data_b)

    print("\nMerging...")
    print(f"Unique problems: {len(merged)}")
    print(f"New problems:    {new}")
    print(f"Solutions replaced (longer): {replaced}")
    print(f"Solutions kept:  {kept}")

    print("\nSaving...")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for pair in merged:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nSaved merged dataset → {OUTPUT}")


if __name__ == "__main__":
    main()