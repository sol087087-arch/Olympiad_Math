#!/usr/bin/env python3
"""
Generate CoT dataset from olympiad problems using Claude API.

Reads olympiad_merged.jsonl, restructures each problem+solution
into Key Insights + Solution Plan + Full Solution + Answer format.

Usage:
  python generate_cot_dataset.py --key sk-ant-YOUR_KEY --input olympiad_merged.jsonl --output cot_dataset.jsonl
  python generate_cot_dataset.py --key sk-ant-YOUR_KEY --input olympiad_merged.jsonl --output cot_dataset.jsonl --limit 10
"""

import json
import asyncio
import argparse
import time
from pathlib import Path
import anthropic

SYSTEM_PROMPT = """You are an expert mathematics educator specializing in olympiad problems.
Your task is to take an olympiad problem and its solution, and restructure
them into a detailed educational format.

Given a problem and its solution, produce exactly this structure:

**Key Insights**
List the core mathematical ideas, theorems, and principles needed to solve
this problem. For each insight, explain WHY it applies here — not just what it is.

**Solution Plan**
A numbered list of steps showing the logical path to the solution.
Each step should be one clear action.

**Full Solution**
The complete solution written following the plan, with all steps explained.
Follow the original solution closely — do not elaborate beyond what is necessary
to make each step clear.
Every non-obvious transition must be justified.

**Answer**
The final answer, clearly stated.

Rules:
- Keep all mathematical notation in LaTeX
- If the original solution appears incomplete or unclear, expand it using correct mathematical reasoning
- If you detect an error in the original solution, correct it and note the correction explicitly with [CORRECTED]
- Do not add information not present in the original solution unless necessary for clarity
- Be precise and rigorous"""


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_checkpoint(checkpoint_path: Path) -> set[int]:
    """Load set of already processed indices."""
    if not checkpoint_path.exists():
        return set()
    done = set()
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(int(line))
                except ValueError:
                    pass
    return done


async def process_record(
    client: anthropic.AsyncAnthropic,
    record: dict,
    index: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    problem = record["messages"][0]["content"]
    solution = record["messages"][1]["content"]

    user_message = f"""Problem:
{problem}

Original solution:
{solution}"""

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                cot_response = response.content[0].text

                return {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": cot_response},
                    ],
                    "original_solution": solution,
                    "index": index,
                }

            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                print(f"  [Rate limit] index={index}, waiting {wait}s...")
                await asyncio.sleep(wait)

            except anthropic.APIError as e:
                print(f"  [API error] index={index}, attempt={attempt+1}: {e}")
                await asyncio.sleep(5)

        print(f"  [FAILED] index={index} after 3 attempts")
        return None


async def main_async(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint")

    records = load_records(input_path)
    if args.limit:
        records = records[:args.limit]

    done_indices = load_checkpoint(checkpoint_path)
    print(f"Total records: {len(records)}")
    print(f"Already done: {len(done_indices)}")
    print(f"Remaining: {len(records) - len(done_indices)}")

    client = anthropic.AsyncAnthropic(api_key=args.key)
    semaphore = asyncio.Semaphore(args.parallel)

    # Open output file in append mode
    out_file = open(output_path, "a", encoding="utf-8")
    checkpoint_file = open(checkpoint_path, "a")

    tasks = []
    indices = []
    for i, record in enumerate(records):
        if i not in done_indices:
            tasks.append(process_record(client, record, i, semaphore))
            indices.append(i)

    completed = 0
    failed = 0
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1

        if result is not None:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            checkpoint_file.write(f"{result['index']}\n")
            checkpoint_file.flush()
        else:
            failed += 1

        if completed % 10 == 0:
            elapsed = time.time() - start_time
            rate = completed / elapsed
            remaining = len(tasks) - completed
            eta = remaining / rate if rate > 0 else 0
            print(f"  Progress: {completed}/{len(tasks)} | "
                  f"Failed: {failed} | "
                  f"Rate: {rate:.1f}/s | "
                  f"ETA: {eta/60:.1f}min")

    out_file.close()
    checkpoint_file.close()

    print(f"\nDone! Completed: {completed - failed}, Failed: {failed}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Anthropic API key")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records (for testing)")
    parser.add_argument("--parallel", type=int, default=5, help="Parallel requests (default: 5)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
