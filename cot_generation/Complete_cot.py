#!/usr/bin/env python3
"""
Complete truncated CoT solutions using Claude API.

Usage:
  python complete_cot.py --key sk-ant-YOUR_KEY --input cot_no_answer.jsonl --output cot_completed.jsonl
"""

import json
import asyncio
import argparse
import time
from pathlib import Path
import anthropic

SYSTEM_PROMPT = """You are an expert mathematics educator specializing in olympiad problems.
Continue and complete the solution that has been started.

Rules:
- Do NOT repeat or restate what was already written — continue seamlessly from where it cuts off
- Keep all mathematical notation in LaTeX
- Be concise: avoid padding, restatements, or unnecessary elaboration
- The full response (including what was already written) must stay well under 4000 tokens
- The solution must end with a clearly stated **Answer** section
- If the Key Insights and Solution Plan are already complete, go straight to the Full Solution or Answer
- If you detect an error in the partial solution, correct it and note [CORRECTED]"""


def load_records(path):
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


def load_checkpoint(checkpoint_path):
    if not Path(checkpoint_path).exists():
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


def estimate_tokens(text):
    return len(text) // 4


async def process_record(client, record, semaphore):
    problem = record["messages"][0]["content"]
    partial = record["messages"][1]["content"].rstrip()
    idx = record["index"]

    partial_tokens = estimate_tokens(partial)
    max_new_tokens = max(800, min(3000, 4000 - partial_tokens - 200))

    user_msg = (
        f"{problem}\n\n"
        f"A solution has been partially written below. "
        f"Continue it seamlessly from exactly where it cuts off. "
        f"Do NOT repeat anything already written.\n\n"
        f"--- PARTIAL SOLUTION (do not repeat) ---\n{partial}"
    )

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=max_new_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                full_response = partial + response.content[0].text
                return {
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": full_response},
                    ],
                    "index": idx,
                }

            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                print(f"  [RateLimit] index={idx}, waiting {wait}s...")
                await asyncio.sleep(wait)

            except anthropic.APIError as e:
                print(f"  [APIError] index={idx}, attempt={attempt+1}: {e}")
                await asyncio.sleep(5)

        print(f"  [FAILED] index={idx}")
        return None


async def main_async(args):
    input_path = args.input
    output_path = args.output
    checkpoint_path = str(Path(output_path).with_suffix(".checkpoint"))

    records = load_records(input_path)
    if args.limit:
        records = records[:args.limit]

    done_indices = load_checkpoint(checkpoint_path)
    remaining = [r for r in records if r["index"] not in done_indices]

    print(f"Total: {len(records)} | Done: {len(done_indices)} | Remaining: {len(remaining)}")

    client = anthropic.AsyncAnthropic(api_key=args.key)
    semaphore = asyncio.Semaphore(args.parallel)

    out_file = open(output_path, "a", encoding="utf-8")
    ckpt_file = open(checkpoint_path, "a")

    tasks = [process_record(client, r, semaphore) for r in remaining]
    completed = 0
    failed = 0
    start = time.time()

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if result is not None:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            ckpt_file.write(f"{result['index']}\n")
            ckpt_file.flush()
        else:
            failed += 1

        if completed % 10 == 0 or completed == len(tasks):
            elapsed = time.time() - start
            rate = completed / elapsed
            eta = (len(tasks) - completed) / rate if rate > 0 else 0
            print(f"  {completed}/{len(tasks)} | failed={failed} | {rate:.1f}/s | ETA {eta/60:.1f}min")

    out_file.close()
    ckpt_file.close()
    print(f"\nDone. Completed: {completed - failed}, Failed: {failed}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
