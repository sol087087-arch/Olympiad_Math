#!/usr/bin/env python3

import json
import asyncio
import argparse
import time
from pathlib import Path
from openai import AsyncOpenAI

SYSTEM_PROMPT = """You are an expert mathematics educator specializing in olympiad problems.
Your task is to take an olympiad problem and its solution, and restructure
them into a detailed educational format.

Given a problem and its solution, produce exactly this structure:

**Key Insights**
List the core mathematical ideas, theorems, and principles needed.

**Solution Plan**
Numbered logical steps.

**Full Solution**
Complete rigorous derivation.

**Answer**
Final result clearly stated.

Rules:
- Keep LaTeX
- Fix mistakes if present
- Be precise
"""


def load_records(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records


def load_checkpoint(path: Path):
    if not path.exists():
        return set()

    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(int(line.strip()))
            except:
                pass
    return done


async def process_record(client, record, index, semaphore):

    problem = record["messages"][0]["content"]
    solution = record["messages"][1]["content"]

    user_message = f"""Problem:
{problem}

Original solution:
{solution}
"""

    async with semaphore:
        for attempt in range(3):
            try:

                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=4000,
                )

                cot = response.choices[0].message.content

                return {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": cot},
                    ],
                    "original_solution": solution,
                    "index": index,
                }

            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"[retry] index={index} {e} wait={wait}s")
                await asyncio.sleep(wait)

        print(f"[FAILED] {index}")
        return None


async def main_async(args):

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint")

    records = load_records(input_path)

    if args.limit:
        records = records[:args.limit]

    done = load_checkpoint(checkpoint_path)

    print("Total:", len(records))
    print("Already done:", len(done))
    print("Remaining:", len(records) - len(done))

    client = AsyncOpenAI(
        api_key=args.key,
        base_url="https://api.deepseek.com"
    )

    semaphore = asyncio.Semaphore(args.parallel)

    out = open(output_path, "a", encoding="utf-8")
    checkpoint = open(checkpoint_path, "a")

    tasks = []
    indices = []

    for i, r in enumerate(records):
        if i not in done:
            tasks.append(process_record(client, r, i, semaphore))
            indices.append(i)

    start = time.time()
    completed = 0
    failed = 0

    for coro in asyncio.as_completed(tasks):

        result = await coro
        completed += 1

        if result:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            checkpoint.write(str(result["index"]) + "\n")
            checkpoint.flush()

        else:
            failed += 1

        if completed % 10 == 0:

            elapsed = time.time() - start
            rate = completed / elapsed

            remaining = len(tasks) - completed
            eta = remaining / rate if rate else 0

            print(
                f"Progress: {completed}/{len(tasks)} | "
                f"Failed: {failed} | "
                f"Rate: {rate:.2f}/s | "
                f"ETA: {eta/60:.1f}min"
            )

    out.close()
    checkpoint.close()

    print("\nDone!")
    print("Completed:", completed - failed)
    print("Failed:", failed)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--key", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel", type=int, default=10)

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()