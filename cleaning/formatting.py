import json
from pathlib import Path

INPUT_PATH = r"C:\lora_training\OLympiad\clean_merged888.jsonl"
OUTPUT_PATH = r"C:\lora_training\OLympiad\converted_for_sft.jsonl"

SYSTEM_PROMPT = """You are a mathematical olympiad solver. Given a problem, produce a complete and rigorous solution.
Rules:
- Be mathematically correct and justify every nontrivial step.
- Do not include failed attempts or self-corrections.
- Use precise mathematical language.
- End with a clearly stated final answer. For numeric answers, use \\boxed{}."""

def convert_example(item):
    if not isinstance(item, dict) or "messages" not in item:
        return None
    messages = item["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    user_text = None
    assistant_text = None

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = str(m.get("content", "")).strip()

        if role == "user" and user_text is None:
            user_text = content
        elif role == "assistant":
            assistant_text = content

    if not user_text or not assistant_text:
        return None

    full_text = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
Problem:
{user_text}
<|im_end|>
<|im_start|>assistant
Solution:
{assistant_text}
<|im_end|>"""

    return {"text": full_text}


print("Converting...")
count_in = 0
count_out = 0

with open(INPUT_PATH, encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        count_in += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        converted = convert_example(item)
        if converted is None:
            continue

        fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
        count_out += 1

print(f"Done.")
print(f"Input lines: {count_in}")
print(f"Valid converted: {count_out}")
print(f"Saved to: {OUTPUT_PATH}")