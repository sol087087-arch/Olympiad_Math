#!/usr/bin/env python3
"""
repair_dataset_v2.py — CoT dataset cleaner (Helga edition)

Фиксит:
- кривой JSON (escape, склейка)
- дубликаты
- мусорные записи

НЕ ломает:
- proof-only задачи
- нормальный reasoning с самопоправками

Детектирует:
- TRUNCATED  — решение не завершено
- MULTIPLE_ANSWERS — несколько боксов с разными ответами
- REFUTES_PROBLEM — модель говорит что задача неверна
- SELF_DOUBTING — модель противоречит сама себе
"""

import json
import re
import hashlib
from pathlib import Path

# ====================== PATHS ======================

INPUT_PATH     = Path(r"C:\lora_training\OLympiad\cot_with_answer.jsonl")
OUTPUT_PATH    = Path(r"C:\lora_training\OLympiad\cot_with_answer_clean.jsonl")
FLAGGED_PATH   = Path(r"C:\lora_training\OLympiad\cot_with_answer_flagged.jsonl")
TRUNCATED_PATH = Path(r"C:\lora_training\OLympiad\cot_with_answer_truncated.jsonl")

# ====================== CONFIG ======================

KEEP_NO_BOXED = True
MIN_LEN = 100

REFUTE_PHRASES = [
    "the statement is false",
    "this is false",
    "is incorrect as stated",
    "the problem is incorrect",
    "no such",           # "no such n exists" когда модель опровергает
]

SELF_DOUBT_PHRASES = [
    "wait, that's wrong",
    "actually, i made an error", 
    "[corrected",
    "corrected to",
    "i was wrong",
    "this is incorrect",
]

FINAL_MARKERS = [
    r"\boxed",
    r"\blacksquare",
    r"\square",
    "**Answer**",
    "**answer**",
    "**Conclusion**",
    "**conclusion**",
    "the answer is",
    "thus the answer",
    "hence the answer",
    "therefore the answer",
    "we conclude",
    "thus we get",
    "hence we get",
    "the minimum is",
    "the maximum is",
    "the largest is",
    "the smallest is",
    "in summary",
    "to summarize",
    "qed",
    "q.e.d",
    "(A)", "(B)", "(C)", "(D)", "(E)",  # multiple choice answers
"the answer is (", 
"∎"
]

# ====================== JSON FIX ======================

def fix_invalid_escapes(s: str) -> str:
    result = []
    i = 0
    bs = "\\"
    valid = set('"' + bs + "/bfnrtu")

    while i < len(s):
        ch = s[i]
        if ch == bs:
            if i + 1 < len(s):
                nxt = s[i + 1]
                if nxt in valid:
                    if nxt == "u":
                        hex_part = s[i + 2 : i + 6]
                        if len(hex_part) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_part):
                            result.append(ch)
                            result.append(nxt)
                            i += 2
                        else:
                            result.append(bs + bs)
                            i += 1
                    else:
                        result.append(ch)
                        result.append(nxt)
                        i += 2
                else:
                    result.append(bs + bs)
                    i += 1
            else:
                result.append(bs + bs)
                i += 1
        else:
            result.append(ch)
            i += 1

    return "".join(result)


def split_json(line: str):
    objs = []
    depth = 0
    in_str = False
    escape = False
    start = 0

    for i, ch in enumerate(line):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
        if not in_str:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objs.append(line[start:i + 1])

    return objs


def parse_line(line: str):
    line = line.strip()
    if not line:
        return []

    try:
        return [json.loads(line)]
    except Exception:
        pass

    fixed = fix_invalid_escapes(line)
    try:
        return [json.loads(fixed)]
    except Exception:
        pass

    parts = split_json(fixed)
    out = []
    for p in parts:
        try:
            out.append(json.loads(fix_invalid_escapes(p)))
        except Exception:
            continue

    return out

# ====================== HELPERS ======================

def get_user_hash(entry):
    if "messages" not in entry:
        return None
    for m in entry["messages"]:
        if m.get("role") == "user":
            return hashlib.md5((m.get("content") or "").encode("utf-8")).hexdigest()
    return None


def audit(entry):
    flags = []

    if "messages" not in entry:
        flags.append("NO_MESSAGES")
        return flags

    asst = next((m.get("content") or "" for m in entry["messages"] if m.get("role") == "assistant"), "")

    if len(asst) < MIN_LEN:
        flags.append("TOO_SHORT")

    if r"\boxed" not in asst:
        flags.append("NO_BOXED")

    low = asst.lower()

    if any(p in low for p in REFUTE_PHRASES):
        flags.append("REFUTES_PROBLEM")

    if any(p in low for p in SELF_DOUBT_PHRASES):
        flags.append("SELF_DOUBTING")

    # ── Multiple boxed answers ────────────────────────────────────
    boxed_count = len(re.findall(r"\\boxed\{", asst))
    if boxed_count > 2:
        flags.append("MULTIPLE_ANSWERS")

    # ── Truncation check ──────────────────────────────────────────
    # Только два надёжных сигнала: нет финального маркера ИЛИ
    # несбалансированные скобки в хвосте (явный обрыв в формуле)
    tail = asst[-120:] if len(asst) >= 120 else asst
    has_final_marker = any(m in asst for m in FINAL_MARKERS)
    tail_unbalanced  = tail.count("{") > tail.count("}")

    if not has_final_marker or tail_unbalanced:
        flags.append("TRUNCATED")

    return flags

# ====================== MAIN ======================

def main():
    print(f"Loading: {INPUT_PATH}")

    lines = INPUT_PATH.read_text(encoding="utf-8", errors="replace").splitlines()

    all_entries = []
    for line in lines:
        all_entries.extend(parse_line(line))

    print(f"Parsed: {len(all_entries)} entries")

    # ===== Dedup =====
    seen = set()
    dedup = []

    for e in all_entries:
        key = e.get("index") or get_user_hash(e)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        dedup.append(e)

    print(f"After dedup: {len(dedup)}")

    # ===== Audit =====
    clean = []
    flagged = []
    truncated = []
    lengths = []
    flag_counts = {}

    for e in dedup:
        flags = audit(e)
        e["_flags"] = flags

        asst = ""
        if "messages" in e:
            asst = next((m.get("content") or "" for m in e["messages"] if m.get("role") == "assistant"), "")
            lengths.append(len(asst))

        for fl in flags:
            flag_counts[fl] = flag_counts.get(fl, 0) + 1

        # Routing:
        # TRUNCATED → отдельный файл, не в clean
        # severe (всё кроме NO_BOXED) → flagged
        # остальное → clean

        if "TRUNCATED" in flags:
            truncated.append(e)
            continue

        severe = set(flags) - {"NO_BOXED", "MISSING_INDEX"}
        if KEEP_NO_BOXED:
            severe -= {"NO_BOXED"}

        if severe:
            flagged.append(e)
        else:
            clean.append(e)

    # ===== Save =====
    def write_jsonl(path, entries):
        with path.open("w", encoding="utf-8") as f:
            for e in entries:
                out = {k: v for k, v in e.items() if k != "_flags"}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    write_jsonl(OUTPUT_PATH, clean)
    write_jsonl(FLAGGED_PATH, flagged)
    write_jsonl(TRUNCATED_PATH, truncated)

    # ===== Stats =====
    print("\n===== STATS =====")
    print(f"Clean:     {len(clean)}")
    print(f"Flagged:   {len(flagged)}")
    print(f"Truncated: {len(truncated)}")

    if lengths:
        print(f"Avg len:   {sum(lengths) // len(lengths)}")
        print(f"Max len:   {max(lengths)}")

    if flag_counts:
        print("\n===== FLAG BREAKDOWN =====")
        for k, v in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
            pct = 100 * v / max(len(dedup), 1)
            print(f"  {k:<22} {v:>6}  ({pct:5.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()