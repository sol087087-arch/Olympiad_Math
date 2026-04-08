"""
repair_dataset.py — AIMO dataset repair + audit
Usage:
    python repair_dataset.py --input your_dataset.jsonl --output clean.jsonl

Fixes:
  - Invalid JSON escape sequences (bare LaTeX \\qquad, \\frac etc.)
  - Concatenated JSON objects on one line
  - Deduplication by 'index' field

Flags (written to flagged.jsonl + summary report):
  - NO_BOXED          — no \boxed{} in solution (proof-only or missing answer)
  - REFUTES_PROBLEM   — model says problem is false / gives counterexample
  - SELF_DOUBTING     — model contradicts itself mid-solution
  - MISSING_INDEX     — no 'index' field
  - TOO_SHORT         — solution under 300 chars (probably garbage)
  - NO_MESSAGES       — entry has no messages field
"""

import argparse
import json
import sys
from pathlib import Path


# ── JSON repair ──────────────────────────────────────────────────────────────

def fix_invalid_escapes(s: str) -> str:
    """Double backslashes that form invalid JSON escape sequences."""
    result = []
    i = 0
    bs = "\\"
    valid_after_bs = set('"' + bs + "/bfnrtu")
    while i < len(s):
        ch = s[i]
        if ch == bs:
            if i + 1 < len(s):
                next_ch = s[i + 1]
                if next_ch in valid_after_bs:
                    if next_ch == "u":
                        hex_part = s[i + 2 : i + 6]
                        if len(hex_part) == 4 and all(
                            c in "0123456789abcdefABCDEF" for c in hex_part
                        ):
                            result.append(ch)
                            result.append(next_ch)
                            i += 2
                            continue
                        else:
                            result.append(bs + bs)
                            i += 1
                            continue
                    result.append(ch)
                    result.append(next_ch)
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


def split_concatenated_json(line: str) -> list[str]:
    """Split a line that may contain multiple back-to-back JSON objects."""
    objects = []
    depth = 0
    in_str = False
    escape = False
    start = 0
    bs = "\\"
    for i, ch in enumerate(line):
        if escape:
            escape = False
            continue
        if ch == bs and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objects.append(line[start : i + 1])
                    start = i + 1
    return objects


def try_parse(line: str) -> tuple[list[dict], list[str]]:
    """
    Try to parse one raw line into one or more JSON objects.
    Returns (objects, errors).
    """
    line = line.strip()
    if not line:
        return [], []

    # 1. Direct parse
    try:
        return [json.loads(line)], []
    except json.JSONDecodeError:
        pass

    # 2. Fix escapes, try again
    fixed = fix_invalid_escapes(line)
    try:
        return [json.loads(fixed)], []
    except json.JSONDecodeError:
        pass

    # 3. Split concatenated objects, fix each part
    parts = split_concatenated_json(fixed)
    if len(parts) > 1:
        objects, errors = [], []
        for part in parts:
            part_fixed = fix_invalid_escapes(part)
            try:
                objects.append(json.loads(part_fixed))
            except json.JSONDecodeError as e:
                errors.append(f"concat-part parse failed: {e} | snippet: {part[:80]}")
        return objects, errors

    return [], [f"unfixable: {line[:80]}"]


# ── Content audit ─────────────────────────────────────────────────────────────

REFUTE_PHRASES = [
    "the statement is false",
    "the problem is false",
    "this is false",
    "a counterexample",
    "is incorrect as stated",
    "as stated is false",
    "no such",
    "cannot be proved",
]

SELF_DOUBT_PHRASES = [
    "wait, that's wrong",
    "actually, i made an error",
    "let me reconsider",
    "i was wrong",
    "this is incorrect",
]


def audit_entry(obj: dict) -> list[str]:
    flags = []

    if "messages" not in obj:
        flags.append("NO_MESSAGES")
        return flags

    if "index" not in obj:
        flags.append("MISSING_INDEX")

    msgs = obj["messages"]
    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    asst_lower = asst.lower()

    if len(asst) < 300:
        flags.append("TOO_SHORT")

    if r"\boxed" not in asst:
        flags.append("NO_BOXED")

    if any(p in asst_lower for p in REFUTE_PHRASES):
        flags.append("REFUTES_PROBLEM")

    if any(p in asst_lower for p in SELF_DOUBT_PHRASES):
        flags.append("SELF_DOUBTING")

    return flags


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output clean JSONL file")
    parser.add_argument(
        "--flagged",
        default=None,
        help="Output file for flagged entries (default: <output>.flagged.jsonl)",
    )
    parser.add_argument(
        "--flag-filter",
        nargs="*",
        default=None,
        help="Only write entries with these flags to flagged file (default: all flagged)",
    )
    parser.add_argument(
        "--keep-no-boxed",
        action="store_true",
        help="Keep NO_BOXED entries in clean output (proof-only problems are OK)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    flagged_path = Path(args.flagged) if args.flagged else output_path.with_suffix(".flagged.jsonl")

    # ── Read + repair ─────────────────────────────────────────────────────────
    raw_lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()

    all_entries: list[dict] = []
    parse_errors: list[tuple[int, list[str]]] = []

    print(f"Reading {input_path} ({len(raw_lines)} raw lines)...")
    for lineno, line in enumerate(raw_lines, 1):
        objects, errors = try_parse(line)
        all_entries.extend(objects)
        if errors:
            parse_errors.append((lineno, errors))

    print(f"  Parsed: {len(all_entries)} objects from {len(raw_lines)} lines")
    if parse_errors:
        print(f"  Parse errors on {len(parse_errors)} lines:")
        for lineno, errs in parse_errors[:10]:
            for e in errs:
                print(f"    Line {lineno}: {e}")
        if len(parse_errors) > 10:
            print(f"    ... and {len(parse_errors) - 10} more")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    seen_indices: set = set()
    unique_entries: list[dict] = []
    duplicates = 0

    for entry in all_entries:
        idx = entry.get("index")
        if idx is not None:
            if idx in seen_indices:
                duplicates += 1
                continue
            seen_indices.add(idx)
        unique_entries.append(entry)

    print(f"  Duplicates removed: {duplicates}")
    print(f"  Unique entries: {len(unique_entries)}")

    # ── Audit ─────────────────────────────────────────────────────────────────
    flag_counts: dict[str, int] = {}
    clean_entries: list[dict] = []
    flagged_entries: list[dict] = []

    for entry in unique_entries:
        flags = audit_entry(entry)
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1

        # Decide if entry goes to clean output
        blocking_flags = set(flags)
        if args.keep_no_boxed:
            blocking_flags.discard("NO_BOXED")
            blocking_flags.discard("MISSING_INDEX")  # keep it but note it

        severe = blocking_flags - {"MISSING_INDEX", "NO_BOXED"}
        # Always block: REFUTES_PROBLEM, SELF_DOUBTING, TOO_SHORT, NO_MESSAGES
        is_blocked = bool(severe - {"NO_BOXED"})
        if args.keep_no_boxed:
            is_blocked = bool(severe)

        if is_blocked:
            entry["_flags"] = flags
            flagged_entries.append(entry)
        else:
            if flags:
                entry["_flags"] = flags  # annotate but keep
            clean_entries.append(entry)

        # Write to flagged file if any flags
        # (we'll do this separately)

    # Flagged file = all entries that have any flags
    filter_set = set(args.flag_filter) if args.flag_filter else None
    flagged_for_file = []
    for entry in unique_entries:
        flags = entry.get("_flags", [])
        if flags:
            if filter_set is None or filter_set & set(flags):
                entry["_flags"] = flags
                flagged_for_file.append(entry)

    # ── Write outputs ─────────────────────────────────────────────────────────
    with output_path.open("w", encoding="utf-8") as f:
        for entry in clean_entries:
            # Remove internal audit key before writing
            out = {k: v for k, v in entry.items() if k != "_flags"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    with flagged_path.open("w", encoding="utf-8") as f:
        for entry in flagged_for_file:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("AUDIT REPORT")
    print("=" * 60)
    print(f"Total input lines:    {len(raw_lines)}")
    print(f"Total parsed objects: {len(all_entries)}")
    print(f"After dedup:          {len(unique_entries)}")
    print(f"Clean output:         {len(clean_entries)}  → {output_path}")
    print(f"Flagged:              {len(flagged_for_file)}  → {flagged_path}")
    print()
    print("Flag breakdown:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(len(unique_entries), 1)
        bar = "█" * int(pct / 2)
        print(f"  {flag:<20} {count:>6}  ({pct:5.1f}%)  {bar}")
    print()
    if parse_errors:
        print(f"⚠ {len(parse_errors)} lines could not be parsed at all (see above)")
    print("Done.")


if __name__ == "__main__":
    main()
