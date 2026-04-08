#!/usr/bin/env python3
"""
Parse all Evan Chen .tex olympiad files into a JSONL training dataset.

Usage:
    python parse_tex_to_dataset.py --input ./evanchen_tex --output ./my_dataset.jsonl
    python parse_tex_to_dataset.py --input ./evanchen_tex  # output defaults to ./olympiad_dataset.jsonl

Supports all known Evan Chen formats:
    - IMO/USAMO/EGMO/JMO/APMO-YYYY-notes.tex
    - sols-TST-IMO-YYYY.tex
    - sols-TSTST-YYYY.tex
    - sols-ELMO-YYYY.tex
    - any other file in the same mdframed/tcolorbox format
"""

import os
import re
import json
import glob
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# LaTeX normalization
# ─────────────────────────────────────────────────────────────

def clean_latex(text):
    # Evan Chen macros → standard LaTeX
    text = text.replace(r'\dang', r'\angle')
    text = re.sub(r'\\dg(?![a-zA-Z])', r'^{\\circ}', text)
    text = re.sub(r'\\ol\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\ul\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\wt\{([^}]*)\}', r'\\widetilde{\1}', text)
    text = re.sub(r'\\wh\{([^}]*)\}', r'\\widehat{\1}', text)
    text = re.sub(r'\\vocab\{([^}]*)\}', r'\1', text)  # vocabulary terms
    text = re.sub(r'\\paragraph\{([^}]*)\}', r'\n\1\n', text)

    # Blackboard bold shortcuts
    for cmd, letter in [('RR','R'),('ZZ','Z'),('QQ','Q'),('NN','N'),('CC','C'),('FF','F'),('KK','K')]:
        text = text.replace(f'\\{cmd}', f'\\mathbb{{{letter}}}')

    # Common Evan Chen abbreviations
    text = text.replace(r'\half', r'\frac{1}{2}')
    text = re.sub(r'\\cbrt\{([^}]*)\}', r'\\sqrt[3]{\1}', text)

    # Remove leftover figure environments (asymptote, tikz, etc.)
    text = re.sub(r'\\begin\{center\}.*?\\end\{center\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{asy\}.*?\\end\{asy\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', text, flags=re.DOTALL)

    # Remove \label, \index commands
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\index\{[^}]*\}', '', text)

    # Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ─────────────────────────────────────────────────────────────
# Problem/solution extraction
# ─────────────────────────────────────────────────────────────

# Patterns for problem boxes Evan Chen uses
PROBLEM_BOX_PATTERNS = [
    # mdframed with frametitle containing "Problem"
    r'\\begin\{mdframed\}[^\n]*?Problem[^\n]*\n(.*?)\\end\{mdframed\}',
    # tcolorbox with title containing "Problem"  
    r'\\begin\{tcolorbox\}[^\n]*?Problem[^\n]*\n(.*?)\\end\{tcolorbox\}',
    # plain mdframed (no title)
    r'\\begin\{mdframed\}(.*?)\\end\{mdframed\}',
    # problem environment
    r'\\begin\{problem\}(.*?)\\end\{problem\}',
    # boxed problem
    r'\\begin\{boxedproblem\}(.*?)\\end\{boxedproblem\}',
]

# Compiled once
PROBLEM_BOX_RE = [re.compile(p, re.DOTALL | re.IGNORECASE) for p in PROBLEM_BOX_PATTERNS]

# Where a solution block ends
SOLUTION_END_RE = re.compile(
    r'\\(?:pagebreak|newpage|clearpage)|\\subsection\{|\\section\{|\\end\{document\}',
    re.IGNORECASE
)


def extract_pairs_from_file(filepath):
    """
    Extract (problem, solution) pairs from a single .tex file.
    Returns list of dicts with 'messages' key.
    """
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  [!] Could not read {filepath}: {e}")
        return []

    source_name = Path(filepath).stem
    pairs = []

    # Split into blocks by \subsection (each subsection = one problem)
    blocks = re.split(r'\\subsection\{', content)

    for block in blocks[1:]:  # skip preamble
        # Try each pattern to find the problem box
        problem_text = None
        problem_end = 0

        for pattern in PROBLEM_BOX_RE:
            m = pattern.search(block)
            if m:
                problem_text = clean_latex(m.group(1).strip())
                problem_end = m.end()
                break

        if not problem_text or len(problem_text) < 40:
            continue

        # Solution = everything after problem box until next major delimiter
        after = block[problem_end:]
        end_m = SOLUTION_END_RE.search(after)
        solution_raw = after[:end_m.start()] if end_m else after
        solution_text = clean_latex(solution_raw)

        if len(solution_text) < 80:
            continue

        pairs.append({
            "source": source_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"Solve the following olympiad problem:\n\n{problem_text}"
                },
                {
                    "role": "assistant",
                    "content": solution_text
                }
            ]
        })

    return pairs


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse Evan Chen .tex files into JSONL dataset")
    parser.add_argument("--input", "-i", default=".",
                        help="Directory containing .tex files (searched recursively)")
    parser.add_argument("--output", "-o", default="./olympiad_dataset.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--min-problem-len", type=int, default=40,
                        help="Skip problems shorter than this (chars)")
    parser.add_argument("--min-solution-len", type=int, default=80,
                        help="Skip solutions shorter than this (chars)")
    parser.add_argument("--with-source", action="store_true",
                        help="Include 'source' field in output (default: messages only)")
    args = parser.parse_args()

    # Find all .tex files
    tex_files = sorted(glob.glob(os.path.join(args.input, "**", "*.tex"), recursive=True))
    if not tex_files:
        print(f"No .tex files found in {args.input}")
        return

    print(f"Found {len(tex_files)} .tex files in {args.input}")
    print()

    all_pairs = []
    seen_keys = set()  # dedup within this run
    total_dupes = 0

    for fp in tex_files:
        fname = os.path.basename(fp)
        pairs = extract_pairs_from_file(fp)

        new_count = 0
        for p in pairs:
            key = p['messages'][0]['content'][:120]
            if key in seen_keys:
                total_dupes += 1
                continue
            seen_keys.add(key)
            all_pairs.append(p)
            new_count += 1

        status = f"{new_count} pairs" if new_count > 0 else "0 pairs (nothing parsed)"
        print(f"  {fname}: {status}")

    print()
    print(f"Total pairs:     {len(all_pairs)}")
    print(f"Dupes skipped:   {total_dupes}")
    print(f"Writing to:      {args.output}")

    with open(args.output, 'w', encoding='utf-8') as f:
        for p in all_pairs:
            out = {"messages": p["messages"]}
            if args.with_source:
                out["source"] = p["source"]
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

    print("Done.")


if __name__ == "__main__":
    main()
