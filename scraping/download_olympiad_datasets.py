"""
Скачивает олимпийские датасеты с HuggingFace и конвертирует в JSONL формат.
Запуск: python download_olympiad_datasets.py

Требования:
    pip install datasets huggingface_hub tqdm
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("C:/lora_training/olympiad_new")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_DATASET = Path("C:/lora_training/combined_dataset_clean_2.jsonl")


def load_existing_keys(path):
    """Загружает ключи существующего датасета для дедупликации."""
    keys = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    keys.add(d["messages"][0]["content"][:120])
                except:
                    pass
    print(f"Existing dataset: {len(keys)} entries")
    return keys


def to_sft_entry(question, solution):
    """Конвертирует пару вопрос/решение в SFT формат."""
    return {
        "messages": [
            {"role": "user", "content": f"Solve the following olympiad problem:\n\n{question.strip()}"},
            {"role": "assistant", "content": solution.strip()}
        ]
    }


def is_valid(question, solution, min_solution_len=100):
    """Базовая проверка качества."""
    if not question or not solution:
        return False
    if len(solution) < min_solution_len:
        return False
    # Выбрасываем если решение содержит Python-код (скорее всего мусор)
    if "```python" in solution and "sympy" in solution:
        return False
    if "Traceback" in solution or "Error" in solution:
        return False
    return True


# ============================================================
# 1. Omni-MATH (~4k задач, чистый олимпийский уровень)
# ============================================================
def download_omni_math(existing_keys):
    print("\n=== Omni-MATH ===")
    from datasets import load_dataset

    try:
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    except Exception as e:
        print(f"Failed: {e}")
        return []

    print(f"Downloaded: {len(ds)} entries")
    print("Columns:", ds.column_names)
    print("Sample:", {k: str(v)[:80] for k, v in ds[0].items()})

    entries = []
    skipped_dupe = 0
    skipped_invalid = 0

    for row in tqdm(ds, desc="Omni-MATH"):
        question = row.get("problem", "") or row.get("question", "")
        solution = row.get("solution", "") or row.get("answer", "")

        if not is_valid(question, solution):
            skipped_invalid += 1
            continue

        key = f"Solve the following olympiad problem:\n\n{question.strip()}"[:120]
        if key in existing_keys:
            skipped_dupe += 1
            continue

        entries.append(to_sft_entry(question, solution))
        existing_keys.add(key)

    print(f"New entries: {len(entries)} | Dupes: {skipped_dupe} | Invalid: {skipped_invalid}")
    return entries


# ============================================================
# 2. NuminaMath-CoT (860k, фильтруем только олимпийские источники)
# ============================================================
NUMINA_OLYMPIAD_SOURCES = {
    "amc_aime", "aops_forum", "cn_contest", "olympiads",
    "imo_shortlist", "usamo", "putnam"
}

def download_numina(existing_keys, max_entries=5000):
    print("\n=== NuminaMath-CoT (olympiad subset) ===")
    from datasets import load_dataset

    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    except Exception as e:
        print(f"Failed: {e}")
        return []

    entries = []
    skipped_dupe = 0
    skipped_source = 0
    skipped_invalid = 0
    seen = 0

    for row in tqdm(ds, desc="NuminaMath", total=max_entries):
        if len(entries) >= max_entries:
            break

        seen += 1

        # Фильтр по источнику
        source = (row.get("source", "") or "").lower()
        if not any(s in source for s in NUMINA_OLYMPIAD_SOURCES):
            skipped_source += 1
            continue

        question = row.get("problem", "") or row.get("question", "")
        solution = row.get("solution", "") or row.get("messages", [{}])[-1].get("content", "")

        if not is_valid(question, solution, min_solution_len=200):
            skipped_invalid += 1
            continue

        key = f"Solve the following olympiad problem:\n\n{question.strip()}"[:120]
        if key in existing_keys:
            skipped_dupe += 1
            continue

        entries.append(to_sft_entry(question, solution))
        existing_keys.add(key)

    print(f"Scanned: {seen} | New: {len(entries)} | Wrong source: {skipped_source} | Invalid: {skipped_invalid} | Dupes: {skipped_dupe}")
    return entries


# ============================================================
# 3. MATH dataset (Hendrycks) — только уровни 4-5
# ============================================================
MATH_HARD_SUBJECTS = {
    "algebra", "number_theory", "geometry",
    "counting_and_probability", "intermediate_algebra", "precalculus"
}

def download_math_hard(existing_keys):
    print("\n=== MATH dataset (level 4-5) ===")
    from datasets import load_dataset

    try:
        ds = load_dataset("hendrycks/competition_math", split="train+test")
    except Exception as e:
        print(f"Failed: {e}")
        return []

    print(f"Downloaded: {len(ds)} entries")

    entries = []
    skipped = 0

    for row in tqdm(ds, desc="MATH"):
        level = str(row.get("level", "")).replace("Level ", "").strip()
        if level not in ("4", "5"):
            skipped += 1
            continue

        subject = (row.get("type", "") or "").lower().replace(" ", "_")
        # Пропускаем precalculus — слишком вычислительный
        if subject == "precalculus":
            skipped += 1
            continue

        question = row.get("problem", "")
        solution = row.get("solution", "")

        if not is_valid(question, solution):
            skipped += 1
            continue

        key = f"Solve the following olympiad problem:\n\n{question.strip()}"[:120]
        if key in existing_keys:
            skipped += 1
            continue

        entries.append(to_sft_entry(question, solution))
        existing_keys.add(key)

    print(f"New entries: {len(entries)} | Skipped: {skipped}")
    return entries


# ============================================================
# Основной запуск
# ============================================================
def main():
    existing_keys = load_existing_keys(EXISTING_DATASET)

    all_new = []

    # Качаем по очереди
    all_new += download_omni_math(existing_keys)
    all_new += download_math_hard(existing_keys)
    all_new += download_numina(existing_keys, max_entries=5000)

    print(f"\nTotal new entries: {len(all_new)}")

    # Сохраняем отдельно (не перезаписываем существующий датасет)
    out_path = OUTPUT_DIR / "hf_olympiad_new.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in all_new:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved to: {out_path}")
    print(f"\nЧтобы добавить в основной датасет:")
    print(f"  cat {out_path} >> {EXISTING_DATASET}")
    print(f"\nПосле этого проверь дедупликацию:")
    print(f"  python dedup_check.py")


if __name__ == "__main__":
    main()
