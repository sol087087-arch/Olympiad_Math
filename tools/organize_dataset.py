"""
organize_dataset.py
====================
Разбирает хаос в C:\\lora_training\\OLympiad\\ :

1. Раскладывает файлы по папкам:
   _raw/              — сырые источники (не трогаем содержимое)
   _merged_protected/ — все файлы с "merged" в имени (не трогаем)
   complete/          — файлы с (в основном) полными решениями
   olympiad/          — олимпиадные файлы
   broken/            — обрублённые / без ответа
   flagged/           — помечены как плохие / rejected

2. Читает ВСЕ файлы кроме _raw и _merged_protected,
   для каждого примера определяет качество решения (heuristic):
     GOOD   = есть \boxed{} + достаточно длинное + нормально заканчивается
     BROKEN = обрублено / слишком короткое / нет \boxed{}

3. Дедуплицирует по хешу текста задачи.

4. Пишет в OUTPUT_DIR:
   good_solutions_deduped.jsonl   — уникальные хорошие решения
   broken_solutions_deduped.jsonl — уникальные сломанные
   dpo_pairs.jsonl                — DPO пары (та же задача: good vs broken)

ВАЖНО: оригинальные файлы НЕ удаляются и НЕ перемещаются.
       Скрипт только ЧИТАЕТ их и пишет новые файлы в OUTPUT_DIR.
       После проверки результатов — сами решите что делать с оригиналами.
"""

import os
import re
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ===================== CONFIG =====================
BASE_DIR    = Path(r"C:\lora_training\OLympiad")
OUTPUT_DIR  = Path(r"C:\lora_training\OLympiad\_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Файлы которые мы ПРОПУСКАЕМ (сырые источники) ───────────────────────────
RAW_FILES = {
    "khan_raw_dataset.jsonl",
    "math_dataset.jsonl",
    "olympiad_dataset.jsonl",
    "aops_dataset.jsonl",
    "elmo_dataset.jsonl",
    "elmo_dataset_chatml.jsonl",
    "batch_requests.jsonl",
    "khan_olympiad_subset.jsonl",
    "khan_filtered_dataset.jsonl",
    "cot_gptoss.jsonl",
    "cot_gptoss - Copy.jsonl",
}

# ─── Категории файлов (для информации / будущего перемещения) ─────────────────
FILE_CATEGORIES = {
    # complete — чистые, полные решения
    "complete": {
        "cot_with_answer_clean.jsonl",
        "cot_with_answer1_clean.jsonl",
        "cot_with_answer3_clean.jsonl",
        "cot_with_answer773_clean_flagged_clean.jsonl",
        "resolved_by_sonnet_clean.jsonl",
        "cot_broken_clean.jsonl",
        "cot_broken2_clean.jsonl",
        "cot_completed7777_clean.jsonl",
        "cot_completed7777_clean1_clean.jsonl",
        "cot_completed77773_clean.jsonl",
        "cot_completed7777_re_fixed.jsonl",
        "combined_dataset_clean_2(1).jsonl",
        "combined_dataset_clean(2).jsonl",
        "converted_for_sft.jsonl",
        "cot_with_answer.jsonl",
    },
    # olympiad — олимпиадные задачи
    "olympiad": {
        "olympiad_final_MATH1.jsonl",
        "olympiad_final_MATH.jsonl",
        "olympiad_new.jsonl",
        "olympiad_final.jsonl",
        "olympiad_cot.jsonl",
    },
    # broken — обрублённые / без ответа
    "broken": {
        "cot_broken.jsonl",
        "cot_broken_truncated.jsonl",
        "cot_broken_clean_lagged.jsonl",
        "cot_with_answer_truncated.jsonl",
        "cot_with_answer1_truncated.jsonl",
        "cot_with_answer1_lagged.jsonl",
        "cot_no_answer.jsonl",
        "cot_no_answer1_truncated.jsonl",
        "cot_no_answer1_clean.jsonl",
        "cot_no_answer1_lagged.jsonl",
        "cot_broken73_clean_flagged_clean.jsonl",
        "cot3_truncated.jsonl",
        "cot_completed7777_clean_truncated.jsonl",
        "cot_completed7777_clean_lagged.jsonl",
    },
    # flagged — помечены как проблемные / rejected
    "flagged": {
        "cot_with_answer_flagged.jsonl",
        "cot_with_answer_clean_flagged.jsonl",
        "cot_with_answer773_clean_flagged.jsonl",
        "cot_with_answer773_clean_flagged_manual_check.jsonl",
        "cot_with_answer773_clean_flagged_sample30.jsonl",
        "cot_with_answer773_clean_flagged_auto_drop.jsonl",
        "cot_broken73_clean_flagged.jsonl",
        "cot_broken73_clean_flagged_manual_check.jsonl",
        "cot_broken73_clean_flagged_sample30.jsonl",
        "cot_broken73_clean_flagged_auto_drop.jsonl",
        "resolved_by_sonnet_flagged.jsonl",
        "resolved_by_sonnet_truncated.jsonl",
        "resolved_failed.jsonl",
        "cot_completed7777_rejected.jsonl",
        "cot_completed77773_clean.flagged.jsonl",
        "cot_completed7777_clean.flagged.jsonl",
        "cot_completed7777_re.jsonl",
        "cot_completed7777_filtered.jsonl",
        "cot_completed7777.jsonl",
    },
}

# ===================== HELPERS =====================

def extract_parts(example: dict):
    """
    Извлекает (system, user, assistant) из примера.
    Поддерживает два формата:
      1. {"text": "<|im_start|>..."} — ChatML строка
      2. {"messages": [{"role": "user", ...}, ...]} — HuggingFace messages
    """
    # Формат 1: text = ChatML строка
    if "text" in example:
        text = example["text"]
        system_m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", text, re.DOTALL)
        user_m   = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>",   text, re.DOTALL)
        asst_m   = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
        system    = system_m.group(1).strip() if system_m else ""
        user      = user_m.group(1).strip()   if user_m   else ""
        assistant = asst_m.group(1).strip()   if asst_m   else ""
        return system, user, assistant

    # Формат 2: messages list
    if "messages" in example:
        msgs = example["messages"]
        system    = next((m["content"] for m in msgs if m.get("role") == "system"),    "")
        user      = next((m["content"] for m in msgs if m.get("role") == "user"),      "")
        assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        return (system or "").strip(), (user or "").strip(), (assistant or "").strip()

    return "", "", ""


def to_chatml_text(system: str, user: str, assistant: str) -> str:
    """Конвертирует parts обратно в ChatML строку (единый формат вывода)."""
    out = ""
    if system:
        out += f"<|im_start|>system\n{system}<|im_end|>\n"
    out += f"<|im_start|>user\nProblem:\n{user}<|im_end|>\n"
    out += f"<|im_start|>assistant\nSolution:\n{assistant}<|im_end|>\n"
    return out


def problem_hash(user_text: str) -> str:
    """SHA256 от нормализованного текста задачи."""
    normalized = re.sub(r"\s+", " ", user_text.strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def score_solution(solution: str) -> dict:
    """
    Оценивает качество решения.
    Возвращает словарь с флагами и итоговым классом: 'good' | 'broken'.
    """
    words      = solution.split()
    word_count = len(words)
    has_boxed  = r"\boxed{" in solution or "\\boxed{" in solution

    # Проверяем нормальное завершение (последние 300 символов)
    tail = solution[-300:].lower()
    ending_signals = [
        r"\boxed{", "therefore", "thus", "hence", "∎", "q.e.d",
        "the answer is", "the solution is", "we conclude",
        "finally,", "in conclusion", "answer:"
    ]
    has_ending = any(s in tail for s in ending_signals)

    # Признаки обрыва / проблемы
    truncation_signals = [
        "...", "… ", "let me", "wait,", "hmm,", "actually,\ni",
        "i need to reconsider", "i made an error",
    ]
    seems_truncated = any(t in solution[-100:].lower() for t in truncation_signals)

    # Решение: GOOD если достаточно длинное + есть нормальный конец
    # (или хотя бы \boxed{} и не слишком короткое)
    if word_count >= 120 and has_ending and not seems_truncated:
        quality = "good"
    elif word_count >= 80 and has_boxed and not seems_truncated:
        quality = "good"
    else:
        quality = "broken"

    return {
        "quality":        quality,
        "word_count":     word_count,
        "has_boxed":      has_boxed,
        "has_ending":     has_ending,
        "seems_truncated": seems_truncated,
    }


def load_jsonl(path: Path):
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠ JSON error line {i+1} in {path.name}: {e}")
    except Exception as e:
        print(f"  ✗ Cannot read {path.name}: {e}")
    return examples


def write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✔ Written {len(records):,} examples → {path.name}")

# ===================== MAIN =====================

def main():
    all_files = sorted(BASE_DIR.glob("*.jsonl"))

    # Статистика
    stats = defaultdict(int)
    skipped_files = []

    # Три пула:
    #   good_pool[problem_hash]  = лучший "good" пример (с метаданными)
    #   broken_pool[problem_hash]= лучший "broken" пример
    good_pool   = {}   # hash → {"text": ..., "word_count": ..., "source": ...}
    broken_pool = {}   # hash → {"text": ..., "word_count": ..., "source": ...}

    print(f"\n{'='*60}")
    print(f"Сканируем {len(all_files)} файлов в {BASE_DIR}")
    print(f"{'='*60}\n")

    for fpath in all_files:
        fname = fpath.name

        # Пропускаем защищённые merged файлы
        if "merged" in fname.lower():
            print(f"[PROTECTED] {fname}")
            skipped_files.append(fname)
            stats["skipped_merged"] += 1
            continue

        # Пропускаем сырые источники
        if fname in RAW_FILES:
            print(f"[RAW]       {fname}")
            skipped_files.append(fname)
            stats["skipped_raw"] += 1
            continue

        # Определяем категорию файла
        file_cat = "unknown"
        for cat, names in FILE_CATEGORIES.items():
            if fname in names:
                file_cat = cat
                break

        examples = load_jsonl(fpath)
        good_count   = 0
        broken_count = 0
        skipped_ex   = 0

        for ex in examples:
            _, user_text, assistant_text = extract_parts(ex)

            # Если не распарсилось — пропускаем
            if not user_text or not assistant_text:
                skipped_ex += 1
                continue

            # Убираем префикс "Problem:\n" если есть
            problem_text = re.sub(r"^Problem:\s*", "", user_text, flags=re.IGNORECASE).strip()
            p_hash       = problem_hash(problem_text)

            # Убираем префикс "Solution:\n" из ответа
            solution = re.sub(r"^Solution:\s*", "", assistant_text, flags=re.IGNORECASE).strip()

            sc = score_solution(solution)

            # Единый формат вывода — ChatML text
            chatml = to_chatml_text("You are a mathematical olympiad solver. Given a problem, produce a complete and rigorous solution.\nRules:\n- Be mathematically correct and justify every nontrivial step.\n- Do not include failed attempts or self-corrections.\n- Use precise mathematical language.\n- End with a clearly stated final answer. For numeric answers, use \\boxed{}.",
                                    problem_text, solution)

            if sc["quality"] == "good":
                good_count += 1
                existing = good_pool.get(p_hash)
                if not existing or sc["word_count"] > existing["word_count"]:
                    good_pool[p_hash] = {
                        "text":       chatml,
                        "word_count": sc["word_count"],
                        "source":     fname,
                        "category":   file_cat,
                        "has_boxed":  sc["has_boxed"],
                    }
            else:
                broken_count += 1
                existing_broken = broken_pool.get(p_hash)
                if not existing_broken or sc["word_count"] > existing_broken["word_count"]:
                    broken_pool[p_hash] = {
                        "text":       chatml,
                        "word_count": sc["word_count"],
                        "source":     fname,
                        "category":   file_cat,
                    }

        total = len(examples) - skipped_ex
        pct_good = f"{good_count/total*100:.0f}%" if total else "n/a"
        print(f"[{file_cat:8s}] {fname:55s}  "
              f"{len(examples):5,} ex  →  "
              f"good {good_count:4,} ({pct_good})  broken {broken_count:4,}  "
              f"skip {skipped_ex}")
        stats["total_examples"] += len(examples)
        stats["total_good"]     += good_count
        stats["total_broken"]   += broken_count

    # ─── Выходные файлы ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Записываем результаты...")
    print(f"{'='*60}\n")

    # 1. Уникальные хорошие решения
    good_records = [v["text"] for v in good_pool.values()]
    write_jsonl(OUTPUT_DIR / "good_solutions_deduped.jsonl",
                [{"text": t} for t in good_records])

    # 2. Только сломанные (задача не встречается в good_pool)
    broken_only_records = [
        v for h, v in broken_pool.items()
        if h not in good_pool
    ]
    write_jsonl(OUTPUT_DIR / "broken_only_deduped.jsonl",
                [{"text": v["text"]} for v in broken_only_records])

    # 3. DPO пары: задача есть и в good, и в broken
    dpo_pairs = []
    for h, good_v in good_pool.items():
        if h in broken_pool:
            # text уже в ChatML — парсим обратно
            _, problem_text, good_sol    = extract_parts({"text": good_v["text"]})
            _, _,            broken_sol  = extract_parts({"text": broken_pool[h]["text"]})
            problem_text = re.sub(r"^Problem:\s*", "", problem_text, flags=re.IGNORECASE).strip()

            dpo_pairs.append({
                "problem":         problem_text,
                "chosen":          re.sub(r"^Solution:\s*", "", good_sol,   flags=re.IGNORECASE).strip(),
                "rejected":        re.sub(r"^Solution:\s*", "", broken_sol, flags=re.IGNORECASE).strip(),
                "chosen_source":   good_v["source"],
                "rejected_source": broken_pool[h]["source"],
            })

    write_jsonl(OUTPUT_DIR / "dpo_pairs.jsonl", dpo_pairs)

    # 4. Сводная таблица по источникам (для ревью)
    source_stats = defaultdict(lambda: {"good": 0, "broken": 0})
    for v in good_pool.values():
        source_stats[v["source"]]["good"] += 1
    for v in broken_pool.values():
        source_stats[v["source"]]["broken"] += 1

    report_lines = ["source_file,good_unique,broken_unique"]
    for src, cnt in sorted(source_stats.items()):
        report_lines.append(f"{src},{cnt['good']},{cnt['broken']}")
    report_path = OUTPUT_DIR / "source_report.csv"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  ✔ Source report → {report_path.name}")

    # ─── Итог ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ИТОГ:")
    print(f"  Файлов обработано:       {len(all_files) - stats['skipped_merged'] - stats['skipped_raw']}")
    print(f"  Файлов пропущено (raw):  {stats['skipped_raw']}")
    print(f"  Файлов пропущено (merged): {stats['skipped_merged']}")
    print(f"  Всего примеров прочитано: {stats['total_examples']:,}")
    print(f"  Уникальных задач (good):  {len(good_pool):,}")
    print(f"  Уникальных задач (broken только): {len(broken_only_records):,}")
    print(f"  DPO пар:                 {len(dpo_pairs):,}")
    print(f"\nРезультаты в: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    print("Рекомендации по структуре папок:")
    print("  Создай вручную (или скажи мне — создам скрипт перемещения):")
    print("  _raw/              ← khan_raw, math_dataset, aops, elmo, batch...")
    print("  _merged_protected/ ← все *merged*.jsonl")
    print("  complete/          ← cot_with_answer*_clean, resolved_by_sonnet_clean...")
    print("  olympiad/          ← olympiad_final*, olympiad_new, olympiad_cot...")
    print("  broken/            ← *_truncated, cot_no_answer*, *_lagged...")
    print("  flagged/           ← *_flagged*, *_rejected, resolved_failed...")
    print()


if __name__ == "__main__":
    main()
