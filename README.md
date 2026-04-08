# Olympiad Math — CoT Distillation for Local LLMs

Fine-tuning local 9B models to reason through olympiad-level math problems,
using Chain-of-Thought solutions distilled from stronger models (Claude, GPT via OpenRouter)
on top of **human-authored** problem+answer pairs.

## Goal

Improve the reasoning capabilities of small local models (GLM-Z1-9B, Qwen3.5-9B)
on competition mathematics by training them on high-quality CoT traces.

Key insight: human solutions provide correct answers and proof structure —
stronger models fill in the intermediate reasoning steps.

## Pipeline

```
1. scraping/          Raw olympiad problems (AoPS, IMO, AMC, geometry, ...)
        ↓
2. cot_generation/    CoT traces via Claude API / GPT via OpenRouter
        ↓
3. cleaning/          Quality filtering, hallucination detection, truncation repair
        ↓
4. merging/           Deduplication, combining sources
        ↓
        → training dataset (ChatML format, ~22K examples)
        ↓
5. training/          LoRA SFT via unsloth (GLM-Z1-9B, Qwen3.5-9B)
```

## Repository Structure

```
├── scraping/
│   ├── scrape_olympiad.py             # multi-source olympiad scraper (IMO, AMC, ...)
│   ├── scrape_volume.py               # high-volume scraper with rate limiting
│   └── download_olympiad_datasets.py  # HuggingFace / public dataset downloader
│
├── cot_generation/
│   ├── generate_cot_dataset.py        # CoT generation via Claude API (async, checkpoint)
│   ├── generate_cot_openrouter.py     # CoT generation via OpenRouter (GPT-4o, etc.)
│   └── Complete_cot.py                # complete truncated/unfinished solutions
│
├── cleaning/
│   ├── repair_dataset.py              # fix malformed JSON, encoding issues
│   ├── repair_dataset_v2.py           # truncation detection, flag-based filtering
│   ├── dedup_filter.py                # deduplication by problem hash
│   ├── formatting.py                  # convert to ChatML SFT format
│   └── parse_tex_to_dataset.py        # extract problems from LaTeX sources
│
├── merging/
│   └── merge_olympiad_datasets.py     # merge sources, prefer longer solutions, dedup
│
├── training/
│   ├── train_glm_z1.py                # LoRA SFT for GLM-Z1-9B (unsloth)
│   └── train_qwen_math.py             # LoRA SFT for Qwen3.5-9B (unsloth)
│
└── tools/
    └── organize_dataset.py            # classify good/broken, build DPO pairs
```

## Dataset

Data files are not included in this repo (too large).
Sources: AoPS, IMO Shortlist, AMC/AIME, Khan Academy olympiad subset, ELMO.

| Split | Examples | Format |
|-------|----------|--------|
| SFT — good solutions | ~22,990 | ChatML `{"text": ...}` |
| Broken / incomplete  | ~6,518  | same |
| DPO pairs (chosen / rejected) | ~4,393 | `{"problem", "chosen", "rejected"}` |

## Models Trained

| Model | Base | Train Loss |
|-------|------|------------|
| GLM-Z1-9B-LoRA | `zai-org/GLM-Z1-9B-0414` | 0.603 |
| Qwen3.5-9B-LoRA | `Qwen/Qwen3.5-9B-Base` | in progress |

LoRA config: `r=32, alpha=64, dropout=0.05`  
Target modules: q/k/v/o projections + gate/up/down MLP

## Training Setup

- GPU: NVIDIA RTX 4070 Ti Super (16 GB VRAM)
- Quantization: 4-bit NF4 (bitsandbytes) — frozen weights only, adapters in bf16
- Max sequence length: 8192 tokens
- Optimizer: adamw_8bit, lr=8e-6, cosine schedule, warmup 3%

## Usage

> **Note:** Update `BASE_DIR` / `INPUT` / `OUTPUT` paths in each script before running.  
> Never hardcode API keys — pass via `--key` argument or `ANTHROPIC_API_KEY` env variable.

```bash
# 1. Scrape problems
python scraping/scrape_olympiad.py --output data/raw_problems.jsonl

# 2. Generate CoT (Claude)
python cot_generation/generate_cot_dataset.py \
  --key $ANTHROPIC_API_KEY \
  --input data/raw_problems.jsonl \
  --output data/cot_dataset.jsonl

# 3. Complete truncated solutions
python cot_generation/Complete_cot.py \
  --key $ANTHROPIC_API_KEY \
  --input data/cot_no_answer.jsonl \
  --output data/cot_completed.jsonl

# 4. Clean and deduplicate
python cleaning/repair_dataset_v2.py
python cleaning/dedup_filter.py
python cleaning/formatting.py

# 5. Merge sources
python merging/merge_olympiad_datasets.py

# 6. Organize (good / broken / DPO)
python tools/organize_dataset.py

# 7. Train
python training/train_glm_z1.py
python training/train_qwen_math.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## License

MIT — scripts and training code only. Dataset not included.
