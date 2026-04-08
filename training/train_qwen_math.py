import os
from pathlib import Path
from datetime import datetime

# Force single-process mode
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

# ===================== CONFIG =====================
RESUME_FROM = None
BASE_MODEL = r"C:\models\Qwen3.5-9B-Base"
DATASET_PATH = r"C:\lora_training\OLympiad\converted_for_sft.jsonl"

MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True  # 19GB model, 16GB VRAM — need 4bit

# ===================== OUTPUT DIR =====================
if RESUME_FROM:
    OUTPUT_DIR = str(Path(RESUME_FROM).parent)
    print(f"RESUMING from: {RESUME_FROM}")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"C:/lora_training/lora_MATH_output/qwen_run_{timestamp}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"NEW RUN -> {OUTPUT_DIR}\n")

# ===================== LOAD DATASET =====================
# Dataset is already in ChatML format — perfect for Qwen, no conversion needed
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Loaded {len(dataset)} examples")

print("\nFirst example preview (first 400 chars):")
print(repr(dataset[0]["text"][:400]), "\n")

# ===================== MODEL =====================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)

# ===================== COMPLETION MASKING =====================
# Mask everything up to and including "Solution:\n"
# so only the actual solution contributes to the loss.
print("Tokenizing + applying completion-only masking...")

response_template = "Solution:\n"
response_ids = text_tokenizer(response_template, add_special_tokens=False)["input_ids"]

def tokenize_and_mask(example):
    tokenized = text_tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    input_ids      = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels         = input_ids.copy()

    # Find LAST occurrence of response_template
    start_idx = None
    for i in range(len(input_ids) - len(response_ids), -1, -1):
        if input_ids[i : i + len(response_ids)] == response_ids:
            start_idx = i + len(response_ids)
            break

    if start_idx is None:
        labels = [-100] * len(input_ids)
    else:
        labels[:start_idx] = [-100] * start_idx

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

dataset = dataset.map(tokenize_and_mask, remove_columns=["text"])

sample = dataset[0]
num_train_tokens = sum(1 for x in sample["labels"] if x != -100)
print(f"Sanity check: trainable tokens in first sample = {num_train_tokens}\n")

if num_train_tokens == 0:
    raise ValueError("Masking failed: 0 trainable tokens. Check that dataset contains 'Solution:\\n'.")

# ===================== TRAINING =====================
bf16_ok = is_bfloat16_supported()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1.0,
    learning_rate=8e-6,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    bf16=bf16_ok,
    fp16=not bf16_ok,
    report_to="none",
    seed=42,
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=text_tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

# Patch unsloth meta-tensor crash
import unsloth_zoo.tokenizer_utils
unsloth_zoo.tokenizer_utils.fix_untrained_tokens = lambda *args, **kwargs: (None, None)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Starting training...\n")
if RESUME_FROM:
    trainer.train(resume_from_checkpoint=RESUME_FROM)
else:
    trainer.train()

# ===================== SAVE =====================
print("\nSaving final model...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
text_tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Done! Saved to: {OUTPUT_DIR}/final")
