import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0, 2"
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# =========================
# CONFIG
# =========================
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DATA_PATH = "data/corpus/articles.jsonl"   # from your pipeline
OUTPUT_DIR = "model_output/llama_lora"
_cuda_ok = torch.cuda.is_available()
_num_gpus = torch.cuda.device_count() if _cuda_ok else 0


MAX_LENGTH = 512

# =========================
# LOAD TOKENIZER + MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# =========================
# APPLY LoRA
# =========================
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =========================
# LOAD DATA
# =========================
def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            article = json.loads(line)

            # Skip if no analysis
            if "analysis" not in article:
                continue

            text = article["text"][:400]
            new_words = article["analysis"].get("new_words", [])[:5]
            coverage = article["analysis"].get("coverage_ratio", 0)

            example = {
                "instruction": f"User is intermediate and interested in {article['title']}",
                "context": text,
                "response": f"""
Summary: {text[:150]}

New Words: {new_words}

Difficulty: {'easy' if coverage > 0.9 else 'medium' if coverage > 0.7 else 'hard'}

Explanation: This article introduces new vocabulary while remaining understandable.
"""
            }

            data.append(example)

    return data

raw_data = load_data(DATA_PATH)

# Convert to HuggingFace dataset
dataset = Dataset.from_list(raw_data)

# =========================
# FORMAT + TOKENIZE
# =========================
def format_example(example):
    prompt = f"""Instruction: {example['instruction']}
Context: {example['context']}
Response: {example['response']}"""

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(format_example)

# Train/val split
dataset = dataset.train_test_split(test_size=0.1)

# =========================
# TRAINING SETUP
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,   # REQUIRED
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    bf16=False,
    optim="paged_adamw_32bit"

)

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE MODEL
# =========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete and model saved.")
