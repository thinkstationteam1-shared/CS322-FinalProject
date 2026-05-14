import os
#os.environ["NCCL_P2P_DISABLE"] = "1"
#os.environ["NCCL_IB_DISABLE"] = "1"
#os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =========================
# CONFIG & DEVICE PROBE
# =========================
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DATA_PATH = "data/corpus/articles.jsonl"
OUTPUT_DIR = "model_output/llama_lora"

# Set fragmentation fix
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MAX_LENGTH = 512
PRIMARY_GPU_CAP = "15GiB"
SECONDARY_GPU_CAP = "4GiB"
CPU_CAP = "15GiB"
max_mem = {1: "18GiB", 0: "5GiB", "cpu": "4GiB"}
def get_max_memory():
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        return None
    
    # We set conservative limits based on your error log (approx 18-20GB per P40)
    # If your environment limits you to 4GB, change these to "3.5GiB"
    return {0: "10GiB", 1: "10GiB", "cpu": "15GiB"}

# =========================
# LOAD MODEL WITH SHARDING
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Tesla P40 requires float16 (Pascal doesn't support bf16)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map= {"":0}, #"auto",             # Automatically shards layers
   #max_memory=max_mem,   # Forces specific VRAM limits per card
    trust_remote_code=True,
    attn_implementation="eager"
)

# CRITICAL for quantized training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# =========================
# APPLY LoRA
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Expanded targets
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False # Required for gradient checkpointing


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
# TRAINING SETUP (VRAM OPTIMIZED)
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # Keep low to avoid OOM
    gradient_accumulation_steps=16,    # Higher value = stable training
    gradient_checkpointing=True,      # CRITICAL: Recomputes weights to save VRAM
    gradient_checkpointing_kwargs={"use_reentrant":False},
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    #save_strategy="epoch",
    evaluation_strategy="no", #"epoch",
    fp16=True,                        # Required for P40
    optim="paged_adamw_8bit",          # Saves optimizer state memory
    bf16=False,
    report_to="none"
)

# =========================
# DATA & TRAINER (Standard)
# =========================
# ... [Insert your existing load_data and dataset mapping code here] ...

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete and model saved.")
