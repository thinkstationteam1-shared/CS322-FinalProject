import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from json_repair import repair_json
import random
import argparse
import logging
import time
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Setup logging
logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

log.info(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    log.info(f"Device {i}: {torch.cuda.get_device_name(i)}")

# --- Vocabulary Levels (from original data_pipeline) ---
VOCAB_LEVELS = {
    "A1": 500, "A2": 1000, "B1": 2000, "B2": 4000, "C1": 8000, "C2": 15000
}

# --- Prompts (Identical to original) ---
SILVER_SYSTEM = (
    "You are a curriculum designer creating training data for a vocabulary-aware "
    "reading recommender. Given an article passage and vocabulary level, generate "
    "a JSON training example. Output ONLY valid JSON with these keys:\n"
    '  "instruction" : student query describing interests and vocabulary level\n'
    '  "context"     : the article passage\n'
    '  "response"    : object with keys:\n'
    '      "recommended_title"   : string\n'
    '      "summary"             : 2-3 sentence summary\n'
    '      "new_vocabulary"      : list of up to 5 {"word","definition","example"}\n'
    '      "difficulty_rating"   : integer 1-10\n'
    '      "confidence_score"    : float 0.0-1.0\n'
    '      "why_good_next_read"  : 1-2 sentence explanation\n'
    "No markdown, no preamble — JSON only."
)

SILVER_USER = (
    "Article title: {title}\n"
    "Vocabulary level: {level} (~{size} known words)\n"
    "Coverage ratio: {coverage:.1%}\n"
    "New words for this student: {new_words}\n\n"
    "Passage:\n{passage}\n\n"
    "Generate the training example JSON:"
)

class LocalLLM:
    """Handles local Llama 3.1 inference with 4-bit quantization."""
    def __init__(self, model_id: str, token: str):
        log.info(f"Initializing local model: {model_id}")
        
        # Configure 4-bit quantization to fit on consumer GPUs
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        # Ensure pad token is set for batch-less generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, system: str, user: str, max_tokens: int = 256) -> Optional[str]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        # Apply Llama 3.1 Instruct Chat Template
        encodings = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **encodings,
                max_new_tokens=256,
                min_new_tokens=120,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        # Decode only the newly generated tokens
        input_ids = encodings['input_ids']
        response_ids = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

def generate_example(llm: LocalLLM, article: Dict, level: str) -> Optional[Dict]:
    """Helper to prepare the prompt and parse JSON response."""
    analysis = article.get("analysis", {})
    new_words = analysis.get("new_words", [])[:10]
    coverage = analysis.get("coverage_ratio", 0.93)
    
    # Pick a random 256-word passage
    words = article["text"].split()
    start = random.randint(0, max(0, len(words) - 256))
    passage = " ".join(words[start : start + 256])

    raw_response = llm.generate(
        SILVER_SYSTEM,
        SILVER_USER.format(
            title=article["title"],
            level=level,
            size=VOCAB_LEVELS[level],
            coverage=coverage,
            new_words=", ".join(new_words) or "none",
            passage=passage,
        )
    )

    if not raw_response:
        return None

    # Clean the raw string (remove markdown JSON tags if present)
    clean_json = re.sub(r"```(?:json)?|```", "", raw_response).strip()

    try:
        fixed_json = repair_json(clean_json)
        parsed = json.loads(fixed_json)
        # Add metadata for tracking
        parsed["_meta"] = {
            "article_id": article.get("id"),
            "vocab_level": level,
            "source_title": article["title"]
        }
        return parsed
    except Exception as e:
        log.warning(f"Failed to parse model output: {e}")
        log.warning("RAW MODEL OUTPUT:")
        log.warning("=" * 80)
        log.warning(clean_json)
        log.warning("=" * 80)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default="data/corpus/candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/finetune")
    parser.add_argument("--num_pairs", type=int, default=5000)
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--token", type=str, default="TOKEN HERE")
    args = parser.parse_args()

    # Load candidates
    if not os.path.exists(args.corpus_file):
        log.error(f"File {args.corpus_file} not found.")
        return

    candidates = []
    with open(args.corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))
    
    log.info(f"Loaded {len(candidates)} candidates. Shuffling...")
    random.seed(42)
    random.shuffle(candidates)

    # Calculate splits (80% train, 10% val, 10% test)
    n_test_articles = max(1, int(len(candidates) * 0.10))
    test_articles = candidates[-n_test_articles:]
    train_articles = candidates[:-n_test_articles]

    llm = LocalLLM(args.model_id, args.token)
    levels = list(VOCAB_LEVELS.keys())
    per_level = args.num_pairs // len(levels)
    
    all_train_val = []
    
    # Generate Training/Val examples
    for level in levels:
        log.info(f"Processing level {level}...")
        count = 0
        attempts = 0
        start_time = time.time()

        for art in train_articles:
            if count >= per_level:
                break

            attempts += 1
            ex = generate_example(llm, art, level)

            if ex:
                all_train_val.append(ex)
                count += 1

            if attempts % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / attempts
                success_rate = count / attempts if attempts else 0.0

                if success_rate > 0:
                    remaining_attempts = (per_level - count) / success_rate
                    eta_seconds = remaining_attempts * avg_time
                    eta_minutes = eta_seconds / 60
                else:
                    eta_minutes = float("inf")

                log.info(
                    f"{level}: {count}/{per_level} valid | "
                    f"{attempts} attempts | "
                    f"{success_rate:.1%} success | "
                    f"{avg_time:.1f}s/attempt | "
                    f"ETA {eta_minutes:.1f} min"
                )

        total_elapsed = time.time() - start_time
        final_success = count / attempts if attempts else 0.0
        log.info(
            f"Completed {level}: {count} examples in {attempts} attempts "
            f"({final_success:.1%} success) in {total_elapsed/60:.1f} minutes"
        )

    # Shuffle and split train/val
    random.shuffle(all_train_val)
    split_idx = int(len(all_train_val) * 0.88) # roughly 80% of total
    train_data = all_train_val[:split_idx]
    val_data = all_train_val[split_idx:]

    # Generate Test examples from held-out articles
    test_data = []
    for art in test_articles:
        if len(test_data) >= (args.num_pairs // 10): break
        ex = generate_example(llm, art, random.choice(levels))
        if ex: test_data.append(ex)

    # Save outputs
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def save_jsonl(filename, data):
        with open(out_path / filename, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl("train.jsonl", train_data)
    save_jsonl("val.jsonl", val_data)
    save_jsonl("test.jsonl", test_data)

    stats = {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "total": len(train_data) + len(val_data) + len(test_data)
    }
    with open(out_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"Dataset construction complete. Stats: {stats}")

if __name__ == "__main__":
    main()
