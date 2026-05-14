#!/usr/bin/env python3
"""
build_finetune_data.py
Generates 5,000+ instruction-response pairs via the Hugging Face API in a single file.
"""

import json
import os
import random
import time
import argparse
import logging
import urllib.request
import urllib.error
import re
from pathlib import Path
from typing import Dict, List, Optional

# --- INLINED CONSTANTS ---
VOCAB_LEVELS = {
    "A1": 500,    # Beginner
    "A2": 1000,   # Elementary
    "B1": 2000,   # Intermediate
    "B2": 4000,   # Upper-Intermediate
    "C1": 8000,   # Advanced
    "C2": 15000   # Mastery
}

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

# --- HUGGING FACE API CONFIG ---
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"

def _call_hf(system: str, user: str, api_key: str, max_tokens: int = 1024) -> Optional[str]:
    """POST to Hugging Face /v1/chat/completions, return text or None."""
    payload = json.dumps({
        "model":      HF_MODEL,
        "max_tokens": max_tokens,
        "messages":   [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.1,
    }).encode("utf-8")

    req = urllib.request.Request(
        HF_API_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 5
                log.warning(f"HF Rate Limit (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            return None
        except Exception as e:
            log.warning(f"HF API error: {e}")
            return None
    return None

# --- PROMPT TEMPLATES ---
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

def _generate_example(api_key: str, article: Dict, level: str, passage: str) -> Optional[Dict]:
    analysis  = article.get("analysis", {})
    new_words = analysis.get("new_words", [])[:10]
    coverage  = analysis.get("coverage_ratio", 0.93)

    raw = _call_hf(
        SILVER_SYSTEM,
        SILVER_USER.format(
            title=article["title"],
            level=level,
            size=VOCAB_LEVELS.get(level, 1000),
            coverage=coverage,
            new_words=", ".join(new_words) or "none",
            passage=passage[:1500],
        ),
        api_key,
    )
    if not raw: return None

    # Cleaning: Remove markdown blocks and trim to actual JSON object
    raw = re.sub(r"json)?", "", raw).replace("```","").strip()
    try:
        start_idx, end_idx = raw.find('{'), raw.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            parsed = json.loads(raw[start_idx:end_idx])
            parsed["_article_id"]  = article.get("id", "")
            parsed["_vocab_level"] = level
            return parsed
    except json.JSONDecodeError:
        return None
    return None

def build_dataset(corpus_dir: str, output_dir: str, num_pairs: int = 5000):
    api_key = "TOKEN HERE"
    if not api_key:
        log.error("HF_TOKEN env var not set. Export it first.")
        return

    cand_path = Path(corpus_dir) / "candidates.jsonl"
    if not cand_path.exists():
        log.error(f"Missing {cand_path}")
        return

    candidates = []
    with open(cand_path, encoding="utf-8") as f:
        for line in f: candidates.append(json.loads(line))
    
    random.seed(42)
    random.shuffle(candidates)
    
    levels = list(VOCAB_LEVELS.keys())
    per_level = num_pairs // len(levels)
    all_examples, errors = [], 0
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for level in levels:
        log.info(f"Generating for {level}...")
        count = 0
        for art in candidates:
            if count >= per_level: break
            passage = " ".join(art.get("text", "").split()[:256])
            ex = _generate_example(api_key, art, level, passage)
            if ex:
                all_examples.append(ex)
                count += 1
            else:
                errors += 1
            time.sleep(0.4) # Rate limit safety

    # Simple 80/10/10 split
    random.shuffle(all_examples)
    n = len(all_examples)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)

    for name, data in [("train", all_examples[:train_end]), 
                       ("val", all_examples[train_end:val_end]), 
                       ("test", all_examples[val_end:])]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in data: f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log.info(f"Complete. Generated {len(all_examples)} pairs. Errors: {errors}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir", default="data/corpus")
    p.add_argument("--output_dir", default="data/finetune")
    p.add_argument("--num_pairs", type=int, default=5000)
    args = p.parse_args()
    build_dataset(args.corpus_dir, args.output_dir, args.num_pairs)
