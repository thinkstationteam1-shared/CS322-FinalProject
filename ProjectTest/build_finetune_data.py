#!/usr/bin/env python3
"""
build_finetune_data.py  —  Deliverable 3 (dataset construction)
Generates 5 000+ instruction-response pairs via the Anthropic API.

External dependencies: NONE beyond stdlib  (calls API via urllib)

Usage:
    python build_finetune_data.py \
        --corpus_dir data/corpus --output_dir data/finetune --num_pairs 5000
"""

import json
import os
import random
import time
import argparse
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

from data_pipeline import VOCAB_LEVELS

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic API — raw urllib (no SDK)
# ---------------------------------------------------------------------------

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"


def _call_anthropic(system: str, user: str, api_key: str, max_tokens: int = 1024) -> Optional[str]:
    """POST to Anthropic /v1/messages, return the assistant text or None."""
    payload = json.dumps({
        "model":      ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   [{"role": "user", "content": user}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        log.warning(f"Anthropic API HTTP {e.code}: {body[:200]}")
        return None
    except Exception as e:
        log.warning(f"Anthropic API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Silver-data generation prompt
# ---------------------------------------------------------------------------

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


def _generate_example(
    api_key: str,
    article: Dict,
    level: str,
    passage: str,
) -> Optional[Dict]:
    analysis  = article.get("analysis", {})
    new_words = analysis.get("new_words", [])[:10]
    coverage  = analysis.get("coverage_ratio", 0.93)

    raw = _call_anthropic(
        SILVER_SYSTEM,
        SILVER_USER.format(
            title=article["title"],
            level=level,
            size=VOCAB_LEVELS[level],
            coverage=coverage,
            new_words=", ".join(new_words) or "none",
            passage=passage[:1500],
        ),
        api_key,
    )
    if not raw:
        return None

    import re
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        parsed = json.loads(raw)
        parsed["_article_id"]  = article.get("id", "")
        parsed["_vocab_level"] = level
        return parsed
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    corpus_dir: str,
    output_dir: str,
    num_pairs: int = 5_000,
    levels: Optional[List[str]] = None,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
):
    if levels is None:
        levels = list(VOCAB_LEVELS.keys())

    api_key = "TOKEN HERE"
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    cand_path = Path(corpus_dir) / "candidates.jsonl"
    if not cand_path.exists():
        raise FileNotFoundError(f"{cand_path} not found. Run data_pipeline.py first.")

    candidates: List[Dict] = []
    with open(cand_path, encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))
    log.info(f"Loaded {len(candidates):,} candidate articles")

    random.seed(42)
    random.shuffle(candidates)

    # Reserve held-out articles for the test split
    n_test_articles = max(1, int(len(candidates) * (1 - train_ratio - val_ratio)))
    test_articles   = candidates[-n_test_articles:]
    train_articles  = candidates[:-n_test_articles]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _pick_passage(article: Dict) -> str:
        words = article["text"].split()
        if len(words) < 100:
            return article["text"]
        start = random.randint(0, max(0, len(words) - 256))
        return " ".join(words[start : start + 256])

    per_level = num_pairs // len(levels)
    all_examples: List[Dict] = []
    errors = 0

    for level in levels:
        log.info(f"Generating {per_level} examples for level={level}…")
        count = 0
        for art in train_articles:
            if count >= per_level:
                break
            ex = _generate_example(api_key, art, level, _pick_passage(art))
            if ex:
                all_examples.append(ex)
                count += 1
            else:
                errors += 1
            # Brief rate-limit pause
            time.sleep(0.3)
        log.info(f"  → {count} examples  (errors: {errors})")

    # Test split examples
    test_examples: List[Dict] = []
    n_test = int(num_pairs * (1 - train_ratio - val_ratio))
    for art in test_articles:
        if len(test_examples) >= n_test:
            break
        level = random.choice(levels)
        ex = _generate_example(api_key, art, level, _pick_passage(art))
        if ex:
            ex["_split"] = "test"
            test_examples.append(ex)
        time.sleep(0.3)

    # Train / val split
    random.shuffle(all_examples)
    n_train = int(len(all_examples) * train_ratio / (train_ratio + val_ratio))
    train_ex = all_examples[:n_train]
    val_ex   = all_examples[n_train:]

    def _write(path: Path, data: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    _write(out / "train.jsonl", train_ex)
    _write(out / "val.jsonl",   val_ex)
    _write(out / "test.jsonl",  test_examples)

    stats = {
        "total":   len(all_examples) + len(test_examples),
        "train":   len(train_ex),
        "val":     len(val_ex),
        "test":    len(test_examples),
        "errors":  errors,
        "levels":  levels,
    }
    with open(out / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Dataset built: {stats}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir",  type=str, default="data/corpus")
    p.add_argument("--output_dir",  type=str, default="data/finetune")
    p.add_argument("--num_pairs",   type=int, default=5000)
    args = p.parse_args()
    build_dataset(args.corpus_dir, args.output_dir, args.num_pairs)
