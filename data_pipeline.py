#!/usr/bin/env python3
"""
data_pipeline.py  —  Deliverable 1
Wikipedia corpus download, cleaning, tokenization, and vocabulary analysis.

Usage:
    python data_pipeline.py --num_articles 50000 --output_dir data/corpus
"""

import os
import re
import json
import unicodedata
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import spacy
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
import textstat
from wordfreq import word_frequency

# ---------------------------------------------------------------------------
# Download NLTK data once
# ---------------------------------------------------------------------------
for _pkg in ["punkt", "stopwords", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ---------------------------------------------------------------------------
# Vocabulary level presets  (word : frequency_rank threshold)
# We use wordfreq to determine familiarity.
# ---------------------------------------------------------------------------
VOCAB_LEVELS = {
    "beginner":     1_000,   # ~1 000 most common English words
    "intermediate": 3_000,   # ~3 000
    "advanced":     6_000,   # ~6 000
}

# Minimum wordfreq score to count as "known" at each level
# wordfreq returns a float; higher = more common.
# Approximate thresholds (English, Zipfian distribution):
FREQ_THRESHOLDS = {
    "beginner":     1e-4,   # very common
    "intermediate": 2e-5,
    "advanced":     5e-6,
}


class VocabularyList:
    """
    A configurable known-word list supporting preset levels
    (beginner / intermediate / advanced) or a custom word set.
    """

    def __init__(self, level: str = "intermediate", custom_words: Optional[List[str]] = None):
        if custom_words is not None:
            self.known_words = set(w.lower() for w in custom_words)
            self.level = "custom"
        elif level in VOCAB_LEVELS:
            self.level = level
            self.known_words = self._build_from_freq(FREQ_THRESHOLDS[level])
        else:
            raise ValueError(f"Unknown level '{level}'. Choose from {list(VOCAB_LEVELS)}")

        logger.info(f"VocabularyList [{self.level}]: {len(self.known_words):,} known words")

    def _build_from_freq(self, threshold: float) -> set:
        """
        Build a known-word set from the top-N most frequent English words
        using wordfreq.  We sample ~20 000 common English tokens and
        keep those above the frequency threshold.
        """
        # wordfreq top_n_list gives the n most common words
        from wordfreq import top_n_list
        # Use 3× the target size to be safe, then threshold
        n = max(VOCAB_LEVELS.values()) * 3
        candidates = top_n_list("en", n)
        known = {w for w in candidates if word_frequency(w, "en") >= threshold}
        return known

    def is_known(self, word: str) -> bool:
        return word.lower() in self.known_words

    def coverage_ratio(self, tokens: List[str]) -> float:
        """Fraction of content tokens that are in the known-word list."""
        content = [t for t in tokens if t.isalpha()]
        if not content:
            return 1.0
        known_count = sum(1 for t in content if self.is_known(t))
        return known_count / len(content)

    def new_words(self, tokens: List[str]) -> List[str]:
        """Return unique alphabetic tokens not in the known-word list."""
        return list({t.lower() for t in tokens if t.isalpha() and not self.is_known(t)})


# ---------------------------------------------------------------------------
# Wikipedia corpus loader
# ---------------------------------------------------------------------------

def load_wikipedia_corpus(num_articles: int = 50_000, language: str = "en") -> List[Dict]:
    """
    Stream English Wikipedia via HuggingFace datasets.
    Returns a list of dicts with keys: id, title, text.
    """
    logger.info(f"Loading {num_articles:,} Wikipedia articles (streaming)…")
    dataset = load_dataset(
        "wikipedia", "20220301.en",
        split="train", streaming=True,
        trust_remote_code=True
    )
    articles = []
    for article in tqdm(dataset, total=num_articles, desc="Downloading"):
        if len(articles) >= num_articles:
            break
        # Skip stubs, disambiguation pages, and redirects
        text = article.get("text", "")
        title = article.get("title", "")
        if _is_valid_article(title, text):
            articles.append({
                "id":    article.get("id", ""),
                "title": title,
                "text":  text,
            })
    logger.info(f"Loaded {len(articles):,} valid articles.")
    return articles


def _is_valid_article(title: str, text: str) -> bool:
    """Filter out stubs, disambiguation, redirects, and very short articles."""
    if len(text) < 500:
        return False
    stub_markers = ["(disambiguation)", "#REDIRECT", "may refer to:", "This article is a stub"]
    for marker in stub_markers:
        if marker.lower() in (title + text).lower():
            return False
    return True


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_article(text: str) -> str:
    """
    Strip Wikipedia markup, normalize Unicode, collapse whitespace.
    """
    # Remove references [1], [2], [citation needed], etc.
    text = re.sub(r"\[\d+\]|\[citation needed\]|\[note \d+\]", "", text)
    # Remove {{...}} template markup
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove [[File:...]] image tags
    text = re.sub(r"\[\[File:[^\]]*\]\]", "", text)
    # Convert [[Link|Display]] → Display; [[Link]] → Link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Remove bold/italic wiki markup
    text = re.sub(r"'{2,}", "", text)
    # Remove section headers (== Header ==)
    text = re.sub(r"={2,}[^=]+=+", " ", text)
    # Normalize Unicode to NFC
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_nlp = None  # lazy-loaded spaCy model


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def word_tokenize(text: str) -> List[str]:
    """Word-level tokenization using NLTK."""
    return nltk.word_tokenize(text)


def subword_tokenize(text: str, tokenizer) -> List[int]:
    """
    Subword tokenization using a HuggingFace tokenizer.
    Returns a list of token IDs (truncated to 512 for indexing).
    tokenizer is passed in to avoid importing transformers at module level.
    """
    return tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors=None,
    )["input_ids"]


# ---------------------------------------------------------------------------
# Vocabulary analysis
# ---------------------------------------------------------------------------

def analyze_article(
    article: Dict,
    vocab_list: VocabularyList,
) -> Dict:
    """
    Compute all vocabulary metrics for a single article.

    Returns a dict with:
        title, word_count, unique_word_count, coverage_ratio,
        new_words, readability_flesch, readability_coleman_liau
    """
    text = article["text"]
    tokens = word_tokenize(text)

    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    unique_tokens = list(set(alpha_tokens))

    coverage = vocab_list.coverage_ratio(alpha_tokens)
    new_wds  = vocab_list.new_words(alpha_tokens)

    # Readability scores (textstat operates on raw text)
    fk_score  = textstat.flesch_reading_ease(text)
    cl_score  = textstat.coleman_liau_index(text)

    return {
        "id":                    article["id"],
        "title":                 article["title"],
        "word_count":            len(alpha_tokens),
        "unique_word_count":     len(unique_tokens),
        "coverage_ratio":        round(coverage, 4),
        "new_word_count":        len(new_wds),
        "new_words":             new_wds[:200],   # cap stored list
        "readability_flesch":    round(fk_score, 2),
        "readability_coleman":   round(cl_score, 2),
    }


def is_candidate(
    analysis: Dict,
    coverage_min: float = 0.90,
    coverage_max: float = 0.97,
) -> bool:
    """
    Return True if the article's coverage ratio falls within the target window.
    Default: 90 %–97 % of words are known.
    """
    r = analysis["coverage_ratio"]
    return coverage_min <= r <= coverage_max


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    num_articles: int,
    output_dir: str,
    vocab_level: str = "intermediate",
    coverage_min: float = 0.90,
    coverage_max: float = 0.97,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus
    articles = load_wikipedia_corpus(num_articles)

    # 2. Clean
    logger.info("Cleaning articles…")
    for a in tqdm(articles, desc="Cleaning"):
        a["text"] = clean_article(a["text"])

    # 3. Vocabulary analysis
    vocab = VocabularyList(level=vocab_level)
    logger.info("Analyzing vocabulary…")
    analyzed = []
    candidates = []
    for a in tqdm(articles, desc="Analyzing"):
        meta = analyze_article(a, vocab)
        analyzed.append(meta)
        a["analysis"] = meta
        if is_candidate(meta, coverage_min, coverage_max):
            candidates.append(a)

    logger.info(
        f"Candidate articles (coverage {coverage_min*100:.0f}–{coverage_max*100:.0f}%): "
        f"{len(candidates):,} / {len(articles):,}"
    )

    # 4. Save
    articles_path = out / "articles.jsonl"
    logger.info(f"Saving {len(articles):,} articles → {articles_path}")
    with open(articles_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    candidates_path = out / "candidates.jsonl"
    logger.info(f"Saving {len(candidates):,} candidate articles → {candidates_path}")
    with open(candidates_path, "w", encoding="utf-8") as f:
        for a in candidates:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # Corpus statistics summary
    stats = {
        "total_articles":    len(articles),
        "candidate_articles": len(candidates),
        "vocab_level":       vocab_level,
        "coverage_window":   [coverage_min, coverage_max],
        "avg_word_count":    sum(a["analysis"]["word_count"] for a in articles) / len(articles),
        "avg_coverage":      sum(a["analysis"]["coverage_ratio"] for a in articles) / len(articles),
    }
    with open(out / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Corpus stats: {stats}")

    return articles, candidates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vocabulary Recommender — Data Pipeline")
    parser.add_argument("--num_articles",  type=int,   default=50_000)
    parser.add_argument("--output_dir",    type=str,   default="data/corpus")
    parser.add_argument("--vocab_level",   type=str,   default="intermediate",
                        choices=list(VOCAB_LEVELS))
    parser.add_argument("--coverage_min",  type=float, default=0.90)
    parser.add_argument("--coverage_max",  type=float, default=0.97)
    args = parser.parse_args()

    run_pipeline(
        num_articles=args.num_articles,
        output_dir=args.output_dir,
        vocab_level=args.vocab_level,
        coverage_min=args.coverage_min,
        coverage_max=args.coverage_max,
    )
