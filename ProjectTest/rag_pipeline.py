#!/usr/bin/env python3
"""
rag_pipeline.py  —  Deliverable 2
Two-stage RAG pipeline:
  Stage 1 — Cosine-similarity retrieval from numpy vector store (top-50)
  Stage 2 — BM25 re-ranking + vocabulary-aware coverage filter (top-5)
  Generation — LLaMA 3.1 8B Instruct produces structured JSON recommendation

External dependencies: torch, transformers  (no cross-encoder library)
BM25 is implemented in pure Python below.
"""

import json
import re
import math
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from data_pipeline import VocabularyList, word_tokenize
from embeddings import VectorStore
from multi_gpu_strategy import get_asymmetric_device_map

log = logging.getLogger(__name__)

LLAMA_MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BROAD_K          = 50
FINAL_K          = 5
COVERAGE_MIN     = 0.90
COVERAGE_MAX     = 0.97

# ---------------------------------------------------------------------------
# BM25 re-ranker — pure Python, replaces cross-encoder library
# ---------------------------------------------------------------------------
# BM25 is a strong sparse-retrieval baseline that works well as a re-ranker
# when the initial candidates are from dense retrieval (complementary signals).

class BM25Reranker:
    """
    Okapi BM25 re-ranker.
    k1=1.5, b=0.75 are standard defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

    #BM25 score determines the relevance of a document based on the key word frequency
    def score(self, query: str, documents: List[str]) -> List[float]:
        """Return a BM25 score for each document given a query."""
        q_terms = [t.lower() for t in word_tokenize(query)]
        doc_tokens = [[t.lower() for t in word_tokenize(d)] for d in documents]

        N      = len(documents)
        avgdl  = sum(len(d) for d in doc_tokens) / max(N, 1)

        # Document frequency per query term
        df: Dict[str, int] = {}
        for term in set(q_terms):
            df[term] = sum(1 for d in doc_tokens if term in d)

        scores = []
        for doc in doc_tokens:
            tf   = Counter(doc)
            dl   = len(doc)
            sc   = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1))
                sc += idf * (num / den)
            scores.append(sc)
        return scores

    #rerank essentially makes a new Okapi BM25 score based on the relevance of the document with respect to the vocabulary scores
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        vocab_list: VocabularyList,
        coverage_min: float = COVERAGE_MIN,
        coverage_max: float = COVERAGE_MAX,
        top_k: int = FINAL_K,
    ) -> List[Dict]:
        """
        Score with BM25, apply a large penalty to articles outside the
        vocabulary-coverage window, then return top_k by final score.
        """
        docs   = [c["text"] for c in candidates]
        bm25_s = self.score(query, docs)

        for c, sc in zip(candidates, bm25_s):
            analysis  = c.get("analysis", {})
            coverage  = analysis.get("coverage_ratio", 1.0)
            in_window = coverage_min <= coverage <= coverage_max
            c["bm25_score"]      = sc
            c["vocab_in_window"] = in_window
            c["final_score"]     = sc if in_window else sc - 1e6

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# LLaMA 3.1 8B Instruct prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a vocabulary-aware reading assistant. "
    "Given a student's query, their vocabulary level, and retrieved article passages, "
    "produce a structured JSON recommendation with exactly these keys:\n"
    '  "recommended_title"    : string\n'
    '  "summary"              : string (2-3 sentences)\n'
    '  "new_vocabulary"       : list of {"word": str, "definition": str, "example": str}\n'
    '  "difficulty_rating"    : integer 1-10\n'
    '  "confidence_score"     : float 0.0-1.0\n'
    '  "why_good_next_read"   : string (1-2 sentences)\n'
    "Output ONLY the JSON object."
)

USER_TEMPLATE = (
    "Student query: {query}\n"
    "Vocabulary level: {vocab_level}\n"
    "New words for this student: {new_words}\n\n"
    "Retrieved passages:\n{passages}\n\n"
    "JSON recommendation:"
)


def _llama31_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _parse_json(text: str) -> Dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {
        "recommended_title": "Parse error",
        "summary":           text[:300],
        "new_vocabulary":    [],
        "difficulty_rating": 5,
        "confidence_score":  0.0,
        "why_good_next_read": "Could not parse model output.",
    }


# ---------------------------------------------------------------------------
# LLaMA generator
# ---------------------------------------------------------------------------

#actually calls the llama model, the above code builds the prompt
class LlamaGenerator:
    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 512,
        gpu0_gb: int = 20,
        gpu1_gb: int = 22,
    ):
        log.info(f"Loading {LLAMA_MODEL_ID} from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            max_memory=get_asymmetric_device_map(gpu0_gb, gpu1_gb),
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
        )

    def generate(
        self, query: str, vocab_level: str, new_words: List[str], passages: List[str]
    ) -> Dict:
        user_msg = USER_TEMPLATE.format(
            query=query,
            vocab_level=vocab_level,
            new_words=", ".join(new_words[:30]) or "none",
            passages="\n\n---\n\n".join(
                f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages)
            ),
        )
        prompt = _llama31_prompt(SYSTEM_PROMPT, user_msg)
        output = self.pipe(prompt)[0]["generated_text"].strip()
        return _parse_json(output)


# ---------------------------------------------------------------------------
# Full RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(
        self,
        vectorstore_dir: str,
        model_dir: str,
        device: str = "cuda",
        gpu0_gb: int = 20,
        gpu1_gb: int = 22,
    ):
        log.info("Initialising RAG pipeline…")
        self.vectorstore = VectorStore(vectorstore_dir, device=device) #vector store keeps the numerical value of the cuda
        self.reranker    = BM25Reranker()
        self.generator   = LlamaGenerator(model_dir, gpu0_gb=gpu0_gb, gpu1_gb=gpu1_gb)

    def recommend(
        self,
        query: str,
        vocab_list: VocabularyList,
        coverage_min: float = COVERAGE_MIN,
        coverage_max: float = COVERAGE_MAX,
        broad_k: int = BROAD_K,
        final_k: int = FINAL_K,
    ) -> Dict:
        # Stage 1: dense retrieval
        candidates = self.vectorstore.search(query, top_k=broad_k)

        # Stage 2: BM25 rerank + vocab filter
        reranked = self.reranker.rerank(
            query, candidates, vocab_list,
            coverage_min=coverage_min,
            coverage_max=coverage_max,
            top_k=final_k,
        )
        if not reranked:
            return {"error": "No articles found matching the vocabulary window."}

        top       = reranked[0]
        new_words = top.get("analysis", {}).get("new_words", [])
        passages  = [r["text"] for r in reranked]

        rec = self.generator.generate(
            query=query,
            vocab_level=vocab_list.level,
            new_words=new_words,
            passages=passages,
        )
        rec["_retrieved_articles"] = [
            {
                "title":           r["title"],
                "article_id":      r["article_id"],
                "coverage_ratio":  r.get("analysis", {}).get("coverage_ratio"),
                "bm25_score":      r.get("bm25_score"),
                "vocab_in_window": r.get("vocab_in_window"),
            }
            for r in reranked
        ]
        rec["_query"]       = query
        rec["_vocab_level"] = vocab_list.level
        return rec


# ---------------------------------------------------------------------------
# Retrieval evaluation — pure Python
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    pipeline_obj: RAGPipeline,
    queries_with_relevant: List[Tuple[str, List[str]]],
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """Precision@k, Recall@k, MRR@k — no external eval libraries."""
    metrics: Dict[int, Dict] = {k: {"p": [], "r": [], "rr": []} for k in k_values}

    for query, relevant_ids in queries_with_relevant:
        candidates    = pipeline_obj.vectorstore.search(query, top_k=max(k_values))
        retrieved_ids = [c["article_id"] for c in candidates]
        rel_set       = set(relevant_ids)

        for k in k_values:
            top_k = retrieved_ids[:k]
            hits  = sum(1 for rid in top_k if rid in rel_set)
            metrics[k]["p"].append(hits / k)
            metrics[k]["r"].append(hits / len(rel_set) if rel_set else 0.0)
            rr = next(
                (1.0 / (rank + 1) for rank, rid in enumerate(top_k) if rid in rel_set),
                0.0,
            )
            metrics[k]["rr"].append(rr)

    return {
        key: sum(vals) / len(vals)
        for k in k_values
        for key, vals in [
            (f"Precision@{k}", metrics[k]["p"]),
            (f"Recall@{k}",    metrics[k]["r"]),
            (f"MRR@{k}",       metrics[k]["rr"]),
        ]
    }
