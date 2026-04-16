#!/usr/bin/env python3
"""
embeddings.py  —  Deliverable 1 (embedding generation & vector store)

External dependencies: torch, transformers, numpy  (no faiss, no sentence-transformers)

Vector store: all embeddings stored as a single numpy .npy file.
Search: cosine similarity via numpy matrix multiply (vectors are L2-normalised).

For 50 k articles × ~4 chunks each ≈ 200 k vectors × 384 dims ≈ 288 MB on disk,
~580 MB in RAM — feasible on a 24 GB GPU server.

Usage:
    python embeddings.py --corpus_dir data/corpus --output_dir data/vectorstore
"""

import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, 22 MB weights
CHUNK_WORDS  = 256    # words per chunk
OVERLAP      = 32     # overlap between consecutive chunks
BATCH_SIZE   = 128


def _progress(i: int, n: int, label: str = "") -> None:
    pct = i / n * 100 if n else 0
    bar = ("#" * (i * 30 // n)).ljust(30) if n else " " * 30
    print(f"\r  {label}  [{bar}] {i}/{n}  ({pct:.1f}%)", end="", flush=True)
    if i >= n:
        print()


# ---------------------------------------------------------------------------
# Chunking — pure Python
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_WORDS, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Encoder — uses transformers directly (no sentence-transformers)
# ---------------------------------------------------------------------------

class TextEncoder:
    """
    Mean-pool encoder over a small transformer model.
    Equivalent to sentence-transformers but using the raw HuggingFace API.
    """

    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cuda"):
        self.device    = device if torch.cuda.is_available() else "cpu"
        log.info(f"Loading encoder: {model_name}  device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, dim)."""
        all_embs: List[np.ndarray] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            enc   = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            out    = self.model(**enc)
            # Mean pooling over non-padding tokens
            mask   = enc["attention_mask"].unsqueeze(-1).float()
            emb    = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb    = emb.cpu().numpy()
            # L2 normalise so cosine sim == dot product
            norms  = np.linalg.norm(emb, axis=1, keepdims=True)
            emb   /= np.where(norms == 0, 1, norms)
            all_embs.append(emb)
        return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Build vector store
# ---------------------------------------------------------------------------

def build_vectorstore(
    corpus_dir: str,
    output_dir: str,
    model_name: str = EMBED_MODEL,
    device: str = "cuda",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load articles
    articles: List[Dict] = []
    with open(Path(corpus_dir) / "articles.jsonl", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    log.info(f"Loaded {len(articles):,} articles")

    # Chunk everything
    log.info("Chunking articles…")
    all_texts:  List[str]  = []
    chunk_meta: List[Dict] = []
    for i, art in enumerate(articles):
        for j, chunk in enumerate(chunk_text(art["text"])):
            all_texts.append(chunk)
            chunk_meta.append({
                "article_id":  art["id"],
                "title":       art["title"],
                "chunk_index": j,
                "text":        chunk,
                "analysis":    art.get("analysis", {}),
            })
        _progress(i + 1, len(articles), "Chunking  ")
    log.info(f"Total chunks: {len(all_texts):,}")

    # Encode
    encoder    = TextEncoder(model_name, device)
    log.info("Encoding chunks…")
    embeddings = np.zeros((len(all_texts), 384), dtype=np.float32)
    n_batches  = (len(all_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, len(all_texts))
        embeddings[s:e] = encoder.encode(all_texts[s:e])
        _progress(b + 1, n_batches, "Encoding  ")

    # Save
    np.save(str(out / "embeddings.npy"), embeddings)
    with open(out / "chunk_meta.pkl", "wb") as f:
        pickle.dump(chunk_meta, f)
    log.info(f"Saved {embeddings.shape} embeddings → {out}")


# ---------------------------------------------------------------------------
# VectorStore — query-time cosine search via numpy
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Flat cosine-similarity store backed by a numpy array.
    No FAISS required; fast enough for ≤500 k vectors on a modern CPU/GPU.
    """

    def __init__(self, vectorstore_dir: str, model_name: str = EMBED_MODEL, device: str = "cuda"):
        vdir = Path(vectorstore_dir)
        log.info("Loading vector store…")
        self.embeddings = np.load(str(vdir / "embeddings.npy"))   # (N, dim)
        with open(vdir / "chunk_meta.pkl", "rb") as f:
            self.chunk_meta: List[Dict] = pickle.load(f)
        self.encoder = TextEncoder(model_name, device)
        log.info(f"Vector store: {self.embeddings.shape[0]:,} chunks loaded")

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """Return top-k chunks by cosine similarity."""
        q   = self.encoder.encode([query])[0]           # (dim,)
        sim = self.embeddings @ q                        # (N,) dot products
        idx = np.argpartition(sim, -top_k)[-top_k:]     # top-k indices (unordered)
        idx = idx[np.argsort(sim[idx])[::-1]]           # sort by score descending
        results = []
        for i in idx:
            meta = dict(self.chunk_meta[i])
            meta["score"] = float(sim[i])
            results.append(meta)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir",  type=str, default="data/corpus")
    p.add_argument("--output_dir",  type=str, default="data/vectorstore")
    p.add_argument("--model",       type=str, default=EMBED_MODEL)
    p.add_argument("--no_gpu",      action="store_true")
    args = p.parse_args()
    build_vectorstore(
        args.corpus_dir, args.output_dir,
        model_name=args.model,
        device="cpu" if args.no_gpu else "cuda",
    )
