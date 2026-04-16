#!/usr/bin/env python3
"""
embeddings.py  —  Deliverable 1 (embedding generation)
Generate dense sentence embeddings for every article chunk and store
them in a FAISS index.

Usage:
    python embeddings.py --corpus_dir data/corpus --output_dir data/vectorstore
"""

import os
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------
# We split each article into overlapping chunks of CHUNK_SIZE tokens
# with OVERLAP tokens overlap.  Rationale:
#   - 256-token chunks give semantically coherent passages for retrieval.
#   - 32-token overlap prevents losing context at chunk boundaries.
#   - Longer chunks (512+) reduce context fragmentation but hurt recall
#     because a single off-topic sentence can dilute the embedding.

CHUNK_SIZE    = 256   # words
OVERLAP       = 32    # words
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, fast
BATCH_SIZE    = 256


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Split *text* into overlapping word-level chunks.
    Returns a list of chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_articles(corpus_dir: str) -> List[Dict]:
    """Load articles from the JSONL file produced by data_pipeline.py."""
    path = Path(corpus_dir) / "articles.jsonl"
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    logger.info(f"Loaded {len(articles):,} articles from {path}")
    return articles


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def embed_articles(
    articles: List[Dict],
    model_name: str = EMBED_MODEL,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Chunk every article and encode all chunks.

    Returns:
        embeddings  — np.ndarray of shape (N_chunks, embed_dim)
        chunk_meta  — list of dicts: {article_id, title, chunk_index, text, analysis}
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    all_texts: List[str] = []
    chunk_meta: List[Dict] = []

    logger.info("Chunking articles…")
    for art in tqdm(articles, desc="Chunking"):
        chunks = chunk_text(art["text"])
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            chunk_meta.append({
                "article_id":   art["id"],
                "title":        art["title"],
                "chunk_index":  idx,
                "text":         chunk,
                "analysis":     art.get("analysis", {}),
            })

    logger.info(f"Total chunks: {len(all_texts):,}")
    logger.info("Encoding chunks (this may take a while)…")

    embeddings = model.encode(
        all_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings, chunk_meta


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """
    Build an IVF-PQ FAISS index for approximate nearest-neighbor search.

    Index choice:
        IndexIVFFlat with nlist=1024 gives good recall/speed trade-off
        for 50 k–500 k vectors. For very large corpora, switch to IVFPQ.
    """
    dim = embeddings.shape[1]
    nlist = 1024   # number of Voronoi cells

    quantizer = faiss.IndexFlatIP(dim)   # inner product (vectors are L2-normalized)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    if use_gpu and faiss.get_num_gpus() > 0:
        logger.info("Moving FAISS index to GPU…")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    logger.info("Training FAISS index…")
    index.train(embeddings.astype(np.float32))
    logger.info("Adding vectors to index…")
    index.add(embeddings.astype(np.float32))
    index.nprobe = 64   # probe 64 cells at query time (accuracy/speed trade-off)

    logger.info(f"FAISS index: {index.ntotal:,} vectors, dim={dim}")
    return index


def save_index(
    index: faiss.Index,
    chunk_meta: List[Dict],
    output_dir: str,
):
    """Persist the FAISS index and metadata to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # If index is on GPU, convert back to CPU before saving
    try:
        index_cpu = faiss.index_gpu_to_cpu(index)
    except Exception:
        index_cpu = index

    faiss.write_index(index_cpu, str(out / "faiss.index"))
    with open(out / "chunk_meta.pkl", "wb") as f:
        pickle.dump(chunk_meta, f)

    logger.info(f"Index saved → {out / 'faiss.index'}")
    logger.info(f"Metadata saved → {out / 'chunk_meta.pkl'}")


def load_index(vectorstore_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    """Load a previously saved FAISS index and its metadata."""
    vdir = Path(vectorstore_dir)
    index = faiss.read_index(str(vdir / "faiss.index"))
    with open(vdir / "chunk_meta.pkl", "rb") as f:
        chunk_meta = pickle.load(f)
    logger.info(f"Loaded FAISS index with {index.ntotal:,} vectors.")
    return index, chunk_meta


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around FAISS for semantic retrieval."""

    def __init__(self, vectorstore_dir: str, model_name: str = EMBED_MODEL, device: str = "cuda"):
        self.index, self.chunk_meta = load_index(vectorstore_dir)
        logger.info(f"Loading query encoder: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        Encode *query* and return the top-k most similar chunks.
        Each result is the chunk_meta dict augmented with a 'score' field.
        """
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self.chunk_meta[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vocabulary Recommender — Embeddings")
    parser.add_argument("--corpus_dir",   type=str, default="data/corpus")
    parser.add_argument("--output_dir",   type=str, default="data/vectorstore")
    parser.add_argument("--model",        type=str, default=EMBED_MODEL)
    parser.add_argument("--no_gpu",       action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.no_gpu else "cuda"
    articles   = load_articles(args.corpus_dir)
    embeddings, chunk_meta = embed_articles(articles, model_name=args.model, device=device)
    index = build_faiss_index(embeddings, use_gpu=(not args.no_gpu))
    save_index(index, chunk_meta, args.output_dir)
    logger.info("Done. Vector store ready.")
