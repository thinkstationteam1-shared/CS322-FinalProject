from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from embeddings import VectorStore


class RetrievalPipeline:
    def __init__(self, vectorstore_dir: str, device: str = "cpu"):
        print("Loading retrieval pipeline...")
        self.vector_store = VectorStore(vectorstore_dir, device=device)

        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device
        )

    def _normalize_coverage(self, coverage: Any):
        """
        Convert coverage_ratio into decimal form if possible.

        Examples:
            0.94   -> 0.94
            94.0   -> 0.94
            "94.0" -> 0.94
            None   -> None
        """
        if coverage is None:
            return None

        try:
            coverage = float(coverage)
        except (TypeError, ValueError):
            return None

        # If stored as percentage, convert to decimal
        if coverage > 1.0:
            coverage = coverage / 100.0

        return coverage

    def filter_by_vocab(
        self,
        results: List[Dict],
        min_cov: float = 0.90,
        max_cov: float = 0.97,
        allow_missing: bool = True
    ) -> List[Dict]:
        """
        Keep only results whose coverage ratio is within [min_cov, max_cov].

        If allow_missing=True, results with missing/invalid coverage are kept
        so the rest of the retrieval pipeline can still be tested.
        """
        filtered = []

        print("\n[DEBUG] First 10 coverage values:")
        for i, r in enumerate(results[:10], 1):
            analysis = r.get("analysis", {})
            raw_coverage = analysis.get("coverage_ratio", None)
            normalized_coverage = self._normalize_coverage(raw_coverage)
            print(
                f"  Result {i}: raw={raw_coverage} "
                f"normalized={normalized_coverage} "
                f"type={type(raw_coverage).__name__} "
                f"analysis_keys={list(analysis.keys())[:8]}"
            )

        missing_count = 0
        kept_in_range_count = 0
        rejected_count = 0

        for r in results:
            analysis = r.get("analysis", {})
            raw_coverage = analysis.get("coverage_ratio", None)
            coverage = self._normalize_coverage(raw_coverage)

            if coverage is None:
                missing_count += 1
                if allow_missing:
                    filtered.append(r)
                continue

            if min_cov <= coverage <= max_cov:
                kept_in_range_count += 1
                # store normalized value for easier later use
                item = dict(r)
                item["normalized_coverage_ratio"] = coverage
                filtered.append(item)
            else:
                rejected_count += 1

        print("\n[DEBUG] Vocabulary filter summary:")
        print(f"  Missing/invalid coverage kept: {missing_count if allow_missing else 0}")
        print(f"  In-range coverage kept:        {kept_in_range_count}")
        print(f"  Rejected by coverage window:   {rejected_count}")

        return filtered

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank FAISS results using a cross-encoder.
        """
        if not results:
            return results

        pairs = [(query, r["text"]) for r in results]
        scores = self.reranker.predict(pairs)

        reranked_results = []
        for r, score in zip(results, scores):
            item = dict(r)
            item["rerank_score"] = float(score)
            reranked_results.append(item)

        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Full retrieval pipeline:
        1. Broad FAISS retrieval
        2. Cross-encoder reranking
        3. Vocabulary filtering
        """
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        # Stage 1: broad retrieval
        initial_results = self.vector_store.search(query, top_k=50)
        print(f"[Stage 1] Initial FAISS results: {len(initial_results)}")

        if initial_results:
            print("\n[DEBUG] Sample result before reranking:")
            print(f"  Title: {initial_results[0].get('title')}")
            print(f"  FAISS Score: {initial_results[0].get('score')}")
            print(f"  Analysis: {initial_results[0].get('analysis', {})}")

        # Stage 2: rerank
        reranked_results = self.rerank(query, initial_results)
        print(f"[Stage 2] After reranking: {len(reranked_results)}")

        if reranked_results:
            print("\n[DEBUG] Top reranked result:")
            print(f"  Title: {reranked_results[0].get('title')}")
            print(f"  FAISS Score: {reranked_results[0].get('score')}")
            print(f"  Rerank Score: {reranked_results[0].get('rerank_score')}")

        # Stage 3: vocab filter
        filtered_results = self.filter_by_vocab(
            reranked_results,
            min_cov=0.90,
            max_cov=0.97,
            allow_missing=True
        )
        print(f"[Stage 3] After vocabulary filtering: {len(filtered_results)}")

        final_results = filtered_results[:top_k]
        print(f"[Final] Returning top {len(final_results)} results")
        print("=" * 80 + "\n")

        return final_results


def print_results(results: List[Dict]) -> None:
    """
    Pretty-print retrieval results for debugging.
    """
    if not results:
        print("No results returned.")
        return

    for i, r in enumerate(results, 1):
        title = r.get("title", "N/A")
        chunk_index = r.get("chunk_index", "N/A")
        faiss_score = r.get("score", None)
        rerank_score = r.get("rerank_score", None)

        raw_coverage = r.get("analysis", {}).get("coverage_ratio", None)
        normalized_coverage = r.get("normalized_coverage_ratio", None)

        text_preview = r.get("text", "")[:300].replace("\n", " ").strip()

        print(f"Result #{i}")
        print(f"Title: {title}")
        print(f"Chunk Index: {chunk_index}")
        print(f"FAISS Score: {faiss_score}")
        print(f"Rerank Score: {rerank_score}")
        print(f"Raw Coverage Ratio: {raw_coverage}")
        print(f"Normalized Coverage Ratio: {normalized_coverage}")
        print(f"Text Preview: {text_preview}")
        print("-" * 80)


if __name__ == "__main__":
    VECTORSTORE_DIR = "data/vectorstore"

    pipeline = RetrievalPipeline(VECTORSTORE_DIR, device="cpu")

    test_queries = [
        "space and planets",
        "ancient Rome",
        "machine learning",
        "volcanoes"
    ]

    for query in test_queries:
        results = pipeline.retrieve(query, top_k=5)
        print_results(results)
