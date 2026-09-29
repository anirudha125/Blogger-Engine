"""
Pre-build and cache BM25Okapi index over the canonical deduplicated corpus (81,123 chunks).
Saves to indexes/bm25_chunked_dedup.pkl for sub-second startup during evaluation and serving.
"""

import time
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.hybrid_retriever import BM25Index

DB_PATH = REPO_ROOT / "data" / "blogger_dedup.db"
CACHE_PATH = REPO_ROOT / "indexes" / "bm25_chunked_dedup.pkl"


def main():
    print("=" * 80)
    print("BUILDING BM25 INDEX FOR FULL CANONICAL CORPUS (81,123 CHUNKS)")
    print("=" * 80)

    assert DB_PATH.exists(), f"Database not found: {DB_PATH}"

    t0 = time.time()
    bm25_index = BM25Index(db_path=str(DB_PATH), cache_path=str(CACHE_PATH))
    total_time = time.time() - t0

    print(f"\nBM25 Index Built & Persisted:")
    print(f"  Source Database: {DB_PATH}")
    print(f"  Target Cache:    {CACHE_PATH}")
    print(f"  Total Chunks:    {bm25_index.total_docs:,}")
    print(f"  Cache Size:      {CACHE_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  Elapsed Time:    {total_time:.2f}s")

    # Quick test query
    test_q = "What is Direct Preference Optimization (DPO) used for in LLM alignment?"
    t1 = time.time()
    results = bm25_index.search(test_q, top_k=5)
    query_time = (time.time() - t1) * 1000
    print(f"\nTest Search ('{test_q[:50]}...'):")
    print(f"  Query time: {query_time:.2f} ms")
    print(f"  Top-5 results: {results}")

    assert len(results) > 0, "BM25 search returned empty results!"
    assert bm25_index.total_docs == 81123, f"Expected 81123 chunks, got {bm25_index.total_docs}"
    print("\nBM25 INDEX VERIFICATION SUCCESSFUL!")


if __name__ == "__main__":
    main()
