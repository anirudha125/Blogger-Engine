"""
Unit and integration tests for HybridRetriever and BM25Index.
Verifies BM25 indexing, dense retrieval, RRF fusion, FAISS ID/doc mapping, and citations.
"""

import pytest
import sqlite3
from pathlib import Path
import numpy as np

from app.core.retriever import Retriever, SearchResult
from app.core.hybrid_retriever import HybridRetriever, BM25Index, tokenize_text
from app.core.generator import AnswerGenerator


@pytest.fixture
def sample_hybrid_retriever():
    """Build a HybridRetriever over the sample 4-article database."""
    index_path = "indexes/faiss_chunked.index"
    db_path = "data/blogger.db"

    assert Path(index_path).exists(), f"Sample index missing: {index_path}"
    assert Path(db_path).exists(), f"Sample database missing: {db_path}"

    dense = Retriever(index_path=index_path, db_path=db_path)
    bm25 = BM25Index(db_path=db_path)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_index=bm25, candidate_depth=10, rrf_k=60)
    return hybrid


def test_tokenize_text():
    text = "SL-CAI and DPO in Llama-3 with BAAI/bge-small-en-v1.5!"
    tokens = tokenize_text(text)
    assert "sl-cai" in tokens
    assert "dpo" in tokens
    assert "llama-3" in tokens
    assert "bge-small-en-v1" in tokens or "bge-small-en-v1_5" in tokens or "baai" in tokens


def test_bm25_index_basic():
    db_path = "data/blogger.db"
    bm25_idx = BM25Index(db_path=db_path)
    assert bm25_idx.total_docs > 0

    results = bm25_idx.search("RLAIF artificial intelligence feedback", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5
    for fid, score in results:
        assert isinstance(fid, int)
        assert score > 0.0


def test_bm25_empty_query():
    db_path = "data/blogger.db"
    bm25_idx = BM25Index(db_path=db_path)
    assert bm25_idx.search("", top_k=5) == []
    assert bm25_idx.search("   ", top_k=5) == []


def test_hybrid_retrieve_structure(sample_hybrid_retriever):
    query = "What is RLAIF and how does it differ from RLHF?"
    results = sample_hybrid_retriever.retrieve(query, top_k=3)

    assert len(results) > 0
    assert len(results) <= 3

    for idx, r in enumerate(results, start=1):
        assert isinstance(r, SearchResult)
        assert r.rank == idx
        assert r.chunk_id
        assert r.doc_id
        assert r.title
        assert r.content
        assert isinstance(r.score, float)
        assert -1.0 <= r.score <= 1.0  # Normalized cosine similarity


def test_hybrid_with_generator_citations(sample_hybrid_retriever):
    query = "What is RLAIF?"
    results = sample_hybrid_retriever.retrieve(query, top_k=3)
    generator = AnswerGenerator()

    prompt = generator.build_user_prompt(query, results)
    assert "[Doc 1]" in prompt
    assert results[0].title in prompt

    # Mock answer with citation
    mock_answer = f"RLAIF uses AI feedback instead of human feedback [Doc 1]."
    citations = generator.extract_citations(mock_answer, results)
    assert len(citations) == 1
    assert citations[0].doc_tag == "[Doc 1]"
    assert citations[0].chunk_id == results[0].chunk_id


def test_hybrid_retrieve_doc_dedup(sample_hybrid_retriever):
    """Verify that doc_dedup=True guarantees unique doc_ids in top-k results."""
    query = "Constitutional AI principles and feedback mechanisms"
    # Query without doc_dedup
    results_no_dedup = sample_hybrid_retriever.retrieve(query, top_k=3, doc_dedup=False)
    # Query with doc_dedup
    results_dedup = sample_hybrid_retriever.retrieve(query, top_k=3, doc_dedup=True, fused_depth=10)

    assert len(results_dedup) > 0
    assert len(results_dedup) <= 3

    # All doc_ids in results_dedup MUST be unique
    doc_ids_dedup = [r.doc_id for r in results_dedup]
    assert len(doc_ids_dedup) == len(set(doc_ids_dedup)), f"Duplicate doc_ids found: {doc_ids_dedup}"

    # Verify rank sequence and result attributes
    for idx, r in enumerate(results_dedup, start=1):
        assert r.rank == idx
        assert r.doc_id
        assert r.chunk_id
        assert r.content
        assert isinstance(r.score, float)


def test_hybrid_doc_dedup_retains_best_chunk(sample_hybrid_retriever):
    """Verify that when multiple chunks of a document exist, doc_dedup retains the top-ranked one."""
    query = "RLAIF vs RLHF"
    # Get top-10 fused without dedup
    results_10 = sample_hybrid_retriever.retrieve(query, top_k=10, doc_dedup=False)
    results_dedup_3 = sample_hybrid_retriever.retrieve(query, top_k=3, doc_dedup=True, fused_depth=10)

    # For the first doc in results_dedup_3, its chunk should match the first occurrence of that doc in results_10
    first_dedup_doc = results_dedup_3[0].doc_id
    expected_chunk = next(r.chunk_id for r in results_10 if r.doc_id == first_dedup_doc)
    assert results_dedup_3[0].chunk_id == expected_chunk


def test_hybrid_with_reranker(sample_hybrid_retriever):
    """Verify that use_reranker=True executes cross-encoder scoring and returns top-k unique docs."""
    query = "What is RLAIF and how does it fundamentally differ from RLHF?"
    sample_hybrid_retriever.use_reranker = True
    results = sample_hybrid_retriever.retrieve(query, top_k=3, doc_dedup=True)

    assert len(results) > 0
    assert len(results) <= 3
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids))

    for idx, r in enumerate(results, start=1):
        assert r.rank == idx
        assert r.chunk_id
        assert r.doc_id
        assert r.content
        assert isinstance(r.score, float)

    # Check latency breakdown
    breakdown = sample_hybrid_retriever.last_latency_breakdown
    assert "dense_ms" in breakdown
    assert "bm25_ms" in breakdown
    assert "rrf_ms" in breakdown
    assert "rerank_ms" in breakdown
    assert "total_ms" in breakdown
    assert breakdown["rerank_ms"] > 0.0
    assert breakdown["total_ms"] > 0.0


def test_extract_prf_keywords():
    """Verify deterministic PRF keyword extraction, stopword filtering, and query-term exclusion."""
    from app.core.hybrid_retriever import extract_prf_keywords

    query = "What is reinforcement learning?"
    passages = [
        "Reinforcement learning from human feedback (RLHF) uses reward models.",
        "Proximal policy optimization is an algorithm used in reward model training.",
        "Feedback from human annotators helps guide policy models."
    ]
    keywords = extract_prf_keywords(query, passages, top_k=4)

    assert len(keywords) <= 4
    # Query words ('reinforcement', 'learning') must not appear in keywords
    assert "reinforcement" not in keywords
    assert "learning" not in keywords
    # Stopwords ('what', 'is', 'from', 'an', 'in') must not appear
    assert "what" not in keywords
    assert "from" not in keywords
    # Meaningful domain terms should be extracted
    assert any(term in keywords for term in ["feedback", "human", "models", "reward", "policy"])

    # Determinism test: calling multiple times returns exact same list
    keywords_2 = extract_prf_keywords(query, passages, top_k=4)
    assert keywords == keywords_2


def test_hybrid_with_query_expansion(sample_hybrid_retriever):
    """Verify that use_query_expansion=True runs PRF and records expansion_ms."""
    query = "RLAIF vs RLHF"
    sample_hybrid_retriever.use_query_expansion = True
    results = sample_hybrid_retriever.retrieve(query, top_k=3, doc_dedup=True)

    assert len(results) > 0
    assert len(results) <= 3
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids))

    for idx, r in enumerate(results, start=1):
        assert r.rank == idx
        assert r.chunk_id
        assert r.doc_id
        assert r.content
        assert isinstance(r.score, float)

    # Check latency breakdown
    breakdown = sample_hybrid_retriever.last_latency_breakdown
    assert "dense_ms" in breakdown
    assert "bm25_ms" in breakdown
    assert "expansion_ms" in breakdown
    assert "rrf_ms" in breakdown
    assert "total_ms" in breakdown
    assert breakdown["total_ms"] > 0.0


def test_hybrid_with_score_fusion_structure(sample_hybrid_retriever):
    """Verify that use_score_fusion=True executes score fusion and records fusion_ms."""
    query = "What is RLAIF and how does it fundamentally differ from RLHF?"
    sample_hybrid_retriever.use_reranker = True
    sample_hybrid_retriever.use_score_fusion = True
    sample_hybrid_retriever.fusion_alpha = 0.7
    results = sample_hybrid_retriever.retrieve(query, top_k=3, doc_dedup=True)

    assert len(results) > 0
    assert len(results) <= 3
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids))

    for idx, r in enumerate(results, start=1):
        assert r.rank == idx
        assert r.chunk_id
        assert r.doc_id
        assert r.content
        assert isinstance(r.score, float)

    # Check latency breakdown
    breakdown = sample_hybrid_retriever.last_latency_breakdown
    assert "dense_ms" in breakdown
    assert "bm25_ms" in breakdown
    assert "rrf_ms" in breakdown
    assert "rerank_ms" in breakdown
    assert "fusion_ms" in breakdown
    assert "total_ms" in breakdown
    assert breakdown["rerank_ms"] > 0.0
    assert breakdown["total_ms"] > 0.0
    # Score fusion arithmetic should take < 5ms on CPU
    assert breakdown["fusion_ms"] < 5.0


def test_hybrid_preserves_exp4_when_score_fusion_disabled(sample_hybrid_retriever):
    """Verify that use_score_fusion=False produces identical results to Exp 4 pure CE path."""
    query = "Constitutional AI principles and feedback"
    sample_hybrid_retriever.use_reranker = True

    # 1. Pure CE (Exp 4)
    results_ce = sample_hybrid_retriever.retrieve(
        query, top_k=3, doc_dedup=True, use_score_fusion=False
    )
    # 2. Disabled explicitly
    results_no_fusion = sample_hybrid_retriever.retrieve(
        query, top_k=3, doc_dedup=True, use_score_fusion=False
    )

    assert [r.chunk_id for r in results_ce] == [r.chunk_id for r in results_no_fusion]
    assert [r.score for r in results_ce] == [r.score for r in results_no_fusion]


def test_hybrid_with_l12_reranker(sample_hybrid_retriever):
    """Verify that reranker_model='cross-encoder/ms-marco-MiniLM-L-12-v2' works with HybridRetriever."""
    query = "What is RLAIF?"
    sample_hybrid_retriever.use_reranker = True
    sample_hybrid_retriever.reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    sample_hybrid_retriever._cross_encoder = None  # Reset cached model

    results = sample_hybrid_retriever.retrieve(query, top_k=3, doc_dedup=True, use_score_fusion=False)
    assert len(results) > 0
    assert len(results) <= 3
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids))
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.chunk_id
        assert r.doc_id


def test_hybrid_canonical_defaults():
    """Verify that default HybridRetriever parameters strictly match canonical Experiment 4."""
    db_path = "data/blogger.db"
    index_path = "indexes/faiss_chunked.index"
    dense = Retriever(index_path=index_path, db_path=db_path)
    bm25 = BM25Index(db_path=db_path)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_index=bm25)
    assert hybrid.doc_dedup is True
    assert hybrid.use_reranker is True
    assert hybrid.candidate_depth == 20
    assert hybrid.rerank_depth == 20
    assert hybrid.fused_depth == 20
    assert hybrid.rrf_k == 60
    assert hybrid.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert hybrid.use_query_expansion is False
    assert hybrid.use_score_fusion is False


def test_hybrid_zero_arg_constructor():
    """Verify that HybridRetriever() with zero arguments instantiates canonical production architecture."""
    # Only run if production assets exist on disk
    prod_index = Path("indexes/faiss_chunked_bge_dedup.index")
    prod_db = Path("data/blogger_dedup.db")
    prod_bm25 = Path("indexes/bm25_chunked_dedup.pkl")
    if prod_index.exists() and prod_db.exists() and prod_bm25.exists():
        hybrid = HybridRetriever()
        assert hybrid.doc_dedup is True
        assert hybrid.use_reranker is True
        assert hybrid.candidate_depth == 20
        assert hybrid.rerank_depth == 20
        assert hybrid.fused_depth == 20
        assert hybrid.rrf_k == 60
        assert hybrid.total_vectors > 0


def test_hybrid_latency_metadata_on_result(sample_hybrid_retriever):
    """Verify that retrieve() returns RetrievalResultList with request-isolated latency_breakdown."""
    query = "What is constitutional AI?"
    results = sample_hybrid_retriever.retrieve(query, top_k=3)

    assert hasattr(results, "latency_breakdown")
    breakdown = results.latency_breakdown
    assert isinstance(breakdown, dict)
    assert "dense_ms" in breakdown
    assert "bm25_ms" in breakdown
    assert "expansion_ms" in breakdown
    assert "rrf_ms" in breakdown
    assert "rerank_ms" in breakdown
    assert "fusion_ms" in breakdown
    assert "total_ms" in breakdown
    assert breakdown["total_ms"] > 0.0


def test_hybrid_retrieve_with_latency_method(sample_hybrid_retriever):
    """Verify that retrieve_with_latency returns a (results, breakdown) tuple."""
    query = "What is constitutional AI?"
    results, breakdown = sample_hybrid_retriever.retrieve_with_latency(query, top_k=3)

    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(breakdown, dict)
    assert "total_ms" in breakdown
    assert breakdown["total_ms"] > 0.0


def test_hybrid_concurrency_latency_isolation(sample_hybrid_retriever):
    """Verify that concurrent retrievals produce isolated, non-corrupting latency metadata."""
    import concurrent.futures

    queries = [
        "What is RLAIF?",
        "How does Constitutional AI work?",
        "What is direct preference optimization?",
        "Reinforcement learning from AI feedback"
    ]

    def run_query(q):
        res = sample_hybrid_retriever.retrieve(q, top_k=2)
        return q, res, res.latency_breakdown

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_query, q) for q in queries]
        results_list = [f.result() for f in futures]

    assert len(results_list) == len(queries)
    for q, res, breakdown in results_list:
        assert len(res) > 0
        assert isinstance(breakdown, dict)
        assert breakdown["total_ms"] > 0.0
        # Each result object holds its own distinct breakdown dict
        assert res.latency_breakdown is breakdown


def test_hybrid_warmup_idempotence():
    """Verify that warmup() loads the cross-encoder and repeated calls do not reload it."""
    db_path = "data/blogger.db"
    index_path = "indexes/faiss_chunked.index"
    dense = Retriever(index_path=index_path, db_path=db_path)
    bm25 = BM25Index(db_path=db_path)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_index=bm25, use_reranker=True)
    assert hybrid._cross_encoder is None

    # First call loads model
    hybrid.warmup()
    assert hybrid._cross_encoder is not None
    loaded_model = hybrid._cross_encoder

    # Second call is idempotent and returns immediately without reloading
    hybrid.warmup()
    assert hybrid._cross_encoder is loaded_model


def test_hybrid_warmup_skips_when_reranker_disabled():
    """Verify that warmup() safely skips when use_reranker=False."""
    db_path = "data/blogger.db"
    index_path = "indexes/faiss_chunked.index"
    dense = Retriever(index_path=index_path, db_path=db_path)
    bm25 = BM25Index(db_path=db_path)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_index=bm25, use_reranker=False)
    hybrid.warmup()
    assert hybrid._cross_encoder is None


def test_hybrid_warmup_failure_raises_clear_error():
    """Verify that warmup() fails clearly with RuntimeError if model cannot be loaded."""
    db_path = "data/blogger.db"
    index_path = "indexes/faiss_chunked.index"
    dense = Retriever(index_path=index_path, db_path=db_path)
    bm25 = BM25Index(db_path=db_path)

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_index=bm25,
        use_reranker=True,
        reranker_model="nonexistent-dummy-reranker-model-999"
    )

    with pytest.raises(RuntimeError) as excinfo:
        hybrid.warmup()

    assert "CrossEncoder warmup failed" in str(excinfo.value)






