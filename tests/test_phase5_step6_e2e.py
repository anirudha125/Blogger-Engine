"""
Phase 5 — Step 6: End-to-End Modernized Production Testing.

Exercises the complete Blogger Engine application using the real canonical
Experiment 4 production architecture:
  - BGE-small-en-v1.5 dense index (81,123 vectors, 384d)
  - BM25Okapi cache (81,123 chunks)
  - Canonical SQLite database (81,123 chunks, 8,242 documents)
  - Cross-Encoder ms-marco-MiniLM-L-6-v2 reranking with lifespan warmup
  - Document-level deduplication down to top_k=3
  - 0.65 confidence refusal gate in AnswerGenerator
"""

import pickle
import faiss
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient

from app.main import app
from app.config import get_settings
from app.schemas import SearchRequest, AskRequest
from app.core.hybrid_retriever import HybridRetriever
from app.core.generator import AnswerGenerator


def test_e2e_canonical_lifespan_and_artifact_identity():
    """Verify application startup loads canonical assets and warms cross-encoder."""
    with TestClient(app) as client:
        retriever = getattr(app.state, "retriever", None)
        generator = getattr(app.state, "generator", None)

        assert retriever is not None, "HybridRetriever not in app.state"
        assert isinstance(retriever, HybridRetriever)
        assert retriever.total_vectors == 81123, f"Expected 81,123 vectors, got {retriever.total_vectors}"
        assert retriever.dense_retriever._index.d == 384
        assert retriever.dense_retriever.embedder.dimension == 384
        assert retriever.use_reranker is True
        assert retriever._cross_encoder is not None, "Cross-encoder was not warmed up during lifespan"

        assert generator is not None, "AnswerGenerator not in app.state"
        assert isinstance(generator, AnswerGenerator)
        assert generator.similarity_threshold == 0.65


def test_e2e_health_check_healthy_and_degraded():
    """Verify /health reflects real index status and handles degraded state."""
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["index_loaded"] is True
        assert data["db_connected"] is True
        assert data["version"] == "1.0.0"

        # Simulate degraded state
        saved_retriever = app.state.retriever
        try:
            app.state.retriever = None
            res_degraded = client.get("/health")
            assert res_degraded.status_code == 200
            assert res_degraded.json()["status"] == "degraded"
            assert res_degraded.json()["index_loaded"] is False
        finally:
            app.state.retriever = saved_retriever


def test_e2e_api_stats_canonical():
    """Verify /api/stats reports accurate counts from canonical assets."""
    with TestClient(app) as client:
        res = client.get("/api/stats")
        assert res.status_code == 200
        data = res.json()
        assert data["total_vectors"] == 81123
        assert data["total_chunks"] == 81123
        assert data["total_documents"] == 8242
        assert "bge-small-en-v1.5" in data["embedding_model"]
        assert data["query_logging_enabled"] is True


def test_e2e_api_search_canonical_queries():
    """Verify /api/search with representative in-domain, entity, and invalid queries."""
    with TestClient(app) as client:
        # A. In-domain technical query
        res_a = client.post("/api/search", json={"query": "What is RLAIF and how does it differ from RLHF?"})
        assert res_a.status_code == 200
        data_a = res_a.json()
        assert data_a["total_results"] <= 3
        assert len(data_a["results"]) == 3
        assert data_a["latency_ms"] > 0
        doc_ids_a = [r["doc_id"] for r in data_a["results"]]
        assert len(set(doc_ids_a)) == len(doc_ids_a), "Document deduplication failed"
        for r in data_a["results"]:
            assert 0.0 <= r["score"] <= 1.0
            assert len(r["content"]) > 0
            assert r["url"].startswith("http")

        # B. Second distinct in-domain query
        res_b = client.post("/api/search", json={"query": "How does Constitutional AI implement critique and revision?"})
        assert res_b.status_code == 200
        data_b = res_b.json()
        assert len(data_b["results"]) == 3
        doc_ids_b = [r["doc_id"] for r in data_b["results"]]
        assert len(set(doc_ids_b)) == len(doc_ids_b)

        # C. Lexical / exact terminology query
        res_c = client.post("/api/search", json={"query": "Direct Preference Optimization DPO loss function"})
        assert res_c.status_code == 200
        data_c = res_c.json()
        assert len(data_c["results"]) == 3

        # D. Whitespace query (must fail with 422)
        res_d = client.post("/api/search", json={"query": "    "})
        assert res_d.status_code == 422


def test_e2e_api_ask_canonical_and_confidence_refusal():
    """Verify /api/ask produces grounded answers for in-domain and enforces 0.65 refusal for OOD."""
    with TestClient(app) as client:
        # In-domain: RLHF / RLAIF
        res_in = client.post("/api/ask", json={"query": "What is RLAIF and how does it work?"})
        assert res_in.status_code == 200
        data_in = res_in.json()
        assert data_in["refused"] is False
        assert len(data_in["answer"]) > 0
        assert data_in["confidence_score"] >= 0.65
        assert len(data_in["sources"]) == 3
        assert isinstance(data_in["citations"], list)

        # Also test citation parsing specifically on retrieved passages
        generator = app.state.generator
        mock_grounded_answer = "RLAIF incorporates AI feedback [Doc 1] to scale alignment [Doc 2]."
        passages = app.state.retriever.retrieve("What is RLAIF and how does it work?", top_k=3)
        parsed_citations = generator.extract_citations(mock_grounded_answer, passages)
        assert len(parsed_citations) == 2
        assert parsed_citations[0].doc_tag == "[Doc 1]"
        assert parsed_citations[1].doc_tag == "[Doc 2]"

        # Out-of-domain: Neapolitan pizza
        res_ood = client.post("/api/ask", json={"query": "How to bake authentic Neapolitan sourdough pizza?"})
        assert res_ood.status_code == 200
        data_ood = res_ood.json()
        assert data_ood["refused"] is True
        assert data_ood["confidence_score"] < 0.65
        assert "below required threshold" in data_ood["answer"].lower() or "confidence" in data_ood["answer"].lower()
        assert len(data_ood["citations"]) == 0


def test_e2e_top_k_coherence_across_layers():
    """Verify top_k defaults to 3 across frontend, schema, settings, and live response."""
    settings = get_settings()
    assert settings.TOP_K == 3
    assert SearchRequest.model_fields["top_k"].default == 3
    assert AskRequest.model_fields["top_k"].default == 3

    html_content = Path("app/static/index.html").read_text(encoding="utf-8")
    assert "const CANONICAL_TOP_K = 3;" in html_content

    with TestClient(app) as client:
        res = client.post("/api/search", json={"query": "Transformer attention mechanism"})
        assert res.status_code == 200
        assert len(res.json()["results"]) == 3


def test_e2e_concurrent_request_safety():
    """Verify concurrent search requests execute safely without state cross-contamination."""
    queries = [
        "What is Low-Rank Adaptation LoRA?",
        "How does FlashAttention optimize memory transfers?",
        "What is the mathematical formulation of DP-SGD?",
        "How do Vision Transformers tokenize image patches?"
    ]

    with TestClient(app) as client:
        def do_search(q):
            return client.post("/api/search", json={"query": q})

        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(do_search, queries))

        for q, res in zip(queries, responses):
            assert res.status_code == 200
            data = res.json()
            assert data["query"] == q
            assert len(data["results"]) == 3
            assert data["latency_ms"] > 0
            doc_ids = [r["doc_id"] for r in data["results"]]
            assert len(set(doc_ids)) == len(doc_ids)


def test_e2e_service_unavailable_503_isolation():
    """Verify clean 503 service unavailable without constructing fallback models."""
    with TestClient(app) as client:
        saved_retriever = app.state.retriever
        saved_generator = app.state.generator

        try:
            # Unset retriever
            app.state.retriever = None
            res_search = client.post("/api/search", json={"query": "test query"})
            assert res_search.status_code == 503
            assert "Search engine is unavailable" in res_search.json()["detail"]

            res_ask = client.post("/api/ask", json={"query": "test query"})
            assert res_ask.status_code == 503

            # Restore retriever, unset generator
            app.state.retriever = saved_retriever
            app.state.generator = None
            res_ask_gen = client.post("/api/ask", json={"query": "test query"})
            assert res_ask_gen.status_code == 503
            assert "Answer generation service is unavailable" in res_ask_gen.json()["detail"]
        finally:
            app.state.retriever = saved_retriever
            app.state.generator = saved_generator


def test_e2e_canonical_artifact_integrity():
    """Directly verify integrity of physical production artifacts on disk."""
    settings = get_settings()

    # 1. FAISS index
    index_path = Path(settings.FAISS_INDEX_PATH)
    assert index_path.exists(), f"Missing {settings.FAISS_INDEX_PATH}"
    index = faiss.read_index(str(index_path))
    assert index.ntotal == 81123
    assert index.d == 384

    # 2. SQLite database
    db_path = Path(settings.SQLITE_DB_PATH)
    assert db_path.exists(), f"Missing {settings.SQLITE_DB_PATH}"
    from app.db.database import get_total_chunks, get_total_documents
    assert get_total_chunks(str(db_path)) == 81123
    assert get_total_documents(str(db_path)) == 8242

    # 3. BM25 cache
    bm25_path = Path(settings.BM25_CACHE_PATH)
    assert bm25_path.exists(), f"Missing {settings.BM25_CACHE_PATH}"
    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)
    assert "bm25" in bm25_data
    assert bm25_data["bm25"].corpus_size == 81123
    assert len(bm25_data["faiss_ids"]) == 81123


def test_e2e_circular_import_regression():
    """Verify database module can be imported without circular dependency error."""
    # Test importing app.db.database directly
    import importlib
    import app.db.database
    importlib.reload(app.db.database)
    from app.db.database import get_total_chunks
    assert callable(get_total_chunks)
