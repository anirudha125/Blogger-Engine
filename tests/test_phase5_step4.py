"""
Integration tests for Phase 5 — Step 4: FastAPI Lifespan & Route Alignment.
Validates that the canonical Experiment 4 retrieval engine is the active live runtime.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from app.main import app
from app.core.hybrid_retriever import HybridRetriever
from app.core.generator import AnswerGenerator


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_lifespan_wires_canonical_hybrid_retriever(client):
    """Verify application startup creates canonical HybridRetriever and attaches to app.state.retriever."""
    retriever = getattr(app.state, "retriever", None)
    assert retriever is not None, "app.state.retriever was not initialized during lifespan startup"
    assert isinstance(retriever, HybridRetriever), f"Expected HybridRetriever, got {type(retriever)}"

    # Verify canonical Experiment 4 parameters
    assert retriever.doc_dedup is True
    assert retriever.use_reranker is True
    assert retriever.candidate_depth == 20
    assert retriever.rerank_depth == 20
    assert retriever.rrf_k == 60
    assert retriever.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert retriever.total_vectors > 0

    # Verify startup called warmup() (cross-encoder weights are in memory)
    assert retriever._cross_encoder is not None, "HybridRetriever was not warmed up during startup"


def test_lifespan_wires_application_scoped_generator(client):
    """Verify application startup creates application-scoped AnswerGenerator with 0.65 threshold."""
    generator = getattr(app.state, "generator", None)
    assert generator is not None, "app.state.generator was not initialized during lifespan startup"
    assert isinstance(generator, AnswerGenerator), f"Expected AnswerGenerator, got {type(generator)}"
    assert generator.similarity_threshold == 0.65


def test_api_search_canonical_execution(client):
    """Verify /api/search runs through HybridRetriever with top_k=3 default and doc dedup."""
    res = client.post("/api/search", json={"query": "What is RLAIF?"})
    assert res.status_code == 200
    data = res.json()

    assert data["query"] == "What is RLAIF?"
    assert len(data["results"]) == 3
    assert data["total_results"] == 3
    assert data["latency_ms"] > 0

    # Verify document-level deduplication: all doc_ids must be unique
    doc_ids = [r["doc_id"] for r in data["results"]]
    assert len(doc_ids) == len(set(doc_ids)), f"Duplicate doc_ids found: {doc_ids}"

    # Verify result structure compatibility with frontend
    first = data["results"][0]
    assert first["rank"] == 1
    assert first["score"] > 0.0
    assert first["chunk_id"] != ""
    assert first["doc_id"] != ""
    assert first["title"] != ""
    assert first["content"] != ""
    assert first["url"].startswith("http")


def test_api_ask_canonical_execution_and_confidence_gate(client):
    """Verify /api/ask executes through HybridRetriever and enforces 0.65 refusal gate."""
    # 1. In-domain query: accepted
    res_in = client.post("/api/ask", json={"query": "What is RLAIF and how does it work?"})
    assert res_in.status_code == 200
    data_in = res_in.json()
    assert data_in["refused"] is False
    assert len(data_in["sources"]) == 3
    assert data_in["confidence_score"] >= 0.65
    assert "answer" in data_in

    # 2. Out-of-domain query: refused cleanly
    res_ood = client.post("/api/ask", json={"query": "How to bake a chocolate sourdough cake?"})
    assert res_ood.status_code == 200
    data_ood = res_ood.json()
    assert data_ood["refused"] is True
    assert data_ood["confidence_score"] < 0.65
    assert "below required threshold" in data_ood["answer"]


def test_missing_retriever_returns_503_no_fallback(client):
    """Verify missing app.state.retriever returns HTTP 503 without constructing fallback Retriever."""
    saved_retriever = app.state.retriever
    try:
        app.state.retriever = None

        # /api/search must return 503
        res_search = client.post("/api/search", json={"query": "test query"})
        assert res_search.status_code == 503
        assert res_search.json()["detail"] == "Search engine is unavailable."

        # /api/ask must return 503
        res_ask = client.post("/api/ask", json={"query": "test query"})
        assert res_ask.status_code == 503
        assert res_ask.json()["detail"] == "Search engine is unavailable."

        # /api/stats must return 503
        res_stats = client.get("/api/stats")
        assert res_stats.status_code == 503
        assert res_stats.json()["detail"] == "Search engine is unavailable."

        # Verify NO fallback retriever was constructed
        assert app.state.retriever is None
    finally:
        app.state.retriever = saved_retriever


def test_missing_generator_returns_503_no_fallback(client):
    """Verify missing app.state.generator returns HTTP 503 on /api/ask without constructing fallback."""
    saved_generator = app.state.generator
    try:
        app.state.generator = None

        res_ask = client.post("/api/ask", json={"query": "test query"})
        assert res_ask.status_code == 503
        assert res_ask.json()["detail"] == "Answer generation service is unavailable."

        assert app.state.generator is None
    finally:
        app.state.generator = saved_generator


def test_health_reports_degraded_when_retriever_missing(client):
    """Verify /health reports status='degraded' and index_loaded=False when retriever is missing."""
    saved_retriever = app.state.retriever
    try:
        app.state.retriever = None

        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "degraded"
        assert data["index_loaded"] is False
    finally:
        app.state.retriever = saved_retriever

    # Restored: reports ok
    res_ok = client.get("/health")
    assert res_ok.status_code == 200
    assert res_ok.json()["status"] == "ok"
    assert res_ok.json()["index_loaded"] is True


def test_cors_configuration_consistency():
    """Verify CORS middleware does not combine wildcard origins with allow_credentials=True."""
    cors_mw = None
    for mw in app.user_middleware:
        if mw.cls == CORSMiddleware:
            cors_mw = mw
            break

    assert cors_mw is not None, "CORSMiddleware not registered on FastAPI app"
    origins = cors_mw.kwargs.get("allow_origins", [])
    allow_credentials = cors_mw.kwargs.get("allow_credentials", False)

    if "*" in origins:
        assert allow_credentials is False, (
            "CORS inconsistency: wildcard '*' origin cannot be used with allow_credentials=True"
        )
