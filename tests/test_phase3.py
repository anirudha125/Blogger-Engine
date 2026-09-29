"""
Integration tests for Phase 3: FastAPI endpoints, query logging, and frontend serving.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings
from app.db.database import get_db_connection


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Blogger Engine" in response.text


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True
    assert data["db_connected"] is True
    assert "version" in data


def test_api_stats(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] > 0
    assert data["total_documents"] > 0
    assert data["total_vectors"] > 0
    assert "bge-small-en-v1.5" in data["embedding_model"]
    assert data["query_logging_enabled"] is True


def test_api_search_success(client):
    payload = {
        "query": "What is RLAIF?",
        "top_k": 3
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is RLAIF?"
    assert len(data["results"]) == 3
    assert data["total_results"] == 3
    assert data["latency_ms"] > 0

    first = data["results"][0]
    assert first["rank"] == 1
    assert first["score"] > 0.0
    assert first["chunk_id"] != ""
    assert first["title"] != ""
    assert first["url"].startswith("http")
    assert len(first["content"]) > 0


def test_api_search_validation_errors(client):
    # Empty query fails validation
    res_empty = client.post("/api/search", json={"query": ""})
    assert res_empty.status_code == 422

    # Excessive top_k (> 20) fails validation
    res_k = client.post("/api/search", json={"query": "test", "top_k": 50})
    assert res_k.status_code == 422


def test_api_ask_success(client):
    payload = {
        "query": "How does RLAIF work with a constitution?",
        "top_k": 3
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "answer" in data
    assert len(data["sources"]) == 3
    assert data["confidence_score"] > 0.0
    assert data["latency_ms"] > 0
    assert "model" in data
    assert "provider" in data


def test_api_ask_refusal_for_out_of_domain(client):
    payload = {
        "query": "How to bake a chocolate sourdough cake?",
        "top_k": 3,
        "similarity_threshold": 0.65
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is True
    assert "below required threshold" in data["answer"]


def test_query_logging_recorded(client):
    settings = get_settings()

    # Query search endpoint
    search_query = "Logging verification query test"
    client.post("/api/search", json={"query": search_query, "top_k": 2})

    # Verify query appears in SQLite query_logs table
    with get_db_connection(settings.SQLITE_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT query, mode FROM query_logs WHERE query = ?", (search_query,))
        row = cursor.fetchone()
        assert row is not None
        assert row["query"] == search_query
        assert row["mode"] == "search"
