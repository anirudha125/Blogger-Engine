"""
Unit and regression tests for Phase 6 LLM configuration hardening and error reporting.

Verifies:
1. Whitespace stripping and normalization for provider, model, api_key, and base_url.
2. Settings model_config str_strip_whitespace behavior.
3. Enhanced HTTPStatusError reporting that preserves provider JSON error messages.
4. Secret redaction and masking in logs, representations, and error responses.
"""

import httpx
import pytest
from unittest.mock import patch, MagicMock

from app.config import Settings
from app.core.generator import AnswerGenerator
from app.core.retriever import SearchResult


def test_generator_input_sanitization():
    """Verify that provider, model, api_key, and base_url are stripped and normalized."""
    gen = AnswerGenerator(
        provider="  GROQ  ",
        model="  openai/gpt-oss-20b  ",
        api_key="  gsk_test_secret_key_12345  ",
        base_url="  https://api.groq.com/openai/v1/  "
    )

    assert gen.provider == "groq"
    assert gen.model == "openai/gpt-oss-20b"
    assert gen.api_key == "gsk_test_secret_key_12345"
    assert gen.base_url == "https://api.groq.com/openai/v1"


def test_generator_default_base_url_normalization_with_whitespace_provider():
    """Verify that a provider with whitespace properly resolves its default base URL."""
    gen = AnswerGenerator(
        provider="  groq  ",
        model="  openai/gpt-oss-20b  ",
        base_url="   "
    )

    assert gen.provider == "groq"
    assert gen.model == "openai/gpt-oss-20b"
    assert gen.base_url == "https://api.groq.com/openai/v1"


def test_settings_str_strip_whitespace(monkeypatch):
    """Verify pydantic-settings strips leading and trailing whitespace from env variables."""
    monkeypatch.setenv("LLM_PROVIDER", "  groq  ")
    monkeypatch.setenv("LLM_MODEL", "  openai/gpt-oss-20b  ")
    monkeypatch.setenv("LLM_API_KEY", "  gsk_sample_key  ")
    monkeypatch.setenv("LLM_BASE_URL", "  https://api.groq.com/openai/v1/  ")

    settings = Settings()
    assert settings.LLM_PROVIDER == "groq"
    assert settings.LLM_MODEL == "openai/gpt-oss-20b"
    assert settings.LLM_API_KEY == "gsk_sample_key"
    assert settings.LLM_BASE_URL == "https://api.groq.com/openai/v1/"


def test_generator_http_status_error_preserves_provider_json_message():
    """Verify that an HTTPStatusError preserves the provider's exact JSON error message."""
    secret_key = "gsk_live_secret_key_99999"
    gen = AnswerGenerator(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_key=secret_key,
        similarity_threshold=0.65
    )

    mock_passage = SearchResult(
        chunk_id="doc1_c0",
        doc_id="doc1",
        chunk_index=0,
        title="Title",
        url="https://example.com/1",
        author="Author",
        content="Passage content here",
        word_count=3,
        score=0.88,
        rank=1,
        faiss_id=0
    )

    # Simulate Groq 404 model_not_found error response
    mock_response = httpx.Response(
        status_code=404,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        json={
            "error": {
                "message": "The model `openai/gpt-oss-20b ` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        }
    )

    with patch("httpx.Client.post", return_value=mock_response):
        result = gen.generate_answer("How does RAG work?", [mock_passage])

    assert result.refused is True
    assert "HTTP 404:" in result.answer
    assert "The model `openai/gpt-oss-20b ` does not exist" in result.answer
    # Ensure secret key was not leaked
    assert secret_key not in result.answer


def test_generator_http_status_error_string_detail_and_redaction():
    """Verify that string error details are extracted and API keys are redacted."""
    secret_key = "gsk_super_confidential_key"
    gen = AnswerGenerator(
        provider="groq",
        model="llama-3.1-8b-instant",
        api_key=secret_key,
        similarity_threshold=0.65
    )

    mock_passage = SearchResult(
        chunk_id="doc1_c0",
        doc_id="doc1",
        chunk_index=0,
        title="Title",
        url="https://example.com/1",
        author="Author",
        content="Passage content here",
        word_count=3,
        score=0.88,
        rank=1,
        faiss_id=0
    )

    # Simulate error response containing the secret key accidentally
    mock_response = httpx.Response(
        status_code=401,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        json={"error": f"Invalid API key provided: {secret_key}"}
    )

    with patch("httpx.Client.post", return_value=mock_response):
        result = gen.generate_answer("How does RAG work?", [mock_passage])

    assert result.refused is True
    assert "HTTP 401:" in result.answer
    assert secret_key not in result.answer
    assert "[REDACTED_API_KEY]" in result.answer


def test_generator_secret_never_exposed_in_repr():
    """Verify that API keys are never exposed in AnswerGenerator representation."""
    secret_key = "gsk_private_key_abc123"
    gen = AnswerGenerator(api_key=f"  {secret_key}  ")
    repr_str = repr(gen)
    assert secret_key not in repr_str
    assert "api_key_set=Yes" in repr_str
