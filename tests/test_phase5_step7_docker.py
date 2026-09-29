"""
Tests for Phase 5 — Step 7: Dockerization & Deployment Packaging.

Validates:
1. Dockerfile existence, syntax, and multi-stage architecture.
2. CPU-only PyTorch installation preventing CUDA package bloat.
3. Deterministic model weight pre-caching and offline environment flags.
4. Non-root user execution and security hardening.
5. Canonical artifact handling for FAISS, SQLite DB, and BM25.
6. Container health check and production entrypoint configuration.
7. .dockerignore rules ensuring clean build context without excluding canonical assets.
"""

from pathlib import Path


def test_dockerfile_structure_and_multistage():
    """Verify Dockerfile uses multi-stage build with Python 3.11 slim."""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists(), "Dockerfile is missing"
    content = dockerfile.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim AS builder" in content
    assert "FROM python:3.11-slim AS runner" in content
    assert "COPY --from=builder" in content


def test_dockerfile_cpu_torch_and_model_precaching():
    """Verify CPU-only PyTorch and model pre-downloading into image layer."""
    dockerfile = Path("Dockerfile")
    content = dockerfile.read_text(encoding="utf-8")

    # CPU-only PyTorch
    assert "--index-url https://download.pytorch.org/whl/cpu" in content

    # Canonical model pre-downloads
    assert "BAAI/bge-small-en-v1.5" in content
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in content

    # Offline runtime flags
    assert "TRANSFORMERS_OFFLINE=1" in content
    assert "HF_HUB_OFFLINE=1" in content


def test_dockerfile_security_and_non_root_user():
    """Verify Dockerfile creates and executes as a dedicated non-root user."""
    dockerfile = Path("Dockerfile")
    content = dockerfile.read_text(encoding="utf-8")

    assert "useradd -r" in content
    assert "USER appuser" in content
    assert "chown" in content


def test_dockerfile_canonical_artifact_packaging():
    """Verify canonical assets (FAISS, SQLite, BM25) are packaged properly."""
    dockerfile = Path("Dockerfile")
    content = dockerfile.read_text(encoding="utf-8")

    assert "indexes/faiss_chunked_bge_dedup.index" in content
    assert "data/blogger_dedup.db" in content
    assert "indexes/bm25_chunked_dedup.pkl" in content
    assert "gzip" in content


def test_dockerfile_healthcheck_and_entrypoint():
    """Verify production HEALTHCHECK and Uvicorn entrypoint configuration."""
    dockerfile = Path("Dockerfile")
    content = dockerfile.read_text(encoding="utf-8")

    assert "HEALTHCHECK" in content
    assert "/health" in content
    assert "uvicorn app.main:create_app --factory" in content
    assert "--host 0.0.0.0" in content
    assert "${PORT:-8000}" in content


def test_dockerignore_exclusions_and_canonical_inclusions():
    """Verify .dockerignore excludes research files while preserving canonical assets."""
    dockerignore = Path(".dockerignore")
    assert dockerignore.exists(), ".dockerignore is missing"
    content = dockerignore.read_text(encoding="utf-8")

    # Required exclusions
    assert ".git" in content
    assert ".env" in content
    assert "notebooks/" in content
    assert "tests/" in content
    assert "scripts/" in content
    assert "data/blogger.db" in content
    assert "indexes/faiss_chunked.index" in content

    # Ensure canonical assets are NOT excluded
    assert "data/blogger_dedup.db\n" not in content
    assert "data/blogger_dedup.db.gz\n" not in content
    assert "indexes/faiss_chunked_bge_dedup.index" not in content
    assert "indexes/bm25_chunked_dedup.pkl" not in content
