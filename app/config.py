"""
Configuration settings for Blogger Engine using pydantic-settings.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Provider Configuration
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""

    # Embedding & Vector Index Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    FAISS_INDEX_PATH: str = "indexes/faiss_chunked_bge_dedup.index"
    SQLITE_DB_PATH: str = "data/blogger_dedup.db"
    BM25_CACHE_PATH: str = "indexes/bm25_chunked_dedup.pkl"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval Pipeline Configuration (Experiment 4 Canonical)
    TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.65
    CANDIDATE_DEPTH: int = 20
    RERANK_DEPTH: int = 20
    RRF_K: int = 60
    DOC_DEDUP: bool = True
    USE_RERANKER: bool = True

    # Application & Logging
    ENABLE_QUERY_LOGGING: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    APP_VERSION: str = "1.0.0"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
