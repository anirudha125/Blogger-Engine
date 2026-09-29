"""
Pydantic request and response schemas for Blogger Engine API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Input payload for semantic search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query string")
    top_k: int = Field(3, ge=1, le=20, description="Maximum number of passages to retrieve")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional minimum cosine similarity filter")

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace-only.")
        return cleaned


class SearchResultItem(BaseModel):
    """Individual ranked passage retrieved from vector index."""
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    url: str
    author: str
    content: str
    word_count: int
    faiss_id: int


class SearchResponse(BaseModel):
    """Response payload for semantic search."""
    query: str
    total_results: int
    latency_ms: float
    results: List[SearchResultItem]


class AskRequest(BaseModel):
    """Input payload for RAG question answering."""
    query: str = Field(..., min_length=1, max_length=500, description="Question for the RAG engine")
    top_k: int = Field(3, ge=1, le=20, description="Number of context passages to supply to the LLM")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override minimum confidence threshold")

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace-only.")
        return cleaned


class CitationItem(BaseModel):
    """Source reference cited in the generated answer."""
    doc_tag: str
    chunk_id: str
    title: str
    url: str
    score: float


class AskResponse(BaseModel):
    """Response payload for RAG question answering with citations."""
    query: str
    answer: str
    refused: bool
    confidence_score: float
    citations: List[CitationItem]
    sources: List[SearchResultItem]
    latency_ms: float
    model: str
    provider: str


class StatsResponse(BaseModel):
    """Metadata statistics about the indexed corpus and active models."""
    total_documents: int
    total_chunks: int
    total_vectors: int
    embedding_model: str
    llm_provider: str
    llm_model: str
    query_logging_enabled: bool


class HealthResponse(BaseModel):
    """Health check status and service liveness."""
    status: str
    index_loaded: bool
    db_connected: bool
    version: str
