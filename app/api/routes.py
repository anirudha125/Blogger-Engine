"""
FastAPI route handlers for Blogger Engine RAG platform.
"""

import time
import logging
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, Depends

from app.config import Settings, get_settings
from app.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    AskRequest,
    AskResponse,
    CitationItem,
    StatsResponse,
    HealthResponse,
)
from app.core.retriever import SearchResult
from app.core.hybrid_retriever import HybridRetriever
from app.core.generator import AnswerGenerator, GeneratedAnswer
from app.core.preview import extract_document_preview
from app.db.database import (
    get_total_chunks,
    get_total_documents,
    get_first_chunks_by_doc_ids,
    log_query,
)

logger = logging.getLogger("blogger_engine.api")
router = APIRouter()


def get_retriever(request: Request) -> HybridRetriever:
    """Retrieve the canonical HybridRetriever stored in app state or raise 503."""
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine is unavailable."
        )
    return retriever


def get_generator(request: Request) -> AnswerGenerator:
    """Retrieve the AnswerGenerator instance stored in app state or raise 503."""
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Answer generation service is unavailable."
        )
    return generator


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Liveness probe verifying that the vector index and database are available."""
    retriever: Optional[HybridRetriever] = getattr(request.app.state, "retriever", None)
    index_loaded = retriever is not None and retriever.total_vectors > 0
    try:
        # Check SQLite connectivity
        total_chunks = get_total_chunks(settings.SQLITE_DB_PATH)
        db_connected = total_chunks >= 0
    except Exception as exc:
        logger.error(f"Health check database failure: {exc}")
        db_connected = False

    status = "ok" if (index_loaded and db_connected) else "degraded"
    return HealthResponse(
        status=status,
        index_loaded=index_loaded,
        db_connected=db_connected,
        version=settings.APP_VERSION
    )


@router.get("/api/stats", response_model=StatsResponse, tags=["Analytics"])
def get_stats(
    settings: Settings = Depends(get_settings),
    retriever: HybridRetriever = Depends(get_retriever),
    generator: AnswerGenerator = Depends(get_generator)
) -> StatsResponse:
    """Return summary statistics of the indexed corpus, models, and system status."""
    try:
        total_docs = get_total_documents(settings.SQLITE_DB_PATH)
        total_chunks = get_total_chunks(settings.SQLITE_DB_PATH)
    except Exception as exc:
        logger.error(f"Error querying statistics: {exc}")
        total_docs = 0
        total_chunks = 0

    return StatsResponse(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_vectors=retriever.total_vectors,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_provider=generator.provider,
        llm_model=generator.model,
        query_logging_enabled=settings.ENABLE_QUERY_LOGGING
    )


@router.post("/api/search", response_model=SearchResponse, tags=["Search"])
def search(
    payload: SearchRequest,
    settings: Settings = Depends(get_settings),
    retriever: HybridRetriever = Depends(get_retriever)
) -> SearchResponse:
    """Perform hybrid dense + BM25 search with cross-encoder reranking across indexed blog passages."""
    start_time = time.time()

    query_str = payload.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        results: List[SearchResult] = retriever.retrieve(
            query=query_str,
            top_k=payload.top_k,
            min_score=payload.min_score
        )
    except Exception as exc:
        logger.error(f"Retrieval error during search: {exc}")
        raise HTTPException(status_code=500, detail=f"Search retrieval error: {str(exc)}")

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # Map to schema items with first-chunk document preview
    doc_ids = [r.doc_id for r in results]
    first_chunks_map = {}
    try:
        first_chunks_map = get_first_chunks_by_doc_ids(doc_ids, db_path=settings.SQLITE_DB_PATH)
    except Exception as exc:
        logger.warning(f"Failed to fetch first chunks for preview: {exc}")

    items = []
    for r in results:
        first_chunk = first_chunks_map.get(r.doc_id)
        first_content = first_chunk["content"] if first_chunk else None
        preview_text = extract_document_preview(
            first_chunk_content=first_content,
            fallback_content=r.content,
            title=r.title
        )
        items.append(
            SearchResultItem(
                rank=r.rank,
                score=round(r.score, 4),
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                chunk_index=r.chunk_index,
                title=r.title,
                url=r.url,
                author=r.author,
                content=r.content,
                word_count=r.word_count,
                faiss_id=r.faiss_id,
                preview=preview_text
            )
        )

    # Optional query logging: failure never breaks response
    try:
        retrieved_ids = [r.chunk_id for r in results]
        log_query(
            query=query_str,
            retrieved_ids=retrieved_ids,
            latency_ms=elapsed_ms,
            mode="search",
            db_path=settings.SQLITE_DB_PATH,
            enabled=settings.ENABLE_QUERY_LOGGING
        )
    except Exception as log_exc:
        logger.warning(f"Optional query logging skipped: {log_exc}")

    return SearchResponse(
        query=query_str,
        total_results=len(items),
        latency_ms=elapsed_ms,
        results=items
    )


@router.post("/api/ask", response_model=AskResponse, tags=["RAG QA"])
def ask(
    payload: AskRequest,
    settings: Settings = Depends(get_settings),
    retriever: HybridRetriever = Depends(get_retriever),
    generator: AnswerGenerator = Depends(get_generator)
) -> AskResponse:
    """
    Execute full RAG QA: retrieves passages, formats grounded prompt,
    calls configurable LLM with strict citation constraints, and checks refusal threshold.
    """
    start_time = time.time()

    query_str = payload.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Question query cannot be empty.")

    # 1. Retrieve top passages
    try:
        passages: List[SearchResult] = retriever.retrieve(
            query=query_str,
            top_k=payload.top_k
        )
    except Exception as exc:
        logger.error(f"Retrieval error during ask: {exc}")
        raise HTTPException(status_code=500, detail=f"QA retrieval error: {str(exc)}")

    # 2. Generate grounded answer
    try:
        generated: GeneratedAnswer = generator.generate_answer(
            query=query_str,
            passages=passages,
            similarity_threshold=payload.similarity_threshold
        )
    except Exception as exc:
        logger.error(f"Generation error during ask: {exc}")
        raise HTTPException(status_code=500, detail=f"QA generation error: {str(exc)}")

    total_latency_ms = round((time.time() - start_time) * 1000, 2)

    # Map sources and citations
    citation_items = [
        CitationItem(
            doc_tag=c.doc_tag,
            chunk_id=c.chunk_id,
            title=c.title,
            url=c.url,
            score=round(c.score, 4)
        )
        for c in generated.citations
    ]

    source_items = [
        SearchResultItem(
            rank=r.rank,
            score=round(r.score, 4),
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            chunk_index=r.chunk_index,
            title=r.title,
            url=r.url,
            author=r.author,
            content=r.content,
            word_count=r.word_count,
            faiss_id=r.faiss_id
        )
        for r in passages
    ]

    # Optional query logging: failure never breaks response
    try:
        retrieved_ids = [r.chunk_id for r in passages]
        log_query(
            query=query_str,
            retrieved_ids=retrieved_ids,
            latency_ms=total_latency_ms,
            mode="ask",
            db_path=settings.SQLITE_DB_PATH,
            enabled=settings.ENABLE_QUERY_LOGGING
        )
    except Exception as log_exc:
        logger.warning(f"Optional query logging skipped: {log_exc}")

    return AskResponse(
        query=query_str,
        answer=generated.answer,
        refused=generated.refused,
        confidence_score=generated.confidence_score,
        citations=citation_items,
        sources=source_items,
        latency_ms=total_latency_ms,
        model=generated.model,
        provider=generated.provider
    )
