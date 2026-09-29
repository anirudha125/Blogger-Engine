"""Database module for Blogger Engine."""
from app.db.database import (
    init_db,
    insert_chunks,
    get_chunk_by_id,
    get_chunks_by_ids,
    get_chunks_by_faiss_ids,
    get_total_chunks,
    get_total_documents,
    log_query,
)

__all__ = [
    "init_db",
    "insert_chunks",
    "get_chunk_by_id",
    "get_chunks_by_ids",
    "get_chunks_by_faiss_ids",
    "get_total_chunks",
    "get_total_documents",
    "log_query",
]
