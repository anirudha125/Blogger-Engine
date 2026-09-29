"""
SQLite database interface for Blogger Engine.

Stores chunked blog content and metadata. Supports optional, ephemeral-safe query logging.
"""

import sqlite3
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Dict, Any

if TYPE_CHECKING:
    from app.core.chunker import DocumentChunk

logger = logging.getLogger("blogger_engine.db")


@contextmanager
def get_db_connection(db_path: str = "data/blogger.db"):
    """Context manager for SQLite connections that guarantees closing."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = "data/blogger.db") -> None:
    """Initialize database tables and indexes."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Documents/chunks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                faiss_id INTEGER UNIQUE,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                author TEXT NOT NULL,
                content TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Performance indexes for metadata lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_url ON documents(url)"
        )

        # Check existing table for backwards-compatibility migration before indexing
        cursor.execute("PRAGMA table_info(documents)")
        existing_cols = [row["name"] for row in cursor.fetchall()]
        if "faiss_id" not in existing_cols:
            cursor.execute("ALTER TABLE documents ADD COLUMN faiss_id INTEGER")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_faiss_id ON documents(faiss_id)"
        )

        # Query logging table (optional/ephemeral-safe)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                retrieved_ids TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                mode TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def insert_chunks(
    chunks: List["DocumentChunk"],
    db_path: str = "data/blogger.db",
    batch_size: int = 500
) -> int:
    """
    Bulk insert or replace document chunks in SQLite with deterministic faiss_id.

    Returns:
        Number of chunks inserted.
    """
    if not chunks:
        return 0

    init_db(db_path)

    query = """
        INSERT OR REPLACE INTO documents (
            id, faiss_id, doc_id, chunk_index, title, url, author, content, word_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    inserted = 0
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            data = [
                (
                    c.chunk_id,
                    c.faiss_id,
                    c.doc_id,
                    c.chunk_index,
                    c.title,
                    c.url,
                    c.author,
                    c.content,
                    c.word_count
                )
                for c in batch
            ]
            cursor.executemany(query, data)
            inserted += len(data)
        conn.commit()

    return inserted


def get_chunk_by_id(
    chunk_id: str,
    db_path: str = "data/blogger.db"
) -> Optional[Dict[str, Any]]:
    """Retrieve a single document chunk by its ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_chunks_by_ids(
    chunk_ids: List[str],
    db_path: str = "data/blogger.db"
) -> List[Dict[str, Any]]:
    """Retrieve multiple document chunks preserving requested order."""
    if not chunk_ids:
        return []

    placeholders = ",".join("?" for _ in chunk_ids)
    query = f"SELECT * FROM documents WHERE id IN ({placeholders})"

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, chunk_ids)
        rows = cursor.fetchall()
        row_map = {row["id"]: dict(row) for row in rows}

    # Return in order of chunk_ids
    return [row_map[cid] for cid in chunk_ids if cid in row_map]


def get_chunks_by_faiss_ids(
    faiss_ids: List[int],
    db_path: str = "data/blogger.db"
) -> List[Dict[str, Any]]:
    """
    Retrieve document chunks by explicit, deterministic faiss_id.
    Safely falls back to (rowid - 1) if faiss_id is unpopulated.
    Preserves exact query order.
    """
    if not faiss_ids:
        return []

    placeholders = ",".join("?" for _ in faiss_ids)
    query = f"""
        SELECT COALESCE(faiss_id, rowid - 1) AS faiss_id, *
        FROM documents
        WHERE faiss_id IN ({placeholders}) OR (faiss_id IS NULL AND (rowid - 1) IN ({placeholders}))
    """

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, faiss_ids + faiss_ids)
        rows = cursor.fetchall()
        row_map = {row["faiss_id"]: dict(row) for row in rows}

    return [row_map[fid] for fid in faiss_ids if fid in row_map]


def get_total_chunks(db_path: str = "data/blogger.db") -> int:
    """Return count of total indexed chunks."""
    if not Path(db_path).exists():
        return 0
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]


def get_total_documents(db_path: str = "data/blogger.db") -> int:
    """Return count of unique original documents."""
    if not Path(db_path).exists():
        return 0
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM documents")
        return cursor.fetchone()[0]


def log_query(
    query: str,
    retrieved_ids: List[str],
    latency_ms: float,
    mode: str = "search",
    db_path: str = "data/blogger.db",
    enabled: bool = True
) -> bool:
    """
    Log an executed search or QA query to SQLite.

    Safe for ephemeral environments:
    - If disabled or if the file system is read-only, catches errors gracefully without raising.
    """
    if not enabled:
        return False

    try:
        init_db(db_path)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_logs (query, retrieved_ids, latency_ms, mode)
                VALUES (?, ?, ?, ?)
                """,
                (
                    query,
                    json.dumps(retrieved_ids),
                    latency_ms,
                    mode
                )
            )
            conn.commit()
            return True
    except Exception as exc:
        logger.warning("Failed to record query log (ephemeral disk or permission issue): %s", exc)
        return False
