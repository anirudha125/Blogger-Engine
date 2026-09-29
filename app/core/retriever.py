"""
Retriever module for Blogger Engine.

Loads the chunked FAISS vector index, embeds queries on CPU, retrieves top-k passage IDs,
fetches the full text and metadata from SQLite, and returns ranked results with similarity scores.
"""

import os
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import faiss

from app.core.embedder import QueryEmbedder, get_default_embedder
from app.db.database import get_chunks_by_faiss_ids

logger = logging.getLogger("blogger_engine.retriever")


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    url: str
    author: str
    content: str
    word_count: int
    score: float
    rank: int
    faiss_id: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Retriever:
    """Orchestrates FAISS vector search and SQLite chunk retrieval."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        db_path: Optional[str] = None,
        embedder: Optional[QueryEmbedder] = None,
        default_min_score: Optional[float] = None
    ):
        self.index_path = Path(index_path or os.getenv("FAISS_INDEX_PATH", "indexes/faiss_chunked.index"))
        self.db_path = str(db_path or os.getenv("SQLITE_DB_PATH", "data/blogger.db"))
        self.embedder = embedder or get_default_embedder()
        self.default_min_score = default_min_score
        self._index: Optional[faiss.Index] = None
        self._load_index()

    def _load_index(self) -> None:
        """Load and validate FAISS index and SQLite database from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_path}'. Run scripts/ingest.py first."
            )
        db_file = Path(self.db_path)
        if not db_file.exists():
            raise FileNotFoundError(
                f"SQLite database not found at '{self.db_path}'. Run scripts/ingest.py first."
            )
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self._index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors.")

    @property
    def total_vectors(self) -> int:
        """Total vectors stored in the FAISS index."""
        return self._index.ntotal if self._index else 0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Perform dense semantic retrieval for a user query.

        Args:
            query: The user query string.
            top_k: Number of top passages to retrieve.
            min_score: Optional minimum cosine similarity threshold.

        Returns:
            List of SearchResult objects sorted by score descending.
        """
        if not query or not query.strip():
            return []

        if self._index is None or self.total_vectors == 0:
            logger.warning("Retriever index is empty or not loaded.")
            return []

        # 1. Embed query on CPU
        query_vec = self.embedder.embed_query(query)

        # 2. Search FAISS index
        effective_k = min(top_k, self.total_vectors)
        scores, indices = self._index.search(query_vec, effective_k)

        # 3. Filter valid IDs (FAISS returns -1 for empty/insufficient slots)
        faiss_ids: List[int] = []
        id_score_map: Dict[int, float] = {}

        effective_min = min_score if min_score is not None else self.default_min_score
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                float_score = float(score)
                if effective_min is not None and float_score < effective_min:
                    continue
                int_idx = int(idx)
                faiss_ids.append(int_idx)
                id_score_map[int_idx] = float_score

        if not faiss_ids:
            return []

        # 4. Fetch corresponding passage text & metadata from SQLite
        chunk_records = get_chunks_by_faiss_ids(faiss_ids, db_path=self.db_path)

        # 5. Build SearchResult objects
        results: List[SearchResult] = []
        for rank, record in enumerate(chunk_records, start=1):
            fid = record["faiss_id"]
            results.append(
                SearchResult(
                    chunk_id=record["id"],
                    doc_id=record["doc_id"],
                    chunk_index=record["chunk_index"],
                    title=record["title"],
                    url=record["url"],
                    author=record["author"],
                    content=record["content"],
                    word_count=record["word_count"],
                    score=id_score_map.get(fid, 0.0),
                    rank=rank,
                    faiss_id=fid
                )
            )

        # Ensure sorted by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results, start=1):
            r.rank = i

        return results
