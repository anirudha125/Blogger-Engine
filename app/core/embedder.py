"""
Query embedding module for Blogger Engine.

Wraps SentenceTransformer for fast, CPU-friendly single-query and batch inference.
Supports instruction-prefixed models (e.g., BGE) where queries require a task-specific
prefix for optimal retrieval, while document passages are encoded without the prefix.
Reuses the exact L2-normalized embedding representation used during offline ingestion
to ensure exact cosine similarity calculation in FAISS.
"""

import os
import logging
from typing import List, Union, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("blogger_engine.embedder")

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Query instruction prefixes for retrieval-optimized models.
# Models not listed here (e.g. all-MiniLM-L6-v2) use no prefix.
# The prefix is prepended ONLY to queries, never to document passages.
QUERY_INSTRUCTION_MAP = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class QueryEmbedder:
    """CPU-friendly query and passage embedder with optional instruction prefix support."""

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
        self.device = device
        self.query_instruction = QUERY_INSTRUCTION_MAP.get(self.model_name, "")
        logger.info(f"Loading QueryEmbedder model '{self.model_name}' on {self.device.upper()}...")
        if self.query_instruction:
            logger.info(f"  Query instruction prefix: '{self.query_instruction}'")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dimension = self.model.get_embedding_dimension() or 384
        else:
            self._dimension = self.model.get_sentence_embedding_dimension() or 384

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (384 for MiniLM/BGE-small, 768 for base models)."""
        return self._dimension

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single search query into an L2-normalized 2D vector for FAISS search.

        For instruction-prefixed models (e.g., BGE), the query instruction is automatically
        prepended. Document passages should use embed_texts() which does NOT add the prefix.

        Args:
            query: The search text.

        Returns:
            np.ndarray of shape (1, dimension) with dtype float32, L2-normalized.
        """
        clean_query = query.strip()
        if not clean_query:
            # Fallback zero vector for empty query
            zero_vec = np.zeros((1, self._dimension), dtype=np.float32)
            return zero_vec

        # Prepend instruction prefix for retrieval-optimized models (e.g. BGE)
        prefixed_query = self.query_instruction + clean_query

        vec = self.model.encode(
            [prefixed_query],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype(np.float32)

        faiss.normalize_L2(vec)
        return vec

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple passages/documents with L2 normalization.

        No instruction prefix is applied — passages are encoded as-is, matching
        the ingestion-time encoding used to build the FAISS index.

        Args:
            texts: List of text strings.
            batch_size: Batch size for inference.

        Returns:
            np.ndarray of shape (len(texts), dimension) with dtype float32, L2-normalized.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        clean_texts = [t.strip() for t in texts]
        vectors = self.model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype(np.float32)

        faiss.normalize_L2(vectors)
        return vectors


# Global default embedder cache for reuse across requests
_default_embedder = None


def get_default_embedder() -> QueryEmbedder:
    """Return a shared singleton instance of QueryEmbedder to avoid reloading model weights."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = QueryEmbedder()
    return _default_embedder
