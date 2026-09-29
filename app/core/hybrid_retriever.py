"""
Hybrid Dense + BM25 Retriever with Reciprocal Rank Fusion (RRF) for Blogger Engine.

Combines semantic dense retrieval (FAISS IndexFlatIP with BGE embeddings) and
lexical sparse retrieval (rank-bm25 BM25Okapi over canonical SQLite chunk texts).
Fuses candidate rankings using standard Reciprocal Rank Fusion (RRF, k=60)
to achieve superior retrieval recall on both semantic and exact keyword/acronym queries.
"""

import os
import re
import time
import pickle
import logging
import sqlite3
import threading
from contextvars import ContextVar
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.retriever import Retriever, SearchResult
from app.db.database import get_chunks_by_faiss_ids

logger = logging.getLogger("blogger_engine.hybrid_retriever")

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_DEPTH = 20
DEFAULT_RERANK_DEPTH = 20
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Request-scoped latency breakdown storage using contextvars to eliminate race conditions
_request_latency_var: ContextVar[Dict[str, float]] = ContextVar(
    "request_latency_breakdown",
    default={
        "dense_ms": 0.0,
        "bm25_ms": 0.0,
        "expansion_ms": 0.0,
        "rrf_ms": 0.0,
        "rerank_ms": 0.0,
        "fusion_ms": 0.0,
        "total_ms": 0.0
    }
)


class RetrievalResultList(list):
    """
    Subclass of list holding SearchResult items along with per-retrieval
    latency breakdown metadata.
    Subclasses list so all indexing, iteration, slicing, and list operations
    remain 100% identical to standard List[SearchResult].
    """
    def __init__(
        self,
        items: Optional[List[SearchResult]] = None,
        latency_breakdown: Optional[Dict[str, float]] = None
    ):
        super().__init__(items or [])
        self.latency_breakdown: Dict[str, float] = latency_breakdown or {
            "dense_ms": 0.0,
            "bm25_ms": 0.0,
            "expansion_ms": 0.0,
            "rrf_ms": 0.0,
            "rerank_ms": 0.0,
            "fusion_ms": 0.0,
            "total_ms": 0.0
        }


# Standard English stopwords for Pseudo-Relevance Feedback (PRF)
PRF_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "using", "use", "used", "uses", "one", "two",
    "many", "new", "like", "well", "may", "can", "first", "second", "often"
}


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into alphanumeric words and hyphens.
    Preserves technical terms, acronyms, and model names (e.g., 'sl-cai', 'dpo', 'llama-3').
    """
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def extract_prf_keywords(
    query: str,
    chunk_texts: List[str],
    top_k: int = 4
) -> List[str]:
    """
    Extract top PRF keywords from initial retrieval passages.
    Filters out query tokens, stopwords, numbers, and short tokens (<3 chars).
    Ranks terms by term frequency across top passages with alphabetical tie-breaking for determinism.
    """
    if not chunk_texts or top_k <= 0:
        return []
    query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
    term_counts: Counter = Counter()
    for text in chunk_texts:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        for w in words:
            if w not in PRF_STOPWORDS and w not in query_tokens:
                term_counts[w] += 1

    # Sort deterministically: frequency descending, then alphabetical ascending
    sorted_terms = sorted(term_counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in sorted_terms[:top_k]]


class BM25Index:
    """In-memory BM25 index built over canonical SQLite chunk texts."""

    def __init__(self, db_path: str, cache_path: Optional[str] = None):
        self.db_path = str(db_path)
        self.cache_path = Path(cache_path) if cache_path else None
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_ids: List[int] = []
        self._load_or_build()

    def _load_or_build(self) -> None:
        """Load pre-built BM25 index from cache if available; otherwise build from SQLite."""
        if self.cache_path and self.cache_path.exists():
            try:
                t0 = time.time()
                logger.info(f"Loading cached BM25 index from {self.cache_path}...")
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.faiss_ids = data["faiss_ids"]
                logger.info(
                    f"Loaded BM25 index ({len(self.faiss_ids):,} chunks) in {time.time()-t0:.2f}s."
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load cached BM25 index: {e}. Rebuilding...")

        self._build_from_db()

    def _build_from_db(self) -> None:
        """Build BM25Okapi index directly from SQLite chunk texts in faiss_id order."""
        t0 = time.time()
        logger.info(f"Building BM25 index from database: {self.db_path}...")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT faiss_id, content FROM documents ORDER BY faiss_id ASC")
        rows = c.fetchall()
        conn.close()

        if not rows:
            raise ValueError(f"No documents found in {self.db_path} to build BM25 index!")

        self.faiss_ids = [r[0] for r in rows]
        corpus_tokens = [tokenize_text(r[1]) for r in rows]
        t_tok = time.time()
        logger.info(f"Tokenized {len(rows):,} chunks in {t_tok - t0:.2f}s.")

        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Built BM25Okapi index in {time.time() - t_tok:.2f}s.")

        # Save cache if path provided
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump({"bm25": self.bm25, "faiss_ids": self.faiss_ids}, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Cached BM25 index to {self.cache_path} ({self.cache_path.stat().st_size / (1024*1024):.1f} MB).")
            except Exception as e:
                logger.warning(f"Failed to cache BM25 index: {e}")

    @property
    def total_docs(self) -> int:
        return len(self.faiss_ids)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Search BM25 index and return list of (faiss_id, bm25_score) tuples.
        Only returns documents with positive BM25 scores.
        """
        if self.bm25 is None or not query.strip():
            return []

        q_tokens = tokenize_text(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        effective_k = min(top_k, len(scores))
        if effective_k == 0:
            return []

        # Find top_k indices efficiently
        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        # Sort descending by score
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[Tuple[int, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.faiss_ids[idx], score))

        return results


class HybridRetriever:
    """
    Orchestrates Dense + BM25 hybrid retrieval fused via Reciprocal Rank Fusion (RRF).
    Fully compatible with the standard Retriever interface.
    """

    def __init__(
        self,
        dense_retriever: Optional[Retriever] = None,
        bm25_index: Optional[BM25Index] = None,
        index_path: Optional[str] = None,
        db_path: Optional[str] = None,
        bm25_cache_path: Optional[str] = None,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
        default_min_score: Optional[float] = None,
        doc_dedup: bool = True,
        fused_depth: int = 20,
        use_reranker: bool = True,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
        use_query_expansion: bool = False,
        expansion_keywords: int = 4,
        expansion_feedback_depth: int = 3,
        use_score_fusion: bool = False,
        fusion_alpha: float = 0.7
    ):
        if dense_retriever is None:
            index_p = index_path or os.getenv("FAISS_INDEX_PATH", "indexes/faiss_chunked_bge_dedup.index")
            db_p = db_path or os.getenv("SQLITE_DB_PATH", "data/blogger_dedup.db")
            self.dense_retriever = Retriever(
                index_path=index_p,
                db_path=db_p,
                default_min_score=default_min_score
            )
        else:
            self.dense_retriever = dense_retriever

        self.index_path = getattr(self.dense_retriever, "index_path", None)
        self.db_path = str(db_path or self.dense_retriever.db_path)
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k
        self.default_min_score = default_min_score
        self.doc_dedup = doc_dedup
        self.fused_depth = fused_depth
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.rerank_depth = rerank_depth
        self.use_query_expansion = use_query_expansion
        self.expansion_keywords = expansion_keywords
        self.expansion_feedback_depth = expansion_feedback_depth
        self.use_score_fusion = use_score_fusion
        self.fusion_alpha = fusion_alpha
        self._cross_encoder = None
        self._lock = threading.Lock()

        if bm25_index is not None:
            self.bm25_index = bm25_index
        else:
            cache_p = bm25_cache_path or os.getenv("BM25_CACHE_PATH", "indexes/bm25_chunked_dedup.pkl")
            self.bm25_index = BM25Index(db_path=self.db_path, cache_path=cache_p)

    @property
    def total_vectors(self) -> int:
        return self.dense_retriever.total_vectors

    @property
    def cross_encoder(self):
        """Thread-safe lazy loader for cross-encoder model."""
        if self._cross_encoder is None and self.use_reranker:
            with self._lock:
                if self._cross_encoder is None:
                    from sentence_transformers import CrossEncoder
                    self._cross_encoder = CrossEncoder(self.reranker_model, device="cpu")
        return self._cross_encoder

    def warmup(self) -> None:
        """
        Explicitly pre-load cross-encoder model weights into memory.
        Safe to call once during application startup (lifespan) to eliminate
        first-request cold-start latency.

        Requirements:
        - Uses the configured `self.reranker_model`
        - Idempotent: repeated calls do not reload the model
        - Fails clearly with descriptive RuntimeError if weights cannot load
        - Does NOT perform fake warmup inference or pollute metrics
        """
        if not self.use_reranker:
            logger.info("Warmup skipped: reranker is disabled (use_reranker=False).")
            return

        if self._cross_encoder is not None:
            logger.debug(f"CrossEncoder already warmed up with model '{self.reranker_model}'.")
            return

        with self._lock:
            if self._cross_encoder is not None:
                return
            t0 = time.time()
            logger.info(f"Warming up CrossEncoder with model '{self.reranker_model}' on CPU...")
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.reranker_model, device="cpu")
                logger.info(f"CrossEncoder warmed up successfully in {time.time() - t0:.2f}s.")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model '{self.reranker_model}': {e}")
                raise RuntimeError(
                    f"CrossEncoder warmup failed for model '{self.reranker_model}': {e}"
                ) from e

    @property
    def last_latency_breakdown(self) -> Dict[str, float]:
        """
        Request-isolated latency breakdown for the current execution context.
        Note: For concurrent execution safety, callers should read `results.latency_breakdown`
        directly from the returned `RetrievalResultList` object.
        """
        return _request_latency_var.get()


    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
        doc_dedup: Optional[bool] = None,
        fused_depth: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        rerank_depth: Optional[int] = None,
        use_query_expansion: Optional[bool] = None,
        expansion_keywords: Optional[int] = None,
        expansion_feedback_depth: Optional[int] = None,
        candidate_depth: Optional[int] = None,
        use_score_fusion: Optional[bool] = None,
        fusion_alpha: Optional[float] = None
    ) -> RetrievalResultList:
        """
        Perform hybrid dense + BM25 retrieval with Reciprocal Rank Fusion (RRF),
        optional Pseudo-Relevance Feedback (PRF) query expansion, optional Cross-Encoder reranking,
        and optional RRF + Cross-Encoder Score Fusion (Experiment 7a).

        Algorithm:
        1. Retrieve top-N candidates from BM25 sparse index (default N=20).
        2. If use_query_expansion:
           - Extract top keywords from top feedback chunks of initial BM25 search.
           - Formulate expanded query: query + ' ' + keywords.
        3. Retrieve top-N candidates from Dense Retriever using (expanded) query.
        4. Compute RRF score for all candidates: score(d) = sum(1 / (k + rank_i(d))).
        5. Rank by RRF score descending.
        6. If use_reranker:
           - Pool top `rerank_depth` candidate chunks (default 20).
           - Score all (query, chunk_content) pairs via CrossEncoder using original query.
           - If use_score_fusion:
             - Compute Min-Max normalized CE and RRF scores across pool.
             - Compute fused_score = alpha * norm_CE + (1-alpha) * norm_RRF.
             - Sort pool by fused_score descending.
           - Else:
             - Sort pool by cross-encoder relevance score descending.
           - If doc_dedup: retain highest scoring chunk per unique doc_id.
           - Return top-k unique documents.
        7. Else if doc_dedup:
           - Pool top `fused_depth` candidates (default 10).
           - Retain highest RRF-ranked chunk per unique doc_id.
           - Return top-k unique documents.
        8. Else:
           - Return top-k candidates directly.
        9. Compute/preserve exact cosine similarity for confidence gating & citations.
        """
        if not query or not query.strip():
            empty_breakdown = {
                "dense_ms": 0.0,
                "bm25_ms": 0.0,
                "expansion_ms": 0.0,
                "rrf_ms": 0.0,
                "rerank_ms": 0.0,
                "fusion_ms": 0.0,
                "total_ms": 0.0
            }
            _request_latency_var.set(empty_breakdown)
            return RetrievalResultList([], latency_breakdown=empty_breakdown)

        use_doc_dedup = self.doc_dedup if doc_dedup is None else doc_dedup
        pool_depth = self.fused_depth if fused_depth is None else fused_depth
        do_rerank = self.use_reranker if use_reranker is None else use_reranker
        ce_depth = self.rerank_depth if rerank_depth is None else rerank_depth
        do_expansion = self.use_query_expansion if use_query_expansion is None else use_query_expansion
        n_keywords = self.expansion_keywords if expansion_keywords is None else expansion_keywords
        fb_depth = self.expansion_feedback_depth if expansion_feedback_depth is None else expansion_feedback_depth
        do_score_fusion = self.use_score_fusion if use_score_fusion is None else use_score_fusion
        alpha = self.fusion_alpha if fusion_alpha is None else fusion_alpha

        t_start = time.time()
        expansion_ms = 0.0
        rerank_ms = 0.0
        fusion_ms = 0.0
        dense_query = query

        c_depth = self.candidate_depth if candidate_depth is None else candidate_depth

        # 1. BM25 retrieval
        t_b0 = time.time()
        bm25_results = self.bm25_index.search(query, top_k=c_depth)
        bm25_ms = (time.time() - t_b0) * 1000

        bm25_rank_map: Dict[int, int] = {
            fid: rank for rank, (fid, _score) in enumerate(bm25_results, start=1)
        }

        # 2. PRF Query Expansion (if enabled)
        if do_expansion and bm25_results:
            t_exp0 = time.time()
            top_fb_fids = [fid for fid, _ in bm25_results[:fb_depth]]
            fb_records = get_chunks_by_faiss_ids(top_fb_fids, db_path=self.db_path)
            fb_texts = [r["content"] for r in fb_records]
            prf_keywords = extract_prf_keywords(query, fb_texts, top_k=n_keywords)
            if prf_keywords:
                dense_query = f"{query} {' '.join(prf_keywords)}"
            expansion_ms = (time.time() - t_exp0) * 1000

        # 3. Dense retrieval with (possibly expanded) query
        t_d0 = time.time()
        dense_results = self.dense_retriever.retrieve(dense_query, top_k=c_depth)
        dense_ms = (time.time() - t_d0) * 1000

        dense_rank_map: Dict[int, int] = {
            r.faiss_id: rank for rank, r in enumerate(dense_results, start=1)
        }
        dense_obj_map: Dict[int, SearchResult] = {r.faiss_id: r for r in dense_results}

        # 3. Reciprocal Rank Fusion (RRF)
        t_r0 = time.time()
        all_candidate_ids = set(dense_rank_map.keys()) | set(bm25_rank_map.keys())
        if not all_candidate_ids:
            no_cand_breakdown = {
                "dense_ms": round(dense_ms, 2),
                "bm25_ms": round(bm25_ms, 2),
                "expansion_ms": round(expansion_ms, 2),
                "rrf_ms": 0.0,
                "rerank_ms": 0.0,
                "fusion_ms": 0.0,
                "total_ms": round((time.time() - t_start) * 1000, 2)
            }
            _request_latency_var.set(no_cand_breakdown)
            return RetrievalResultList([], latency_breakdown=no_cand_breakdown)

        rrf_scores: Dict[int, float] = {}
        for fid in all_candidate_ids:
            score = 0.0
            if fid in dense_rank_map:
                score += 1.0 / (self.rrf_k + dense_rank_map[fid])
            if fid in bm25_rank_map:
                score += 1.0 / (self.rrf_k + bm25_rank_map[fid])
            rrf_scores[fid] = score

        # 4. Sort candidates by RRF score descending
        # Secondary sort key: dense rank if present, else 9999
        sorted_fids = sorted(
            all_candidate_ids,
            key=lambda fid: (rrf_scores[fid], -dense_rank_map.get(fid, 9999)),
            reverse=True
        )
        rrf_ms = (time.time() - t_r0) * 1000

        rerank_ms = 0.0

        if do_rerank and self.cross_encoder is not None:
            # 5. Cross-Encoder Reranking over top fused candidate pool
            t_ce0 = time.time()
            ce_pool_fids = sorted_fids[:ce_depth]

            # Fetch SQLite metadata for all pool candidates so content and doc_id are known
            missing_fids = [fid for fid in ce_pool_fids if fid not in dense_obj_map]
            if missing_fids:
                fetched_records = get_chunks_by_faiss_ids(missing_fids, db_path=self.db_path)
                for rec in fetched_records:
                    fid = rec["faiss_id"]
                    dense_obj_map[fid] = SearchResult(
                        chunk_id=rec["id"],
                        doc_id=rec["doc_id"],
                        chunk_index=rec["chunk_index"],
                        title=rec["title"],
                        url=rec["url"],
                        author=rec["author"],
                        content=rec["content"],
                        word_count=rec["word_count"],
                        score=0.0,
                        rank=0,
                        faiss_id=fid
                    )

            # Score each (query, candidate_passage) pair
            pairs = [(query, dense_obj_map[fid].content) for fid in ce_pool_fids]
            ce_scores = self.cross_encoder.predict(pairs)
            ce_score_map = {fid: float(score) for fid, score in zip(ce_pool_fids, ce_scores)}

            if do_score_fusion:
                # Experiment 7a: RRF + Cross-Encoder Score Fusion
                t_f0 = time.time()
                ce_vals = [ce_score_map[fid] for fid in ce_pool_fids]
                rrf_vals = [rrf_scores[fid] for fid in ce_pool_fids]

                ce_min, ce_max = min(ce_vals), max(ce_vals)
                rrf_min, rrf_max = min(rrf_vals), max(rrf_vals)

                fused_score_map: Dict[int, float] = {}
                for fid in ce_pool_fids:
                    norm_ce = (ce_score_map[fid] - ce_min) / (ce_max - ce_min) if ce_max > ce_min else 1.0
                    norm_rrf = (rrf_scores[fid] - rrf_min) / (rrf_max - rrf_min) if rrf_max > rrf_min else 1.0
                    fused_score_map[fid] = alpha * norm_ce + (1.0 - alpha) * norm_rrf

                # Re-order candidates by fused score descending (tie-breaker: raw CE score)
                ranked_ce_fids = sorted(
                    ce_pool_fids,
                    key=lambda fid: (fused_score_map[fid], ce_score_map[fid]),
                    reverse=True
                )
                fusion_ms = (time.time() - t_f0) * 1000
            else:
                # Re-order candidates strictly by cross-encoder score descending (Exp 4)
                ranked_ce_fids = sorted(ce_pool_fids, key=lambda fid: ce_score_map[fid], reverse=True)
            rerank_ms = (time.time() - t_ce0) * 1000

            if use_doc_dedup:
                # Retain best CE-scoring chunk per unique doc_id
                seen_doc_ids = set()
                top_fids = []
                for fid in ranked_ce_fids:
                    doc_id = dense_obj_map[fid].doc_id
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        top_fids.append(fid)
                        if len(top_fids) == top_k:
                            break
            else:
                top_fids = ranked_ce_fids[:top_k]

        elif not use_doc_dedup:
            top_fids = sorted_fids[:top_k]
            # Fetch SQLite metadata for candidates not already in dense_obj_map
            missing_fids = [fid for fid in top_fids if fid not in dense_obj_map]
            if missing_fids:
                fetched_records = get_chunks_by_faiss_ids(missing_fids, db_path=self.db_path)
                for rec in fetched_records:
                    fid = rec["faiss_id"]
                    dense_obj_map[fid] = SearchResult(
                        chunk_id=rec["id"],
                        doc_id=rec["doc_id"],
                        chunk_index=rec["chunk_index"],
                        title=rec["title"],
                        url=rec["url"],
                        author=rec["author"],
                        content=rec["content"],
                        word_count=rec["word_count"],
                        score=0.0,  # placeholder, calculated below
                        rank=0,
                        faiss_id=fid
                    )
        else:
            # Document-level deduplication over top fused candidate pool (default depth=10)
            pool_fids = sorted_fids[:pool_depth]

            # Fetch SQLite metadata for all pool candidates so doc_id is known
            missing_fids = [fid for fid in pool_fids if fid not in dense_obj_map]
            if missing_fids:
                fetched_records = get_chunks_by_faiss_ids(missing_fids, db_path=self.db_path)
                for rec in fetched_records:
                    fid = rec["faiss_id"]
                    dense_obj_map[fid] = SearchResult(
                        chunk_id=rec["id"],
                        doc_id=rec["doc_id"],
                        chunk_index=rec["chunk_index"],
                        title=rec["title"],
                        url=rec["url"],
                        author=rec["author"],
                        content=rec["content"],
                        word_count=rec["word_count"],
                        score=0.0,
                        rank=0,
                        faiss_id=fid
                    )

            # Deduplicate by unique doc_id, retaining highest RRF-ranked chunk per doc
            seen_doc_ids = set()
            top_fids = []
            for fid in pool_fids:
                doc_id = dense_obj_map[fid].doc_id
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    top_fids.append(fid)
                    if len(top_fids) == top_k:
                        break

            # Fallback if pool_depth contained fewer than top_k unique docs
            if len(top_fids) < top_k and len(sorted_fids) > pool_depth:
                for fid in sorted_fids[pool_depth:]:
                    if fid not in dense_obj_map:
                        records = get_chunks_by_faiss_ids([fid], db_path=self.db_path)
                        if records:
                            rec = records[0]
                            dense_obj_map[fid] = SearchResult(
                                chunk_id=rec["id"],
                                doc_id=rec["doc_id"],
                                chunk_index=rec["chunk_index"],
                                title=rec["title"],
                                url=rec["url"],
                                author=rec["author"],
                                content=rec["content"],
                                word_count=rec["word_count"],
                                score=0.0,
                                rank=0,
                                faiss_id=fid
                            )
                    if fid in dense_obj_map:
                        doc_id = dense_obj_map[fid].doc_id
                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            top_fids.append(fid)
                            if len(top_fids) == top_k:
                                break

        # 6. Ensure cosine similarity score is accurately assigned
        # For dense candidates, score is already cosine similarity.
        # For BM25-only candidates, compute dot product with query vector.
        query_vec: Optional[np.ndarray] = None
        faiss_idx = self.dense_retriever._index

        results: List[SearchResult] = []
        for rank, fid in enumerate(top_fids, start=1):
            base_res = dense_obj_map[fid]
            sim_score = base_res.score

            if fid not in dense_rank_map and faiss_idx is not None:
                # Compute exact cosine similarity via FAISS reconstruction & dot product
                if query_vec is None:
                    query_vec = self.dense_retriever.embedder.embed_query(query)
                vec = faiss_idx.reconstruct(fid)
                sim_score = float(np.dot(query_vec[0], vec))

            effective_min = min_score if min_score is not None else self.default_min_score
            if effective_min is not None and sim_score < effective_min:
                continue

            results.append(
                SearchResult(
                    chunk_id=base_res.chunk_id,
                    doc_id=base_res.doc_id,
                    chunk_index=base_res.chunk_index,
                    title=base_res.title,
                    url=base_res.url,
                    author=base_res.author,
                    content=base_res.content,
                    word_count=base_res.word_count,
                    score=round(sim_score, 4),
                    rank=rank,
                    faiss_id=fid
                )
            )

        total_ms = (time.time() - t_start) * 1000
        latency_breakdown = {
            "dense_ms": round(dense_ms, 2),
            "bm25_ms": round(bm25_ms, 2),
            "expansion_ms": round(expansion_ms, 2),
            "rrf_ms": round(rrf_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "fusion_ms": round(fusion_ms, 2),
            "total_ms": round(total_ms, 2)
        }
        _request_latency_var.set(latency_breakdown)
        return RetrievalResultList(results, latency_breakdown=latency_breakdown)

    def retrieve_with_latency(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Perform hybrid retrieval and return a tuple of (results, latency_breakdown).
        Guarantees request-isolated latency metadata for callers.
        """
        results = self.retrieve(query, top_k=top_k, **kwargs)
        return list(results), results.latency_breakdown
