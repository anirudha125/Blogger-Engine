"""
Data ingestion and indexing pipeline for Blogger Engine.

Reads raw blog JSON data, applies sliding-window chunking (~400 words, ~50 overlap),
stores chunk records in SQLite, embeds passages using SentenceTransformers,
and serializes a normalized Inner-Product (cosine) FAISS index.

Supports CPU locally and auto-detects CUDA (e.g., Kaggle T4 GPU) for high-throughput batching.
Preserves existing whole-document baseline indexes without overwriting them.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from app.core.chunker import chunk_document, DocumentChunk
from app.db.database import init_db, insert_chunks, get_total_chunks, get_total_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("blogger_engine.ingest")


def preserve_baseline_index(baseline_target: str = "indexes/faiss_baseline.index", source_index: str = "indexes/faiss.index") -> None:
    """
    Ensure the existing whole-document index is preserved as a separate baseline artifact.
    """
    src = Path(source_index)
    dst = Path(baseline_target)

    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Preserved existing baseline FAISS index: {src} -> {dst}")
    elif dst.exists():
        logger.info(f"Baseline index already preserved at {dst}")


def load_raw_documents(input_file: str) -> List[Dict[str, Any]]:
    """Load and normalize raw document entries from JSON file."""
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Handle dict wrapping a list (e.g. {"blogs": [...]})
        for key in ["blogs", "documents", "data", "items"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        if isinstance(data, dict):
            data = [data]

    normalized = []
    for item in data:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        # Fallbacks for content/body/text
        content = item.get("content") or item.get("body") or item.get("text") or ""
        author = item.get("author") or "Unknown"

        if content and content.strip():
            normalized.append({
                "title": title or "Untitled Article",
                "url": url,
                "author": author,
                "content": content
            })

    return normalized


def run_ingestion(
    input_path: str = "data/sample.json",
    output_index_path: str = "indexes/faiss_chunked.index",
    output_db_path: str = "data/blogger.db",
    model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    batch_size: Optional[int] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute full ingestion: load -> chunk -> save SQLite -> embed -> save FAISS.
    """
    start_time = time.time()

    # Step 0: Preserve baseline artifact
    preserve_baseline_index()

    # Step 1: Detect hardware acceleration
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running ingestion on device: {device.upper()}")

    if batch_size is None:
        batch_size = 128 if device == "cuda" else 32

    # Step 2: Load raw documents
    logger.info(f"Loading raw articles from {input_path}...")
    raw_docs = load_raw_documents(input_path)
    logger.info(f"Loaded {len(raw_docs)} valid raw articles.")

    if not raw_docs:
        raise ValueError(f"No valid articles found in {input_path}")

    # Step 3: Chunk documents
    logger.info(f"Chunking articles (target: ~{chunk_size} words, overlap: ~{chunk_overlap} words)...")
    all_chunks: List[DocumentChunk] = []
    for doc in raw_docs:
        chunks = chunk_document(
            title=doc["title"],
            url=doc["url"],
            content=doc["content"],
            author=doc["author"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        all_chunks.extend(chunks)

    # Assign deterministic, explicit faiss_id matching vector index order
    for idx, chunk in enumerate(all_chunks):
        chunk.faiss_id = idx

    logger.info(f"Generated {len(all_chunks)} chunks from {len(raw_docs)} articles.")

    # Step 4: Persist chunks and metadata in SQLite
    logger.info(f"Storing chunks in SQLite database: {output_db_path}...")
    init_db(output_db_path)
    inserted = insert_chunks(all_chunks, db_path=output_db_path)
    logger.info(f"Inserted/updated {inserted} chunks in {output_db_path}.")

    # Step 5: Encode passages with SentenceTransformer
    logger.info(f"Loading embedding model '{model_name}' on {device.upper()}...")
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"Encoding {len(all_chunks)} passage texts (batch_size={batch_size})...")
    chunk_texts = [c.content for c in all_chunks]
    embeddings = model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    # Step 6: Build normalized Inner-Product (cosine) FAISS Index
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP (dimensions={dim}, count={embeddings.shape[0]})...")
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Step 7: Persist FAISS Index
    out_idx_path = Path(output_index_path)
    out_idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_idx_path))
    logger.info(f"Successfully saved FAISS index to {out_idx_path} ({out_idx_path.stat().st_size} bytes)")

    elapsed = time.time() - start_time
    total_db_chunks = get_total_chunks(output_db_path)
    total_db_docs = get_total_documents(output_db_path)

    summary = {
        "device": device,
        "raw_articles": len(raw_docs),
        "chunks_generated": len(all_chunks),
        "total_db_chunks": total_db_chunks,
        "total_db_documents": total_db_docs,
        "embedding_dimensions": dim,
        "faiss_index_path": str(out_idx_path),
        "sqlite_db_path": output_db_path,
        "elapsed_seconds": round(elapsed, 2)
    }

    logger.info("=" * 60)
    logger.info("Ingestion completed successfully!")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest blog articles, chunk, store in SQLite, and build FAISS index.")
    parser.add_argument("--input", default="data/sample.json", help="Path to input JSON file with blog articles")
    parser.add_argument("--output-index", default="indexes/faiss_chunked.index", help="Output path for the FAISS index")
    parser.add_argument("--output-db", default="data/blogger.db", help="Output path for the SQLite database")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--chunk-size", type=int, default=400, help="Target word count per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Word overlap between consecutive chunks")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Inference device")

    args = parser.parse_args()

    run_ingestion(
        input_path=args.input,
        output_index_path=args.output_index,
        output_db_path=args.output_db,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
