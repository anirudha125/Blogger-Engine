"""
Text chunking module for Blogger Engine.

Splits blog articles into ~400-word passages with a ~50-word sliding window overlap,
preserving document metadata (title, URL, author, doc_id, chunk_index).
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
import hashlib
import re


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    url: str
    author: str
    content: str
    word_count: int
    faiss_id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def generate_doc_id(url: str, title: str = "") -> str:
    """Generate a deterministic, clean document ID based on URL or title."""
    seed = url.strip() or title.strip()
    if not seed:
        seed = "unknown_document"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    if not text:
        return ""
    # Replace multiple whitespaces/newlines with single space
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split a raw text string into chunks of ~chunk_size words with ~chunk_overlap words overlap.

    Args:
        text: Input raw text string.
        chunk_size: Desired target word count per chunk (default ~400 words).
        chunk_overlap: Overlap in words between consecutive chunks (default ~50 words).

    Returns:
        List of text chunks.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    total_words = len(words)

    if total_words <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= total_words:
            break
        start += step

    return chunks


def chunk_document(
    title: str,
    url: str,
    content: str,
    author: Optional[str] = None,
    doc_id: Optional[str] = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[DocumentChunk]:
    """
    Break an article into DocumentChunk instances preserving all metadata.

    Args:
        title: Title of the blog post.
        url: Original URL of the blog post.
        content: Full text content of the blog post.
        author: Author of the blog post (defaults to 'Unknown').
        doc_id: Unique document identifier. If not provided, computed from URL/title.
        chunk_size: Target word count per chunk (~400 words).
        chunk_overlap: Overlap in words between consecutive chunks (~50 words).

    Returns:
        List of DocumentChunk objects.
    """
    effective_doc_id = doc_id or generate_doc_id(url, title)
    effective_author = author.strip() if author and author.strip() else "Unknown"
    text_chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    document_chunks = []
    for idx, chunk_content in enumerate(text_chunks):
        word_count = len(chunk_content.split(" "))
        chunk_id = f"{effective_doc_id}_c{idx}"
        document_chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=effective_doc_id,
                chunk_index=idx,
                title=title.strip(),
                url=url.strip(),
                author=effective_author,
                content=chunk_content,
                word_count=word_count
            )
        )

    return document_chunks
