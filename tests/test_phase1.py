"""
Unit tests for Phase 1 components: chunker and SQLite database layer.
"""

import os
import tempfile
import pytest
from app.core.chunker import chunk_text, chunk_document, clean_text, DocumentChunk
from app.db.database import (
    init_db,
    insert_chunks,
    get_chunk_by_id,
    get_chunks_by_ids,
    get_total_chunks,
    get_total_documents,
    log_query,
)


def test_clean_text():
    raw = "  Hello   world \n\n this is \t a test.   "
    assert clean_text(raw) == "Hello world this is a test."
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short():
    words = ["word"] * 100
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_sliding_window():
    # 900 words, chunk_size=400, chunk_overlap=50 -> step = 350
    # Chunk 0: 0..400
    # Chunk 1: 350..750
    # Chunk 2: 700..900
    words = [f"w{i}" for i in range(900)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)

    assert len(chunks) == 3
    assert chunks[0].startswith("w0")
    assert chunks[0].endswith("w399")
    assert chunks[1].startswith("w350")
    assert chunks[1].endswith("w749")
    assert chunks[2].startswith("w700")
    assert chunks[2].endswith("w899")


def test_chunk_document_metadata_preservation():
    content = " ".join([f"token_{i}" for i in range(850)])
    chunks = chunk_document(
        title="Test Blog Title",
        url="https://example.com/blog/test",
        content=content,
        author="John Doe",
        chunk_size=400,
        chunk_overlap=50
    )

    assert len(chunks) == 3
    for idx, c in enumerate(chunks):
        assert isinstance(c, DocumentChunk)
        assert c.title == "Test Blog Title"
        assert c.url == "https://example.com/blog/test"
        assert c.author == "John Doe"
        assert c.chunk_index == idx
        assert c.chunk_id == f"{c.doc_id}_c{idx}"
        assert c.word_count > 0


def test_database_crud_and_logging():
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = os.path.join(tmp_dir, "test_blogger.db")

        # 1. Initialize
        init_db(test_db)
        assert get_total_chunks(test_db) == 0
        assert get_total_documents(test_db) == 0

        # 2. Insert chunks with explicit faiss_ids
        sample_chunks = [
            DocumentChunk(
                chunk_id="doc1_c0",
                doc_id="doc1",
                chunk_index=0,
                title="Article 1",
                url="https://example.com/1",
                author="Alice",
                content="This is the first chunk.",
                word_count=5,
                faiss_id=0
            ),
            DocumentChunk(
                chunk_id="doc1_c1",
                doc_id="doc1",
                chunk_index=1,
                title="Article 1",
                url="https://example.com/1",
                author="Alice",
                content="This is the second chunk.",
                word_count=5,
                faiss_id=1
            ),
            DocumentChunk(
                chunk_id="doc2_c0",
                doc_id="doc2",
                chunk_index=0,
                title="Article 2",
                url="https://example.com/2",
                author="Bob",
                content="This is article 2 chunk 0.",
                word_count=6,
                faiss_id=2
            )
        ]

        inserted = insert_chunks(sample_chunks, db_path=test_db)
        assert inserted == 3
        assert get_total_chunks(test_db) == 3
        assert get_total_documents(test_db) == 2

        # 3. Retrieve single chunk
        c0 = get_chunk_by_id("doc1_c0", db_path=test_db)
        assert c0 is not None
        assert c0["title"] == "Article 1"
        assert c0["author"] == "Alice"
        assert c0["content"] == "This is the first chunk."
        assert c0["faiss_id"] == 0

        # 4. Retrieve multiple chunks by string ID
        multi = get_chunks_by_ids(["doc2_c0", "doc1_c1"], db_path=test_db)
        assert len(multi) == 2
        assert multi[0]["id"] == "doc2_c0"
        assert multi[1]["id"] == "doc1_c1"

        # 5. Retrieve multiple chunks by deterministic faiss_id
        from app.db.database import get_chunks_by_faiss_ids
        by_faiss = get_chunks_by_faiss_ids([2, 0], db_path=test_db)
        assert len(by_faiss) == 2
        assert by_faiss[0]["id"] == "doc2_c0"
        assert by_faiss[0]["faiss_id"] == 2
        assert by_faiss[1]["id"] == "doc1_c0"
        assert by_faiss[1]["faiss_id"] == 0

        # 6. Test query logging enabled and disabled
        assert log_query("test query", ["doc1_c0"], 12.5, mode="search", db_path=test_db, enabled=True) is True
        assert log_query("disabled query", ["doc2_c0"], 10.0, mode="ask", db_path=test_db, enabled=False) is False
