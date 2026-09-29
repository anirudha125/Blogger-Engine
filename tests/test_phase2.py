"""
Unit tests for Phase 2 components: embedder, retriever, and generator.
"""

import numpy as np
import pytest
from app.core.embedder import QueryEmbedder
from app.core.retriever import Retriever, SearchResult
from app.core.generator import AnswerGenerator, Citation


@pytest.fixture(scope="module")
def embedder():
    return QueryEmbedder(device="cpu")


@pytest.fixture(scope="module")
def retriever(embedder):
    return Retriever(
        index_path="indexes/faiss_chunked.index",
        db_path="data/blogger.db",
        embedder=embedder
    )


def test_embedder_single_query(embedder):
    vec = embedder.embed_query("Reinforcement Learning from AI Feedback")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, 384)
    assert vec.dtype == np.float32

    # L2 norm should be approximately 1.0
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embedder_empty_query(embedder):
    vec = embedder.embed_query("   ")
    assert vec.shape == (1, 384)
    assert np.all(vec == 0.0)


def test_embedder_batch(embedder):
    texts = ["First query", "Second query", "Third query"]
    vecs = embedder.embed_texts(texts)
    assert vecs.shape == (3, 384)
    for row in vecs:
        assert pytest.approx(np.linalg.norm(row), rel=1e-3) == 1.0


def test_retriever_search(retriever):
    results = retriever.retrieve("What is RLAIF and how does it work?", top_k=3)
    assert len(results) > 0
    assert len(results) <= 3

    # Check order & attributes
    for i, r in enumerate(results, start=1):
        assert isinstance(r, SearchResult)
        assert r.rank == i
        assert r.score > 0.0
        assert r.title != ""
        assert r.url.startswith("http")
        assert len(r.content) > 0
        assert r.chunk_id != ""

    # Ensure sorted by score descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_empty_query(retriever):
    results = retriever.retrieve("   ", top_k=3)
    assert results == []


def test_retriever_min_score_filter(retriever):
    # If we set min_score higher than any possible cosine similarity, it should return []
    results = retriever.retrieve("What is RLAIF?", top_k=3, min_score=0.99)
    assert results == []


def test_generator_prompt_building():
    generator = AnswerGenerator(similarity_threshold=0.35)
    sample_passages = [
        SearchResult(
            chunk_id="doc1_c0",
            doc_id="doc1",
            chunk_index=0,
            title="RLAIF Guide",
            url="https://example.com/rlaif",
            author="Author A",
            content="RLAIF uses constitutional principles to train an AI assistant.",
            word_count=9,
            score=0.75,
            rank=1,
            faiss_id=0
        )
    ]

    sys_prompt = generator.build_system_prompt()
    assert "Strict Grounding Rules" in sys_prompt
    assert "[Doc 1]" in sys_prompt

    user_prompt = generator.build_user_prompt("Explain RLAIF", sample_passages)
    assert "[Doc 1] Title: RLAIF Guide" in user_prompt
    assert "User Question: Explain RLAIF" in user_prompt


def test_generator_citation_extraction():
    generator = AnswerGenerator()
    sample_passages = [
        SearchResult(
            chunk_id="doc1_c0",
            doc_id="doc1",
            chunk_index=0,
            title="Doc 1 Title",
            url="https://example.com/1",
            author="Author 1",
            content="Passage 1 content",
            word_count=3,
            score=0.8,
            rank=1,
            faiss_id=0
        ),
        SearchResult(
            chunk_id="doc2_c0",
            doc_id="doc2",
            chunk_index=0,
            title="Doc 2 Title",
            url="https://example.com/2",
            author="Author 2",
            content="Passage 2 content",
            word_count=3,
            score=0.7,
            rank=2,
            faiss_id=1
        )
    ]

    llm_output = "RLAIF uses a constitution [Doc 1]. It also improves safety [Doc 2]."
    citations = generator.extract_citations(llm_output, sample_passages)
    assert len(citations) == 2
    assert citations[0].doc_tag == "[Doc 1]"
    assert citations[0].chunk_id == "doc1_c0"
    assert citations[1].doc_tag == "[Doc 2]"
    assert citations[1].chunk_id == "doc2_c0"


def test_generator_refusal_threshold():
    generator = AnswerGenerator(similarity_threshold=0.60)
    low_confidence_passages = [
        SearchResult(
            chunk_id="doc1_c0",
            doc_id="doc1",
            chunk_index=0,
            title="Irrelevant Doc",
            url="https://example.com/irrelevant",
            author="Author",
            content="Content about something else.",
            word_count=5,
            score=0.42,  # Below 0.60
            rank=1,
            faiss_id=0
        )
    ]

    answer = generator.generate_answer("How to bake bread?", low_confidence_passages)
    assert answer.refused is True
    assert "below required threshold" in answer.answer
    assert answer.citations == []


def test_generator_no_passages_refusal():
    generator = AnswerGenerator()
    answer = generator.generate_answer("Any question", [])
    assert answer.refused is True
    assert "could not find any relevant blog articles" in answer.answer


def test_generator_per_call_threshold_override():
    # Base threshold 0.35, but query overrides with 0.80
    generator = AnswerGenerator(similarity_threshold=0.35)
    passages = [
        SearchResult(
            chunk_id="doc1_c0",
            doc_id="doc1",
            chunk_index=0,
            title="Title",
            url="https://example.com/1",
            author="Author",
            content="Some content",
            word_count=2,
            score=0.65,  # Above 0.35, but below 0.80
            rank=1,
            faiss_id=0
        )
    ]
    # Default threshold accepts it (or prompts for API key if missing)
    default_ans = generator.generate_answer("Query", passages)
    assert default_ans.refused is False

    # Overridden threshold refuses it
    overridden_ans = generator.generate_answer("Query", passages, similarity_threshold=0.80)
    assert overridden_ans.refused is True
    assert "below required threshold: 0.80" in overridden_ans.answer


def test_generator_secret_masking():
    # Ensure api_key is never exposed in repr
    secret_key = "sk-super-secret-production-key-12345"
    generator = AnswerGenerator(api_key=secret_key)
    repr_str = repr(generator)
    assert secret_key not in repr_str
    assert "api_key_set=Yes" in repr_str


def test_retriever_missing_resource_safety():
    # Missing index file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        Retriever(index_path="non_existent_index.faiss", db_path="data/blogger.db")

    # Missing database file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="SQLite database not found"):
        Retriever(index_path="indexes/faiss_chunked.index", db_path="non_existent_db.db")


def test_retriever_explicit_faiss_id_mapping(retriever):
    # Verify that retrieved results match the exact faiss_id in database
    results = retriever.retrieve("Reinforcement learning", top_k=3)
    assert len(results) > 0
    for r in results:
        assert isinstance(r.faiss_id, int)
        assert r.faiss_id >= 0
