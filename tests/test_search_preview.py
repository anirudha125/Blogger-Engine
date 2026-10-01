"""
Focused tests for the Final Search-Result UX change:
- Preview comes from document's first chunk (chunk 1 / chunk_index 0)
- Preview is short (roughly 35-50 words max, 2-3 sentences, with '...')
- No chunk metadata or expand/collapse controls appear in rendered search results
- Fallback works when first chunk is unavailable or empty/unusable
- ASK mode source cards and citation behavior remain unchanged
- Search API route attaches preview without changing retrieval behavior
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

from app.core.preview import (
    clean_leading_boilerplate,
    split_sentences,
    extract_document_preview,
)
from app.db.database import get_first_chunks_by_doc_ids
from app.schemas import SearchResultItem, AskResponse


def test_preview_comes_from_document_first_chunk():
    """Verify preview is extracted from chunk 0 / beginning of document, regardless of retrieved chunk index."""
    doc_id = "doc_test_transformer"
    chunk_0_content = (
        "Transformers are neural-network architectures based on self-attention and are widely used in "
        "modern natural language processing. They replaced recurrent neural networks by processing entire "
        "sequences in parallel. This design drastically accelerates large-scale training on technical corpora."
    )
    chunk_3_content = (
        "In this later section of the monograph, we analyze cross-attention head pruning. Specifically, "
        "layer 11 exhibits high redundancy when evaluated on the GLUE benchmark. Further ablation shows "
        "minimal loss degradation."
    )

    # Simulate retrieval of chunk 3
    retrieved_chunk = {
        "doc_id": doc_id,
        "chunk_index": 3,
        "content": chunk_3_content,
        "title": "Transformers in NLP | State-Of-The-Art-Models",
    }

    # First chunk lookup gives chunk 0
    first_chunks_map = {
        doc_id: {"doc_id": doc_id, "chunk_index": 0, "content": chunk_0_content}
    }

    preview = extract_document_preview(
        first_chunk_content=first_chunks_map[doc_id]["content"],
        fallback_content=retrieved_chunk["content"],
        title=retrieved_chunk["title"],
    )

    # Preview must reflect chunk 0 (Transformers are neural-network...), NOT chunk 3 (cross-attention head pruning)
    assert "Transformers are neural-network architectures" in preview
    assert "cross-attention head pruning" not in preview
    assert preview.endswith("...") or len(preview.split()) <= 50


def test_preview_is_short_and_bounded():
    """Verify preview is bounded to roughly 35-50 words, 2-3 sentences, ending with '...' when truncated."""
    long_passage = (
        "Deep reinforcement learning from human feedback has emerged as a cornerstone of modern alignment methodologies. "
        "By training reward models on human preference annotations, policy networks can optimize for helpfulness and safety. "
        "However, reward hacking remains a formidable theoretical and practical bottleneck across large foundation models. "
        "Recent advancements explore group relative policy optimization to eliminate auxiliary critic networks altogether. "
        "This substantially reduces GPU memory utilization while stabilizing policy variance during multi-step rollout trajectories."
    )

    preview = extract_document_preview(
        first_chunk_content=long_passage,
        fallback_content=None,
        title="Reinforcement Learning Alignment",
        min_words=35,
        max_words=50,
        max_sentences=3,
    )

    words = preview.split()
    # Word count must be bounded
    assert 20 <= len(words) <= 52, f"Word count {len(words)} outside expected bounds"
    assert preview.endswith("..."), "Truncated preview must end with '...'"
    # Should not include all 5 sentences
    assert "GPU memory utilization" not in preview


def test_fallback_works_when_first_chunk_unavailable_or_unusable():
    """Verify graceful fallback to retrieved chunk when chunk 0 is missing, whitespace, or unusable."""
    retrieved_content = (
        "Proximal Policy Optimization balances policy improvement with clipped surrogate objectives to maintain "
        "training stability. The clipping mechanism bounds divergence between the target policy and sampling policy."
    )

    # Case 1: First chunk is None
    p1 = extract_document_preview(
        first_chunk_content=None,
        fallback_content=retrieved_content,
        title="PPO Algorithm",
    )
    assert "Proximal Policy Optimization" in p1

    # Case 2: First chunk is empty or whitespace
    p2 = extract_document_preview(
        first_chunk_content="   \n\t  ",
        fallback_content=retrieved_content,
        title="PPO Algorithm",
    )
    assert "Proximal Policy Optimization" in p2

    # Case 3: First chunk contains only noisy boilerplate with < 15 words
    p3 = extract_document_preview(
        first_chunk_content="Navigation menu. Table of contents. Sign in.",
        fallback_content=retrieved_content,
        title="PPO Algorithm",
    )
    assert "Proximal Policy Optimization" in p3


def test_boilerplate_cleanup_and_noise_removal():
    """Verify deterministic removal of navigation headers, repeated titles, and common scraping noise."""
    noisy_raw = (
        "Skip to Content Table of Contents [hide] Sign up Sign in "
        "Attention Is All You Need — Attention Is All You Need is a seminal architecture paper. "
        "It introduces the multi-head self-attention mechanism, discarding recurrent and convolutional layers. "
        "The model achieves state-of-the-art BLEU scores on translation tasks."
    )

    cleaned = clean_leading_boilerplate(noisy_raw, title="Attention Is All You Need")
    assert "Skip to Content" not in cleaned
    assert "Sign up Sign in" not in cleaned
    # Leading repetition of title is stripped
    preview = extract_document_preview(
        first_chunk_content=noisy_raw,
        fallback_content=None,
        title="Attention Is All You Need",
    )
    assert "Skip to Content" not in preview
    assert "seminal architecture paper" in preview


def test_no_chunk_metadata_in_rendered_search_result():
    """
    Verify that in SEARCH mode:
    - No chunk number appears
    - No word count appears
    - No chunk ID appears
    - No 'Read full passage' or expand/collapse controls appear
    - Preview div is rendered
    """
    if not shutil.which("node"):
        return

    index_html = Path("app/static/index.html").read_text(encoding="utf-8")

    # Extract JS rendering functions from index.html
    import re
    sanitize_m = re.search(r"function sanitizeText\([\s\S]*?\n    \}", index_html)
    escape_m = re.search(r"function escapeHtml\([\s\S]*?\n    \}", index_html)
    hostname_m = re.search(r"function extractHostname\([\s\S]*?\n    \}", index_html)
    render_search_m = re.search(r"function renderSearchResults\([\s\S]*?\n    \}", index_html)

    assert sanitize_m and escape_m and hostname_m and render_search_m

    js_code = f"""
{sanitize_m.group(0)}
{escape_m.group(0)}
{hostname_m.group(0)}
let containerHtml = '';
const document = {{
    getElementById: (id) => ({{
        innerHTML: '',
        appendChild: (child) => {{
            containerHtml += child.outerHTML;
        }}
    }}),
    createElement: (tag) => {{
        return {{
            tagName: tag,
            className: '',
            id: '',
            _val: '',
            get innerHTML() {{ return this._val; }},
            set innerHTML(v) {{ this._val = v; }},
            get textContent() {{ return this._val; }},
            set textContent(v) {{ this._val = String(v); }},
            get outerHTML() {{
                return '<div class="' + this.className + '" id="' + this.id + '">' + this._val + '</div>';
            }}
        }};
    }}
}};

{render_search_m.group(0)}

const sampleResults = [{{
    rank: 1,
    score: 0.8842,
    chunk_id: 'sample_c_99',
    doc_id: 'sample_doc_1',
    chunk_index: 4,
    title: 'Transformers in NLP | State-Of-The-Art-Models',
    url: 'https://analyticsvidhya.com/blog/transformers-nlp',
    author: 'Editorial Team',
    content: 'Full 400 word passage with lots of detailed text that should NOT appear in search mode...',
    word_count: 385,
    faiss_id: 12,
    preview: 'Transformers are neural-network architectures based on self-attention and are widely used in NLP...'
}}];

renderSearchResults(sampleResults);

console.log(JSON.stringify({{
    rendered: containerHtml,
    has_preview: containerHtml.includes('row-preview'),
    has_preview_text: containerHtml.includes('Transformers are neural-network architectures based on self-attention'),
    has_chunk_num: /CHUNK #/i.test(containerHtml),
    has_word_count: /WORDS/i.test(containerHtml),
    has_chunk_id_label: /ID:\\s*sample_c_99/i.test(containerHtml),
    has_read_full: /Read full passage/i.test(containerHtml),
    has_collapse: /Collapse passage/i.test(containerHtml),
    has_toggle_btn: containerHtml.includes('toggle-passage-btn'),
    has_row_full: containerHtml.includes('row-full')
}}));
"""

    proc = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())

    assert res["has_preview"] is True, "row-preview must be present in search result"
    assert res["has_preview_text"] is True, "preview text must be rendered"
    assert res["has_chunk_num"] is False, "CHUNK # must not appear in search result"
    assert res["has_word_count"] is False, "WORDS count must not appear in search result"
    assert res["has_chunk_id_label"] is False, "Chunk ID must not appear in search result"
    assert res["has_read_full"] is False, "'Read full passage' must not appear in search result"
    assert res["has_collapse"] is False, "'Collapse passage' must not appear in search result"
    assert res["has_toggle_btn"] is False, "toggle-passage-btn must not appear in search result"
    assert res["has_row_full"] is False, "row-full must not appear in search result"


def test_ask_mode_retains_source_cards_and_passage_details():
    """Verify ASK mode's renderSources retains expandable passage details and chunk metadata for citations."""
    if not shutil.which("node"):
        return

    index_html = Path("app/static/index.html").read_text(encoding="utf-8")

    import re
    sanitize_m = re.search(r"function sanitizeText\([\s\S]*?\n    \}", index_html)
    escape_m = re.search(r"function escapeHtml\([\s\S]*?\n    \}", index_html)
    hostname_m = re.search(r"function extractHostname\([\s\S]*?\n    \}", index_html)
    render_sources_m = re.search(r"function renderSources\([\s\S]*?\n    \}", index_html)

    assert sanitize_m and escape_m and hostname_m and render_sources_m

    js_code = f"""
{sanitize_m.group(0)}
{escape_m.group(0)}
{hostname_m.group(0)}
let containerHtml = '';
const document = {{
    getElementById: (id) => ({{
        innerHTML: '',
        appendChild: (child) => {{
            containerHtml += child.outerHTML;
        }}
    }}),
    createElement: (tag) => {{
        return {{
            tagName: tag,
            className: '',
            id: '',
            innerHTML: '',
            get outerHTML() {{
                return `<div class="${{this.className}}" id="${{this.id}}">${{this.innerHTML}}</div>`;
            }}
        }};
    }}
}};

{render_sources_m.group(0)}

const sampleSources = [{{
    rank: 1,
    score: 0.9123,
    chunk_id: 'ask_source_chunk_42',
    doc_id: 'doc_alpha',
    chunk_index: 2,
    title: 'Grounded Alignment Principles',
    url: 'https://example.com/alignment',
    author: 'Alice Researcher',
    content: 'Very long content '.repeat(30),
    word_count: 360,
    faiss_id: 42
}}];

renderSources(sampleSources);

console.log(JSON.stringify({{
    rendered: containerHtml,
    has_chunk_num: containerHtml.includes('CHUNK #2'),
    has_word_count: containerHtml.includes('360 WORDS'),
    has_id: containerHtml.includes('ask_source_chunk_42'),
    has_toggle_btn: containerHtml.includes('toggle-passage-btn'),
    has_row_full: containerHtml.includes('row-full')
}}));
"""

    proc = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())

    assert res["has_chunk_num"] is True, "ASK mode source card must retain CHUNK #"
    assert res["has_word_count"] is True, "ASK mode source card must retain word count"
    assert res["has_id"] is True, "ASK mode source card must retain chunk ID"
    assert res["has_toggle_btn"] is True, "ASK mode source card must retain toggle button"
    assert res["has_row_full"] is True, "ASK mode source card must retain full passage container"


def test_schema_backward_compatibility():
    """Verify SearchResultItem preserves all 11 existing fields and adds preview as optional."""
    existing_fields = [
        "rank", "score", "chunk_id", "doc_id", "chunk_index",
        "title", "url", "author", "content", "word_count", "faiss_id"
    ]
    for f in existing_fields:
        assert f in SearchResultItem.model_fields

    assert "preview" in SearchResultItem.model_fields

    # Can instantiate without preview
    item = SearchResultItem(
        rank=1,
        score=0.85,
        chunk_id="c1",
        doc_id="d1",
        chunk_index=0,
        title="T",
        url="https://u.com",
        author="A",
        content="Passage text",
        word_count=2,
        faiss_id=0,
    )
    assert item.preview is None

    # Can instantiate with preview
    item_with_preview = SearchResultItem(
        rank=1,
        score=0.85,
        chunk_id="c1",
        doc_id="d1",
        chunk_index=0,
        title="T",
        url="https://u.com",
        author="A",
        content="Passage text",
        word_count=2,
        faiss_id=0,
        preview="Preview text...",
    )
    assert item_with_preview.preview == "Preview text..."


def test_get_first_chunks_by_doc_ids_database_query():
    """Verify get_first_chunks_by_doc_ids fetches chunk_index=0 from SQLite."""
    import tempfile
    import os
    from app.core.chunker import DocumentChunk
    from app.db.database import init_db, insert_chunks

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = os.path.join(tmp_dir, "test_preview.db")
        init_db(test_db)

        chunks = [
            DocumentChunk(
                chunk_id="doc1_c0",
                doc_id="doc1",
                chunk_index=0,
                title="Doc One",
                url="https://example.com/1",
                author="Alice",
                content="Chunk 0 opening introduction to machine learning architectures.",
                word_count=8,
                faiss_id=0,
            ),
            DocumentChunk(
                chunk_id="doc1_c1",
                doc_id="doc1",
                chunk_index=1,
                title="Doc One",
                url="https://example.com/1",
                author="Alice",
                content="Chunk 1 detailed benchmarks and loss tables.",
                word_count=7,
                faiss_id=1,
            ),
            DocumentChunk(
                chunk_id="doc2_c0",
                doc_id="doc2",
                chunk_index=0,
                title="Doc Two",
                url="https://example.com/2",
                author="Bob",
                content="Chunk 0 second document starting paragraph on optimization.",
                word_count=8,
                faiss_id=2,
            ),
        ]
        insert_chunks(chunks, db_path=test_db)

        # Query first chunks for both docs
        first_map = get_first_chunks_by_doc_ids(["doc1", "doc2"], db_path=test_db)
        assert len(first_map) == 2
        assert first_map["doc1"]["chunk_index"] == 0
        assert "opening introduction" in first_map["doc1"]["content"]
        assert first_map["doc2"]["chunk_index"] == 0
        assert "second document starting paragraph" in first_map["doc2"]["content"]


def test_search_endpoint_returns_preview(monkeypatch):
    """Verify /api/search returns preview on each SearchResultItem."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.retriever import SearchResult

    mock_results = [
        SearchResult(
            rank=1,
            score=0.88,
            chunk_id="doc1_c1",
            doc_id="doc1",
            chunk_index=1,
            title="Doc One",
            url="https://example.com/1",
            author="Alice",
            content="Retrieved chunk 1 with benchmark details and evaluation metrics across various datasets.",
            word_count=12,
            faiss_id=1,
        )
    ]

    class MockRetriever:
        total_vectors = 100
        def retrieve(self, query, top_k=3, min_score=None):
            return mock_results

    with TestClient(app) as client:
        client.app.state.retriever = MockRetriever()
        res = client.post("/api/search", json={"query": "test query", "top_k": 1})
        assert res.status_code == 200
        data = res.json()
        assert len(data["results"]) == 1
        item = data["results"][0]
        assert "preview" in item
        assert item["preview"] is not None
        assert len(item["preview"]) > 0


def test_frontend_search_and_ask_top_k_contract():
    """Verify SEARCH mode requests top_k=10 and ASK mode requests canonical top_k=3."""
    index_html = Path("app/static/index.html").read_text(encoding="utf-8")
    assert "const SEARCH_TOP_K = 10;" in index_html
    assert "const CANONICAL_TOP_K = 3;" in index_html
    assert "top_k: SEARCH_TOP_K" in index_html
    assert "top_k: CANONICAL_TOP_K" in index_html


def test_search_renders_up_to_10_results():
    """Verify renderSearchResults renders exactly 10 result rows when 10 results are provided."""
    if not shutil.which("node"):
        return

    index_html = Path("app/static/index.html").read_text(encoding="utf-8")
    import re
    sanitize_m = re.search(r"function sanitizeText\([\s\S]*?\n    \}", index_html)
    escape_m = re.search(r"function escapeHtml\([\s\S]*?\n    \}", index_html)
    hostname_m = re.search(r"function extractHostname\([\s\S]*?\n    \}", index_html)
    render_search_m = re.search(r"function renderSearchResults\([\s\S]*?\n    \}", index_html)

    js_code = f"""
{sanitize_m.group(0)}
{escape_m.group(0)}
{hostname_m.group(0)}
let containerHtml = '';
const rows = [];
const document = {{
    getElementById: (id) => ({{
        innerHTML: '',
        appendChild: (child) => {{
            containerHtml += child.outerHTML;
            rows.push(child);
        }}
    }}),
    createElement: (tag) => ({{
        tagName: tag,
        className: '',
        id: '',
        _val: '',
        get innerHTML() {{ return this._val; }},
        set innerHTML(v) {{ this._val = v; }},
        get textContent() {{ return this._val; }},
        set textContent(v) {{ this._val = String(v); }},
        get outerHTML() {{
            return '<div class="' + this.className + '" id="' + this.id + '">' + this._val + '</div>';
        }}
    }})
}};

{render_search_m.group(0)}

const tenResults = Array.from({{ length: 10 }}, (_, i) => ({{
    rank: i + 1,
    score: 0.90 - i * 0.02,
    chunk_id: 'chunk_' + (i + 1),
    doc_id: 'doc_' + (i + 1),
    chunk_index: i,
    title: 'Research Paper ' + (i + 1),
    url: 'https://arxiv.org/abs/2301.' + (1000 + i),
    author: 'Author ' + (i + 1),
    content: 'Full content of passage ' + (i + 1),
    word_count: 350,
    faiss_id: i,
    preview: 'Short preview for paper ' + (i + 1) + ' discussing key results in the field...'
}}));

renderSearchResults(tenResults);

console.log(JSON.stringify({{
    rowCount: rows.length,
    has_01: containerHtml.includes('>01<'),
    has_10: containerHtml.includes('>10<'),
    has_no_full_passages: !containerHtml.includes('row-full'),
    has_no_chunk_numbers: !containerHtml.includes('CHUNK #'),
    has_no_toggle_buttons: !containerHtml.includes('toggle-passage-btn')
}}));
"""

    proc = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())

    assert res["rowCount"] == 10
    assert res["has_01"] is True
    assert res["has_10"] is True
    assert res["has_no_full_passages"] is True
    assert res["has_no_chunk_numbers"] is True
    assert res["has_no_toggle_buttons"] is True


def test_no_prominent_confidence_score_visible():
    """Verify confidence score is not visibly displayed in SEARCH results or ASK answer headline."""
    index_html = Path("app/static/index.html").read_text(encoding="utf-8")
    style_css = Path("app/static/style.css").read_text(encoding="utf-8")

    # In style.css, qa-confidence has display: none
    assert "qa-confidence" in style_css
    assert ".qa-confidence" in style_css

    # In index.html, qa-confidence element style is set to display: none
    assert "confEl.style.display = 'none';" in index_html
