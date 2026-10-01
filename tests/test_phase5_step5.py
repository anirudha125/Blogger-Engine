"""
Tests for Phase 5 — Step 5: Frontend Alignment & UX Hardening.

Verifies:
1. Production request payloads default to canonical top_k=3 in frontend JS.
2. No active frontend search/ask code hardcodes top_k: 5.
3. No browser alert() calls exist for API error handling.
4. Non-blocking in-page editorial notice component exists with accessibility attributes.
5. All JSON fields consumed by the frontend exist on backend Pydantic schemas.
6. The root path '/' serves the updated frontend with CSS and status elements.
"""

from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import (
    SearchResponse,
    AskResponse,
    SearchResultItem,
    CitationItem,
    HealthResponse,
    StatsResponse,
)


def test_root_serves_updated_html():
    """Verify root endpoint serves the updated HTML containing the status notice."""
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        html = res.text

        # Verify branding and editorial structure
        assert "Blogger Engine" in html
        assert 'id="status-notice"' in html
        assert 'role="alert"' in html
        assert 'id="results-container"' in html
        assert 'id="qa-box"' in html


def test_frontend_static_top_k_alignment():
    """Verify frontend JavaScript enforces CANONICAL_TOP_K = 3 and has no top_k: 5."""
    index_path = Path("app/static/index.html")
    assert index_path.exists(), "index.html missing"
    content = index_path.read_text(encoding="utf-8")

    assert "const CANONICAL_TOP_K = 3;" in content
    assert "top_k: CANONICAL_TOP_K" in content
    assert "top_k: 5" not in content


def test_frontend_no_alert_calls():
    """Verify all alert() error popups have been removed from the frontend."""
    index_path = Path("app/static/index.html")
    content = index_path.read_text(encoding="utf-8")

    assert "alert(" not in content


def test_frontend_css_editorial_notice_classes():
    """Verify style.css contains the non-blocking editorial notice design tokens."""
    css_path = Path("app/static/style.css")
    assert css_path.exists(), "style.css missing"
    content = css_path.read_text(encoding="utf-8")

    assert ".editorial-notice" in content
    assert ".editorial-notice.notice-error" in content
    assert ".editorial-notice.notice-warning" in content
    assert ".editorial-notice.notice-info" in content
    assert ".notice-title" in content
    assert ".notice-body" in content
    assert ".notice-dismiss" in content


def test_frontend_backend_schema_field_compatibility():
    """Verify every field consumed by frontend JS exists in backend Pydantic schemas."""
    # SearchResponse fields consumed
    search_consumed = ["total_results", "latency_ms", "results"]
    for field in search_consumed:
        assert field in SearchResponse.model_fields, f"Missing {field} in SearchResponse"

    # SearchResultItem fields consumed
    item_consumed = [
        "rank", "score", "chunk_id", "doc_id", "chunk_index",
        "title", "url", "author", "content", "word_count", "faiss_id"
    ]
    for field in item_consumed:
        assert field in SearchResultItem.model_fields, f"Missing {field} in SearchResultItem"

    # AskResponse fields consumed
    ask_consumed = [
        "query", "answer", "refused", "confidence_score",
        "citations", "sources", "latency_ms", "model", "provider"
    ]
    for field in ask_consumed:
        assert field in AskResponse.model_fields, f"Missing {field} in AskResponse"

    # CitationItem fields consumed
    citation_consumed = ["doc_tag", "chunk_id", "title", "url", "score"]
    for field in citation_consumed:
        assert field in CitationItem.model_fields, f"Missing {field} in CitationItem"

    # HealthResponse fields consumed
    health_consumed = ["status", "index_loaded", "db_connected", "version"]
    for field in health_consumed:
        assert field in HealthResponse.model_fields, f"Missing {field} in HealthResponse"

    # StatsResponse fields consumed
    stats_consumed = ["total_documents", "total_chunks", "total_vectors", "embedding_model"]
    for field in stats_consumed:
        assert field in StatsResponse.model_fields, f"Missing {field} in StatsResponse"


def test_frontend_markdown_rendering_layer_defined():
    """Verify index.html defines the safe Markdown, math, and citation rendering pipeline."""
    index_path = Path("app/static/index.html")
    assert index_path.exists(), "index.html missing"
    content = index_path.read_text(encoding="utf-8")

    assert "function renderGroundedAnswer(" in content
    assert "function formatMath(" in content
    assert "function handleInlineCitationClick(" in content
    assert "renderGroundedAnswer(data.answer, data.citations)" in content
    assert "textContent = sanitizeText(data.answer)" not in content


def test_frontend_css_markdown_and_math_tokens():
    """Verify style.css includes styling tokens for tables, math, code blocks, and citations."""
    css_path = Path("app/static/style.css")
    assert css_path.exists(), "style.css missing"
    css_content = css_path.read_text(encoding="utf-8")

    assert ".qa-table" in css_content
    assert ".qa-table-wrapper" in css_content
    assert ".math-inline" in css_content
    assert ".math-display" in css_content
    assert ".inline-citation" in css_content
    assert ".qa-code-block" in css_content
    assert ".qa-inline-code" in css_content
    assert ".qa-blockquote" in css_content
    assert ".qa-heading" in css_content


def test_frontend_safe_markdown_and_math_execution():
    """Verify renderGroundedAnswer accurately transforms Markdown, tables, math, and prevents XSS."""
    import re
    import json
    import shutil
    import subprocess

    if not shutil.which("node"):
        return

    index_path = Path("app/static/index.html")
    content = index_path.read_text(encoding="utf-8")

    sanitize_m = re.search(r"function sanitizeText\([\s\S]*?\n    \}", content)
    format_math_m = re.search(r"function formatMath\([\s\S]*?\n    \}", content)
    render_m = re.search(r"function renderGroundedAnswer\([\s\S]*?\n    \}", content)
    assert sanitize_m and format_math_m and render_m

    js_funcs = f"{sanitize_m.group(0)}\n{format_math_m.group(0)}\n{render_m.group(0)}"

    test_runner = f"""
{js_funcs}
const sample = '**GRPO** | Metric | Score |\\n|---|---|\\n| Accuracy | 0.95 |\\n\\nInline math: \\\\( \\\\pi_\\\\theta \\\\) and [Doc 1].\\n<script>alert("xss")</script>\\n[link](javascript:alert("bad"))';
const out = renderGroundedAnswer(sample, [{{ doc_tag: '[Doc 1]', chunk_id: 'c1' }}]);
console.log(JSON.stringify({{
    has_bold: out.includes('<strong>GRPO</strong>'),
    has_table: out.includes('<table class="qa-table">'),
    has_math: out.includes('<span class="math-inline">π<sub>θ</sub></span>'),
    has_citation: out.includes('class="inline-citation"'),
    xss_neutralized: !out.includes('<script>') && out.includes('&lt;script&gt;'),
    bad_link_neutralized: !out.includes('javascript:')
}}));
"""
    proc = subprocess.run(["node", "-e", test_runner], capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())
    assert res["has_bold"] is True
    assert res["has_table"] is True
    assert res["has_math"] is True
    assert res["has_citation"] is True
    assert res["xss_neutralized"] is True
    assert res["bad_link_neutralized"] is True
