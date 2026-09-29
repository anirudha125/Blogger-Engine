# PROJECT HANDOFF: Blogger Engine RAG Platform

> **Target Audience**: Incoming AI Coding & Research Agent / Engineering Reviewer / Interview Evaluator  
> **Repository Root**: `d:\Blogger-Engine-main`  
> **Authoritative Companions**: [`README.md`](file:///d:/Blogger-Engine-main/README.md), [`PROGRESS.md`](file:///d:/Blogger-Engine-main/PROGRESS.md)  
> **Current Status**: Phase 5 Complete (Steps 1–8 Verified). Zero pending code changes.

---

## 1. Executive Summary & Project Purpose

### What Blogger Engine Is
**Blogger Engine** is an open-source, production-hardened Retrieval-Augmented Generation (RAG) platform. It provides high-precision semantic passage search and citation-backed question answering over **8,242 technical machine learning publications** (**81,123 passage chunks**).

### Problem Solved
Traditional naive RAG implementations face severe bottlenecks when applied to deep technical corpora:
1. **Silent Embedding Truncation**: Older bi-encoders (e.g., `all-MiniLM-L6-v2`) truncate text beyond 256 tokens (~180 words). Whole-document embeddings silently discard the body of articles, resulting in a dismal **9.3%** baseline Document Recall@3.
2. **Dense Semantic Crowding**: In saturated technical subfields (e.g. RLHF, DPO, PPO), dense vector search returns multiple chunks from the same or rival articles, crowding out the exact answer passage.
3. **Entity & Acronym Blindness**: Pure dense retrieval struggles with exact mathematical symbols, author citations, and library names (e.g., *DP-SGD*, *FlashAttention*, *Ivison et al.*), where BM25 excels.
4. **Context Window Exhaustion & Hallucination**: Feeding entire articles into LLM context prompts inflates token costs by >90%, causes "lost-in-the-middle" attention degradation, and leads to ungrounded hallucinations without source verification.

### System Solution
Blogger Engine implements **Canonical Experiment 4**:
- **Two-Stage Hybrid Retrieval**: Dense Bi-Encoder (`BAAI/bge-small-en-v1.5`) + Sparse Lexical (`BM25Okapi`) fused via Reciprocal Rank Fusion (`RRF`, $k=60$).
- **Neural Cross-Attention Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` evaluating top-20 fused candidate pairs.
- **Document-Level Deduplication**: Enforces strict inter-document diversity, ensuring the final top-3 context window contains 3 distinct documents.
- **Calibrated Confidence Gating**: Pre-generation threshold gate ($0.65$) refusing out-of-domain queries before invoking the LLM.
- **Grounded LLM Generation**: Provider-agnostic synthesis (Groq / Gemini / OpenAI / Ollama) with deterministic `[Doc X]` bracketed citation extraction.
- **$0 Recurring Operational Cost**: Runs 100% on standard x86 multi-core CPU with zero production GPUs and zero paid managed vector databases.

---

## 2. Current Production Architecture

### High-Level Architecture Flowchart

```
========================================================================================
                          OFFLINE INGESTION & CANONICAL ARTIFACTS
========================================================================================
  Raw Scraped Blog Corpus (8,925 articles / 8,242 unique canonical articles)
          │
          ▼
   app/core/chunker.py
     • Sliding window: ~400 words per chunk, ~50 words overlap
     • Deterministic SHA-256 doc_id (12-char hex) & chunk_id ({doc_id}_c{idx})
     • Sequential, deterministic faiss_id assignment [0..81122]
          │
          ├───► SQLite DB: data/blogger_dedup.db (81,123 canonical chunks)
          │       • documents table (id, faiss_id, doc_id, chunk_index, title, url, author, content, word_count)
          │       • query_logs table (query, retrieved_ids, latency_ms, mode, timestamp)
          │
          ├───► BAAI/bge-small-en-v1.5 (d=384, 512-token sequence capacity)
          │       • FAISS IndexFlatIP: indexes/faiss_chunked_bge_dedup.index (81,123 vectors)
          │       • Exact Inner Product = Cosine Similarity on L2-normalized vectors
          │
          └───► rank-bm25 BM25Okapi
                  • Pre-tokenized cache: indexes/bm25_chunked_dedup.pkl (81,123 passages)

========================================================================================
                               ONLINE SERVING PIPELINE
========================================================================================
   Web Browser Client (app/static/index.html & style.css)
          │  HTTP POST /api/search | HTTP POST /api/ask (JSON: query, top_k=3)
          ▼
   FastAPI Application Runtime (app/main.py)
     • Lifespan startup: preloads HybridRetriever and AnswerGenerator into app.state
     • Pre-warms cross-encoder weights to eliminate first-request cold-start latency
          │
          ├──► app/core/hybrid_retriever.py (HybridRetriever)
          │      │
          │      ├─► Step 1: Dense Retrieval (BGE-small + FAISS IndexFlatIP)
          │      │     • Query encoded with prefix: "Represent this sentence for searching relevant passages: "
          │      │     • FAISS search -> Top-20 dense candidates (latency: ~26 ms)
          │      │
          │      ├─► Step 2: Sparse Lexical Retrieval (BM25Okapi)
          │      │     • Tokenized query scored against 81,123 chunks -> Top-20 sparse candidates (latency: ~435 ms)
          │      │
          │      ├─► Step 3: Reciprocal Rank Fusion (RRF, k=60)
          │      │     • Fuses dense and sparse rankings -> Top-20 fused candidates
          │      │
          │      ├─► Step 4: Neural Cross-Encoder Reranking
          │      │     • cross-encoder/ms-marco-MiniLM-L-6-v2 evaluates 20 (query, passage) pairs
          │      │     • Joint transformer cross-attention -> Sigmoid scores [0, 1] (latency: ~872 ms)
          │      │
          │      ├─► Step 5: Document-Level Deduplication
          │      │     • Groups candidates by doc_id; retains highest-scoring chunk per document
          │      │
          │      └─► Step 6: Final Slicing & SQLite Hydration
          │            • Slices top_k=3 passages and returns RetrievalResultList
          │
          ├──► app/core/generator.py (AnswerGenerator) [QA Mode Only]
          │      │
          │      ├─► Gate 1: Check if passages list is empty -> Refuse
          │      ├─► Gate 2: Check if max(score) < 0.65 -> Refuse
          │      │     • Refusal message: "I do not have enough confidence in the indexed blog articles..."
          │      │     • Consumes zero LLM tokens on out-of-domain queries
          │      │
          │      └─► Synthesis: Top score >= 0.65
          │            • Formats grounded prompt with numbered [Doc 1], [Doc 2], [Doc 3] passages
          │            • Calls LLM API (Groq llama-3.1-8b-instant / Gemini / OpenAI)
          │            • Regex parses [Doc X] citations and maps to source URLs and scores
          │
          ├──► app/db/database.py (log_query)
          │      • Non-blocking, ephemeral-safe query logging (read-only filesystem safe)
          │
          ▼
   JSON Response (answer, citations, sources, confidence_score, refused, latency_ms)
```

---

## 3. Canonical Models, Paths, and Settings

The application runtime is strictly bound to the following configuration:

### Canonical Models
- **Dense Embedding Model**: `BAAI/bge-small-en-v1.5`
  - Dimensions: 384
  - Max Sequence Length: 512 tokens
  - Asymmetric Query Prefix: `"Represent this sentence for searching relevant passages: "` (passages embedded without prefix).
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - 6-layer transformer cross-encoder trained on MS MARCO.
  - Scores query and passage jointly via cross-attention.
  - Output mapped to $[0, 1]$ via sigmoid.
- **Default LLM**: `llama-3.1-8b-instant` via Groq (pluggable to Gemini, OpenAI, or local Ollama).

### Canonical Production Artifacts
- **FAISS Index**: [`indexes/faiss_chunked_bge_dedup.index`](file:///d:/Blogger-Engine-main/indexes/faiss_chunked_bge_dedup.index)
  - Type: `faiss.IndexFlatIP`
  - Vector count: 81,123 vectors ($d=384$)
  - Size: 124.6 MB
- **SQLite Database**: [`data/blogger_dedup.db`](file:///d:/Blogger-Engine-main/data/blogger_dedup.db)
  - Rows: 81,123 chunks across 8,242 unique documents
  - Size: 328.9 MB (Compressed: [`data/blogger_dedup.db.gz`](file:///d:/Blogger-Engine-main/data/blogger_dedup.db.gz), 69.1 MB)
  - Index: `idx_faiss_id` covering `faiss_id` column $[0..81122]$
- **BM25 Cache**: [`indexes/bm25_chunked_dedup.pkl`](file:///d:/Blogger-Engine-main/indexes/bm25_chunked_dedup.pkl)
  - Serialized `rank-bm25 BM25Okapi` instance containing pre-tokenized corpus
  - Size: 184.6 MB

### Canonical Runtime Settings ([`app/config.py`](file:///d:/Blogger-Engine-main/app/config.py))
- `TOP_K`: `3`
- `SIMILARITY_THRESHOLD`: `0.65`
- `CANDIDATE_DEPTH`: `20`
- `RERANK_DEPTH`: `20`
- `RRF_K`: `60`
- `DOC_DEDUP`: `True`
- `USE_RERANKER`: `True`
- `ENABLE_QUERY_LOGGING`: `True`

---

## 4. Final Validated Evaluation Metrics

All metrics were evaluated offline on the frozen 60-query benchmark dataset ([`data/eval_dataset_full.json`](file:///d:/Blogger-Engine-main/data/eval_dataset_full.json): 54 in-domain + 6 out-of-domain) at final output depth $k=3$:

| Metric | Baseline (Exp 0: MiniLM) | Final Production (Exp 4: BGE+BM25+RRF+CE+Dedup) | Absolute Gain | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Document Recall@3** | `46.3%` (25/54) | **`66.7%` (36/54)** | **`+20.4 pp`** | **`+44.1%`** |
| **Passage Recall@3** | `27.8%` (15/54) | **`46.3%` (25/54)** | **`+18.5 pp`** | **`+66.5%`** |
| **Citation Source Accuracy** | `77.2%` | **`84.6%`** | **`+7.4 pp`** | **`+9.6%`** |
| **Lexical Groundedness** | `83.0%` | **`82.1%`** | `-0.9 pp` | `-1.1%` |
| **Context Compression** | `90.7%` | **`90.4%`** | `-0.3 pp` | ~1,948 prompt tokens vs ~20,199 in whole-doc |
| **Mean Retrieval Latency (CPU)** | `18.52 ms` | **`1,469.89 ms`** (~1.47 s) | `+1.45 s` | CPU cross-attention trade-off |
| **Mean Unique Docs in Top-3** | `2.50 docs` | **`3.00 docs`** | **`+0.50 docs`** | **100% inter-document diversity** |

### Benchmark vs. Smoke-Test Latency Distinction
- **Offline Benchmark Latency (~1,469.89 ms)**: Measured across all 60 queries in batch evaluation on multi-core CPU. Breakdown:
  - Dense BGE encoding + FAISS search: ~26 ms
  - Sparse BM25 tokenization + scoring: ~435 ms
  - RRF fusion: <1 ms
  - Cross-Encoder reranking (20 pairs): ~872 ms
  - SQLite metadata hydration + deduplication: ~136 ms
- **Online Single-Query Smoke Tests (~1,120 ms)**: Live FastAPI requests with pre-warmed models and memory-resident caches typically complete in ~1.1 to 1.3 seconds.

---

## 5. Confidence Threshold Calibration

### Methodology
To prevent test-set leakage, threshold calibration was conducted on a separate held-out dataset of 30 queries ([`data/calibration_dataset.json`](file:///d:/Blogger-Engine-main/data/calibration_dataset.json): 20 in-domain + 10 out-of-domain) that does not overlap with the 60-query benchmark. All 20 in-domain calibration queries were sampled from documents strictly disjoint from the 38 target documents of the benchmark.

### Calibration Distribution
- In Experiment 0 (MiniLM), raw cosine scores clustered low, necessitating a threshold of `0.35`.
- In Experiment 4, BGE dense embeddings and cross-encoder sigmoid scores shifted systematically higher (~+0.25 to +0.35 upward shift).
- Measured calibration scores:
  - **In-Domain Minimum Score**: `0.7423` (Median: `0.8355`, Mean: `0.8356`, Max: `0.9046`)
  - **Out-of-Domain Maximum Score**: `0.6464` (Median: `0.6095`, Mean: `0.6047`, Min: `0.5441`)
  - **Empirical Separation Margin**: `+0.0959` clean gap between highest OOD probe (`0.6464`) and lowest valid in-domain question (`0.7423`).

### Canonical Threshold: `0.65`
- Clears the highest OOD probe (`0.6464`) with safety headroom, while preserving a **+0.0923 buffer** below the lowest in-domain calibration score (`0.7423`).
- **In-Domain Preservation**: **0 false refusals** across all 54 in-domain benchmark queries (100% valid acceptance).
- **Out-of-Domain Rejection**: Successfully rejects 5 of 6 benchmark OOD queries (e.g., sourdough pizza baking, quantum dilution refrigeration) and all 10 held-out calibration OOD probes.
- **Honest Limitation**: A similarity score gate is a heuristic guardrail, not an infallible mathematical proof of OOD status. Queries near the semantic boundary of machine learning concepts may score close to 0.65.

---

## 6. Application Runtime Architecture

### FastAPI Lifespan & Lifecycle ([`app/main.py`](file:///d:/Blogger-Engine-main/app/main.py))
- **Single Lifespan Initialization**: Instantiates `HybridRetriever` and `AnswerGenerator` once during application startup and attaches them to `app.state.retriever` and `app.state.generator`.
- **Startup Warmup**: Invokes `retriever.warmup()` during lifespan, triggering a dummy cross-encoder inference pass to pre-load weights into CPU cache and compile execution graphs.
- **Zero Request-Path Instantiations**: No models, tokenizers, or FAISS indexes are constructed during HTTP request handling.

### API Routes & 503 Handling ([`app/api/routes.py`](file:///d:/Blogger-Engine-main/app/api/routes.py))
- **Strict Dependency Injection**: Routes depend on `get_retriever()` and `get_generator()`.
- **Explicit 503 Degradation**: If models fail to load or indexes are absent, endpoints raise `HTTPException(503, "Search engine is unavailable.")` rather than silently crashing or falling back to toy sample indexes.
- **Health Probing (`GET /health`)**: Returns `status="ok"` and `index_loaded=true` when healthy; returns `status="degraded"` and `index_loaded=false` when retriever is uninitialized.
- **CORS Consistency**: Configured to allow credentials only with explicit origins; defaults to wildcard origins with `allow_credentials=False` to strictly satisfy the Fetch specification.

### Concurrency Safety & Latency Tracking ([`app/core/hybrid_retriever.py`](file:///d:/Blogger-Engine-main/app/core/hybrid_retriever.py))
- **`RetrievalResultList` + `ContextVar`**: Per-request latency breakdowns are stored in a custom list subclass attached to the returned results and backed by Python's `contextvars.ContextVar`. Concurrent requests across multiple threads cannot overwrite each other's latency telemetry.
- **Thread-Safe Warmup**: Protected by a double-checked threading lock (`threading.Lock()`), guaranteeing idempotency even if multiple worker threads call `warmup()` simultaneously.

### Frontend Alignment ([`app/static/index.html`](file:///d:/Blogger-Engine-main/app/static/index.html), [`app/static/style.css`](file:///d:/Blogger-Engine-main/app/static/style.css))
- **Top-K Synchronization**: Client-side query submissions are hardcoded to `CANONICAL_TOP_K = 3`.
- **Zero Browser Popups**: Browser `alert()` calls have been completely eliminated.
- **Non-Blocking In-Page Notices**: Replaced with `.editorial-notice` banners styled to match the luxury dark/gold aesthetic (`notice-warning`, `notice-error`, `notice-info`).
- **Resilient Error Parsing**: `parseApiError()` cleanly handles HTTP 503, 400/422 validation errors, and network disconnects without producing `[object Object]` artifacts.

---

## 7. Automated Test Suite

The repository contains **84 automated tests** across 9 test suites, all passing with zero failures:

| Test File | Test Count | Scope & Coverage |
| :--- | :---: | :--- |
| [`tests/test_phase1.py`](file:///d:/Blogger-Engine-main/tests/test_phase1.py) | 6 | Chunker sliding window, whitespace cleaning, metadata preservation, SQLite CRUD, logging |
| [`tests/test_phase2.py`](file:///d:/Blogger-Engine-main/tests/test_phase2.py) | 14 | Embeddings, batch encoding, retriever ranking, score filtering, prompt assembly, citation regex, refusal thresholds, secret masking |
| [`tests/test_phase3.py`](file:///d:/Blogger-Engine-main/tests/test_phase3.py) | 8 | FastAPI routes (/health, /api/stats, /api/search, /api/ask), Pydantic validation, query logging |
| [`tests/test_phase4.py`](file:///d:/Blogger-Engine-main/tests/test_phase4.py) | 6 | Benchmark dataset integrity, baseline artifacts, lexical groundedness, citation source accuracy |
| [`tests/test_hybrid_retriever.py`](file:///d:/Blogger-Engine-main/tests/test_hybrid_retriever.py) | 21 | HybridRetriever canonical defaults, candidate expansion, RRF fusion, cross-encoder reranking, document deduplication, thread-safe warmup, latency metadata concurrency isolation |
| [`tests/test_phase5_step4.py`](file:///d:/Blogger-Engine-main/tests/test_phase5_step4.py) | 8 | FastAPI lifespan wiring, cross-encoder pre-warming, application-scoped generator, canonical /api/search & /api/ask execution, 503 error handling, CORS consistency |
| [`tests/test_phase5_step5.py`](file:///d:/Blogger-Engine-main/tests/test_phase5_step5.py) | 5 | Frontend static HTML serving, top_k=3 contract, absence of alert(), CSS notice classes, schema field compatibility |
| [`tests/test_phase5_step6_e2e.py`](file:///d:/Blogger-Engine-main/tests/test_phase5_step6_e2e.py) | 10 | Live application startup against canonical artifacts, real search/ask execution, out-of-domain refusal, concurrent request isolation, database circular import regression |
| [`tests/test_phase5_step7_docker.py`](file:///d:/Blogger-Engine-main/tests/test_phase5_step7_docker.py) | 6 | Dockerfile multi-stage syntax, CPU PyTorch flags, model pre-caching, non-root user execution, canonical artifact paths, healthcheck entrypoint, .dockerignore rules |
| **TOTAL** | **84** | **84 passed, 0 failed, 1 warning** |

---

## 8. Docker Packaging & Validation Status

### Packaging Implementation
- **Dockerfile** ([`Dockerfile`](file:///d:/Blogger-Engine-main/Dockerfile)): Multi-stage build (`builder` -> `runner`) based on `python:3.11-slim`.
  - Installs CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`), shedding ~2 GB of CUDA bloat.
  - Pre-caches `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` during build.
  - Configures `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` in the runner stage.
  - Runs as unprivileged user `appuser:appgroup` (UID 10001).
  - Production `HEALTHCHECK` probing `/health` via Python standard library `urllib.request`.
  - Conforms to dynamic cloud `$PORT` assignment.
- **Dockerignore** ([`.dockerignore`](file:///d:/Blogger-Engine-main/.dockerignore)): Strictly excludes `.git`, `.venv`, `tests/`, `notebooks/`, research scripts, and development databases, while preserving canonical production assets.
- **Asset Portability**:
  - Handles compressed database `data/blogger_dedup.db.gz` (decompresses at build time).
  - Automatically regenerates `bm25_chunked_dedup.pkl` if not committed directly.

### Strict Verification Disclosure
> **IMPORTANT VERIFICATION NOTICE**:  
> The `Dockerfile` and `.dockerignore` have been statically verified and validated via automated specification tests ([`tests/test_phase5_step7_docker.py`](file:///d:/Blogger-Engine-main/tests/test_phase5_step7_docker.py)).  
> However, **an actual local `docker build` and container runtime execution were NOT performed on the development machine** because the Docker CLI is not installed in this Windows environment (`docker: The term 'docker' is not recognized`).  
> Cloud deployment must be executed on a standard Linux CI/CD host or container platform.

---

## 9. Production Runtime vs. Research-Only Assets

To prevent confusion, repository assets are strictly partitioned:

### Production Runtime Assets (Required for Live Serving)
- **`app/`**: Core application codebase (`api/`, `core/`, `db/`, `static/`, `config.py`, `main.py`, `schemas.py`).
- **`data/blogger_dedup.db`**: Canonical SQLite database (81,123 chunks, 8,242 articles).
- **`indexes/faiss_chunked_bge_dedup.index`**: Canonical BGE FAISS IndexFlatIP (81,123 vectors).
- **`indexes/bm25_chunked_dedup.pkl`**: Pre-tokenized BM25 cache.
- **`requirements.txt`**, **`Dockerfile`**, **`.dockerignore`**: Production build definitions.

### Research-Only & Historical Artifacts (Preserved for Traceability)
- **`indexes/faiss_chunked_dedup.index`**: Historical Experiment 0 MiniLM chunk index (384d, 81,123 vectors).
- **`indexes/faiss_baseline_dedup.index`**: Historical whole-document MiniLM baseline index (384d, 8,242 vectors).
- **`indexes/faiss.index`**, **`indexes/faiss_baseline.index`**, **`indexes/metadata_full.pkl`**: Preserved raw Kaggle ingestion artifacts.
- **`data/blogger.db`**, **`indexes/faiss_chunked.index`**: Local 46-chunk development sample dataset.
- **`data/eval_dataset_full.json`**, **`data/calibration_dataset.json`**: Frozen evaluation and calibration benchmarks.
- **`experiments/`**, **`calibration/`**, **`scripts/`**: Offline evaluation, experiment comparison, and data curation scripts.

---

## 10. How to Run Locally

### 1. Environment Setup
```bash
# Clone and enter repository
cd d:\Blogger-Engine-main

# Activate virtual environment
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
cp .env.example .env
# Set LLM_API_KEY (e.g. Groq API key) if live answer generation is desired.
# If left empty, retrieval and citations work transparently with an offline notice.
```

### 3. Launch Application
```bash
# Launch production FastAPI server via application factory
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```
- Web UI: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`

### 4. Run Test Suite
```bash
python -m pytest -v
```

---

## 11. System Trade-Offs & Known Limitations

1. **CPU Retrieval Latency (~1.47 s)**: Running 20 candidate passage pairs through a 6-layer transformer cross-encoder on CPU takes ~870 ms. Total retrieval latency is ~1.47 s. In a GPU environment, this would take ~25 ms.
2. **Memory Footprint (~650–700 MB RSS)**: The FAISS index (124 MB), BM25 cache (185 MB), and model weights (~350 MB) require at least 1 GB of system RAM. Deploying on a 512 MB micro-tier instance will trigger an Out-Of-Memory (OOM) kill.
3. **Artifact Size**: Canonical production assets occupy ~650 MB on disk. Standard GitHub restricts files >100 MB without Git LFS; the Dockerfile supports building with `blogger_dedup.db.gz` and build-time BM25 regeneration.
4. **External LLM Dependency**: Full question answering requires an active external LLM provider API key (Groq, Gemini, OpenAI) or local Ollama instance.
5. **Recall Upper Bound**: Document Recall@3 is 66.7%. Diagnostic analysis proved that 13.0% of target documents do not appear in the top-20 fused candidate pool, setting an 83.3% theoretical ceiling for Stage 2 reranking.

---

## 12. Final Project Status

- **Research**: COMPLETE
- **Retrieval Optimization**: COMPLETE (Canonical Experiment 4 validated)
- **Confidence Calibration**: COMPLETE (Threshold 0.65 validated on held-out dataset)
- **Backend Productionization**: COMPLETE (Lifespan, 503 handling, concurrency isolation)
- **Frontend Hardening**: COMPLETE (top_k=3, non-blocking in-page editorial notice, zero alerts)
- **End-to-End Validation**: COMPLETE (84 passing tests)
- **Docker Configuration**: COMPLETE (Dockerfile + .dockerignore created & spec-tested)
- **Documentation**: COMPLETE (README, PROJECT_HANDOFF, PROGRESS synchronized)
- **Actual Cloud Deployment**: NOT YET PERFORMED (Local Docker CLI unavailable; ready for cloud deployment)
