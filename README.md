# Blogger Engine: Precision RAG Search & QA Platform

> A production-structured, CPU-compatible Retrieval-Augmented Generation (RAG) platform delivering grounded semantic search and verifiable citation-backed question answering over 8,242 technical machine learning publications (81,123 passage chunks).

---

## 1. Project Overview

### What Blogger Engine Is
**Blogger Engine** is an end-to-end, open-source RAG search and question-answering system. It transforms an unchunked, exploratory scraping script into a modular, production-hardened platform featuring:
- **Hybrid Retrieval**: Dense bi-encoder search combined with BM25 lexical term matching fused via Reciprocal Rank Fusion (RRF).
- **Neural Cross-Encoder Reranking**: Full cross-attention reranking over candidate passages on standard CPU.
- **Document-Level Deduplication**: Guaranteed inter-document diversity in the top-3 context window.
- **Grounded Synthesis & Citations**: Provider-agnostic LLM answer generation with strict `[Doc X]` bracketed citation attribution.
- **Calibrated Confidence Refusal Gate**: Automatic refusal of out-of-domain or low-confidence queries before generation.
- **$0/Month Serving Architecture**: Fully CPU-compatible inference requiring zero production GPUs and zero paid cloud databases.

### The Problem Being Solved
Standard naive RAG implementations suffer from compounding failure modes when applied to technical corpus domains:
1. **Silent Embedding Truncation**: Standard bi-encoders (e.g. `all-MiniLM-L6-v2`) truncate texts beyond 256 tokens (~180 words). Whole-document embeddings silently discard the body of long technical articles, collapsing document recall to **9.3%**.
2. **Topical Semantic Crowding**: In dense technical domains (e.g. RLHF, DPO, PPO), multiple articles cover identical concepts. Dense retrieval frequently returns multiple chunks from rival articles or the same article, crowding out the specific answering text.
3. **Vocabulary Mismatch & Entity Blindness**: Dense embeddings often struggle with exact mathematical acronyms, author citations, and library names (e.g., *DP-SGD*, *FlashAttention*, *Ivison et al.*), which BM25 captures with precision.
4. **Context Window Exhaustion & Hallucination**: Feeding entire articles into LLM context prompts inflates token costs by >90%, causes "lost-in-the-middle" attention degradation, and leads to ungrounded hallucinations without source verification.

---

## 2. High-Level Architecture

```
User Query
    │
    ▼
Frontend Client (app/static/index.html & style.css)
    │  HTTP POST /api/search  |  HTTP POST /api/ask  (JSON: query, top_k=3)
    ▼
FastAPI Application Runtime (app/main.py)
    │
    ├──────────────────────────┬──────────────────────────┐
    ▼                          ▼                          ▼
BGE Dense Retrieval         BM25 Lexical Retrieval     Query Logging (SQLite)
(BAAI/bge-small-en-v1.5)    (rank-bm25 BM25Okapi)      (ephemeral-safe)
FAISS IndexFlatIP (384d)    Tokenized Chunks Cache
81,123 vectors              81,123 passages
[Top-20 Candidates]         [Top-20 Candidates]
    │                          │
    └──────────────┬───────────┘
                   ▼
       Reciprocal Rank Fusion (RRF, k=60)
       [Top-20 Fused Candidates]
                   │
                   ▼
       Cross-Encoder Reranking
       (cross-encoder/ms-marco-MiniLM-L-6-v2)
       [20 (query, passage) pairs scored jointly via cross-attention]
                   │
                   ▼
       Document-Level Deduplication
       [Groups by unique doc_id; retains highest-scoring chunk per article]
                   │
                   ▼
       Top-3 Final Diverse Passages (top_k=3)
                   │
                   ├─────────────────────────────────────────────────┐
                   │ (Search Mode)                                   │ (QA Mode)
                   ▼                                                 ▼
          Search Response JSON                         Confidence Gate (0.65)
          (ranked passages, scores,                      │
           metadata, latency breakdown)                  ├── Top Score < 0.65:
                                                         │     Refuse Answer
                                                         │     (Zero LLM calls)
                                                         │
                                                         └── Top Score >= 0.65:
                                                               Grounded Prompt Assembly
                                                               │
                                                               ▼
                                                               LLM Generation
                                                               (Groq / Gemini / OpenAI)
                                                               │
                                                               ▼
                                                               Answer + Verified Citations
```

---

## 3. Data Pipeline & Ingestion

1. **Corpus Extraction & Deduplication**:
   - Starting from 8,925 raw scraped web articles, MD5-hashed URL normalization and content hashing identified and consolidated duplicate mirror posts.
   - Result: **8,242 unique canonical technical articles**.
2. **Sliding-Window Chunking (`app/core/chunker.py`)**:
   - Chunks articles into ~400-word passages with a 50-word sliding overlap.
   - Preserves complete article coverage while eliminating silent truncation.
   - Generates **81,123 canonical passage chunks**.
   - Every chunk is deterministically identified by SHA-256 derived `doc_id` and indexed chunk ID (`{doc_id}_c{idx}`).
3. **Relational & Vector Persistence**:
   - **SQLite (`data/blogger_dedup.db`)**: Stores full text, titles, URLs, authors, word counts, and deterministic `faiss_id` mappings [0..81122].
   - **FAISS IndexFlatIP (`indexes/faiss_chunked_bge_dedup.index`)**: Stores 81,123 L2-normalized 384-dimensional dense vectors for exact inner product cosine search.
   - **BM25 Cache (`indexes/bm25_chunked_dedup.pkl`)**: In-memory serialized token index for sub-second startup.

---

## 4. Retrieval Architecture (Canonical Experiment 4)

Blogger Engine employs a two-stage hybrid retrieval and reranking pipeline:

### Component Breakdown
1. **Dense Bi-Encoder (`BAAI/bge-small-en-v1.5`)**:
   - 384 dimensions, 512-token sequence capacity (2x larger than MiniLM).
   - Generates semantic query representations with task-specific prefix: `"Represent this sentence for searching relevant passages: "`. Document passages are embedded raw.
   - L2 normalized so FAISS `IndexFlatIP` performs exact cosine similarity in ~26 ms on CPU.
2. **Sparse Lexical Retrieval (`rank-bm25 BM25Okapi`)**:
   - Tokenizes text with lowercasing and punctuation stripping.
   - Computes BM25 inverse document frequency scores over all 81,123 chunks in ~430 ms on CPU.
   - Solves entity blindness, recovering exact technical terms (e.g. *LoRA*, *DP-SGD*, *ViT*).
3. **Reciprocal Rank Fusion (RRF)**:
   - Fuses dense rank and sparse rank without requiring brittle score normalization:
     $$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   - Candidate pool depth: $N=20$.
4. **Cross-Encoder Neural Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**:
   - Unlike bi-encoders that encode query and document separately, the cross-encoder feeds `(query, passage)` pairs jointly through full transformer cross-attention.
   - Evaluates all 20 candidates on CPU in ~870 ms.
   - Maps raw logits through sigmoid to produce calibrated relevance scores in $[0, 1]$.
5. **Document-Level Deduplication**:
   - Groups reranked candidates by `doc_id`.
   - Retains only the single highest-scoring chunk per document, guaranteeing that the top-3 slots represent 3 distinct articles rather than redundant chunks from a single article.
6. **Confidence Refusal Gate (`similarity_threshold = 0.65`)**:
   - If the top passage relevance score is below `0.65`, the system refuses immediately:
     *"I do not have enough confidence in the indexed blog articles to answer this question."*
   - Prevents ungrounded synthesis and hallucination on out-of-domain queries while consuming zero LLM tokens.

---

## 5. Why CPU-Only Serving Is Possible

Blogger Engine requires **zero production GPUs**:
- **Lightweight Models**: `bge-small-en-v1.5` has 33M parameters; `ms-marco-MiniLM-L-6-v2` has 22M parameters. Both run efficiently on modern x86 CPU cores.
- **Exact Vector Search Efficiency**: At 81,123 vectors of 384 dimensions, a brute-force `IndexFlatIP` inner product matrix multiplication requires only ~31 million floating-point operations. On CPU with OpenMP (`libgomp`), this completes in **~26 ms**.
- **Stage 2 Bounded Scoring**: Cross-attention reranking is restricted to the top-20 fused candidates, keeping CPU reranking time under **~870 ms**.
- **Total Mean Retrieval Latency**: **~1.47 seconds** per query on a standard multi-core CPU.

---

## 6. API Endpoints

The FastAPI server provides strict, schema-validated REST endpoints:

### `GET /health`
Liveness probe checking vector engine and database connectivity.
```json
{
  "status": "ok",
  "index_loaded": true,
  "db_connected": true,
  "version": "1.0.0"
}
```
*(Returns `status: "degraded"` and `index_loaded: false` if models or indexes fail to initialize).*

### `GET /api/stats`
Corpus and active configuration analytics.
```json
{
  "total_documents": 8242,
  "total_chunks": 81123,
  "total_vectors": 81123,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "llm_provider": "groq",
  "llm_model": "llama-3.1-8b-instant",
  "query_logging_enabled": true
}
```

### `POST /api/search`
Execute hybrid dense + BM25 search with cross-encoder reranking and document deduplication.
- **Request**:
  ```json
  {
    "query": "What is Direct Preference Optimization DPO?",
    "top_k": 3,
    "min_score": null
  }
  ```
- **Response**:
  ```json
  {
    "query": "What is Direct Preference Optimization DPO?",
    "total_results": 3,
    "latency_ms": 1142.3,
    "results": [
      {
        "rank": 1,
        "score": 0.8245,
        "chunk_id": "a1b2c3d4e5f6_c2",
        "doc_id": "a1b2c3d4e5f6",
        "chunk_index": 2,
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "url": "https://example.com/blog/dpo-overview",
        "author": "Eric Mitchell",
        "content": "Direct Preference Optimization (DPO) optimizes the policy directly...",
        "word_count": 395,
        "faiss_id": 14205
      }
    ]
  }
  ```

### `POST /api/ask`
Execute full RAG question answering with citations and refusal gating.
- **Request**:
  ```json
  {
    "query": "What is RLAIF and how does it differ from RLHF?",
    "top_k": 3,
    "similarity_threshold": 0.65
  }
  ```
- **Response (In-Domain)**:
  ```json
  {
    "query": "What is RLAIF and how does it differ from RLHF?",
    "answer": "RLAIF (Reinforcement Learning from AI Feedback) uses an LLM to generate preference labels [Doc 1], whereas traditional RLHF relies on human annotators [Doc 2]...",
    "refused": false,
    "confidence_score": 0.7709,
    "citations": [
      {
        "doc_tag": "[Doc 1]",
        "chunk_id": "8f3b2e1a9c0d_c1",
        "title": "RLAIF: Scaling Reinforcement Learning from AI Feedback",
        "url": "https://example.com/blog/rlaif-scaling",
        "score": 0.7709
      }
    ],
    "sources": [ ... ],
    "latency_ms": 1380.5,
    "model": "llama-3.1-8b-instant",
    "provider": "groq"
  }
  ```
- **Response (Out-of-Domain Refusal)**:
  ```json
  {
    "query": "How to bake authentic Neapolitan sourdough pizza?",
    "answer": "I do not have enough confidence in the indexed blog articles to answer this question (top similarity: 0.58 is below required threshold: 0.65).",
    "refused": true,
    "confidence_score": 0.5841,
    "citations": [],
    "sources": [ ... ],
    "latency_ms": 1050.2,
    "model": "llama-3.1-8b-instant",
    "provider": "groq"
  }
  ```

---

## 7. Configuration & Environment Variables

All settings are managed via `pydantic-settings` in [app/config.py](file:///d:/Blogger-Engine-main/app/config.py) and can be overridden via environment variables or a local `.env` file:

| Environment Variable | Canonical Default | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense bi-encoder SentenceTransformer model |
| `FAISS_INDEX_PATH` | `indexes/faiss_chunked_bge_dedup.index` | Path to production FAISS IndexFlatIP index |
| `SQLITE_DB_PATH` | `data/blogger_dedup.db` | Path to canonical SQLite chunks database |
| `BM25_CACHE_PATH` | `indexes/bm25_chunked_dedup.pkl` | Path to pre-tokenized BM25 cache file |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder neural reranker model |
| `TOP_K` | `3` | Default number of final context passages |
| `SIMILARITY_THRESHOLD` | `0.65` | Confidence gate refusal threshold |
| `CANDIDATE_DEPTH` | `20` | Depth for dense and BM25 candidate retrieval |
| `RERANK_DEPTH` | `20` | Depth for cross-encoder reranking |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant ($k$) |
| `DOC_DEDUP` | `True` | Group by document and drop intra-doc duplicates |
| `USE_RERANKER` | `True` | Enable neural cross-encoder reranking stage |
| `LLM_PROVIDER` | `groq` | Supported: `groq`, `gemini`, `openai`, `ollama` |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Model identifier for generation |
| `LLM_API_KEY` | `""` | API key for external LLM provider |
| `LLM_BASE_URL` | `""` | Custom OpenAI-compatible base URL |
| `CORS_ORIGINS` | `""` | Comma-separated allowed origins (defaults to wildcard) |
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `8000` | Server port binding (dynamically overridable) |

---

## 8. Running Locally

### Prerequisites
- Python 3.11+
- Git

### Quickstart
```bash
# 1. Clone the repository
git clone https://github.com/your-username/blogger-engine.git
cd blogger-engine

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate   # Windows

# 3. Install dependencies
# On Linux/macOS, install CPU-only PyTorch first for minimal footprint:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Optional: Configure LLM API credentials
cp .env.example .env
# Edit .env to set GROQ_API_KEY or OPENAI_API_KEY (optional; returns offline notice if omitted)

# 5. Start the production FastAPI server
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```
- Open your browser to `http://localhost:8000` to interact with the luxury editorial UI.
- Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 9. Docker Deployment

### Packaging Strategy
A production multi-stage [Dockerfile](file:///d:/Blogger-Engine-main/Dockerfile) and [.dockerignore](file:///d:/Blogger-Engine-main/.dockerignore) are provided:
- **Multi-Stage Build**: Isolates build tools in a temporary builder stage; final runner image contains only `python:3.11-slim` and `libgomp1`.
- **Pre-Cached Models**: Model weights for BGE-small and Cross-Encoder MiniLM are pre-downloaded during build. The container runs with `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` for deterministic, network-free startup.
- **Non-Root User**: Runs as `appuser:appgroup` (UID 10001).
- **Dynamic Port Support**: Conforms to cloud platform requirements (`Render`, `Railway`, `Fly.io`, `Google Cloud Run`) via `PORT` environment variable.

### Building & Running the Container
```bash
# Build the production image
docker build -t blogger-engine .

# Run the container locally (listen on 8000)
docker run -p 8000:8000 -e LLM_API_KEY="your-api-key" blogger-engine
```

> **Deployment Validation Notice**:
> The `Dockerfile` and `.dockerignore` were developed, linted, and verified via automated static specification tests ([`tests/test_phase5_step7_docker.py`](file:///d:/Blogger-Engine-main/tests/test_phase5_step7_docker.py)). However, because Docker CLI was not installed on the local Windows development machine, local `docker build` and container runtime execution were not performed on this machine. Cloud builds should be executed in standard Linux CI/CD environments with at least **1 GB of RAM**.

---

## 10. Quantitative Retrieval Evaluation

Retrieval performance was evaluated against the frozen 60-query benchmark dataset ([`data/eval_dataset_full.json`](file:///d:/Blogger-Engine-main/data/eval_dataset_full.json), 54 in-domain + 6 out-of-domain probes) at final output depth $k=3$:

| Metric | Baseline (Exp 0: MiniLM) | Canonical Production (Exp 4) | Absolute Delta | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Document Recall@3** | `46.3%` (25/54) | **`66.7%` (36/54)** | **`+20.4 pp`** | **`+44.1%`** |
| **Passage Recall@3** | `27.8%` (15/54) | **`46.3%` (25/54)** | **`+18.5 pp`** | **`+66.5%`** |
| **Citation Source Accuracy** | `77.2%` | **`84.6%`** | **`+7.4 pp`** | **`+9.6%`** |
| **Lexical Groundedness** | `83.0%` | **`82.1%`** | `-0.9 pp` | `-1.1%` |
| **Context Compression** | 90.7% | **90.4%** | `-0.3 pp` | ~1,948 prompt tokens vs ~20,199 in whole-doc |
| **Mean Retrieval Latency (CPU)** | `18.52 ms` | **`1,469.89 ms`** | `+1.45 s` | Joint cross-attention trade-off |
| **Mean Unique Docs in Top-3** | 2.50 docs | **3.00 docs** | **`+0.50 docs`** | **100% inter-document diversity** |

*Note on Latency Accounting*: The canonical evaluation latency (~1.47s) reflects full offline benchmarking on CPU (Dense BGE: 26ms + Sparse BM25: 435ms + Cross-Encoder 20 pairs: 872ms + SQLite dedup: 136ms). This is distinct from local single-query smoke tests (~1.12s).

---

## 11. Confidence Threshold Calibration

### Methodology
To calibrate the refusal gate without corrupting the frozen 60-query benchmark, a separate held-out calibration dataset of 30 queries (20 in-domain + 10 out-of-domain) was created in `data/calibration_dataset.json`. All 20 in-domain calibration queries were sampled from documents strictly disjoint from the 38 target documents of the benchmark.

### Calibration Analysis
- In Experiment 0 (MiniLM), scores were compressed near zero, using a threshold of `0.35`.
- In Experiment 4, BGE cosine embeddings and cross-encoder sigmoid scores shifted systematically higher (~+0.25 to +0.35 upward shift).
- Calibration distribution:
  - **In-Domain Minimum Score**: `0.7423` (Median: `0.8355`, Mean: `0.8356`, Max: `0.9046`)
  - **Out-of-Domain Maximum Score**: `0.6464` (Median: `0.6095`, Mean: `0.6047`, Min: `0.5441`)
  - **Empirical Separation Margin**: `+0.0959` clean gap between highest OOD probe (`0.6464`) and lowest in-domain query (`0.7423`).
- **Selected Threshold**: **`0.65`**
  - Clears the highest OOD probe (`0.6464`) with safety headroom, while preserving a **+0.0923 buffer** below the lowest in-domain score (`0.7423`).
  - Enforces **0 false refusals** across all 54 in-domain benchmark queries (100% valid acceptance).
  - Eliminates ungrounded synthesis on out-of-domain queries (e.g. baking pizza, quantum dilution).

*Honest Limitation*: A score threshold is a similarity-gating heuristic, not a mathematical proof of out-of-distribution status. Queries near the corpus topic boundary can score close to 0.65.

---

## 12. Research Journey & Rejected Approaches

Retrieval optimization progressed through structured, empirical iterations documented in [PROGRESS.md](file:///d:/Blogger-Engine-main/PROGRESS.md):

1. **Experiment 1 (BGE-small Bi-Encoder)**:
   - Upgraded to 512-token sequence capacity.
   - Result: Document recall stayed flat at 46.3% (due to dense semantic crowding), but passage recall improved by +5.5 pp. Proved bi-encoders alone hit a capacity ceiling.
2. **Experiment 2 (Hybrid BM25 + RRF)**:
   - Added BM25 lexical retrieval and Reciprocal Rank Fusion ($k=60$).
   - Result: Document recall jumped from 46.3% to **57.4% (+11.1 pp)**. Rescued 10/10 technical acronyms and entity queries that dense search missed.
3. **Experiment 3 (Retrieval Depth + Document Deduplication)**:
   - Expanded candidate depth to 10 with document deduplication.
   - Result: Document recall rose to **61.1% (+3.7 pp)**; top-3 slots reached 100% distinct documents (3.00 unique docs).
4. **Experiment 4 (Cross-Encoder Neural Reranking)** $\to$ **Canonical**:
   - Added `cross-encoder/ms-marco-MiniLM-L-6-v2` over top-20 fused candidates.
   - Result: Document recall reached **66.7% (+5.6 pp)** and passage recall surged to **46.3% (+14.8 pp)**.

### Summary of Rejected Approaches
- **Experiment 5 (Pseudo-Relevance Feedback Query Expansion)**: Extracted top BM25 terms to augment dense queries. Result: Zero net gain (+0.0 pp doc recall) and introduced query drift on specific entities. Rejected.
- **Deeper Candidate Pools ($N=50$)**: Doubled cross-encoder scoring from 20 to 50 pairs. Result: Added +1.8s of CPU latency for only +1.8 pp recall gain. Rejected as an inefficient latency trade-off.
- **Linear Score Fusion (CombSUM / CombMNZ)**: Attempted min-max normalized weighted addition of BM25 and dense scores. Result: Required query-dependent parameter tuning and proved inferior to calibration-free RRF. Rejected.
- **Larger Same-Family Reranker (ms-marco-MiniLM-L-12-v2)**: Evaluated 12-layer cross-encoder. Result: Doubled inference time to 1.7s on CPU with zero improvement over the 6-layer model. Rejected.

---

## 13. System Trade-Offs & Known Limitations

- **CPU Latency Trade-Off**: Running full cross-attention over 20 passage pairs on CPU takes ~870 ms, bringing total retrieval latency to ~1.47 s. In a GPU environment, this would take ~25 ms; on CPU, this is the cost of zero recurring infrastructure costs.
- **Memory Footprint**: The in-memory FAISS index (124 MB), BM25 cache (185 MB), and model weights (~350 MB) require **~650–700 MB RSS**. A 512 MB micro VPS will experience OOM during startup; a 1 GB RAM instance is required.
- **Artifact Size**: Canonical assets occupy ~650 MB on disk. Standard GitHub cannot store files >100 MB directly without Git LFS; the Dockerfile provides built-in decompression (`blogger_dedup.db.gz`) and BM25 build-time generation to accommodate repository limits.
- **External LLM Requirement**: In live QA mode, answer generation depends on an external LLM API (e.g. Groq). When offline or unset, the system provides transparent retrieval and citations with a mock notice.
- **Recall Ceiling**: Benchmark Document Recall@3 is 66.7%. Diagnostic analysis revealed that 13.0% of target documents were absent from the top-20 candidate pool entirely, establishing an 83.3% theoretical ceiling for Stage 2 reranking.

---

## 14. Project Directory Structure

```
Blogger-Engine-main/
├── app/
│   ├── api/
│   │   └── routes.py             # FastAPI route handlers (/health, /api/stats, /api/search, /api/ask)
│   ├── core/
│   │   ├── chunker.py            # Sliding-window document chunker & text cleaner
│   │   ├── embedder.py           # BGE-small QueryEmbedder with instruction prefix
│   │   ├── generator.py          # LLM answer generator, citation extractor, refusal gate
│   │   ├── hybrid_retriever.py   # Canonical Exp 4 engine (Dense + BM25 + RRF + Cross-Encoder + Dedup)
│   │   └── retriever.py          # Base FAISS dense retriever wrapper
│   ├── db/
│   │   └── database.py           # SQLite database interface & query logging
│   ├── static/
│   │   ├── index.html            # Single-page editorial user interface
│   │   └── style.css             # Vanilla CSS luxury editorial design system
│   ├── config.py                 # Centralized Pydantic settings & environment configuration
│   ├── main.py                   # FastAPI lifespan management & application factory
│   └── schemas.py                # Pydantic v2 request/response schemas
├── data/
│   ├── blogger_dedup.db          # Canonical SQLite database (81,123 chunks, 8,242 articles)
│   ├── blogger_dedup.db.gz       # Compressed database archive (69 MB)
│   ├── eval_dataset_full.json    # Frozen 60-query benchmark dataset
│   └── calibration_dataset.json      # Held-out 30-query threshold calibration set
├── indexes/
│   ├── faiss_chunked_bge_dedup.index # Canonical FAISS IndexFlatIP (81,123 vectors, 384d)
│   └── bm25_chunked_dedup.pkl    # Serialized BM25Okapi cache (81,123 chunks)
├── scripts/                      # Offline ingestion, evaluation, and research analysis scripts
├── tests/
│   ├── test_phase1.py            # Chunker and SQLite unit tests (6 tests)
│   ├── test_phase2.py            # Embedder and generator unit tests (14 tests)
│   ├── test_phase3.py            # API route integration tests (8 tests)
│   ├── test_phase4.py            # Evaluation harness & metric tests (6 tests)
│   ├── test_hybrid_retriever.py  # HybridRetriever unit & concurrency tests (21 tests)
│   ├── test_phase5_step4.py      # Lifespan wiring & 503 error tests (8 tests)
│   ├── test_phase5_step5.py      # Frontend contract & notice tests (5 tests)
│   ├── test_phase5_step6_e2e.py  # Production end-to-end integration tests (10 tests)
│   └── test_phase5_step7_docker.py # Dockerfile & deployment specification tests (6 tests)
├── Dockerfile                    # Production multi-stage CPU-only container definition
├── .dockerignore                 # Production build exclusion rules
├── requirements.txt              # Production Python dependencies
├── PROGRESS.md                   # Authoritative, chronological engineering and research log
└── PROJECT_HANDOFF.md            # Comprehensive project handoff and architecture guide
```

---

## 15. Engineering & Interview Highlights

- **Deterministic ID Decoupling**: Vector positions in FAISS (`faiss_id`) are decoupled from SQLite internal `rowid`s and explicitly preserved across ingestion, indexing, and reranking.
- **Calibration-Free Rank Fusion**: Using Reciprocal Rank Fusion ($k=60$) avoids the need for empirical weight tuning between dense cosine scores and unbounded BM25 scores.
- **Concurrency-Safe Latency Metadata**: Request-specific latency breakdowns are stored in a custom `RetrievalResultList` and backed by Python's `contextvars.ContextVar`, eliminating race conditions under concurrent async traffic.
- **Application-Scoped Lifecycle**: Heavy vector indexes and neural cross-encoders are loaded once during FastAPI lifespan startup with explicit thread-safe warmup, eliminating first-request latency spikes.
- **Strict Degradation Handling**: Missing indexes or uninitialized services raise explicit HTTP 503 errors rather than falling back to unrepresentative toy sample indexes.
- **100% Test Coverage**: **84 passing tests** across 9 test suites covering chunking, database CRUD, vector search, hybrid fusion, reranking, API contracts, concurrency, and Docker configuration.

---

## 16. License

This project is licensed under the MIT License.
