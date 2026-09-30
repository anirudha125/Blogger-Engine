# Blogger Engine: Production RAG Search & QA Platform

> A production-structured Retrieval-Augmented Generation (RAG) platform delivering grounded semantic search and verifiable citation-backed question answering over 8,242 technical publications (81,123 passage chunks).

---

## 1. Project Overview

### What Blogger Engine Does
**Blogger Engine** is an end-to-end RAG system designed for technical document search and grounded question answering. Given user queries over a corpus of engineering and machine learning publications, it retrieves the most relevant passages through a multi-stage hybrid pipeline, filters unconfident or out-of-domain queries, and synthesizes answers backed by explicit, verifiable in-text citations.

### The Problem It Solves
Standard naive RAG pipelines suffer from compounding failure modes on technical corpora:
- **Silent Truncation**: Standard bi-encoders truncate long documents past 256 or 512 tokens. Embedding whole documents without sliding-window chunking collapses retrieval recall down to single digits.
- **Topical Redundancy & Clustering**: In dense technical domains (e.g., RLHF, DPO, PPO), multiple chunks from the same document often crowd out competing relevant documents in the top-$k$ window.
- **Entity Blindness**: Dense embeddings frequently miss exact identifiers, library names, mathematical acronyms, and author citations (*DP-SGD*, *FlashAttention*, *LoRA*), which lexical search captures reliably.
- **Ungrounded Generation & Hallucination**: Passing unconstrained context or failing to detect queries that the corpus cannot answer causes models to hallucinate plausible but ungrounded answers.

### End-to-End System Summary
Blogger Engine ingests 8,242 deduplicated articles into 81,123 sliding-window chunks, combines BGE dense semantic search with BM25 lexical retrieval via Reciprocal Rank Fusion ($k=60$), refines candidate ordering using a neural cross-encoder, deduplicates results by document ID, guards synthesis with an empirical BGE confidence gate (threshold $0.65$), and generates cited answers using an OpenAI-compatible LLM endpoint with post-generation refusal classification.

---

## 2. Key Features

- **Hybrid Dense + Sparse Retrieval**: Combines dense semantic representations with BM25 keyword matching to handle both conceptual queries and exact entity names.
- **BGE Embeddings**: Uses `BAAI/bge-small-en-v1.5` (384-dimensional normalized vectors) with search-optimized query instruction prefixes.
- **BM25 Lexical Retrieval**: Employs `rank-bm25` (BM25Okapi) over tokenized text for precise term matching.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings ($k=60$) without requiring brittle cross-model score normalization.
- **Cross-Encoder Neural Reranking**: Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` joint cross-attention over candidate passage pairs.
- **Document-Level Deduplication**: Enforces inter-document diversity by keeping only the single highest-scoring chunk per document in the final top-3 context window.
- **Empirical Retrieval-Confidence Gate**: Uses an empirical threshold of $0.65$ on the maximum BGE cosine similarity among top-3 retrieved passages to reject low-confidence queries before generation.
- **Grounded LLM Synthesis**: Formats prompt contexts with structured `[Doc X]` tags and strictly instructs the LLM to synthesize only from provided passages.
- **Citation Extraction**: Parses bracketed citations from generated text, mapping them back to database metadata (title, URL, chunk ID, author).
- **Post-Generation Refusal Handling**: Detects canonical and variant insufficient-evidence statements from the LLM, setting `refused: true` and suppressing empty or hallucinated citations.
- **FastAPI Backend**: Fully async-ready REST API with strict Pydantic schemas, application lifecycle management, and thread-safe warmup.
- **SQLite + FAISS Persistence**: Stores full chunk metadata in SQLite and dense vectors in a FAISS `IndexFlatIP` index with deterministic ID mapping.
- **Dockerized Deployment**: Multi-stage, CPU-optimized container running non-root on Oracle Cloud Infrastructure (ARM64 Ampere A1).

---

## 3. Architecture

```mermaid
flowchart TD
    User([User Query]) --> API[FastAPI /api/ask]

    subgraph Stage1["Stage 1: Multi-Index Candidate Retrieval"]
        API --> Dense[BGE Bi-Encoder + FAISS<br/>Top-20 Dense Candidates]
        API --> Sparse[BM25Okapi Lexical Search<br/>Top-20 Sparse Candidates]
    end

    Dense --> RRF[Reciprocal Rank Fusion k=60<br/>Pool: Top-20 Fused Candidates]
    Sparse --> RRF

    subgraph Stage2["Stage 2: Neural Reranking & Diversity"]
        RRF --> CE[Cross-Encoder ms-marco-MiniLM-L-6-v2<br/>Joint Cross-Attention Scoring]
        CE --> Dedup[Document Deduplication<br/>1 Best Chunk per Unique doc_id]
        Dedup --> Top3[Final Top-3 Passages]
    end

    subgraph Stage3["Stage 3: Gating & Grounded Synthesis"]
        Top3 --> Gate{Max BGE Cosine Sim >= 0.65?}
        Gate -- No --> RefuseGate[Return 200 OK<br/>refused: true<br/>citations: empty]
        Gate -- Yes --> Prompt[Assemble Grounded Context<br/>[Doc 1], [Doc 2], [Doc 3]]
        Prompt --> LLM[LLM Generation<br/>Groq API]
        LLM --> PostGen{Detect Refusal Phrasing?}
        PostGen -- Yes --> RefusePost[Return 200 OK<br/>refused: true<br/>citations: empty]
        PostGen -- No --> Success[Return 200 OK<br/>refused: false<br/>citations: populated]
    end
```

### Request Flow
1. **User Query**: Incoming HTTP `POST /api/ask` request with query string and optional parameter overrides.
2. **Dense Search**: Query is prefixed (`Represent this sentence for searching relevant passages: `) and embedded via `bge-small-en-v1.5`, searching 81,123 vectors in FAISS for top-20 candidates.
3. **Sparse Search**: Tokenized query runs against pre-cached `BM25Okapi` index over 81,123 passage documents for top-20 candidates.
4. **Reciprocal Rank Fusion**: Ranks are fused via $RRF(d) = \sum \frac{1}{60 + \text{rank}(d)}$, creating a unified top-20 candidate pool.
5. **Cross-Encoder Reranking**: All 20 `(query, passage)` pairs are scored jointly via full cross-attention with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
6. **Document Deduplication**: Candidates are grouped by `doc_id`, keeping only the highest-scoring chunk per document to guarantee distinct sources in the context.
7. **Top-3 Selection & Score Mapping**: Top-3 unique documents are selected, preserving exact BGE cosine similarity scores for confidence gating.
8. **BGE Confidence Gate**: Evaluates maximum BGE cosine similarity against the empirical threshold ($0.65$). If $< 0.65$, refuses immediately without calling the LLM.
9. **Grounded Prompt Assembly**: Context chunks are formatted with `[Doc 1]`, `[Doc 2]`, `[Doc 3]` source tags and sent with strict anti-hallucination instructions.
10. **LLM Generation & Refusal Detection**: LLM generates an answer. If generation indicates insufficient context, the system flags `refused: true` and clears citations; otherwise, valid citations are extracted and returned.

---

## 4. Retrieval Design

Blogger Engine implements the finalized, empirically validated **Canonical Experiment 4** configuration:

| Component | Specification | Operational Role |
| :--- | :--- | :--- |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` | 384-dimensional dense semantic vectors with query instruction prefix |
| **Vector Normalization** | L2 Unit Normalization | Allows inner product search in FAISS to compute exact cosine similarity |
| **Vector Index** | FAISS `IndexFlatIP` | Exact brute-force vector search (~26 ms for 81,123 vectors on CPU) |
| **Sparse Index** | `rank-bm25` (BM25Okapi) | Tokenized inverted index for technical terms, entities, and acronyms |
| **Rank Fusion** | Reciprocal Rank Fusion ($k=60$) | Calibration-free rank merging across dense and sparse candidate sets |
| **Candidate Pool Depth** | $N = 20$ | Candidate depth retrieved from dense and BM25 search |
| **Neural Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint cross-attention scoring across the top-20 fused candidate pool |
| **Rerank Depth** | 20 pairs | Bounded cross-encoder scoring depth (~870 ms CPU latency budget) |
| **Deduplication** | Document-Level (`doc_id`) | Retains single highest-scoring chunk per unique article |
| **Output Depth** | $\text{top\_k} = 3$ | Context window provided to the LLM (3 distinct documents) |
| **Confidence Metric** | Max BGE Cosine Similarity | Maximum cosine similarity among the final top-3 retrieved documents |
| **Confidence Threshold** | $0.65$ | Empirical threshold on maximum BGE cosine similarity for gating |

---

## 5. Evaluation

The retrieval pipeline was quantitatively evaluated against a frozen 60-query benchmark dataset (`data/eval_dataset_full.json`), comprising 54 in-domain technical machine learning queries and 6 out-of-domain probes, evaluated at output depth $k=3$:

| Metric | Whole-Doc Baseline (Exp 0) | Canonical Production (Exp 4) | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Document Recall@3** | 46.3% (25/54) | **66.7% (36/54)** | **+20.4 pp** | **+44.1%** |
| **Passage Recall@3** | 27.8% (15/54) | **46.3% (25/54)** | **+18.5 pp** | **+66.5%** |
| **Citation Source Accuracy** | 77.2% | **84.6%** | **+7.4 pp** | **+9.6%** |
| **Lexical Groundedness** | 83.0% | **82.1%** | -0.9 pp | -1.1% |
| **Mean Unique Docs in Top-3** | 2.50 docs | **3.00 docs** | **+0.50 docs** | **100% Inter-Doc Diversity** |
| **Context Compression** | 90.7% | **90.4%** | -0.3 pp | ~1,948 prompt tokens vs ~20,199 baseline |
| **Mean Retrieval Latency (CPU)** | 18.52 ms | **1,469.89 ms** | +1.45 s | Trade-off for cross-attention accuracy |

### Metric Definitions
- **Document Recall@3**: Proportion of in-domain queries where the true source document appears in the final top-3 results.
- **Passage Recall@3**: Proportion of in-domain queries where the specific answering chunk appears in the top-3 results.
- **Citation Source Accuracy**: Proportion of generated citations that match the ground-truth target documents.
- **Lexical Groundedness**: Token-level lexical overlap between generated responses and retrieved context passages.
- **Context Compression**: Token reduction achieved by sliding-window chunking relative to unchunked full-document feeding.

---

## 6. Threshold Calibration

To establish a defensible gating threshold without overfitting the benchmark, a held-out calibration set of **30 queries** (`data/calibration_dataset.json`) was evaluated:
- **20 In-Domain Technical Queries**: Sampled from documents strictly disjoint from benchmark target documents.
- **10 Out-of-Domain (OOD) Probes**: Topics absent from the corpus (e.g., culinary recipes, gardening, quantum mechanics).

### Score Distributions (Maximum BGE Cosine Similarity)

| Partition | Min Score | Mean Score | Median Score | Max Score |
| :--- | :---: | :---: | :---: | :---: |
| **In-Domain (20 queries)** | **0.7423** | 0.8356 | 0.8355 | 0.9046 |
| **Out-of-Domain (10 queries)** | 0.5441 | 0.6047 | 0.6095 | **0.6464** |

### Calibration Conclusion & Selection
- **Observed Separation Margin**: A **+0.0959 gap** exists between the highest OOD probe (`0.6464`) and the lowest in-domain query (`0.7423`).
- **Selected Threshold**: **`0.65`**
  - Positioned above the highest OOD probe (`0.6464`) with safety headroom.
  - Leaves a **+0.0923 buffer** below the lowest in-domain score (`0.7423`), ensuring **0 false refusals** across all 54 in-domain benchmark queries.
- **Defensible Scope**: The 0.65 threshold operates as an **empirical retrieval-confidence gate**, *not* a universal out-of-distribution classifier. Queries that share superficial semantic vocabulary with the corpus can score above 0.65 despite lacking answering facts; such cases are handled by post-generation refusal detection.

---

## 7. LLM Grounding & Refusal Semantics

The `/api/ask` endpoint provides a unified, defensible response contract:

> **`refused: bool` indicates whether the system was unable or declined to provide a factual, grounded answer based on the indexed corpus.**

### Refusal Pathways
```
Case 1: Retrieval Confidence Gate Rejection
  Trigger: Top BGE cosine similarity < 0.65 (or 0 passages found)
  Behavior: Immediate refusal before generation. Zero LLM calls.
  Output: refused: true, citations: []
  Message: "I do not have enough confidence in the indexed blog articles to answer this question..."

Case 2: Post-Generation Evidence Insufficiency
  Trigger: Retrieval passes >= 0.65, but retrieved passages lack facts to answer.
  Behavior: LLM follows prompt Rule 3, outputting standard insufficiency phrasing.
  Detection: AnswerGenerator regex pattern matches canonical & variant refusal phrases.
  Output: refused: true, citations: [] (suppresses hallucinated/extraneous citations)

Case 3: Grounded Answer Generation
  Trigger: Retrieval passes >= 0.65 and context contains answering facts.
  Behavior: LLM generates affirmative answer with [Doc X] bracketed citations.
  Output: refused: false, citations: [CitationItem, ...]
```

---

## 8. Engineering Decisions

| Architectural Choice | Decision | Technical Rationale |
| :--- | :--- | :--- |
| **Retrieval Strategy** | Hybrid (BGE + BM25) | Dense bi-encoders alone miss exact entities, acronyms, and technical citations (*LoRA*, *DP-SGD*). BM25 handles lexical precision while BGE handles semantic concepts. |
| **Fusion Algorithm** | Reciprocal Rank Fusion (RRF) | Avoids fragile min-max score normalization between bounded cosine similarities $[-1, 1]$ and unbounded BM25 scores $[0, \infty)$. Parameter-free ($k=60$). |
| **Reranking Method** | Neural Cross-Encoder | Bi-encoders compress documents into independent vectors; cross-encoders process full query-document token interactions via cross-attention, boosting passage recall from 27.8% to 46.3%. |
| **Deduplication** | Document-Level (`doc_id`) | Without deduplication, top-3 slots frequently contain 2–3 chunks from the same long article. Deduplication guarantees 3 distinct sources in the context window. |
| **Vector Engine** | FAISS `IndexFlatIP` | At 81,123 vectors of 384 dimensions, brute-force inner product requires only ~31M FLOPs, executing in ~26 ms on CPU with OpenMP. Eliminates IVF/HNSW approximation recall loss. |
| **Relational Storage** | SQLite (`blogger_dedup.db`) | Single-file, zero-maintenance relational storage with integer primary keys and indexing. Ideal for local and containerized read-heavy workloads. |
| **Web Framework** | FastAPI | High-performance async ASGI framework with automatic OpenAPI documentation, dependency injection, and Pydantic v2 validation. |
| **Containerization** | Multi-Stage Docker | Pre-caches model weights to run offline (`TRANSFORMERS_OFFLINE=1`), builds with CPU-only wheels, and enforces non-root execution (`appuser:10001`). |

---

## 9. Deployment Architecture

Blogger Engine is deployed and validated on cloud infrastructure:

- **Hosting Platform**: Oracle Cloud Infrastructure (OCI) Virtual Machine
- **Operating System**: Ubuntu 24.04 LTS
- **Architecture**: ARM64 / aarch64 (Ampere A1 Compute, 4 OCPUs, 24 GB RAM)
- **Container Runtime**: Docker Engine running `blogger-engine:arm64`
- **Application Process**: Uvicorn running FastAPI behind a non-root container profile
- **LLM Integration**: OpenAI-compatible chat completion provider (`groq`) using `openai/gpt-oss-20b`
- **Networking**: Container port 8000 exposed to host loopback and public HTTP endpoints

---

## 10. API Usage

### 1. Grounded Question Answering (`POST /api/ask`)

#### Request: In-Domain Query
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Docker and how do containers work?"}'
```

#### Response: Successful Grounded Answer (`refused: false`)
```json
{
  "query": "What is Docker and how do containers work?",
  "answer": "Docker is a container-runtime tool that lets you package an application together with all of its libraries, system tools, and runtime into a single, portable unit called an image... Containers share the host operating-system kernel [Doc 1][Doc 2]...",
  "refused": false,
  "confidence_score": 0.8265,
  "citations": [
    {
      "doc_tag": "[Doc 1]",
      "chunk_id": "a9f3b1c2d0e4_c1",
      "title": "Lecture 11: Deployment & Monitoring - The Full Stack",
      "url": "https://example.com/deployment-monitoring",
      "score": 0.8265
    },
    {
      "doc_tag": "[Doc 2]",
      "chunk_id": "8b2e1f4a9c3d_c0",
      "title": "What is Container Orchestration? Explained",
      "url": "https://example.com/container-orchestration",
      "score": 0.7363
    }
  ],
  "sources": [ ... ],
  "latency_ms": 6084.94,
  "model": "openai/gpt-oss-20b",
  "provider": "groq"
}
```

#### Request: Out-of-Domain or Insufficient-Evidence Query
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I repair a bicycle tire?"}'
```

#### Response: Refusal (`refused: true`)
```json
{
  "query": "How do I repair a bicycle tire?",
  "answer": "I do not have enough information in the provided blog articles to answer this question.",
  "refused": true,
  "confidence_score": 0.6602,
  "citations": [],
  "sources": [ ... ],
  "latency_ms": 5786.93,
  "model": "openai/gpt-oss-20b",
  "provider": "groq"
}
```

### 2. Semantic Search (`POST /api/search`)
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Direct Preference Optimization DPO", "top_k": 3}'
```

### 3. Service Health & Metadata (`GET /health`, `GET /api/stats`)
```bash
curl http://localhost:8000/health
# {"status":"ok","index_loaded":true,"db_connected":true,"version":"1.0.0"}

curl http://localhost:8000/api/stats
# {"total_documents":8242,"total_chunks":81123,"total_vectors":81123,...}
```

---

## 11. Project Structure

```
Blogger-Engine/
├── app/
│   ├── api/
│   │   └── routes.py              # REST routes: /health, /api/stats, /api/search, /api/ask
│   ├── core/
│   │   ├── chunker.py             # 400-word sliding-window chunker with 50-word overlap
│   │   ├── embedder.py            # BGE-small bi-encoder with query task instruction prefix
│   │   ├── generator.py           # LLM generation, citation extraction & refusal classification
│   │   ├── hybrid_retriever.py    # Canonical Exp 4 engine (Dense + BM25 + RRF + Cross-Encoder + Dedup)
│   │   └── retriever.py           # Base FAISS IndexFlatIP dense retriever wrapper
│   ├── db/
│   │   └── database.py            # SQLite connection pool, query logging, chunk retrieval
│   ├── static/
│   │   ├── index.html             # Single-page editorial client interface
│   │   └── style.css              # Custom responsive typography and layout design system
│   ├── config.py                  # Pydantic Settings with whitespace sanitization
│   ├── main.py                    # Lifespan startup, model warmup & application factory
│   └── schemas.py                 # Pydantic v2 request/response schemas with contract docs
├── data/
│   ├── blogger_dedup.db           # Canonical SQLite database (8,242 articles, 81,123 chunks)
│   ├── eval_dataset_full.json     # Frozen 60-query benchmark dataset
│   └── calibration_dataset.json   # 30-query held-out threshold calibration dataset
├── indexes/
│   ├── faiss_chunked_bge_dedup.index # FAISS IndexFlatIP (81,123 vectors, 384d)
│   └── bm25_chunked_dedup.pkl     # Pre-tokenized BM25Okapi cache
├── scripts/                       # Ingestion, benchmarking & calibration analysis scripts
├── tests/
│   ├── test_phase1.py             # Chunker & SQLite CRUD tests (6 tests)
│   ├── test_phase2.py             # Embedder & generator unit tests (14 tests)
│   ├── test_phase3.py             # API route integration tests (8 tests)
│   ├── test_phase4.py             # Evaluation harness tests (6 tests)
│   ├── test_hybrid_retriever.py   # HybridRetriever & concurrency isolation tests (21 tests)
│   ├── test_phase5_step4.py       # Lifespan wiring & degradation tests (8 tests)
│   ├── test_phase5_step5.py       # Frontend contract tests (5 tests)
│   ├── test_phase5_step6_e2e.py   # End-to-end integration tests (10 tests)
│   ├── test_phase5_step7_docker.py# Docker build & packaging specification tests (6 tests)
│   └── test_phase6_llm_hardening.py# LLM config sanitization & refusal regression tests (10 tests)
├── Dockerfile                     # Multi-stage production container specification
├── .dockerignore                  # Strict packaging exclusion rules
├── requirements.txt               # Locked production dependencies
├── PROGRESS.md                    # Detailed engineering and evaluation log
└── PROJECT_HANDOFF.md             # Production handoff and deployment guide
```

*Test Suite Status*: **94 passed tests**, 0 failed across all suites (`pytest -v`).

---

## 12. Limitations

1. **Similarity Threshold is Not a Classifier**: The 0.65 BGE cosine similarity gate is an empirical confidence heuristic. An out-of-domain query that contains engineering buzzwords may produce a score $\ge 0.65$; the system relies on post-generation refusal classification as a secondary defense.
2. **CPU Retrieval Latency**: Computing full cross-attention over 20 candidate pairs using CPU takes ~870 ms, leading to a mean retrieval latency of ~1.47 s. In a GPU environment, this would run in <30 ms.
3. **Candidate-Window Limitation**: 18 of 54 in-domain benchmark queries were missed at the final top-3 stage; forensic analysis found 11 reranker misranks, 6 first-stage retrieval misses, and 1 target outside the cross-encoder's top-20 reranking window.
4. **External Artifact Distribution**: Large binary artifacts (`faiss_chunked_bge_dedup.index` at 119 MB, `blogger_dedup.db` at 314 MB, and `bm25_chunked_dedup.pkl` at 177 MB) exceed standard Git limits and are managed via release artifacts or container builds rather than direct Git tracking.

---

## 13. Research & Evolution Summary

The retrieval pipeline evolved across structured experimental phases:

```
[Baseline: Whole-Doc MiniLM]
  - Doc Recall@3: 46.3% | Passage Recall@3: 27.8%
  - Silent truncation beyond 256 tokens; entity blindness on technical terms.
         │
         ▼
[Exp 1: BGE-small Dense Bi-Encoder]
  - 512-token context capacity, 384 dimensions.
  - Doc Recall@3 remained flat at 46.3% due to dense topical crowding.
         │
         ▼
[Exp 2: Hybrid BGE + BM25Okapi + RRF (k=60)]
  - Doc Recall@3 jumped to 57.4% (+11.1 pp).
  - BM25 rescued exact acronyms and author names missed by dense embeddings.
         │
         ▼
[Exp 3: Document-Level Deduplication]
  - Doc Recall@3 reached 61.1% (+3.7 pp).
  - Enforced 100% inter-document diversity (3.00 unique articles in top-3).
         │
         ▼
[Exp 4: Cross-Encoder Neural Reranking (Canonical Production)]
  - ms-marco-MiniLM-L-6-v2 joint cross-attention over top-20 fused candidates.
  - Final Doc Recall@3: 66.7% (+20.4 pp over baseline).
  - Final Passage Recall@3: 46.3% (+18.5 pp over baseline).
```

### Discarded Approaches
- **Pseudo-Relevance Feedback (Exp 5)**: Augmenting queries with top BM25 keywords yielded 0.0 pp recall gain and caused semantic drift.
- **Deeper Candidate Pools ($N=50$)**: Scored 50 cross-encoder pairs. Increased latency by +1.8 s for a marginal +1.8 pp recall gain.
- **Linear Score Normalization**: Min-max addition of BM25 and dense scores proved unstable across varying query lengths compared to rank-based RRF.
- **12-Layer Cross-Encoder (L-12)**: Doubled reranker latency to 1.7 s on CPU without measurable recall improvement over the 6-layer model.

---

## 14. Quick Start

### Prerequisites
- Python 3.11+
- Git
- 1+ GB RAM available

### Local Installation
```bash
# 1. Clone repository
git clone https://github.com/anirudha125/Blogger-Engine.git
cd Blogger-Engine

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install CPU-only PyTorch and dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Canonical Artifacts Setup
# Ensure data/blogger_dedup.db and indexes/ exist.
# If data/blogger_dedup.db.gz is present:
# python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('data/blogger_dedup.db.gz', 'rb'), open('data/blogger_dedup.db', 'wb'))"

# 5. Configure environment variables (optional for QA mode)
cp .env.example .env
# Edit .env to supply LLM_API_KEY if testing live generation

# 6. Run the test suite
pytest -v

# 7. Start the FastAPI development server
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` to interact with the search and QA frontend, or `http://localhost:8000/docs` for interactive Swagger documentation.

---

## 15. Tech Stack

- **Retrieval & Reranking**: `sentence-transformers` (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`), `faiss-cpu`, `rank-bm25`
- **Application Backend**: `FastAPI`, `uvicorn`, `pydantic v2`, `pydantic-settings`, `httpx`
- **Storage & Indexing**: `SQLite3`, `FAISS IndexFlatIP`, `pickle` (protocol 4)
- **Deployment & Infrastructure**: `Docker` (multi-stage), `Oracle Cloud Infrastructure` (ARM64 Ampere A1), `Linux Ubuntu 24.04`
- **Testing & Quality Assurance**: `pytest`, `pytest-asyncio`, Starlette TestClient

---

## License

This project is licensed under the MIT License.
