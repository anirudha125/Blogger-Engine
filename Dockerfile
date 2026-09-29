# ==============================================================================
# Blogger Engine - Production Dockerfile
# Canonical Architecture: BGE-small + BM25 + RRF + Cross-Encoder Reranker
# CPU-Only Production Container (Multi-Stage Build)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Environment & Artifact Preparation
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Set build-time environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/build/.cache/huggingface

# Install essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency specifications
COPY requirements.txt .

# Install CPU-only PyTorch and production dependencies
# Installing PyTorch from the official CPU index prevents pulling ~2GB of unused CUDA packages
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Pre-download and cache canonical model weights into image layer
# This guarantees deterministic, network-free runtime startup
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Pre-downloading BAAI/bge-small-en-v1.5...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Pre-downloading cross-encoder/ms-marco-MiniLM-L-6-v2...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Canonical model weights cached successfully.')"

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

# Production environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Install runtime OpenMP support required by FAISS CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for security compliance
RUN groupadd -r -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

# Copy virtualenv and cached model weights from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appgroup /build/.cache/huggingface /app/.cache/huggingface

# Create required directory hierarchy
RUN mkdir -p /app/data /app/indexes /app/app

# Copy application source code
COPY --chown=appuser:appgroup app/ /app/app/

# Copy canonical FAISS index
COPY --chown=appuser:appgroup indexes/faiss_chunked_bge_dedup.index /app/indexes/faiss_chunked_bge_dedup.index

# Copy database artifact: prefer decompressed blogger_dedup.db if present, else decompress .gz
COPY --chown=appuser:appgroup data/blogger_dedup.db* /app/data/
RUN if [ ! -f /app/data/blogger_dedup.db ] && [ -f /app/data/blogger_dedup.db.gz ]; then \
        echo "Decompressing data/blogger_dedup.db.gz during build..." && \
        python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('/app/data/blogger_dedup.db.gz', 'rb'), open('/app/data/blogger_dedup.db', 'wb'))" && \
        rm -f /app/data/blogger_dedup.db.gz; \
    fi && \
    rm -f /app/data/blogger_dedup.db.gz

# Copy BM25 cache artifact: copy if present, or regenerate in ~8.5s during build
COPY --chown=appuser:appgroup indexes/bm25_chunked_dedup.pkl* /app/indexes/
RUN if [ ! -f /app/indexes/bm25_chunked_dedup.pkl ]; then \
        echo "Regenerating canonical BM25 index from database during build..." && \
        python -c "\
import pickle; \
from app.db.database import get_db_connection; \
from app.core.hybrid_retriever import tokenize_text, BM25Okapi; \
conn = get_db_connection('/app/data/blogger_dedup.db'); \
cursor = conn.cursor(); \
cursor.execute('SELECT faiss_id, content FROM documents ORDER BY faiss_id ASC'); \
rows = cursor.fetchall(); \
conn.close(); \
faiss_ids = [r['faiss_id'] for r in rows]; \
corpus = [tokenize_text(r['content']) for r in rows]; \
bm25 = BM25Okapi(corpus); \
with open('/app/indexes/bm25_chunked_dedup.pkl', 'wb') as f: \
    pickle.dump({'bm25': bm25, 'faiss_ids': faiss_ids}, f, protocol=4); \
print(f'BM25 index generated with {len(corpus)} documents.')"; \
    fi

# Ensure all files in /app are owned by appuser
RUN chown -R appuser:appgroup /app

# Switch to non-root runtime user
USER appuser

# Expose default HTTP port
EXPOSE 8000

# Container healthcheck using FastAPI liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:' + str('${PORT:-8000}') + '/health').getcode() == 200 else 1)"

# Start production FastAPI application through factory entrypoint
CMD ["sh", "-c", "exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
