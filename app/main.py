import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router as api_router
from app.core.hybrid_retriever import HybridRetriever
from app.core.generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("blogger_engine.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads models, indexes, and warms cross-encoder during startup."""
    settings = get_settings()
    logger.info("Initializing Blogger Engine backend...")

    # 1. Initialize canonical Experiment 4 HybridRetriever
    try:
        app.state.retriever = HybridRetriever(
            index_path=settings.FAISS_INDEX_PATH,
            db_path=settings.SQLITE_DB_PATH,
            bm25_cache_path=settings.BM25_CACHE_PATH,
            candidate_depth=settings.CANDIDATE_DEPTH,
            rrf_k=settings.RRF_K,
            doc_dedup=settings.DOC_DEDUP,
            fused_depth=settings.CANDIDATE_DEPTH,
            use_reranker=settings.USE_RERANKER,
            reranker_model=settings.RERANKER_MODEL,
            rerank_depth=settings.RERANK_DEPTH,
            default_min_score=None,
        )
        logger.info(f"Loaded canonical HybridRetriever with {app.state.retriever.total_vectors} vectors.")

        # 2. Warm up the cross-encoder to eliminate first-request cold-start latency
        app.state.retriever.warmup()
        logger.info("HybridRetriever cross-encoder warmed up successfully.")
    except Exception as exc:
        logger.error(f"HybridRetriever initialization failure: {exc}")
        app.state.retriever = None

    # 3. Initialize application-scoped AnswerGenerator
    try:
        app.state.generator = AnswerGenerator(
            provider=settings.LLM_PROVIDER,
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            similarity_threshold=settings.SIMILARITY_THRESHOLD
        )
        logger.info(f"Loaded generator: {app.state.generator}")
    except Exception as exc:
        logger.error(f"Generator initialization failure: {exc}")
        app.state.generator = None

    yield
    logger.info("Shutting down Blogger Engine backend...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Blogger Engine - RAG Search & QA",
        description="Clean, deployable RAG platform with semantic search, grounded QA, and source citations.",
        version=settings.APP_VERSION,
        lifespan=lifespan
    )

    # Enable CORS for local development and browser UI
    # In accordance with the Fetch specification, wildcard '*' cannot be used with allow_credentials=True.
    cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
    if cors_origins_env:
        origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        allow_creds = True
    else:
        origins = ["*"]
        allow_creds = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router)

    # Serve static assets
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Serve single-page dashboard at root
    @app.get("/", include_in_schema=False)
    def index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {
            "message": "Blogger Engine API is running.",
            "docs": "/docs",
            "health": "/health"
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run("app.main:app", host=cfg.HOST, port=cfg.PORT, reload=True)
