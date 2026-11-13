"""
api/main.py — FastAPI application entry point.

CHANGE from original: load_secrets() called at the very top, before
all local imports, so every os.getenv() in settings.py sees GCP values.

Start:  uvicorn api.main:app --reload --port 8000
Docs:   http://localhost:8000/docs
Health: http://localhost:8000/health
"""

# ── Step 1: Secrets first ─────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(override=False)

from app.config.secrets import load_secrets
load_secrets()

# ── Standard imports (unchanged) ──────────────────────────────────────────────
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.logging.logger import setup_logging
from app.config.settings import (
    CHROMA_PERSIST, CORS_ORIGINS as CFG_CORS,
    API_HOST, API_PORT,
    validate_config,
)
setup_logging()
logger = logging.getLogger("financial_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("RAG API starting up...")
    for warning in validate_config():
        logger.warning("CONFIG: %s", warning)

    import threading
    def _prewarm():
        try:
            from app.retrieval.reranker import _get_model
            _get_model()
            logger.info("Reranker pre-warmed ✓ (first query will be fast)")
        except Exception as e:
            logger.warning("Reranker pre-warm skipped: %s", e)

    threading.Thread(target=_prewarm, daemon=True, name="prewarm").start()

   # Connect audit DB (non-blocking — app works without it)
    from api.middleware.logging import ensure_pool
    await ensure_pool()

    # Create conversation tables in the same DB (Step 4 Part B)
    import api.middleware.logging as _logging_module
    if _logging_module._db_pool:
        from app.services.conversation_service import ensure_conversation_tables
        await ensure_conversation_tables(_logging_module._db_pool)

        from app.services.usage_service import ensure_usage_tables
        await ensure_usage_tables(_logging_module._db_pool)


    logger.info("API ready → http://localhost:8000/docs")
    logger.info("=" * 60)
    yield
    logger.info("RAG API shutting down...")


# ── App (rest is identical to original) ──────────────────────────────────────
app = FastAPI(
    title="RAG Platform API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG_CORS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routers import query, index, auth, admin, conversations, evaluate
app.include_router(query.router)
app.include_router(index.router)
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(conversations.router)
app.include_router(evaluate.router)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}