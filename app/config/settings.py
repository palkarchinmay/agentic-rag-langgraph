"""
app/config/settings.py Single source of truth for all config.

Every service endpoint, path, and credential reads from environment
variables with sensible local defaults. Switching clouds = change .env only.

.env template: see .env.example in project root

Storage abstraction:
  STORAGE_BACKEND = local | gcs | azure_blob | s3
  CACHE_BACKEND   = local | redis
  LOG_BACKEND     = file | cloud (stdout â†’ cloud logging auto)
  Auth is handled by app/providers/auth/ (AUTH_PROVIDER env var), not here.
"""

import os
from pathlib import Path
from typing import Optional


#  Base paths
# All local paths read from env so Docker volumes map cleanly

BASE_DIR         = Path(os.getenv("APP_BASE_DIR",   "."))
DATA_DIR         = Path(os.getenv("DATA_DIR",        "data"))
LOG_DIR          = Path(os.getenv("LOG_DIR",         "logs"))
CHROMA_PERSIST   = os.getenv("CHROMA_PERSIST_DIR",   "chroma_db")


# -- Vector store backend (Step 2) ------------------------------------------
# VECTOR_BACKEND=chroma  -> ChromaDB on local disk (default, local dev)
# VECTOR_BACKEND=qdrant  -> Qdrant Cloud (production)
#
# Single env var switches the entire vector stack.
# ChromaDB path (CHROMA_PERSIST above) is still used for BM25 pickle
# persistence in both modes -- only the dense vector index moves to Qdrant.

VECTOR_BACKEND   = os.getenv("VECTOR_BACKEND", "chroma")

# Qdrant Cloud connection
# QDRANT_URL set as plain env var on Cloud Run (not a secret -- just a URL)
# QDRANT_API_KEY fetched from GCP Secret Manager as 'rag-qdrant-api-key'
QDRANT_URL       = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", "")
CACHE_DIR        = os.getenv("CACHE_DIR",            ".cache")
RAGAS_CACHE_DIR  = os.getenv("RAGAS_CACHE_DIR",      ".ragas_cache")
PROMPTS_FILE     = os.getenv("PROMPTS_FILE",         "prompts.json")


# â”€â”€ API server â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

API_HOST         = os.getenv("API_HOST",    "0.0.0.0")
API_PORT         = int(os.getenv("API_PORT", "8000"))
API_WORKERS      = int(os.getenv("API_WORKERS", "2"))

# URL Streamlit uses to reach the API
# Local dev:       http://localhost:8000
# Docker compose:  http://api:8000
# GCP Cloud Run:   https://rag-api-xxx-uc.a.run.app
# Azure Cont Apps: https://rag-api.azurecontainerapps.io
# AWS ECS:         https://rag-api.yourdomain.com
RAG_API_URL      = os.getenv("RAG_API_URL", "http://localhost:8000")

CORS_ORIGINS     = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501"
).split(",")


# ── Auth ─────────────────────────────────────────────────────────────────────
# Auth is handled entirely by app/providers/auth/ (AUTH_PROVIDER env var:
# firebase | cognito | azure_ad -- see app/providers/auth/factory.py).
# The custom-JWT / users.json system this section used to configure has
# been fully replaced; nothing in the app reads JWT_SECRET, AUTH_BACKEND,
# or SSO_* anymore.
# â”€â”€ Cache backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CACHE_BACKEND=local   â†’ JSON files in CACHE_DIR (default, zero infra)
# CACHE_BACKEND=redis   â†’ Redis (required for multi-container / cloud)

CACHE_BACKEND    = os.getenv("CACHE_BACKEND", "local")

# Redis â€” same URL format works for all clouds:
# Local dev:         redis://localhost:6379
# GCP Memorystore:   redis://:password@10.x.x.x:6379
# Azure Cache:       rediss://:password@name.redis.cache.windows.net:6380
# AWS ElastiCache:   redis://:password@cluster.abc.cache.amazonaws.com:6379
# Upstash (serverless, free tier): rediss://:token@host.upstash.io:6380
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_PASSWORD   = os.getenv("REDIS_PASSWORD", "")

CACHE_TTL_SECONDS   = int(os.getenv("CACHE_TTL_SECONDS",   str(60 * 60)))  # 1 hour
CACHE_SIMILARITY    = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
MAX_CACHE_ENTRIES   = int(os.getenv("MAX_CACHE_ENTRIES", "500"))

# Session TTL (used when Redis sessions are enabled)
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(4 * 60 * 60)))  # 4 hours


# â”€â”€ Storage backend (for vector store + files) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STORAGE_BACKEND=local       â†’ local disk (default)
# STORAGE_BACKEND=gcs         â†’ Google Cloud Storage
# STORAGE_BACKEND=azure_blob  â†’ Azure Blob Storage
# STORAGE_BACKEND=s3          â†’ AWS S3

STORAGE_BACKEND     = os.getenv("STORAGE_BACKEND", "local")

# GCS (Google Cloud Storage)
GCS_BUCKET          = os.getenv("GCS_BUCKET", "")
GCS_PREFIX          = os.getenv("GCS_PREFIX", "rag-data")

# Azure Blob
AZURE_STORAGE_CONN  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_BLOB_CONTAINER= os.getenv("AZURE_BLOB_CONTAINER", "rag-data")

# AWS S3
S3_BUCKET           = os.getenv("S3_BUCKET", "")
S3_PREFIX           = os.getenv("S3_PREFIX", "rag-data")
S3_REGION           = os.getenv("S3_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))


# â”€â”€ Logging backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LOG_BACKEND=file    â†’ writes to LOG_DIR/rag_app.log (default)
# LOG_BACKEND=cloud   â†’ stdout only (cloud logging captures automatically)
#                       GCP: Cloud Logging
#                       Azure: Azure Monitor
#                       AWS: CloudWatch
# LOG_BACKEND=both    â†’ file + stdout

LOG_BACKEND         = os.getenv("LOG_BACKEND", "both")
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT          = os.getenv("LOG_FORMAT", "text")  # text | json

# Audit DB (optional â€” structured SQL queries on logs)
# Works with any PostgreSQL-compatible DB:
# Local:       postgresql://user:pass@localhost:5432/ragdb
# GCP:         postgresql://user:pass@/ragdb?host=/cloudsql/project:region:instance
# Azure:       postgresql://user:pass@server.postgres.database.azure.com:5432/ragdb
# AWS RDS:     postgresql://user:pass@instance.region.rds.amazonaws.com:5432/ragdb
# Neon (serverless): postgresql://user:pass@ep-xxx.neon.tech/ragdb
AUDIT_DB_URL        = os.getenv("AUDIT_DB_URL", "")
AUDIT_LOG_ENABLED   = os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true"


# â”€â”€ LLM / embedding defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DEFAULT_LLM_PROVIDER    = os.getenv("DEFAULT_LLM_PROVIDER",    "openai")
DEFAULT_LLM_MODEL       = os.getenv("DEFAULT_LLM_MODEL",       "gpt-4o-mini")
DEFAULT_EMBED_PROVIDER  = os.getenv("DEFAULT_EMBED_PROVIDER",  "openai")
DEFAULT_EMBED_MODEL     = os.getenv("DEFAULT_EMBED_MODEL",     "text-embedding-3-small")
DEFAULT_OCR_METHOD      = os.getenv("DEFAULT_OCR_METHOD",      "easyocr")

# LLM guard config
LLM_MAX_RETRIES     = int(os.getenv("LLM_MAX_RETRIES",    "3"))
LLM_RETRY_BASE_DELAY= float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
LLM_SESSION_COST_LIMIT = float(os.getenv("LLM_SESSION_COST_LIMIT", "1.00"))


# â”€â”€ Retrieval defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DEFAULT_TOP_K           = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_RETRIEVAL_MODE  = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid")
DEFAULT_RERANK          = os.getenv("DEFAULT_RERANK", "true").lower() == "true"


# â”€â”€ Helper: is Redis configured? â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def redis_enabled() -> bool:
    return CACHE_BACKEND == "redis" and bool(REDIS_URL)

def audit_db_enabled() -> bool:
    return bool(AUDIT_DB_URL) and AUDIT_LOG_ENABLED

def cloud_storage_enabled() -> bool:
    return STORAGE_BACKEND != "local"


# â”€â”€ Validation on startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def validate_config() -> list[str]:
    """
    Returns list of warnings for missing/insecure config.
    Call at startup to catch misconfigurations early.
    """
    warnings = []

    if CACHE_BACKEND == "redis" and not REDIS_URL:
        warnings.append("CACHE_BACKEND=redis but REDIS_URL not set â€” falling back to local")

    if STORAGE_BACKEND == "gcs" and not GCS_BUCKET:
        warnings.append("STORAGE_BACKEND=gcs but GCS_BUCKET not set")

    if STORAGE_BACKEND == "azure_blob" and not AZURE_STORAGE_CONN:
        warnings.append("STORAGE_BACKEND=azure_blob but AZURE_STORAGE_CONNECTION_STRING not set")

    if STORAGE_BACKEND == "s3" and not S3_BUCKET:
        warnings.append("STORAGE_BACKEND=s3 but S3_BUCKET not set")

    return warnings

def qdrant_enabled() -> bool:
    """True when Qdrant Cloud is configured and should be used (Step 2)."""
    return VECTOR_BACKEND == "qdrant" and bool(QDRANT_URL)