"""
api/middleware/logging.py â€” Structured request logging with trace_id.

Every request gets a unique trace_id that flows through all log lines.
Logs to:
  1. Existing app/logging/logger.py (stdout + file)  â€” always
  2. PostgreSQL audit_logs table                      â€” when DB configured

Access control for log queries:
  admin    â†’ all logs
  engineer â†’ logs WHERE role IN ('user', 'engineer')
  user     â†’ own logs only

.env config:
  AUDIT_DB_URL=postgresql://user:pass@localhost:5432/ragdb
  AUDIT_LOG_ENABLED=true   (default: true if AUDIT_DB_URL set)
"""

import json
import logging
import os
import time
import uuid
from typing import Optional

logger = logging.getLogger("financial_rag.api.audit")

# â”€â”€ PostgreSQL setup (optional â€” app works without it) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from app.config.settings import AUDIT_DB_URL, AUDIT_LOG_ENABLED as AUDIT_ENABLED
_db_pool         = None
_db_unavailable  = False   # set True after first connection failure


def _get_pool():
    """Lazy-init async DB pool. Returns None if DB not configured."""
    global _db_pool, _db_unavailable
    if _db_unavailable or not AUDIT_DB_URL or not AUDIT_ENABLED:
        return None
    if _db_pool is not None:
        return _db_pool
    try:
        import asyncpg
        return None   # pool created in async context â€” see ensure_pool()
    except ImportError:
        logger.warning(
            "asyncpg not installed â€” audit DB logging disabled. "
            "pip install asyncpg to enable."
        )
        _db_unavailable = True
        return None


async def ensure_pool():
    """Create asyncpg pool on first use. Call from FastAPI lifespan."""
    global _db_pool, _db_unavailable
    if _db_unavailable or not AUDIT_DB_URL or not AUDIT_ENABLED:
        return
    try:
        import asyncpg
        _db_pool = await asyncpg.create_pool(AUDIT_DB_URL, min_size=1, max_size=5, command_timeout=60, max_inactive_connection_lifetime=300.0, server_settings={"application_name": "rag_platform"})
        logger.info("Audit DB pool created: %s", AUDIT_DB_URL.split("@")[-1])
        await _ensure_table()
    except Exception as e:
        logger.warning("Audit DB unavailable â€” logging to file only: %s", e)
        _db_unavailable = True


async def _ensure_table():
    """Create audit_logs table if it doesn't exist."""
    if not _db_pool:
        return
    await _db_pool.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id              BIGSERIAL PRIMARY KEY,
            trace_id        VARCHAR(16) NOT NULL,
            timestamp       TIMESTAMPTZ DEFAULT NOW(),

            -- Who
            user_id         VARCHAR(100),
            role            VARCHAR(20),
            ip_address      VARCHAR(45),

            -- What
            method          VARCHAR(10),
            endpoint        VARCHAR(200),
            query_text      TEXT,
            query_type      VARCHAR(20),

            -- How
            llm_provider    VARCHAR(30),
            llm_model       VARCHAR(100),

            -- Result
            status_code     INT,
            success         BOOLEAN,
            error_component VARCHAR(100),
            error_message   TEXT,

            -- Performance
            latency_ms      INT,
            tokens_total    INT,
            cost_usd        NUMERIC(10, 6),

            -- RAG quality
            chunks_retrieved INT,
            cache_hit        BOOLEAN,
            faithfulness     NUMERIC(4, 3),
            answer_relevancy NUMERIC(4, 3)
        );

        CREATE INDEX IF NOT EXISTS idx_audit_trace
            ON audit_logs(trace_id);
        CREATE INDEX IF NOT EXISTS idx_audit_user
            ON audit_logs(user_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_errors
            ON audit_logs(success, timestamp DESC)
            WHERE success = FALSE;
        CREATE INDEX IF NOT EXISTS idx_audit_time
            ON audit_logs(timestamp DESC);
    """)
    logger.info("Audit table ready âœ“")


async def close_pool():
    """Call from FastAPI shutdown."""
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None


# â”€â”€ Log entry writer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def write_audit_log(entry: dict):
    """
    Write one audit entry. Always logs to structured logger.
    Also writes to PostgreSQL if configured.
    """
    # 1. Structured log line (always)
    logger.info(
        "[%s] %s %s â†’ %d | user=%s role=%s latency=%dms tokens=%d cost=$%.5f | %s",
        entry.get("trace_id", "-"),
        entry.get("method", "-"),
        entry.get("endpoint", "-"),
        entry.get("status_code", 0),
        entry.get("user_id", "anon"),
        entry.get("role", "-"),
        entry.get("latency_ms", 0),
        entry.get("tokens_total", 0),
        entry.get("cost_usd", 0.0),
        "ERROR: " + entry["error_message"] if entry.get("error_message") else "ok",
    )

    # 2. PostgreSQL (if configured)
    if not _db_pool:
        return
    try:
        await _db_pool.execute("""
            INSERT INTO audit_logs (
                trace_id, user_id, role, ip_address,
                method, endpoint, query_text, query_type,
                llm_provider, llm_model,
                status_code, success, error_component, error_message,
                latency_ms, tokens_total, cost_usd,
                chunks_retrieved, cache_hit, faithfulness, answer_relevancy
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21
            )
        """,
            entry.get("trace_id"),
            entry.get("user_id"),
            entry.get("role"),
            entry.get("ip_address"),
            entry.get("method"),
            entry.get("endpoint"),
            entry.get("query_text"),
            entry.get("query_type"),
            entry.get("llm_provider"),
            entry.get("llm_model"),
            entry.get("status_code"),
            entry.get("success", True),
            entry.get("error_component"),
            entry.get("error_message"),
            entry.get("latency_ms"),
            entry.get("tokens_total"),
            entry.get("cost_usd"),
            entry.get("chunks_retrieved"),
            entry.get("cache_hit"),
            entry.get("faithfulness"),
            entry.get("answer_relevancy"),
        )
    except Exception as e:
        logger.error("Audit DB write failed: %s", e)


# â”€â”€ trace_id context helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def new_trace_id() -> str:
    return str(uuid.uuid4())[:8]


