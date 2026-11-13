"""
app/services/usage_service.py — Per-user quotas and usage aggregation.

WHAT THIS SOLVES
=================
audit_logs (see api/middleware/logging.py) already records cost_usd and
tokens_total per request, computed via app/monitoring/cost_tracker.py's
MODEL_PRICING table — that part already existed. What was missing:
  1. A place to store admin-configurable per-user budgets (this file's
     user_quotas table).
  2. Queries to aggregate audit_logs by user/date/model for dashboards
     and quota enforcement (this file's get_usage_* functions).
  3. A single check_quota() call query.py can use to reject requests once
     a user is over budget — see its usage in api/routers/query.py.

DATABASE SCHEMA
=================
user_quotas:
    user_id                 the uid/sub from the auth token, PK
    monthly_token_limit     NULL = unlimited
    monthly_cost_limit_usd  NULL = unlimited
    monthly_request_limit   NULL = unlimited
    updated_at              timestamp
    updated_by              which admin last changed this

QUOTA PERIOD
=================
Calendar month (1st of month 00:00 UTC to now), not a rolling 30-day
window — simpler to reason about for both admins and end users ("your
quota resets on the 1st"), and avoids the more expensive rolling-window
query. If you need rolling-window later, get_usage_this_period()'s
period_start calculation is the only thing that needs to change.

COST ESTIMATES, NOT INVOICES
=================
Every function here returns cost as estimated_cost_usd — computed from
MODEL_PRICING at request time, not reconciled against actual provider
billing. Label this clearly as "estimated" anywhere it surfaces in the UI.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("financial_rag.usage_service")


# ── Table setup ─────────────────────────────────────────────────────────────

async def ensure_usage_tables(pool) -> None:
    """Create user_quotas table if it doesn't exist. Call from FastAPI lifespan,
    same as ensure_conversation_tables — see api/main.py's startup."""
    if not pool:
        return
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS user_quotas (
            user_id                 VARCHAR(200) PRIMARY KEY,
            monthly_token_limit     BIGINT,           -- NULL = unlimited
            monthly_cost_limit_usd  NUMERIC(10, 2),   -- NULL = unlimited
            monthly_request_limit   INT,              -- NULL = unlimited
            updated_at              TIMESTAMPTZ DEFAULT NOW(),
            updated_by              VARCHAR(200)
        );
    """)
    logger.info("user_quotas table ready ✓")


def _month_start(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# ── Quota CRUD ───────────────────────────────────────────────────────────────

async def get_quota(pool, user_id: str) -> Optional[Dict]:
    """Returns None if no quota is set for this user (i.e. unlimited)."""
    if not pool:
        return None
    row = await pool.fetchrow(
        "SELECT * FROM user_quotas WHERE user_id = $1", user_id,
    )
    return dict(row) if row else None


async def set_quota(
    pool, user_id: str,
    monthly_token_limit: Optional[int] = None,
    monthly_cost_limit_usd: Optional[float] = None,
    monthly_request_limit: Optional[int] = None,
    updated_by: str = "",
) -> Dict:
    """Upsert a user's quota. Pass None (or 0 — treated identically, since
    JSON tools like Swagger UI often pre-fill numeric fields with 0 rather
    than leaving them blank) for any limit to leave it unlimited. This
    always overwrites all three fields on every call — not a partial
    update — since partial-update semantics would be confusing for a
    budget control.
    """
    if not pool:
        raise RuntimeError("Audit DB not configured — quotas require AUDIT_DB_URL")

    # 0 means "no limit" here, same as None/omitted — a genuine zero-tokens
    # quota isn't a real use case (that's just "disable the user", which
    # the existing disable_user() admin action already does properly).
    monthly_token_limit    = monthly_token_limit or None
    monthly_cost_limit_usd = monthly_cost_limit_usd or None
    monthly_request_limit  = monthly_request_limit or None

    row = await pool.fetchrow("""
        INSERT INTO user_quotas
            (user_id, monthly_token_limit, monthly_cost_limit_usd,
             monthly_request_limit, updated_at, updated_by)
        VALUES ($1, $2, $3, $4, NOW(), $5)
        ON CONFLICT (user_id) DO UPDATE SET
            monthly_token_limit    = EXCLUDED.monthly_token_limit,
            monthly_cost_limit_usd = EXCLUDED.monthly_cost_limit_usd,
            monthly_request_limit  = EXCLUDED.monthly_request_limit,
            updated_at              = NOW(),
            updated_by              = EXCLUDED.updated_by
        RETURNING *
    """, user_id, monthly_token_limit, monthly_cost_limit_usd,
         monthly_request_limit, updated_by)
    logger.info("Quota set for user_id=%s by=%s: tokens=%s cost=%s requests=%s",
                user_id, updated_by, monthly_token_limit,
                monthly_cost_limit_usd, monthly_request_limit)
    return dict(row)


async def delete_quota(pool, user_id: str) -> None:
    """Removes any limit — user becomes unlimited again."""
    if not pool:
        return
    await pool.execute("DELETE FROM user_quotas WHERE user_id = $1", user_id)


# ── Usage aggregation (for dashboards) ───────────────────────────────────────

async def get_usage_this_period(pool, user_id: str,
                                 period_start: Optional[datetime] = None) -> Dict:
    """Sums this user's tokens/cost/requests from audit_logs since
    period_start (defaults to start of current calendar month)."""
    if not pool:
        return {"tokens": 0, "cost_usd": 0.0, "requests": 0}

    period_start = period_start or _month_start()
    row = await pool.fetchrow("""
        SELECT
            COALESCE(SUM(tokens_total), 0)  AS tokens,
            COALESCE(SUM(cost_usd), 0.0)    AS cost_usd,
            COUNT(*)                         AS requests
        FROM audit_logs
        WHERE user_id = $1 AND timestamp >= $2
    """, user_id, period_start)
    return {
        "tokens":   int(row["tokens"]),
        "cost_usd": float(row["cost_usd"]),
        "requests": int(row["requests"]),
    }


async def check_quota(pool, user_id: str) -> Dict:
    """The one function query.py calls before running a request. Returns:
        {
            "allowed": bool,
            "reason": str | None,          # technical detail w/ exact numbers,
                                            # for server-side logs only — never
                                            # show this to the end user directly
            "user_message": str | None,    # friendly, safe to show in the UI
            "usage": {tokens, cost_usd, requests},
            "limits": {monthly_token_limit, monthly_cost_limit_usd, monthly_request_limit} | None,
        }
    A user with no row in user_quotas is unlimited — allowed=True always."""
    _FRIENDLY_MESSAGE = (
        "You've reached your usage limit for this billing period. "
        "Please contact your administrator to request an increase."
    )

    quota = await get_quota(pool, user_id)
    usage = await get_usage_this_period(pool, user_id)

    if not quota:
        return {"allowed": True, "reason": None, "user_message": None, "usage": usage, "limits": None}

    limits = {
        "monthly_token_limit":    quota.get("monthly_token_limit"),
        "monthly_cost_limit_usd": float(quota["monthly_cost_limit_usd"]) if quota.get("monthly_cost_limit_usd") is not None else None,
        "monthly_request_limit":  quota.get("monthly_request_limit"),
    }

    if limits["monthly_token_limit"] is not None and usage["tokens"] >= limits["monthly_token_limit"]:
        return {"allowed": False,
                "reason": f"Monthly token limit reached ({usage['tokens']:,}/{limits['monthly_token_limit']:,})",
                "user_message": _FRIENDLY_MESSAGE,
                "usage": usage, "limits": limits}

    if limits["monthly_cost_limit_usd"] is not None and usage["cost_usd"] >= limits["monthly_cost_limit_usd"]:
        return {"allowed": False,
                "reason": f"Monthly cost limit reached (${usage['cost_usd']:.2f}/${limits['monthly_cost_limit_usd']:.2f})",
                "user_message": _FRIENDLY_MESSAGE,
                "usage": usage, "limits": limits}

    if limits["monthly_request_limit"] is not None and usage["requests"] >= limits["monthly_request_limit"]:
        return {"allowed": False,
                "reason": f"Monthly request limit reached ({usage['requests']}/{limits['monthly_request_limit']})",
                "user_message": _FRIENDLY_MESSAGE,
                "usage": usage, "limits": limits}

    return {"allowed": True, "reason": None, "user_message": None, "usage": usage, "limits": limits}


async def get_usage_summary(pool, start: datetime, end: datetime,
                             user_id: Optional[str] = None) -> List[Dict]:
    """Per-user totals for the admin dashboard's main table. If user_id is
    given, returns just that one user's row (still as a list, for a
    consistent return shape)."""
    if not pool:
        return []

    if user_id:
        rows = await pool.fetch("""
            SELECT
                user_id,
                COALESCE(SUM(tokens_total), 0)  AS tokens,
                COALESCE(SUM(cost_usd), 0.0)    AS cost_usd,
                COUNT(*)                         AS requests,
                COUNT(*) FILTER (WHERE success = FALSE) AS errors
            FROM audit_logs
            WHERE user_id = $1 AND timestamp >= $2 AND timestamp < $3
            GROUP BY user_id
        """, user_id, start, end)
    else:
        rows = await pool.fetch("""
            SELECT
                user_id,
                COALESCE(SUM(tokens_total), 0)  AS tokens,
                COALESCE(SUM(cost_usd), 0.0)    AS cost_usd,
                COUNT(*)                         AS requests,
                COUNT(*) FILTER (WHERE success = FALSE) AS errors
            FROM audit_logs
            WHERE timestamp >= $1 AND timestamp < $2
            GROUP BY user_id
            ORDER BY cost_usd DESC
        """, start, end)

    return [
        {
            "user_id":  r["user_id"],
            "tokens":   int(r["tokens"]),
            "cost_usd": float(r["cost_usd"]),
            "requests": int(r["requests"]),
            "errors":   int(r["errors"]),
        }
        for r in rows
    ]


async def get_usage_timeseries(pool, start: datetime, end: datetime,
                                user_id: Optional[str] = None,
                                group_by: str = "day") -> List[Dict]:
    """Usage over time, for the trend chart. group_by: 'day' or 'hour'."""
    if not pool:
        return []
    if group_by not in ("day", "hour"):
        raise ValueError("group_by must be 'day' or 'hour'")

    bucket_expr = f"date_trunc('{group_by}', timestamp)"
    params = [start, end]
    user_filter = ""
    if user_id:
        user_filter = "AND user_id = $3"
        params.append(user_id)

    rows = await pool.fetch(f"""
        SELECT
            {bucket_expr} AS bucket,
            COALESCE(SUM(tokens_total), 0)  AS tokens,
            COALESCE(SUM(cost_usd), 0.0)    AS cost_usd,
            COUNT(*)                         AS requests
        FROM audit_logs
        WHERE timestamp >= $1 AND timestamp < $2 {user_filter}
        GROUP BY bucket
        ORDER BY bucket ASC
    """, *params)

    return [
        {
            "bucket":   r["bucket"].isoformat(),
            "tokens":   int(r["tokens"]),
            "cost_usd": float(r["cost_usd"]),
            "requests": int(r["requests"]),
        }
        for r in rows
    ]


async def get_usage_by_model(pool, start: datetime, end: datetime,
                              user_id: Optional[str] = None) -> List[Dict]:
    """Cost/token breakdown by model, for the 'what's actually expensive'
    question admins usually ask right after seeing total spend."""
    if not pool:
        return []

    params = [start, end]
    user_filter = ""
    if user_id:
        user_filter = "AND user_id = $3"
        params.append(user_id)

    rows = await pool.fetch(f"""
        SELECT
            COALESCE(llm_provider, 'unknown') AS llm_provider,
            COALESCE(llm_model, 'unknown')    AS llm_model,
            COALESCE(SUM(tokens_total), 0)     AS tokens,
            COALESCE(SUM(cost_usd), 0.0)       AS cost_usd,
            COUNT(*)                            AS requests
        FROM audit_logs
        WHERE timestamp >= $1 AND timestamp < $2 {user_filter}
        GROUP BY llm_provider, llm_model
        ORDER BY cost_usd DESC
    """, *params)

    return [
        {
            "llm_provider": r["llm_provider"],
            "llm_model":    r["llm_model"],
            "tokens":       int(r["tokens"]),
            "cost_usd":     float(r["cost_usd"]),
            "requests":     int(r["requests"]),
        }
        for r in rows
    ]