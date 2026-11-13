"""
api/routers/admin.py — Admin log viewer endpoints.

GET /api/v1/admin/logs  — query audit logs with filters
GET /api/v1/admin/stats — summary stats for dashboard

Access control:
  admin    → all logs
  engineer → logs WHERE role IN ('user', 'engineer')
  user     → 403 Forbidden
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.middleware.auth import require_role

logger = logging.getLogger("financial_rag.api.admin")
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ── Response models ───────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    id:               int
    trace_id:         str
    timestamp:        str
    user_id:          Optional[str]
    role:             Optional[str]
    method:           Optional[str]
    endpoint:         Optional[str]
    query_text:       Optional[str]
    status_code:      Optional[int]
    success:          Optional[bool]
    error_component:  Optional[str]
    error_message:    Optional[str]
    latency_ms:       Optional[int]
    tokens_total:     Optional[int]
    cost_usd:         Optional[float]
    cache_hit:        Optional[bool]
    faithfulness:     Optional[float]
    answer_relevancy: Optional[float]


class LogsResponse(BaseModel):
    total:   int
    page:    int
    limit:   int
    entries: list[LogEntry]


class StatsResponse(BaseModel):
    total_requests:    int
    total_errors:      int
    error_rate_pct:    float
    avg_latency_ms:    float
    total_tokens:      int
    total_cost_usd:    float
    cache_hit_rate_pct: float
    top_errors:        list[dict]


# ── Log file fallback (when no DB configured) ─────────────────────────────────

def _read_log_file(limit: int = 50, user_id: str = None,
                   errors_only: bool = False) -> list[dict]:
    """Read from log file when PostgreSQL is not configured."""
    log_path = os.path.join(os.getenv("LOG_DIR", "logs"), "rag_app.log")
    if not os.path.exists(log_path):
        return []

    entries = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-500:]   # last 500 lines

        for line in reversed(lines):
            if "[financial_rag.api.audit]" not in line:
                continue
            if errors_only and "ERROR:" not in line:
                continue
            if user_id and f"user={user_id}" not in line:
                continue
            entries.append({"raw": line.strip()})
            if len(entries) >= limit:
                break
    except Exception as e:
        logger.warning("Could not read log file: %s", e)

    return entries


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/logs",
    summary="Query audit logs",
    description=(
        "**Admin:** all logs | **Engineer:** user + engineer logs only\n\n"
        "Returns recent audit log entries with filtering options."
    ),
)
async def get_logs(
    # Filters
    user_id:     Optional[str]  = Query(None, description="Filter by username"),
    role_filter: Optional[str]  = Query(None, description="Filter by role"),
    errors_only: bool           = Query(False, description="Only show errors"),
    endpoint:    Optional[str]  = Query(None, description="Filter by endpoint path"),
    start_time:  Optional[str]  = Query(None, description="Filter from timestamp (ISO format, e.g. 2026-07-10T00:00:00)"),
    end_time:    Optional[str]  = Query(None, description="Filter to timestamp (ISO format)"),

    # Pagination
    page:  int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),

    # Auth
    current_user: dict = Depends(require_role("admin", "engineer")),
):
    caller_role = current_user.get("role")
    caller_user = current_user.get("sub")

    from api.middleware.logging import _db_pool, AUDIT_DB_URL

    # ── PostgreSQL path ───────────────────────────────────────────────────────
    if _db_pool:
        conditions = ["1=1"]
        params     = []
        i          = 1

        # Engineers can't see other engineers' private queries
        if caller_role == "engineer":
            conditions.append(f"role = ANY(${i}::text[])")
            params.append(["user", "engineer"])
            i += 1
        # Admins see everything — no filter

        if user_id:
            conditions.append(f"user_id = ${i}")
            params.append(user_id); i += 1

        if role_filter:
            conditions.append(f"role = ${i}")
            params.append(role_filter); i += 1

        if errors_only:
            conditions.append("success = false")

        if endpoint:
            conditions.append(f"endpoint ILIKE ${i}")
            params.append(f"%{endpoint}%"); i += 1

        if start_time:
            conditions.append(f"timestamp >= ${i}")
            params.append(start_time); i += 1

        if end_time:
            conditions.append(f"timestamp <= ${i}")
            params.append(end_time); i += 1

        where = " AND ".join(conditions)
        offset = (page - 1) * limit

        total_row = await _db_pool.fetchval(
            f"SELECT COUNT(*) FROM audit_logs WHERE {where}", *params
        )
        rows = await _db_pool.fetch(
            f"SELECT * FROM audit_logs WHERE {where} "
            f"ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}",
            *params,
        )

        entries = []
        for r in rows:
            entries.append(LogEntry(
                id=r["id"],
                trace_id=r["trace_id"] or "",
                timestamp=str(r["timestamp"]),
                user_id=r["user_id"],
                role=r["role"],
                method=r["method"],
                endpoint=r["endpoint"],
                query_text=r["query_text"],
                status_code=r["status_code"],
                success=r["success"],
                error_component=r["error_component"],
                error_message=r["error_message"],
                latency_ms=r["latency_ms"],
                tokens_total=r["tokens_total"],
                cost_usd=float(r["cost_usd"]) if r["cost_usd"] else None,
                cache_hit=r["cache_hit"],
                faithfulness=float(r["faithfulness"]) if r["faithfulness"] else None,
                answer_relevancy=float(r["answer_relevancy"]) if r["answer_relevancy"] else None,
            ))

        return LogsResponse(total=total_row, page=page, limit=limit, entries=entries)

    # ── File fallback (no DB) ─────────────────────────────────────────────────
    raw_entries = _read_log_file(limit=limit, user_id=user_id, errors_only=errors_only)
    return {
        "total":   len(raw_entries),
        "page":    1,
        "limit":   limit,
        "source":  "log_file",
        "note":    "Set AUDIT_DB_URL in .env for full structured log queries.",
        "entries": raw_entries,
    }


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Usage statistics",
    description="Summary metrics: requests, errors, cost, latency, cache hit rate.",
)
async def get_stats(
    hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    current_user: dict = Depends(require_role("admin", "engineer")),
):
    from api.middleware.logging import _db_pool
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stats require AUDIT_DB_URL to be configured in .env",
        )

    row = await _db_pool.fetchrow("""
        SELECT
            COUNT(*)                                        AS total_requests,
            COUNT(*) FILTER (WHERE success = false)         AS total_errors,
            ROUND(AVG(latency_ms)::numeric, 1)              AS avg_latency_ms,
            COALESCE(SUM(tokens_total), 0)                  AS total_tokens,
            COALESCE(SUM(cost_usd), 0)                      AS total_cost_usd,
            COUNT(*) FILTER (WHERE cache_hit = true)        AS cache_hits
        FROM audit_logs
        WHERE timestamp > NOW() - INTERVAL '1 hour' * $1
    """, hours)

    total     = row["total_requests"] or 0
    errors    = row["total_errors"] or 0
    cache_hits = row["cache_hits"] or 0

    top_errors = await _db_pool.fetch("""
        SELECT error_component, error_message, COUNT(*) AS count
        FROM audit_logs
        WHERE success = false
          AND timestamp > NOW() - INTERVAL '1 hour' * $1
        GROUP BY error_component, error_message
        ORDER BY count DESC
        LIMIT 5
    """, hours)

    return StatsResponse(
        total_requests=    total,
        total_errors=      errors,
        error_rate_pct=    round(errors / max(total, 1) * 100, 1),
        avg_latency_ms=    float(row["avg_latency_ms"] or 0),
        total_tokens=      int(row["total_tokens"] or 0),
        total_cost_usd=    float(row["total_cost_usd"] or 0),
        cache_hit_rate_pct= round(cache_hits / max(total, 1) * 100, 1),
        top_errors=[
            {"component": r["error_component"], "message": r["error_message"],
             "count": r["count"]}
            for r in top_errors
        ],
    )

# ── User management (Firebase/Cognito/Azure AD via AuthProvider) ──────────────
# All routes here go through app.providers.auth.factory.get_auth_provider(),
# never a cloud SDK directly — see app/providers/auth/base.py.

from app.providers.auth.base import VALID_ROLES


class UserOut(BaseModel):
    uid:       str
    email:     str
    role:      str
    full_name: str
    disabled:  bool


class UserListOut(BaseModel):
    users:           list[UserOut]
    next_page_token: Optional[str] = None


class CreateUserRequest(BaseModel):
    email:     str
    password:  str
    role:      str = "user"
    full_name: str = ""


class UpdateUserRequest(BaseModel):
    email:     Optional[str] = None
    full_name: Optional[str] = None
    password:  Optional[str] = None


class SetRoleRequest(BaseModel):
    role: str


def _to_user_out(record) -> UserOut:
    return UserOut(uid=record.uid, email=record.email, role=record.role,
                    full_name=record.full_name, disabled=record.disabled)


@router.get("/users", response_model=UserListOut, summary="List users (admin only)")
async def list_users(
    page_token: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    user: dict = Depends(require_role("admin")),
):
    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    users, next_token = provider.list_users(page_token=page_token, limit=limit)
    return UserListOut(users=[_to_user_out(u) for u in users], next_page_token=next_token)


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED,
             summary="Create a new user (admin only)")
async def create_user(
    req: CreateUserRequest,
    user: dict = Depends(require_role("admin")),
):
    if req.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of {VALID_ROLES}")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="password must be at least 6 characters")

    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    try:
        record = provider.create_user(email=req.email, password=req.password,
                                       role=req.role, full_name=req.full_name)
    except Exception as e:
        # Firebase raises its own exception types for "email already exists" etc.
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Admin %s created user email=%s role=%s",
                user.get("email", user.get("sub")), req.email, req.role)
    return _to_user_out(record)


@router.get("/users/{uid}", response_model=UserOut, summary="Get a single user (admin only)")
async def get_user(uid: str, user: dict = Depends(require_role("admin"))):
    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    try:
        record = provider.get_user(uid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _to_user_out(record)


@router.patch("/users/{uid}", response_model=UserOut, summary="Update user profile (admin only)")
async def update_user(uid: str, req: UpdateUserRequest,
                       user: dict = Depends(require_role("admin"))):
    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    try:
        record = provider.update_user(uid, email=req.email, full_name=req.full_name,
                                       password=req.password)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info("Admin %s updated user uid=%s", user.get("email", user.get("sub")), uid)
    return _to_user_out(record)


@router.patch("/users/{uid}/role", response_model=UserOut, summary="Change a user's role (admin only)")
async def set_user_role(uid: str, req: SetRoleRequest,
                         user: dict = Depends(require_role("admin"))):
    if req.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of {VALID_ROLES}")

    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    try:
        provider.set_user_role(uid, req.role)
        record = provider.get_user(uid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    logger.info("Admin %s set role=%s for uid=%s",
                user.get("email", user.get("sub")), req.role, uid)
    return _to_user_out(record)


@router.post("/users/{uid}/disable", response_model=UserOut, summary="Disable a user (admin only)")
async def disable_user(uid: str, user: dict = Depends(require_role("admin"))):
    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    try:
        provider.disable_user(uid)
        record = provider.get_user(uid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    logger.info("Admin %s disabled uid=%s", user.get("email", user.get("sub")), uid)
    return _to_user_out(record)


@router.post("/users/{uid}/enable", response_model=UserOut, summary="Re-enable a user (admin only)")
async def enable_user(uid: str, user: dict = Depends(require_role("admin"))):
    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    try:
        provider.enable_user(uid)
        record = provider.get_user(uid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    logger.info("Admin %s enabled uid=%s", user.get("email", user.get("sub")), uid)
    return _to_user_out(record)


@router.delete("/users/{uid}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a user permanently (admin only)")
async def delete_user(uid: str, user: dict = Depends(require_role("admin"))):
    requester_uid = user.get("sub", "")
    if uid == requester_uid:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")

    from app.providers.auth.factory import get_auth_provider
    provider = get_auth_provider()
    try:
        provider.delete_user(uid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    logger.info("Admin %s deleted uid=%s", user.get("email", user.get("sub")), uid)


# ── FinOps / Usage tracking ────────────────────────────────────────────────
# Aggregation queries live in app/services/usage_service.py — this router
# only handles request parsing, auth, and response shaping. Quota
# ENFORCEMENT happens in api/routers/query.py (check_quota() before a
# request runs), not here — these endpoints are for viewing/configuring,
# not for gating requests.

from datetime import datetime, timedelta, timezone


class QuotaOut(BaseModel):
    user_id:                str
    monthly_token_limit:    Optional[int] = None
    monthly_cost_limit_usd: Optional[float] = None
    monthly_request_limit:  Optional[int] = None
    updated_at:              Optional[str] = None
    updated_by:               Optional[str] = None


class SetQuotaRequest(BaseModel):
    monthly_token_limit:    Optional[int] = None
    monthly_cost_limit_usd: Optional[float] = None
    monthly_request_limit:  Optional[int] = None


class UsageSummaryRow(BaseModel):
    user_id:  str
    tokens:   int
    cost_usd: float
    requests: int
    errors:   int


class UsageSummaryOut(BaseModel):
    period_start: str
    period_end:   str
    rows:         list[UsageSummaryRow]
    note:         str = "cost_usd is an estimate computed from token counts, not a reconciled invoice"


def _get_audit_pool():
    """FinOps endpoints need direct DB access for aggregation queries
    that don't fit the existing get_logs()/get_stats() shape — same
    _db_pool every other DB-backed feature in this app uses."""
    from api.middleware.logging import _db_pool
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Usage tracking requires the audit database (AUDIT_DB_URL) to be configured and reachable.",
        )
    return _db_pool


def _parse_date_range(from_: Optional[str], to: Optional[str]) -> tuple[datetime, datetime]:
    """Defaults to the current calendar month if not specified — matches
    usage_service's quota period, so 'this period' means the same thing
    everywhere in the UI."""
    if from_:
        start = datetime.fromisoformat(from_).replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if to:
        end = datetime.fromisoformat(to).replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc) + timedelta(seconds=1)  # inclusive of "now"

    return start, end


@router.get("/usage", response_model=UsageSummaryOut,
            summary="Usage summary, all users or one (admin only)")
async def get_usage(
    user_id: Optional[str] = Query(None),
    from_: Optional[str] = Query(None, alias="from", description="ISO date, e.g. 2026-08-01"),
    to: Optional[str] = Query(None, description="ISO date, e.g. 2026-08-31"),
    user: dict = Depends(require_role("admin")),
):
    from app.services.usage_service import get_usage_summary

    pool = _get_audit_pool()
    start, end = _parse_date_range(from_, to)
    rows = await get_usage_summary(pool, start, end, user_id=user_id)

    return UsageSummaryOut(
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        rows=[UsageSummaryRow(**r) for r in rows],
    )


@router.get("/usage/timeseries", summary="Usage over time, for trend charts (admin only)")
async def get_usage_timeseries_endpoint(
    user_id: Optional[str] = Query(None),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    group_by: str = Query("day", pattern="^(day|hour)$"),
    user: dict = Depends(require_role("admin")),
):
    from app.services.usage_service import get_usage_timeseries

    pool = _get_audit_pool()
    start, end = _parse_date_range(from_, to)
    points = await get_usage_timeseries(pool, start, end, user_id=user_id, group_by=group_by)
    return {"period_start": start.isoformat(), "period_end": end.isoformat(), "points": points}


@router.get("/usage/by-model", summary="Cost/token breakdown by LLM model (admin only)")
async def get_usage_by_model_endpoint(
    user_id: Optional[str] = Query(None),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    user: dict = Depends(require_role("admin")),
):
    from app.services.usage_service import get_usage_by_model

    pool = _get_audit_pool()
    start, end = _parse_date_range(from_, to)
    rows = await get_usage_by_model(pool, start, end, user_id=user_id)
    return {"period_start": start.isoformat(), "period_end": end.isoformat(), "rows": rows}


@router.get("/usage/{uid}/quota", response_model=QuotaOut,
            summary="Get a user's quota + current-period usage (admin only)")
async def get_user_quota(uid: str, user: dict = Depends(require_role("admin"))):
    from app.services.usage_service import get_quota

    pool = _get_audit_pool()
    quota = await get_quota(pool, uid)
    if not quota:
        return QuotaOut(user_id=uid)  # unlimited — no row set

    return QuotaOut(
        user_id=uid,
        monthly_token_limit=quota.get("monthly_token_limit"),
        monthly_cost_limit_usd=float(quota["monthly_cost_limit_usd"]) if quota.get("monthly_cost_limit_usd") is not None else None,
        monthly_request_limit=quota.get("monthly_request_limit"),
        updated_at=quota["updated_at"].isoformat() if quota.get("updated_at") else None,
        updated_by=quota.get("updated_by"),
    )


@router.patch("/usage/{uid}/quota", response_model=QuotaOut,
              summary="Set a user's monthly quota (admin only)")
async def set_user_quota(uid: str, req: SetQuotaRequest,
                          user: dict = Depends(require_role("admin"))):
    from app.services.usage_service import set_quota

    pool = _get_audit_pool()
    quota = await set_quota(
        pool, uid,
        monthly_token_limit=req.monthly_token_limit,
        monthly_cost_limit_usd=req.monthly_cost_limit_usd,
        monthly_request_limit=req.monthly_request_limit,
        updated_by=user.get("email", user.get("sub", "")),
    )
    logger.info("Admin %s set quota for uid=%s", user.get("email", user.get("sub")), uid)

    return QuotaOut(
        user_id=uid,
        monthly_token_limit=quota.get("monthly_token_limit"),
        monthly_cost_limit_usd=float(quota["monthly_cost_limit_usd"]) if quota.get("monthly_cost_limit_usd") is not None else None,
        monthly_request_limit=quota.get("monthly_request_limit"),
        updated_at=quota["updated_at"].isoformat() if quota.get("updated_at") else None,
        updated_by=quota.get("updated_by"),
    )


@router.delete("/usage/{uid}/quota", status_code=status.HTTP_204_NO_CONTENT,
               summary="Remove a user's quota - back to unlimited (admin only)")
async def delete_user_quota(uid: str, user: dict = Depends(require_role("admin"))):
    from app.services.usage_service import delete_quota

    pool = _get_audit_pool()
    await delete_quota(pool, uid)
    logger.info("Admin %s removed quota for uid=%s", user.get("email", user.get("sub")), uid)