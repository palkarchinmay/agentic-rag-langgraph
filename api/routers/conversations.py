"""
api/routers/conversations.py — Conversation history endpoints.

Endpoints:
  GET    /api/v1/conversations              — list own conversations
  GET    /api/v1/conversations/{id}          — get one conversation + messages
  DELETE /api/v1/conversations/{id}          — delete own conversation
  GET    /api/v1/admin/conversations         — admin: list any/all users' conversations
  DELETE /api/v1/admin/conversations/{id}    — admin: delete any conversation
  POST   /api/v1/admin/conversations/cleanup — admin: trigger retention cleanup manually

Access control mirrors api/routers/admin.py's existing pattern exactly:
  require_auth     → any authenticated user, sees only their own data
  require_role("admin") → full access to all users' conversations

This router reuses the SAME asyncpg pool as audit logging
(api/middleware/logging.py's _db_pool) — no second connection pool,
no second AUDIT_DB_URL. Conversations and audit_logs live in the same
Supabase Postgres instance, just different tables.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.middleware.auth import require_auth, require_role
from app.services import conversation_service as conv_service

logger = logging.getLogger("financial_rag.api.conversations")
router = APIRouter(prefix="/api/v1", tags=["conversations"])


# ── Response models ───────────────────────────────────────────────────────────

class MessageModel(BaseModel):
    role: str
    content: str
    token_count: Optional[int] = None
    created_at: Optional[str] = None


class ConversationModel(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    summary: Optional[str] = None
    created_at: str
    updated_at: str


class ConversationDetailModel(ConversationModel):
    messages: list[MessageModel] = []


class ConversationsListResponse(BaseModel):
    total: int
    page: int
    limit: int
    conversations: list[ConversationModel]


class CleanupResponse(BaseModel):
    deleted_count: int
    retention_days: int


# ── Helper to get the shared DB pool ──────────────────────────────────────────

def _get_pool():
    """
    Reuses the exact same pool as audit logging — conversations and
    audit_logs are different tables in the same Supabase database.
    """
    from api.middleware.logging import _db_pool
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation history requires AUDIT_DB_URL to be configured",
        )
    return _db_pool


# ── User-facing endpoints ──────────────────────────────────────────────────────

@router.get(
    "/conversations",
    response_model=ConversationsListResponse,
    summary="List my conversations",
)
async def list_my_conversations(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(require_auth),
):
    pool = _get_pool()
    user_id = current_user.get("sub")
    result = await conv_service.list_conversations(pool, user_id=user_id, page=page, limit=limit)

    return ConversationsListResponse(
        total=result["total"],
        page=result["page"],
        limit=result["limit"],
        conversations=[
            ConversationModel(
                id=str(c["id"]), user_id=c["user_id"], title=c["title"],
                summary=c.get("summary"),
                created_at=str(c["created_at"]), updated_at=str(c["updated_at"]),
            ) for c in result["conversations"]
        ],
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailModel,
    summary="Get a conversation with full message history",
)
async def get_conversation_detail(
    conversation_id: str,
    current_user: dict = Depends(require_auth),
):
    pool = _get_pool()
    user_id  = current_user.get("sub")
    is_admin = current_user.get("role") == "admin"

    conv = await conv_service.get_conversation(pool, conversation_id, user_id, is_admin)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    rows = await pool.fetch(
        "SELECT role, content, token_count, created_at FROM messages "
        "WHERE conversation_id = $1 ORDER BY created_at ASC",
        conv["id"],
    )

    return ConversationDetailModel(
        id=str(conv["id"]), user_id=conv["user_id"], title=conv["title"],
        summary=conv.get("summary"),
        created_at=str(conv["created_at"]), updated_at=str(conv["updated_at"]),
        messages=[
            MessageModel(role=r["role"], content=r["content"],
                        token_count=r["token_count"], created_at=str(r["created_at"]))
            for r in rows
        ],
    )


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete my conversation",
)
async def delete_my_conversation(
    conversation_id: str,
    current_user: dict = Depends(require_auth),
):
    pool = _get_pool()
    user_id  = current_user.get("sub")
    is_admin = current_user.get("role") == "admin"

    deleted = await conv_service.delete_conversation(pool, conversation_id, user_id, is_admin)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"status": "deleted", "conversation_id": conversation_id}


# ── Admin endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/admin/conversations",
    response_model=ConversationsListResponse,
    summary="Admin: list any user's conversations (or everyone's)",
)
async def admin_list_conversations(
    user_id: Optional[str] = Query(None, description="Filter by username; omit for all users"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(require_role("admin")),
):
    pool = _get_pool()
    result = await conv_service.list_conversations(
        pool, user_id=current_user.get("sub"), is_admin=True,
        target_user_id=user_id, page=page, limit=limit,
    )
    return ConversationsListResponse(
        total=result["total"], page=result["page"], limit=result["limit"],
        conversations=[
            ConversationModel(
                id=str(c["id"]), user_id=c["user_id"], title=c["title"],
                summary=c.get("summary"),
                created_at=str(c["created_at"]), updated_at=str(c["updated_at"]),
            ) for c in result["conversations"]
        ],
    )


@router.delete(
    "/admin/conversations/{conversation_id}",
    summary="Admin: delete any user's conversation",
)
async def admin_delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(require_role("admin")),
):
    pool = _get_pool()
    deleted = await conv_service.delete_conversation(
        pool, conversation_id, user_id=current_user.get("sub"), is_admin=True,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}


@router.post(
    "/admin/conversations/cleanup",
    response_model=CleanupResponse,
    summary="Admin: manually trigger retention cleanup",
    description=(
        "Deletes conversations older than CONVERSATION_RETENTION_DAYS. "
        "Runs automatically per the configured schedule, but can be "
        "triggered manually for testing or immediate cleanup."
    ),
)
async def admin_trigger_cleanup(
    current_user: dict = Depends(require_role("admin")),
):
    pool = _get_pool()
    deleted_count = await conv_service.cleanup_old_conversations(pool)
    return CleanupResponse(
        deleted_count=deleted_count,
        retention_days=conv_service.CONVERSATION_RETENTION_DAYS,
    )