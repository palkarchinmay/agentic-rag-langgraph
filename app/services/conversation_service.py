"""
app/services/conversation_service.py — Conversation persistence & context window management.

WHAT THIS SOLVES
=================
Before this: chat_history was client-side only (Streamlit's st.session_state),
resent in full on every API call, with no server-side memory. A user closing
their browser or switching devices lost everything. There was also no token
counting — prompt_node.py just took the last 6 messages regardless of size.

After this: conversations and messages are persisted in Postgres (same
Supabase instance as audit_logs, Step 4 Part A). A conversation_id can be
passed to /api/v1/query instead of (or alongside) chat_history, and the
server reconstructs a token-aware context window automatically.

DATABASE SCHEMA
=================
conversations:
    id              UUID PK
    user_id         the username/sub from JWT
    title           short auto-generated label (first ~50 chars of first message)
    created_at      timestamp
    updated_at      timestamp (bumped on every new message)

messages:
    id              BIGSERIAL PK
    conversation_id FK -> conversations.id
    role            'user' | 'assistant'
    content         the message text
    token_count     precomputed via tiktoken, so context building never
                    re-tokenizes on every request
    created_at      timestamp

CONTEXT WINDOW STRATEGY
=========================
1. Token counting happens once, at write time (see _count_tokens), not
   repeatedly at read time.
2. build_context() walks messages newest-to-oldest, accumulating token
   counts, and stops BEFORE exceeding max_tokens — a sliding window.
3. If a conversation has grown large enough that even the sliding window
   would drop a meaningful chunk of older context, summarize_old_messages()
   collapses everything older than the window into a single short LLM-
   generated summary, stored back so it isn't regenerated every turn.

RETENTION
==========
CONVERSATION_RETENTION_DAYS (default 2, configurable via env var) controls
how long messages are kept. cleanup_old_conversations() deletes anything
older than that — intended to be called periodically (see Section on
scheduling in the Step 4 documentation).
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("financial_rag.conversation_service")

CONVERSATION_RETENTION_DAYS = int(os.getenv("CONVERSATION_RETENTION_DAYS", "2"))
DEFAULT_MAX_CONTEXT_TOKENS  = int(os.getenv("CONTEXT_MAX_TOKENS", "8000"))
SUMMARIZE_THRESHOLD_TURNS   = int(os.getenv("SUMMARIZE_THRESHOLD_TURNS", "20"))


# ── Token counting ─────────────────────────────────────────────────────────────

_encoding_cache = {}


def _get_encoding(model: str = "gpt-4o-mini"):
    """Cache tiktoken encodings per model — avoids re-loading on every call."""
    if model not in _encoding_cache:
        import tiktoken
        try:
            _encoding_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Unknown/newer model name — fall back to a standard encoding
            _encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache[model]


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using the real tokenizer for the given model."""
    if not text:
        return 0
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


# ── Schema setup (mirrors api/middleware/logging.py's _ensure_table pattern) ───

async def ensure_conversation_tables(pool) -> None:
    """Create conversations/messages tables if they don't exist. Idempotent."""
    if not pool:
        return
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id     VARCHAR(100) NOT NULL,
            title       VARCHAR(200),
            summary     TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS messages (
            id               BIGSERIAL PRIMARY KEY,
            conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role             VARCHAR(20) NOT NULL,
            content          TEXT NOT NULL,
            token_count      INT NOT NULL DEFAULT 0,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, created_at ASC);
    """)
    logger.info("Conversation tables ready ✓")


# ── Conversation CRUD ─────────────────────────────────────────────────────────

async def create_conversation(pool, user_id: str, first_message: str = "") -> str:
    """Create a new conversation, return its id (as string)."""
    title = (first_message[:50] + "...") if len(first_message) > 50 else first_message
    title = title or "New conversation"

    row = await pool.fetchrow(
        """
        INSERT INTO conversations (user_id, title)
        VALUES ($1, $2)
        RETURNING id
        """,
        user_id, title,
    )
    conv_id = str(row["id"])
    logger.info("Conversation created: %s for user=%s", conv_id, user_id)
    return conv_id


async def add_message(pool, conversation_id: str, role: str, content: str,
                       model: str = "gpt-4o-mini") -> int:
    """Store a message, return its token count."""
    tokens = count_tokens(content, model)

    await pool.execute(
        """
        INSERT INTO messages (conversation_id, role, content, token_count)
        VALUES ($1, $2, $3, $4)
        """,
        uuid.UUID(conversation_id), role, content, tokens,
    )
    await pool.execute(
        "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
        uuid.UUID(conversation_id),
    )
    return tokens


async def get_conversation(pool, conversation_id: str, user_id: str,
                            is_admin: bool = False) -> Optional[Dict]:
    """
    Fetch a conversation's metadata. Returns None if not found OR if the
    caller doesn't own it and isn't an admin (prevents cross-user access).
    """
    row = await pool.fetchrow(
        "SELECT * FROM conversations WHERE id = $1", uuid.UUID(conversation_id)
    )
    if not row:
        return None
    if row["user_id"] != user_id and not is_admin:
        logger.warning(
            "Access denied: user=%s tried to access conversation owned by %s",
            user_id, row["user_id"]
        )
        return None
    return dict(row)


async def list_conversations(pool, user_id: str, is_admin: bool = False,
                              target_user_id: Optional[str] = None,
                              page: int = 1, limit: int = 20) -> Dict:
    """
    List conversations. Regular users see only their own.
    Admins can pass target_user_id to view a specific user's conversations,
    or omit it to see everyone's.
    """
    offset = (page - 1) * limit

    if is_admin:
        if target_user_id:
            where, params = "user_id = $1", [target_user_id]
        else:
            where, params = "1=1", []
    else:
        where, params = "user_id = $1", [user_id]

    total = await pool.fetchval(f"SELECT COUNT(*) FROM conversations WHERE {where}", *params)
    rows = await pool.fetch(
        f"""
        SELECT * FROM conversations WHERE {where}
        ORDER BY updated_at DESC LIMIT {limit} OFFSET {offset}
        """,
        *params,
    )
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "conversations": [dict(r) for r in rows],
    }


async def delete_conversation(pool, conversation_id: str, user_id: str,
                               is_admin: bool = False) -> bool:
    """Delete a conversation (cascades to messages). Returns True if deleted."""
    conv = await get_conversation(pool, conversation_id, user_id, is_admin)
    if not conv:
        return False

    await pool.execute("DELETE FROM conversations WHERE id = $1", uuid.UUID(conversation_id))
    logger.info("Conversation deleted: %s by user=%s (admin=%s)",
                conversation_id, user_id, is_admin)
    return True


# ── Context window building (the core Step 4 feature) ─────────────────────────

async def build_context(pool, conversation_id: str,
                         max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
                         model: str = "gpt-4o-mini") -> List[Dict]:
    """
    Build a token-aware sliding-window context for the LLM call.

    Walks messages newest-to-oldest, accumulating token counts, and stops
    BEFORE exceeding max_tokens. Returns messages in chronological order
    (oldest first) ready to feed into prompt_node.py's existing
    chat_history format: [{"role": ..., "content": ...}, ...]

    If a stored summary exists (from summarize_old_messages), it is
    prepended as a synthetic system-role message so older context isn't
    lost entirely even when the raw messages fall outside the window.
    """
    conv_row = await pool.fetchrow(
        "SELECT summary FROM conversations WHERE id = $1", uuid.UUID(conversation_id)
    )
    summary = conv_row["summary"] if conv_row else None

    rows = await pool.fetch(
        """
        SELECT role, content, token_count FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at DESC
        """,
        uuid.UUID(conversation_id),
    )

    selected: List[Dict] = []
    running_total = 0

    for row in rows:
        msg_tokens = row["token_count"] or count_tokens(row["content"], model)
        if running_total + msg_tokens > max_tokens:
            break
        selected.insert(0, {"role": row["role"], "content": row["content"]})
        running_total += msg_tokens

    if summary:
        selected.insert(0, {
            "role": "system",
            "content": f"[Earlier conversation summary]: {summary}",
        })

    logger.debug(
        "Context built for conversation=%s: %d messages, ~%d tokens",
        conversation_id, len(selected), running_total
    )
    return selected


async def should_summarize(pool, conversation_id: str) -> bool:
    """True if a conversation has grown beyond SUMMARIZE_THRESHOLD_TURNS messages."""
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = $1",
        uuid.UUID(conversation_id),
    )
    return count >= SUMMARIZE_THRESHOLD_TURNS


async def summarize_old_messages(pool, conversation_id: str, llm_call_fn,
                                  keep_recent: int = 10) -> Optional[str]:
    """
    Summarize everything except the most recent `keep_recent` messages,
    store the summary on the conversation row, and DELETE the summarized
    raw messages (their content now lives only in the summary).

    llm_call_fn: async callable(prompt: str) -> str — injected so this
    module doesn't import a specific LLM provider directly (matches the
    existing app/providers/llm factory pattern used elsewhere).

    Returns the new summary text, or None if there was nothing to summarize.
    """
    rows = await pool.fetch(
        """
        SELECT id, role, content FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at ASC
        """,
        uuid.UUID(conversation_id),
    )

    if len(rows) <= keep_recent:
        return None

    to_summarize = rows[:-keep_recent]
    existing_summary_row = await pool.fetchrow(
        "SELECT summary FROM conversations WHERE id = $1", uuid.UUID(conversation_id)
    )
    existing_summary = existing_summary_row["summary"] if existing_summary_row else None

    transcript = "\n".join(f"{r['role']}: {r['content']}" for r in to_summarize)
    prior = f"Previous summary: {existing_summary}\n\n" if existing_summary else ""

    prompt = (
        f"{prior}Summarize the following conversation turns in 2-4 concise "
        f"sentences, preserving key facts, names, and numbers mentioned:\n\n"
        f"{transcript}"
    )

    new_summary = await llm_call_fn(prompt)

    await pool.execute(
        "UPDATE conversations SET summary = $1, updated_at = NOW() WHERE id = $2",
        new_summary, uuid.UUID(conversation_id),
    )

    ids_to_delete = [r["id"] for r in to_summarize]
    await pool.execute(
        "DELETE FROM messages WHERE id = ANY($1::bigint[])",
        ids_to_delete,
    )

    logger.info(
        "Summarized %d messages into conversation=%s summary (%d chars)",
        len(to_summarize), conversation_id, len(new_summary)
    )
    return new_summary


# ── Retention cleanup ────────────────────────────────────────────────────────

async def cleanup_old_conversations(pool) -> int:
    """
    Delete conversations whose updated_at is older than
    CONVERSATION_RETENTION_DAYS. Cascades to delete their messages too.
    Returns the number of conversations deleted.

    Intended to be called periodically (e.g. a scheduled Cloud Run job,
    or triggered from an admin endpoint) — see Step 4 documentation for
    scheduling options.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=CONVERSATION_RETENTION_DAYS)

    rows = await pool.fetch(
        "SELECT id FROM conversations WHERE updated_at < $1", cutoff
    )
    if not rows:
        return 0

    ids = [r["id"] for r in rows]
    await pool.execute(
        "DELETE FROM conversations WHERE id = ANY($1::uuid[])", ids
    )
    logger.info(
        "Retention cleanup: deleted %d conversations older than %d days",
        len(ids), CONVERSATION_RETENTION_DAYS
    )
    return len(ids)