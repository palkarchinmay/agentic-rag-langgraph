"""
tests/test_conversation_service.py — Unit tests for Step 4 Part B.

All tests mock the asyncpg pool — no live Postgres connection needed.
Run: pytest tests/test_conversation_service.py -v
"""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.services.conversation_service as cs


def _mock_pool():
    """A MagicMock with the asyncpg pool methods we use, all as AsyncMock."""
    pool = MagicMock()
    pool.execute  = AsyncMock(return_value=None)
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=0)
    pool.fetch    = AsyncMock(return_value=[])
    return pool


# ── Token counting ─────────────────────────────────────────────────────────────

class TestTokenCounting:
    def test_count_tokens_empty_string(self):
        assert cs.count_tokens("") == 0

    def test_count_tokens_returns_positive_int(self):
        n = cs.count_tokens("What is the revenue growth in Q4?")
        assert isinstance(n, int)
        assert n > 0

    def test_count_tokens_longer_text_more_tokens(self):
        short = cs.count_tokens("Hello")
        long  = cs.count_tokens("Hello, this is a much longer sentence with many more words in it.")
        assert long > short

    def test_count_tokens_unknown_model_falls_back(self):
        # Should not raise even for a model tiktoken doesn't recognize
        n = cs.count_tokens("test text", model="some-future-model-xyz")
        assert isinstance(n, int)
        assert n > 0

    def test_encoding_cache_reused(self):
        cs._encoding_cache.clear()
        cs.count_tokens("first call", model="gpt-4o-mini")
        assert "gpt-4o-mini" in cs._encoding_cache
        cached_encoding = cs._encoding_cache["gpt-4o-mini"]
        cs.count_tokens("second call", model="gpt-4o-mini")
        assert cs._encoding_cache["gpt-4o-mini"] is cached_encoding


# ── Schema setup ──────────────────────────────────────────────────────────────

class TestEnsureTables:
    @pytest.mark.asyncio
    async def test_ensure_tables_noop_without_pool(self):
        # Should not raise when pool is None
        await cs.ensure_conversation_tables(None)

    @pytest.mark.asyncio
    async def test_ensure_tables_calls_execute(self):
        pool = _mock_pool()
        await cs.ensure_conversation_tables(pool)
        pool.execute.assert_called_once()
        sql = pool.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS conversations" in sql
        assert "CREATE TABLE IF NOT EXISTS messages" in sql


# ── Conversation CRUD ─────────────────────────────────────────────────────────

class TestConversationCRUD:
    @pytest.mark.asyncio
    async def test_create_conversation_returns_id(self):
        pool = _mock_pool()
        fake_id = uuid.uuid4()
        pool.fetchrow = AsyncMock(return_value={"id": fake_id})

        conv_id = await cs.create_conversation(pool, "alice", "What is the revenue?")
        assert conv_id == str(fake_id)

    @pytest.mark.asyncio
    async def test_create_conversation_truncates_long_title(self):
        pool = _mock_pool()
        fake_id = uuid.uuid4()
        pool.fetchrow = AsyncMock(return_value={"id": fake_id})

        long_msg = "x" * 100
        await cs.create_conversation(pool, "alice", long_msg)

        call_args = pool.fetchrow.call_args[0]
        title_param = call_args[2]
        assert len(title_param) <= 53  # 50 chars + "..."
        assert title_param.endswith("...")

    @pytest.mark.asyncio
    async def test_create_conversation_default_title_when_empty(self):
        pool = _mock_pool()
        fake_id = uuid.uuid4()
        pool.fetchrow = AsyncMock(return_value={"id": fake_id})

        await cs.create_conversation(pool, "alice", "")
        call_args = pool.fetchrow.call_args[0]
        assert call_args[2] == "New conversation"

    @pytest.mark.asyncio
    async def test_add_message_returns_token_count(self):
        pool = _mock_pool()
        conv_id = str(uuid.uuid4())

        tokens = await cs.add_message(pool, conv_id, "user", "What is the revenue growth?")
        assert tokens > 0
        assert pool.execute.call_count == 2  # INSERT message + UPDATE conversations

    @pytest.mark.asyncio
    async def test_get_conversation_returns_none_if_not_found(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value=None)

        result = await cs.get_conversation(pool, str(uuid.uuid4()), "alice")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_conversation_denies_access_to_other_users(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"user_id": "bob", "id": uuid.uuid4()})

        result = await cs.get_conversation(pool, str(uuid.uuid4()), "alice", is_admin=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_conversation_admin_can_access_others(self):
        fake_id = uuid.uuid4()
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"user_id": "bob", "id": fake_id})

        result = await cs.get_conversation(pool, str(fake_id), "alice", is_admin=True)
        assert result is not None
        assert result["user_id"] == "bob"

    @pytest.mark.asyncio
    async def test_get_conversation_owner_can_access_own(self):
        fake_id = uuid.uuid4()
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"user_id": "alice", "id": fake_id})

        result = await cs.get_conversation(pool, str(fake_id), "alice", is_admin=False)
        assert result is not None

    @pytest.mark.asyncio
    async def test_list_conversations_own_only_for_regular_user(self):
        pool = _mock_pool()
        pool.fetchval = AsyncMock(return_value=2)
        pool.fetch    = AsyncMock(return_value=[
            {"id": uuid.uuid4(), "user_id": "alice", "title": "t1",
             "summary": None, "created_at": datetime.now(), "updated_at": datetime.now()},
        ])

        result = await cs.list_conversations(pool, "alice", is_admin=False)
        assert result["total"] == 2
        # Confirm WHERE clause filtered by user_id (param included)
        fetchval_call = pool.fetchval.call_args[0]
        assert "alice" in fetchval_call

    @pytest.mark.asyncio
    async def test_list_conversations_admin_sees_all_without_filter(self):
        pool = _mock_pool()
        pool.fetchval = AsyncMock(return_value=10)
        pool.fetch    = AsyncMock(return_value=[])

        result = await cs.list_conversations(pool, "admin_user", is_admin=True)
        fetchval_sql = pool.fetchval.call_args[0][0]
        assert "1=1" in fetchval_sql  # no user filter applied

    @pytest.mark.asyncio
    async def test_list_conversations_admin_can_filter_by_target_user(self):
        pool = _mock_pool()
        pool.fetchval = AsyncMock(return_value=3)
        pool.fetch    = AsyncMock(return_value=[])

        await cs.list_conversations(pool, "admin_user", is_admin=True, target_user_id="bob")
        fetchval_call = pool.fetchval.call_args[0]
        assert "bob" in fetchval_call

    @pytest.mark.asyncio
    async def test_delete_conversation_returns_false_if_not_found(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value=None)

        result = await cs.delete_conversation(pool, str(uuid.uuid4()), "alice")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_conversation_succeeds_for_owner(self):
        fake_id = uuid.uuid4()
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"user_id": "alice", "id": fake_id})

        result = await cs.delete_conversation(pool, str(fake_id), "alice")
        assert result is True
        # Confirm DELETE was actually called
        delete_calls = [c for c in pool.execute.call_args_list if "DELETE" in c[0][0]]
        assert len(delete_calls) == 1

    @pytest.mark.asyncio
    async def test_delete_conversation_denied_for_non_owner_non_admin(self):
        fake_id = uuid.uuid4()
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"user_id": "bob", "id": fake_id})

        result = await cs.delete_conversation(pool, str(fake_id), "alice", is_admin=False)
        assert result is False


# ── Context window building ───────────────────────────────────────────────────

class TestBuildContext:
    @pytest.mark.asyncio
    async def test_build_context_empty_conversation(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value=None)
        pool.fetch    = AsyncMock(return_value=[])

        context = await cs.build_context(pool, str(uuid.uuid4()))
        assert context == []

    @pytest.mark.asyncio
    async def test_build_context_returns_chronological_order(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"summary": None})
        # fetch returns NEWEST first (as the real SQL does — ORDER BY created_at DESC)
        pool.fetch = AsyncMock(return_value=[
            {"role": "assistant", "content": "Second answer", "token_count": 5},
            {"role": "user",      "content": "Second question", "token_count": 5},
            {"role": "assistant", "content": "First answer", "token_count": 5},
            {"role": "user",      "content": "First question", "token_count": 5},
        ])

        context = await cs.build_context(pool, str(uuid.uuid4()), max_tokens=1000)
        # Should be reversed back to chronological (oldest first)
        assert context[0]["content"] == "First question"
        assert context[-1]["content"] == "Second answer"

    @pytest.mark.asyncio
    async def test_build_context_respects_token_budget(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"summary": None})
        pool.fetch = AsyncMock(return_value=[
            {"role": "user", "content": "msg3", "token_count": 100},
            {"role": "user", "content": "msg2", "token_count": 100},
            {"role": "user", "content": "msg1", "token_count": 100},
        ])

        # Budget only fits 2 messages (200 tokens), not all 3 (300 tokens)
        context = await cs.build_context(pool, str(uuid.uuid4()), max_tokens=200)
        assert len(context) == 2
        # Should keep the MOST RECENT ones (msg2, msg3), drop the oldest (msg1)
        contents = [m["content"] for m in context]
        assert "msg1" not in contents
        assert "msg2" in contents
        assert "msg3" in contents

    @pytest.mark.asyncio
    async def test_build_context_prepends_summary_when_present(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"summary": "User discussed Q4 revenue."})
        pool.fetch    = AsyncMock(return_value=[
            {"role": "user", "content": "recent msg", "token_count": 5},
        ])

        context = await cs.build_context(pool, str(uuid.uuid4()), max_tokens=1000)
        assert context[0]["role"] == "system"
        assert "Q4 revenue" in context[0]["content"]
        assert context[-1]["content"] == "recent msg"

    @pytest.mark.asyncio
    async def test_build_context_no_summary_key_omitted(self):
        pool = _mock_pool()
        pool.fetchrow = AsyncMock(return_value={"summary": None})
        pool.fetch    = AsyncMock(return_value=[
            {"role": "user", "content": "msg", "token_count": 5},
        ])

        context = await cs.build_context(pool, str(uuid.uuid4()))
        assert all(m["role"] != "system" for m in context)


# ── Summarization ──────────────────────────────────────────────────────────────

class TestSummarization:
    @pytest.mark.asyncio
    async def test_should_summarize_false_under_threshold(self):
        pool = _mock_pool()
        pool.fetchval = AsyncMock(return_value=5)

        result = await cs.should_summarize(pool, str(uuid.uuid4()))
        assert result is False

    @pytest.mark.asyncio
    async def test_should_summarize_true_at_threshold(self):
        pool = _mock_pool()
        pool.fetchval = AsyncMock(return_value=cs.SUMMARIZE_THRESHOLD_TURNS)

        result = await cs.should_summarize(pool, str(uuid.uuid4()))
        assert result is True

    @pytest.mark.asyncio
    async def test_summarize_returns_none_when_not_enough_messages(self):
        pool = _mock_pool()
        pool.fetch = AsyncMock(return_value=[
            {"id": 1, "role": "user", "content": "a"},
            {"id": 2, "role": "assistant", "content": "b"},
        ])

        async def fake_llm(prompt):
            return "summary text"

        result = await cs.summarize_old_messages(pool, str(uuid.uuid4()), fake_llm, keep_recent=10)
        assert result is None

    @pytest.mark.asyncio
    async def test_summarize_calls_llm_and_stores_result(self):
        pool = _mock_pool()
        rows = [{"id": i, "role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
                for i in range(15)]
        pool.fetch    = AsyncMock(return_value=rows)
        pool.fetchrow = AsyncMock(return_value={"summary": None})

        captured_prompt = {}
        async def fake_llm(prompt):
            captured_prompt["text"] = prompt
            return "This is a generated summary."

        result = await cs.summarize_old_messages(pool, str(uuid.uuid4()), fake_llm, keep_recent=10)

        assert result == "This is a generated summary."
        assert "msg0" in captured_prompt["text"]  # older messages included in summarization prompt
        # Confirm UPDATE (store summary) and DELETE (remove summarized rows) both happened
        update_calls = [c for c in pool.execute.call_args_list if "UPDATE conversations" in c[0][0]]
        delete_calls = [c for c in pool.execute.call_args_list if "DELETE FROM messages" in c[0][0]]
        assert len(update_calls) == 1
        assert len(delete_calls) == 1

    @pytest.mark.asyncio
    async def test_summarize_keeps_most_recent_messages_untouched(self):
        pool = _mock_pool()
        rows = [{"id": i, "role": "user", "content": f"msg{i}"} for i in range(15)]
        pool.fetch    = AsyncMock(return_value=rows)
        pool.fetchrow = AsyncMock(return_value={"summary": None})

        async def fake_llm(prompt):
            return "summary"

        await cs.summarize_old_messages(pool, str(uuid.uuid4()), fake_llm, keep_recent=10)

        delete_call = [c for c in pool.execute.call_args_list if "DELETE FROM messages" in c[0][0]][0]
        deleted_ids = delete_call[0][1]
        # Should delete ids 0-4 (15 - 10 = 5 oldest), keep ids 5-14
        assert deleted_ids == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_summarize_includes_existing_summary_in_prompt(self):
        pool = _mock_pool()
        rows = [{"id": i, "role": "user", "content": f"msg{i}"} for i in range(15)]
        pool.fetch    = AsyncMock(return_value=rows)
        pool.fetchrow = AsyncMock(return_value={"summary": "Earlier summary text"})

        captured = {}
        async def fake_llm(prompt):
            captured["prompt"] = prompt
            return "new summary"

        await cs.summarize_old_messages(pool, str(uuid.uuid4()), fake_llm, keep_recent=10)
        assert "Earlier summary text" in captured["prompt"]


# ── Retention cleanup ────────────────────────────────────────────────────────

class TestRetentionCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_when_nothing_old(self):
        pool = _mock_pool()
        pool.fetch = AsyncMock(return_value=[])

        count = await cs.cleanup_old_conversations(pool)
        assert count == 0
        # DELETE should not be called if nothing matched
        delete_calls = [c for c in pool.execute.call_args_list if "DELETE" in c[0][0]]
        assert len(delete_calls) == 0

    @pytest.mark.asyncio
    async def test_cleanup_deletes_old_conversations(self):
        pool = _mock_pool()
        old_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
        pool.fetch = AsyncMock(return_value=[{"id": i} for i in old_ids])

        count = await cs.cleanup_old_conversations(pool)
        assert count == 3
        pool.execute.assert_called_once()
        deleted_ids_param = pool.execute.call_args[0][1]
        assert deleted_ids_param == old_ids

    @pytest.mark.asyncio
    async def test_cleanup_uses_configured_retention_days(self):
        pool = _mock_pool()
        pool.fetch = AsyncMock(return_value=[])

        with patch.object(cs, "CONVERSATION_RETENTION_DAYS", 2):
            await cs.cleanup_old_conversations(pool)

        fetch_call = pool.fetch.call_args[0]
        cutoff_param = fetch_call[1]
        expected_cutoff = datetime.now(timezone.utc) - timedelta(days=2)
        # Allow a few seconds of tolerance for test execution time
        assert abs((cutoff_param - expected_cutoff).total_seconds()) < 5