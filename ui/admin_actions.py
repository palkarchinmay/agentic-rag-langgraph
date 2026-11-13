"""
ui/admin_actions.py — "⚙️ Admin Panel" chat profile: system operations.

Covers: usage stats, recent logs, semantic cache management, Qdrant
collection management, and conversation cleanup.

User account management (create/list/promote/disable/delete) moved to its
own "👥 User Management" chat profile — see ui/user_management_actions.py.
Usage tracking and quota management moved to "💰 Usage & FinOps" — see
ui/usage_actions.py.

BUGFIX (vs. original chainlit_app.py): there were two
@cl.action_callback("clear_all_conversations") handlers defined under the
same name (on_clear_all_conversations). Chainlit — and Python itself, since
they shared a name at module scope — only keeps the second one, which was
missing the `role != "admin"` guard the first one had. This version keeps
the guard. If you actually want any authenticated user able to wipe all
conversations, tell me and I'll drop the check back out explicitly (rather
than have it be an accident of definition order).
"""

import logging

import chainlit as cl

from app.services.api_client import get_logs, get_stats, APIError, API_URL
from ui.helpers import convert_ts

logger = logging.getLogger("financial_rag.chainlit")


# ── Admin Panel renderer ───────────────────────────────────────────────────

async def render_admin_panel():
    """Render the admin panel as interactive messages."""
    token = cl.user_session.get("token", "")

    await cl.Message(content="## ⚙️ Admin Panel\n\nSelect an action:").send()

    # Quick stats
    try:
        stats = get_stats(token, hours=24)
        await cl.Message(content=(
            f"### 📊 Usage Statistics (Last 24h)\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Total Requests | {stats.get('total_requests', 0)} |\n"
            f"| Error Rate | {stats.get('error_rate_pct', 0):.1f}% |\n"
            f"| Avg Latency | {stats.get('avg_latency_ms', 0):.0f}ms |\n"
            f"| Total Tokens | {stats.get('total_tokens', 0):,} |\n"
            f"| Total Cost | ${stats.get('total_cost_usd', 0):.4f} |\n"
            f"| Cache Hit Rate | {stats.get('cache_hit_rate_pct', 0):.1f}% |"
        ), actions=[
            cl.Action(name="refresh_stats", label="🔄 Refresh Stats", payload={}),
        ]).send()
    except APIError as e:
        err_str = str(e)
        if "503" in err_str or "AUDIT_DB" in err_str:
            await cl.Message(content=(
                "⚠️ **Stats unavailable** — database not reachable from local environment.\n\n"
                "_Stats will be available once deployed to production (Cloud Run → Supabase)._"
            )).send()
        else:
            await cl.Message(content=f"Could not load stats: {e}").send()

    # Recent logs — auto-loaded on entry
    try:
        result = get_logs(token, limit=10)
        entries = result.get("entries", [])
        total = result.get("total", len(entries))
        if entries:
            from ui.usage_actions import _build_uid_lookup, _display_name
            lookup = await _build_uid_lookup(token)

            lines = [f"### 📋 Recent Logs (last {len(entries)} of {total})\n",
                     "| Trace | User | Time | Endpoint | Latency | Tokens | Status |",
                     "|---|---|---|---|---|---|---|"]
            for e in entries:
                icon = "✅" if e.get("success") else "❌"
                tz_name = cl.user_session.get("config", {}).get("timezone", "Asia/Kolkata")
                timestamp = convert_ts(str(e.get("timestamp", "")), tz_name)
                name = _display_name(e.get("user_id", "?"), lookup)
                lines.append(
                    f"| `{e.get('trace_id','?')}` "
                    f"| {name} "
                    f"| {timestamp} "
                    f"| {e.get('endpoint','?')} "
                    f"| {e.get('latency_ms',0)}ms "
                    f"| {e.get('tokens_total',0)} "
                    f"| {icon} |"
                )
            lines.append(
                "\n💡 **Filter logs using `/logs` command:**\n\n"
                "| Filter | Example | Description |\n"
                "|---|---|---|\n"
                "| `user=X` | `/logs user=admin` | Logs for a specific user |\n"
                "| `endpoint=X` | `/logs endpoint=/query` | Logs for a specific endpoint |\n"
                "| `errors=true` | `/logs errors=true` | Only failed requests |\n"
                "| `from=DATE` | `/logs from=2026-07-10` | Logs from a date (YYYY-MM-DD) |\n"
                "| `to=DATE` | `/logs to=2026-07-13` | Logs up to a date |\n"
                "| `page=N` | `/logs page=2` | Next page of results |\n"
                "| Combined | `/logs user=admin from=2026-07-10 errors=true` | Multiple filters |\n"
                "| `limit=N` | `/logs limit=50` | Number of results (default 20) |"
            )
            await cl.Message(
                content="\n".join(lines),
                actions=[
                    cl.Action(name="view_logs", label="📋 Load More (50)",
                              payload={"limit": 50}),
                ]
            ).send()
        else:
            await cl.Message(content="### 📋 Recent Logs\n\nNo logs yet.").send()
    except APIError as e:
        await cl.Message(content=f"### 📋 Recent Logs\n\nCould not load: {e}").send()

    # Cache management
    await cl.Message(
        content="### 🗄️ Cache Management",
        actions=[
            cl.Action(name="clear_all_cache", label="🗑️ Clear All Semantic Cache",
                      payload={}, tooltip="Clears cached query results for all collections"),
            cl.Action(name="view_cache_keys", label="📋 View Cache Keys",
                      payload={}, tooltip="Show all cached query keys"),
        ]
    ).send()

    # Index management
    await cl.Message(
        content="### 📚 Index Management",
        actions=[
            cl.Action(name="view_collections", label="📊 View Collections",
                      payload={}, tooltip="Show all Qdrant collections and point counts"),
            cl.Action(name="clear_collection", label="🗑️ Clear My Collection",
                      payload={}, tooltip="Delete all indexed documents for current user"),
        ]
    ).send()

    # Conversation management (stays here — system-level data ops, distinct
    # from user account management, which lives in its own "👥 User
    # Management" chat profile now — see ui/user_management_actions.py)
    await cl.Message(
        content="### 💬 Conversation Management",
        actions=[
            cl.Action(name="view_conversations", label="💬 View Conversations", payload={}),
            cl.Action(name="cleanup_conversations", label="🗑️ Cleanup Old (>2 days)", payload={}),
            cl.Action(name="clear_my_conversations", label="🗑️ Clear My Conversations",
                      payload={}, tooltip="Delete ALL your conversations regardless of age"),
            cl.Action(name="clear_all_conversations", label="🗑️ Clear ALL Conversations",
                      payload={}, tooltip="Admin only: delete ALL conversations for all users"),
        ]
    ).send()


# ── Admin Panel action callbacks ───────────────────────────────────────────

@cl.action_callback("clear_all_cache")
async def on_clear_all_cache(action: cl.Action):
    """Clear all semantic cache entries from Redis."""
    try:
        from app.graph.nodes.ragas_node import _get_redis
        r = _get_redis()
        if not r:
            await cl.Message(content="❌ Redis not available (CACHE_BACKEND=local)").send()
            return
        keys = r.keys("semcache:*")
        if keys:
            r.delete(*keys)
            await cl.Message(content=(
                f"✅ **Semantic cache cleared**\n\n"
                f"- Deleted {len(keys)} cache collection(s): {', '.join(f'`{k}`' for k in keys)}\n"
                f"- Next queries will fetch fresh results from Qdrant"
            )).send()
        else:
            await cl.Message(content="ℹ️ Cache is already empty — no entries to clear.").send()
    except Exception as e:
        await cl.Message(content=f"❌ Cache clear failed: {e}").send()


@cl.action_callback("view_cache_keys")
async def on_view_cache_keys(action: cl.Action):
    """Show all semantic cache keys."""
    try:
        from app.graph.nodes.ragas_node import _get_redis
        r = _get_redis()
        if not r:
            await cl.Message(content="❌ Redis not available.").send()
            return
        sem_keys = r.keys("semcache:*")
        ragas_keys = r.keys("ragas:*")
        sess_keys = r.keys("session_state:*")
        lines = ["📋 **Redis Cache Contents**\n"]
        lines.append(f"**Semantic cache** ({len(sem_keys)} collections):")
        for k in sem_keys:
            size = r.hlen(k) if r.type(k) == "hash" else "?"
            lines.append(f"  - `{k}` ({size} entries)")
        lines.append(f"\n**RAGAS results** ({len(ragas_keys)} keys)")
        lines.append(f"**Session state** ({len(sess_keys)} users)")
        await cl.Message(content="\n".join(lines)).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()


@cl.action_callback("view_collections")
async def on_view_collections(action: cl.Action):
    """Show all Qdrant collections and point counts."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from qdrant_client import QdrantClient
        import httpx, os
        token = cl.user_session.get("token", "")
        api_url = cl.user_session.get("config", {}).get("api_url", API_URL)

        # Try via API first
        r = httpx.get(f"{api_url}/api/v1/admin/collections",
                      headers={"Authorization": f"Bearer {token}"}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            collections = data.get("collections", [])
        else:
            # Direct Qdrant access
            client = QdrantClient(
                url=os.getenv("QDRANT_URL", ""),
                api_key=os.getenv("QDRANT_API_KEY", ""),
            )
            cols = client.get_collections().collections
            collections = []
            for c in cols:
                info = client.get_collection(c.name)
                collections.append({
                    "name": c.name,
                    "points": info.points_count,
                    "vectors": info.config.params.vectors.size,
                })

        lines = ["📚 **Qdrant Collections**\n",
                 "| Collection | Points | Vector Dim |",
                 "|---|---|---|"]
        for c in collections:
            lines.append(
                f"| `{c.get('name')}` | {c.get('points', 0):,} | {c.get('vectors', '?')} |"
            )
        total = sum(c.get("points", 0) for c in collections)
        lines.append(f"\n_Total: {len(collections)} collection(s), {total:,} points_")
        await cl.Message(content="\n".join(lines)).send()
    except Exception as e:
        await cl.Message(content=f"❌ Could not load collections: {e}").send()


@cl.action_callback("clear_collection")
async def on_clear_collection(action: cl.Action):
    """Clear the current user's Qdrant collection, cache, and session state."""
    username = cl.user_session.get("username", "default")
    collection = cl.user_session.get("collection_name", f"user_{username}")
    try:
        from app.graph.nodes.ragas_node import _get_redis
        import os
        from qdrant_client import QdrantClient
        # Lazy import to avoid a circular import with settings_actions.py
        from ui.settings_actions import rebuild_settings_widget

        r = _get_redis()
        cache_deleted = 0
        if r:
            keys = r.keys(f"semcache:{collection}*")
            if keys:
                cache_deleted = r.delete(*keys)
            r.delete(f"session_state:{username}")

        client = QdrantClient(
            url=os.getenv("QDRANT_URL", ""),
            api_key=os.getenv("QDRANT_API_KEY", ""),
        )
        try:
            client.delete_collection(collection)
        except Exception:
            pass  # Collection may not exist

        cl.user_session.set("indexed_docs", [])
        cl.user_session.set("collection_name", f"user_{username}")

        await rebuild_settings_widget(cl.user_session.get("config", {}))

        await cl.Message(content=(
            f"✅ **Collection `{collection}` cleared**\n\n"
            f"- Qdrant collection deleted\n"
            f"- Semantic cache cleared ({cache_deleted} entries)\n"
            f"- Session state cleared (indexed docs removed)\n\n"
            f"_Upload new documents to re-index._"
        )).send()
    except Exception as e:
        await cl.Message(content=f"❌ Clear failed: {e}").send()


@cl.action_callback("refresh_stats")
async def on_refresh_stats(action: cl.Action):
    token = cl.user_session.get("token", "")
    try:
        stats = get_stats(token, hours=24)
        await cl.Message(content=(
            f"📊 **Updated Stats (Last 24h)**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Total Requests | {stats.get('total_requests', 0)} |\n"
            f"| Error Rate | {stats.get('error_rate_pct', 0):.1f}% |\n"
            f"| Avg Latency | {stats.get('avg_latency_ms', 0):.0f}ms |\n"
            f"| Total Tokens | {stats.get('total_tokens', 0):,} |\n"
            f"| Total Cost | ${stats.get('total_cost_usd', 0):.4f} |\n"
            f"| Cache Hit Rate | {stats.get('cache_hit_rate_pct', 0):.1f}% |"
        )).send()
    except APIError as e:
        await cl.Message(content=f"Could not refresh stats: {e}").send()


@cl.action_callback("view_logs")
async def on_view_logs(action: cl.Action):
    token = cl.user_session.get("token", "")
    limit = action.payload.get("limit", 20)
    filter_user = action.payload.get("filter_user", "")
    try:
        result = get_logs(token, limit=limit)
        entries = result.get("entries", [])
        if not entries:
            await cl.Message(content="No logs found.").send()
            return

        if filter_user:
            entries = [e for e in entries if e.get("user_id", "") == filter_user]

        all_users = sorted(set(e.get("user_id", "?") for e in result.get("entries", [])))

        from ui.usage_actions import _build_uid_lookup, _display_name
        lookup = await _build_uid_lookup(token)

        lines = [f"📋 **Recent Logs** (last {len(entries)})\n",
                 "| Trace | User | Time | Endpoint | Latency | Tokens | Status |",
                 "|---|---|---|---|---|---|---|"]
        for e in entries:
            icon = "✅" if e.get("success") else "❌"
            timestamp = str(e.get("timestamp", ""))[:16].replace("T", " ")
            user = _display_name(e.get("user_id", "?"), lookup)
            lines.append(
                f"| `{e.get('trace_id','?')}` "
                f"| {user} "
                f"| {timestamp} "
                f"| {e.get('endpoint','?')} "
                f"| {e.get('latency_ms',0)}ms "
                f"| {e.get('tokens_total',0)} "
                f"| {icon} |"
            )

        actions = []
        for u in all_users:
            u_name = _display_name(u, lookup)
            actions.append(cl.Action(
                name="view_logs",
                label=f"👤 {u_name[:25]}",
                payload={"limit": limit, "filter_user": u},  # keep raw uid for filtering
                tooltip=f"Show logs for {u_name} only",
            ))
        if filter_user:
            actions.append(cl.Action(
                name="view_logs",
                label="👥 All Users",
                payload={"limit": limit, "filter_user": ""},
                tooltip="Show logs for all users",
            ))
        actions.append(cl.Action(
            name="view_logs",
            label="📋 Load More (50)",
            payload={"limit": 50, "filter_user": filter_user},
        ))

        header = f"📋 **Logs** — {f'User: {_display_name(filter_user, lookup)}' if filter_user else 'All Users'}\n"
        await cl.Message(
            content=header + "\n".join(lines),
            actions=actions
        ).send()
    except APIError as e:
        await cl.Message(content=f"Could not load logs: {e}").send()


@cl.action_callback("clear_my_conversations")
async def on_clear_my_conversations(action: cl.Action):
    """Delete all conversations for the current user."""
    import httpx
    token = cl.user_session.get("token", "")
    username = cl.user_session.get("username", "")
    api_url = cl.user_session.get("config", {}).get("api_url", API_URL)
    try:
        r = httpx.delete(
            f"{api_url}/api/v1/conversations/user/{username}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code in (200, 204):
            data = r.json() if r.content else {}
            deleted = data.get("deleted_count", "all")
            cl.user_session.set("conversation_id", None)
            await cl.Message(content=(
                f"✅ **Your conversations cleared**\n\n"
                f"- Deleted: {deleted} conversation(s) for user `{username}`\n"
                f"- Current session conversation reset"
            )).send()
        else:
            r2 = httpx.post(
                f"{api_url}/api/v1/admin/conversations/cleanup",
                json={"user_id": username, "retention_days": 0},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if r2.status_code == 200:
                data = r2.json()
                cl.user_session.set("conversation_id", None)
                await cl.Message(content=(
                    f"✅ **Conversations cleared**\n\n"
                    f"Deleted: {data.get('deleted_count', 0)} conversation(s)"
                )).send()
            else:
                await cl.Message(content=(
                    "⚠️ **Direct deletion not available via API**\n\n"
                    "The API doesn't have a per-user delete endpoint yet.\n\n"
                    "**Manual options:**\n"
                    "- Run SQL in Supabase dashboard:\n"
                    "```sql\n"
                    f"DELETE FROM conversations WHERE user_id = '{username}';\n"
                    "```\n"
                    "- Or wait 2 days for automatic cleanup"
                )).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()


@cl.action_callback("clear_all_conversations")
async def on_clear_all_conversations(action: cl.Action):
    """Admin only: delete ALL conversations for all users, regardless of age."""
    import httpx
    token = cl.user_session.get("token", "")
    role = cl.user_session.get("role", "")
    api_url = cl.user_session.get("config", {}).get("api_url", API_URL)

    if role != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    try:
        r = httpx.post(
            f"{api_url}/api/v1/admin/conversations/cleanup",
            json={"retention_days": 0},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            deleted = data.get("deleted_count", 0)
            cl.user_session.set("conversation_id", None)
            await cl.Message(content=(
                f"🗑️ **All Conversations Cleared**\n\n"
                f"- Deleted: {deleted} conversation(s)\n"
                f"- All messages and history removed for all users\n\n"
                f"_Current active conversations may still appear until the next query._"
            )).send()
        else:
            await cl.Message(content=(
                f"⚠️ API returned HTTP {r.status_code}\n\n"
                f"To manually clear all conversations, run:\n"
                f"```sql\n"
                f"DELETE FROM messages;\n"
                f"DELETE FROM conversations;\n"
                f"```\n"
                f"in your Supabase SQL editor at supabase.com"
            )).send()
    except Exception as e:
        await cl.Message(content=(
            f"❌ Could not clear conversations: {e}\n\n"
            f"**Manual option:** Go to supabase.com → SQL Editor and run:\n"
            f"```sql\n"
            f"DELETE FROM messages;\n"
            f"DELETE FROM conversations;\n"
            f"```"
        )).send()




@cl.action_callback("view_conversations")
async def on_view_conversations(action: cl.Action):
    import httpx
    token = cl.user_session.get("token", "")
    api_url = cl.user_session.get("config", {}).get("api_url", API_URL)

    try:
        r = httpx.get(
            f"{api_url}/api/v1/admin/conversations?limit=10",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            convs = data.get("conversations", [])
            total = data.get("total", 0)
            if not convs:
                await cl.Message(content="No conversations found.").send()
                return

            from ui.usage_actions import _build_uid_lookup, _display_name
            lookup = await _build_uid_lookup(token)

            lines = [f"💬 **Conversations** (showing {len(convs)} of {total})\n",
                     "| ID | User | Title | Date | Time |",
                     "|---|---|---|---|---|"]
            for c in convs:
                cid = str(c.get("id", ""))[:8]
                user = _display_name(c.get("user_id", "?"), lookup)
                title = c.get("title", "Untitled")[:35]
                updated_at = str(c.get("updated_at", ""))
                tz_name = cl.user_session.get("config", {}).get("timezone", "Asia/Kolkata")
                ts_converted = convert_ts(updated_at, tz_name)
                date_part = ts_converted[:10] if len(ts_converted) >= 10 else "—"
                time_part = ts_converted[11:] if len(ts_converted) > 10 else "—"
                lines.append(f"| `{cid}...` | {user} | {title} | {date_part} | {time_part} |")
            await cl.Message(content="\n".join(lines)).send()
        elif r.status_code == 503:
            await cl.Message(content=(
                "⚠️ **Conversations unavailable** — database not reachable from local environment.\n\n"
                "_Conversations will be available once deployed to production._"
            )).send()
        else:
            await cl.Message(content=f"Could not load conversations: HTTP {r.status_code}").send()
    except Exception as e:
        await cl.Message(content=f"Error loading conversations: {e}").send()


@cl.action_callback("cleanup_conversations")
async def on_cleanup_conversations(action: cl.Action):
    import httpx
    token = cl.user_session.get("token", "")
    api_url = cl.user_session.get("config", {}).get("api_url", API_URL)

    try:
        r = httpx.post(
            f"{api_url}/api/v1/admin/conversations/cleanup",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            deleted = data.get("deleted_count", 0)
            days = data.get("retention_days", 2)
            await cl.Message(
                content=(
                    f"🗑️ **Cleanup Complete**\n\n"
                    f"- Deleted: {deleted} conversation(s)\n"
                    f"- Retention policy: {days} days\n"
                    f"- Conversations newer than {days} days were kept"
                )
            ).send()
        else:
            await cl.Message(content=f"Cleanup failed: HTTP {r.status_code}").send()
    except Exception as e:
        await cl.Message(content=f"Cleanup error: {e}").send()