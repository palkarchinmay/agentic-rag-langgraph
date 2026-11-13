"""
ui/commands.py — Slash command dispatcher (/stats, /logs, /clear, /help,
/golden, /sidebar, /reset).

Called from ui.chat_handlers.on_message when a message starts with "/".
"""

import logging

import chainlit as cl

from app.services.api_client import get_stats, APIError, API_URL
from ui.helpers import convert_ts, save_evals
from ui.settings_actions import update_sidebar, rebuild_settings_widget
from ui.evaluation_actions import export_evaluations
from ui.query_handler import run_golden_evaluation_interactive

logger = logging.getLogger("financial_rag.chainlit")


async def handle_command(command: str, role: str):
    token = cl.user_session.get("token", "")
    cmd = command.lower().split()[0]

    if cmd == "/stats" and role == "admin":
        try:
            stats = get_stats(token, hours=24)
            await cl.Message(content=(
                f"📊 **Last 24h Stats**\n\n"
                f"- Requests: {stats.get('total_requests', 0)}\n"
                f"- Errors: {stats.get('total_errors', 0)} "
                f"({stats.get('error_rate_pct', 0):.1f}%)\n"
                f"- Avg latency: {stats.get('avg_latency_ms', 0):.0f}ms\n"
                f"- Total tokens: {stats.get('total_tokens', 0):,}\n"
                f"- Total cost: ${stats.get('total_cost_usd', 0):.4f}\n"
                f"- Cache hit rate: {stats.get('cache_hit_rate_pct', 0):.1f}%"
            )).send()
        except APIError as e:
            await cl.Message(content=f"Could not fetch stats: {e}").send()

    elif cmd == "/logs" and role in ("admin", "engineer"):
        # Filter syntax: /logs user=admin endpoint=/query errors=true from=2026-07-10 to=2026-07-11 page=2
        parts = command.strip().split()[1:]  # skip "/logs"
        filters = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                filters[k.lower()] = v

        user_id = filters.get("user", filters.get("user_id"))
        endpoint_f = filters.get("endpoint")
        errors_only = filters.get("errors", "false").lower() == "true"
        start_time = filters.get("from")
        end_time = filters.get("to")
        page = int(filters.get("page", 1))
        limit = int(filters.get("limit", 20))

        active = []
        if user_id:     active.append(f"user=`{user_id}`")
        if endpoint_f:  active.append(f"endpoint=`{endpoint_f}`")
        if errors_only: active.append("errors only")
        if start_time:  active.append(f"from `{start_time}`")
        if end_time:    active.append(f"to `{end_time}`")
        if page > 1:    active.append(f"page {page}")
        filter_desc = " · ".join(active) if active else "all logs"

        try:
            import httpx as _httpx
            api_url = cl.user_session.get("config", {}).get("api_url", API_URL)
            params = {"page": page, "limit": limit, "errors_only": errors_only}
            if user_id:    params["user_id"] = user_id
            if endpoint_f: params["endpoint"] = endpoint_f
            if start_time: params["start_time"] = start_time + "T00:00:00" if "T" not in start_time else start_time
            if end_time:   params["end_time"] = end_time + "T23:59:59" if "T" not in end_time else end_time

            r = _httpx.get(
                f"{api_url}/api/v1/admin/logs",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            result = r.json()
            entries = result.get("entries", [])
            total = result.get("total", len(entries))

            if not entries:
                await cl.Message(content=(
                    f"No logs found for: {filter_desc}\n\n"
                    f"**Filter syntax:** `/logs user=admin endpoint=/query "
                    f"errors=true from=2026-07-10 to=2026-07-11 page=2`"
                )).send()
                return

            lines = [f"📋 **Logs** ({filter_desc}) — {len(entries)} of {total}\n",
                     "| Trace | User | Time | Endpoint | Latency | Tokens | Status |",
                     "|---|---|---|---|---|---|---|"]
            for e in entries:
                icon = "✅" if e.get("success") else "❌"
                tz_name = cl.user_session.get("config", {}).get("timezone", "Asia/Kolkata")
                timestamp = convert_ts(str(e.get("timestamp", "")), tz_name)
                lines.append(
                    f"| `{e.get('trace_id','?')}` "
                    f"| {e.get('user_id','?')} "
                    f"| {timestamp} "
                    f"| {e.get('endpoint','?')} "
                    f"| {e.get('latency_ms',0)}ms "
                    f"| {e.get('tokens_total',0)} "
                    f"| {icon} |"
                )
            if total > limit * page:
                lines.append(f"\n_Showing {len(entries)} of {total}. Type `/logs page={page+1}` for next._")
            lines.append(
                "\n💡 **Available filters:** `/logs user=X endpoint=/query errors=true "
                "from=YYYY-MM-DD to=YYYY-MM-DD page=N limit=N`"
            )
            await cl.Message(content="\n".join(lines)).send()
        except Exception as e:
            await cl.Message(content=f"Could not fetch logs: {e}").send()

    elif cmd == "/clear":
        username = cl.user_session.get("username", "default")
        collection_name = cl.user_session.get("collection_name") or f"user_{username}"

        # Actually delete the server-side collection (Qdrant + BM25 + cache)
        # -- NOT just Chainlit's own session memory of "docs are indexed".
        # Confirmed missing before this fix: chunks from every prior
        # indexing run in a session were silently accumulating in the
        # same collection forever, since the collection name is always
        # user_{username} regardless of session -- meaning retrieval was
        # pulling from a growing pool of old/stale/superseded chunks
        # (old OCR-era text, old DePlot-era chart extractions, etc.)
        # alongside whatever was just re-indexed.
        delete_ok = True
        delete_error = ""
        try:
            from app.services.api_client import delete_collection
            delete_collection(token, collection_name)
        except APIError as e:
            delete_ok = False
            delete_error = str(e)
            logger.warning("Failed to delete collection '%s' during /clear: %s",
                            collection_name, e)

        cl.user_session.set("conversation_id", None)
        cl.user_session.set("indexed_docs", [])
        cl.user_session.set("collection_name", f"user_{username}")
        cl.user_session.set("evaluations", [])
        save_evals(username, [])
        # Also clear Redis session state so profile switches don't restore old data
        try:
            from app.graph.nodes.ragas_node import _get_redis
            r = _get_redis()
            if r:
                r.delete(f"session_state:{username}")
        except Exception:
            pass
        await rebuild_settings_widget(cl.user_session.get("config", {}))

        if delete_ok:
            await cl.Message(content=(
                "🗑️ **Session cleared**\n\n"
                "- Conversation reset\n"
                "- Indexed documents cleared\n"
                "- Evaluations cleared\n"
                "- Server-side collection deleted (no stale chunks left behind)\n\n"
                "_Upload new documents to start fresh._"
            )).send()
        else:
            await cl.Message(content=(
                "🗑️ **Session cleared** (client-side)\n\n"
                "- Conversation reset\n"
                "- Indexed documents cleared\n"
                "- Evaluations cleared\n\n"
                f"⚠️ **Could not delete the server-side collection**: {delete_error}\n"
                "Old chunks from previous uploads may still be searchable until "
                "this is resolved — check the API logs.\n\n"
                "_Upload new documents to start fresh._"
            )).send()

    elif cmd == "/export":
        await export_evaluations()

    elif cmd == "/create_user" and role == "admin":
        parts = command.strip().split()[1:]
        fields = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.lower()] = v

        email = fields.get("email", "")
        password = fields.get("password", "")
        new_role = fields.get("role", "user")
        full_name = fields.get("name", "").replace("_", " ")

        if not email or not password:
            await cl.Message(content=(
                "**Usage:** `/create_user email=user@example.com password=Secret123 "
                "role=user name=Full_Name`\n\n"
                "- `role` defaults to `user` if omitted — one of `admin`, `engineer`, `user`\n"
                "- `name` uses underscores for spaces (e.g. `name=Jane_Doe`) since this "
                "is a space-separated command\n"
                "- `password` must be at least 6 characters"
            )).send()
            return

        from app.services.api_client import create_user as api_create_user
        try:
            user = api_create_user(token, email, password, new_role, full_name)
            await cl.Message(content=(
                f"✅ **User created**\n\n"
                f"- Email: `{user.get('email')}`\n"
                f"- Role: `{user.get('role')}`\n"
                f"- UID: `{user.get('uid')}`"
            )).send()
        except APIError as e:
            await cl.Message(content=f"❌ Could not create user: {e}").send()

    elif cmd == "/usage" and role == "admin":
        parts = command.strip().split()[1:]
        fields = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.lower()] = v

        from app.services.api_client import get_usage as api_get_usage
        from ui.usage_actions import _build_uid_lookup, _display_name
        try:
            result = api_get_usage(
                token,
                user_id=fields.get("user"),
                from_=fields.get("from"),
                to=fields.get("to"),
            )
        except APIError as e:
            await cl.Message(content=f"❌ Could not load usage: {e}").send()
            return

        rows = result.get("rows", [])
        if not rows:
            await cl.Message(content=(
                f"No usage recorded for {result.get('period_start','?')[:10]} "
                f"to {result.get('period_end','?')[:10]}."
            )).send()
            return

        lookup = await _build_uid_lookup(token)

        lines = [f"📊 **Usage** — {result['period_start'][:10]} to {result['period_end'][:10]}\n",
                 "| User | Requests | Tokens | Est. Cost |",
                 "|---|---|---|---|"]
        for r in sorted(rows, key=lambda x: x["cost_usd"], reverse=True):
            name = _display_name(r["user_id"], lookup)
            lines.append(f"| {name} | {r['requests']} | {r['tokens']:,} | ${r['cost_usd']:.5f} |")
        lines.append(
            "\n💡 **Filters:** `/usage from=2026-08-01 to=2026-08-31 user=<uid>` "
            "— all optional, defaults to current month, all users."
        )
        await cl.Message(content="\n".join(lines)).send()

    elif cmd == "/set_quota" and role == "admin":
        parts = command.strip().split()[1:]
        fields = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.lower()] = v

        uid = fields.get("uid", "")
        if not uid:
            await cl.Message(content=(
                "**Usage:** `/set_quota uid=<uid> tokens=500000 cost=50 requests=1000`\n\n"
                "- Get a user's uid from the 💰 Usage & FinOps panel (click their "
                "name) or 👥 User Management → List Users\n"
                "- Any of `tokens`/`cost`/`requests` can be omitted, or set to `0`, "
                "for unlimited on that dimension"
            )).send()
            return

        def _to_num(s, cast):
            if s is None or s == "" or s == "0":
                return None
            return cast(s)

        from app.services.api_client import set_user_quota as api_set_quota
        try:
            quota = api_set_quota(
                token, uid,
                tokens=_to_num(fields.get("tokens"), int),
                cost=_to_num(fields.get("cost"), float),
                requests=_to_num(fields.get("requests"), int),
            )
            await cl.Message(content=(
                f"✅ **Quota updated for `{uid}`**\n\n"
                f"- Tokens: `{quota.get('monthly_token_limit') or 'unlimited'}`\n"
                f"- Cost: `{('$' + str(quota['monthly_cost_limit_usd'])) if quota.get('monthly_cost_limit_usd') else 'unlimited'}`\n"
                f"- Requests: `{quota.get('monthly_request_limit') or 'unlimited'}`"
            )).send()
        except APIError as e:
            await cl.Message(content=f"❌ Could not set quota: {e}").send()

    elif cmd == "/help":
        lines = ["**Available commands:**\n",
                 "- `/clear`   — reset conversation and indexed documents",
                 "- `/reset`   — reset all settings to defaults (use this — the settings "
                 "panel's own Reset button at the bottom doesn't reliably reset everything)",
                 "- `/sidebar` — refresh the session info panel (right side)",
                 "- `/golden`  — run golden dataset evaluation (upload CSV first)",
                 "- `/export`  — download all RAGAS evaluations as Excel",
                 "- `/help`    — show this message"]
        if role == "admin":
            lines += ["- `/stats`   — 24h usage stats (admin only)",
                      "- `/logs`    — query logs with filters",
                      "- `/create_user email=... password=... role=... name=...` — create a new user (admin only)",
                      "- `/usage from=... to=... user=...` — usage & cost summary (admin only)",
                      "- `/set_quota uid=... tokens=... cost=... requests=...` — set a user's monthly quota (admin only)"]
        elif role == "engineer":
            lines += ["- `/logs`    — query logs with filters"]
        await cl.Message(content="\n".join(lines)).send()

    elif cmd == "/golden":
        await run_golden_evaluation_interactive()

    elif cmd == "/sidebar" or cmd == "/status":
        await update_sidebar()
        await cl.Message(content="📋 Session panel refreshed — check the right sidebar.").send()

    elif cmd == "/reset":
        defaults = {
            "llm_provider": "openai", "llm_model": "gpt-4o-mini",
            "temperature": 0.0, "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "retrieval_mode": "hybrid", "top_k": 5, "rerank": True,
            "chunk_size": 500, "chunk_overlap": 50,
            "chunk_method": "recursive", "ocr_method": "easyocr",
            "vision_mode": False,
            "domain": "uploaded documents", "assistant_name": "Document Assistant",
        }
        cl.user_session.set("config", defaults)
        await rebuild_settings_widget(defaults)
        await cl.Message(content="🔄 Settings reset to defaults — open ⚙️ to confirm.").send()

    else:
        await cl.Message(
            content=f"Unknown command `{command}`. Type `/help` for help."
        ).send()