"""
app/chainlit_ui/settings_actions.py — Connections panel, ChatSettings widget,
session sidebar, and settings persistence.

Covers: the "🔌 Connections" chat profile (test_connection), the settings
sidebar (ChatSettings widget + ElementSidebar), @cl.on_settings_update, and
the reset_settings / open_sidebar action callbacks.

rebuild_settings_widget() is also imported by ui.admin_actions
(on_clear_collection) — that's a lazy import there to avoid a circular
import, since this module doesn't need anything from admin_actions.
"""

import os
import logging

import chainlit as cl

from ui.helpers import mask_url

logger = logging.getLogger("financial_rag.chainlit")


# ── Connections panel ───────────────────────────────────────────────────────

# Default connection configs (overridable in session)
_DEFAULT_CONNECTIONS = {
    "api":      {"label": "FastAPI Backend",       "env": "RAG_API_URL",  "default": "http://localhost:8000"},
    "qdrant":   {"label": "Qdrant Vector DB",       "env": "QDRANT_URL",   "default": ""},
    "redis":    {"label": "Redis / Upstash Cache",  "env": "REDIS_URL",    "default": ""},
    "supabase": {"label": "Supabase PostgreSQL",    "env": "AUDIT_DB_URL", "default": ""},
}


async def render_connections_panel():
    """Render the Connections panel — test and update all external services."""
    token = cl.user_session.get("token", "")
    config = cl.user_session.get("config", {})

    await cl.Message(content=(
        "## 🔌 Connections\n\n"
        "Test and update connections to all external services. "
        "Click **🔍 Test** to check each connection, or update the URL and click **💾 Save**."
    )).send()

    for key, meta in _DEFAULT_CONNECTIONS.items():
        current_url = config.get(f"{key}_url", os.getenv(meta["env"], meta["default"]))
        masked_url = mask_url(current_url)

        await cl.Message(
            content=(
                f"### {meta['label']}\n\n"
                f"**Current:** `{masked_url}`\n\n"
                f"_Update via the text input in ⚙️ Settings, or click Test to check the current connection._"
            ),
            actions=[
                cl.Action(
                    name="test_connection",
                    label=f"🔍 Test {meta['label']}",
                    payload={"key": key, "url": current_url, "token": token},
                ),
            ]
        ).send()

    await cl.Message(content=(
        "---\n"
        "### Update Connection URLs\n\n"
        "Use the **⚙️ Settings** panel (top-right gear icon) to update connection URLs. "
        "The Connections tab above will reflect changes after you save."
    )).send()


@cl.action_callback("test_connection")
async def on_test_connection(action: cl.Action):
    """Test a specific connection and report status."""
    key = action.payload.get("key", "")
    url = action.payload.get("url", "")
    token = action.payload.get("token", "")

    thinking = cl.Message(content=f"🔍 Testing {_DEFAULT_CONNECTIONS.get(key, {}).get('label', key)}...")
    await thinking.send()

    result = await _test_single_connection(key, url, token)
    await thinking.remove()
    await cl.Message(content=result).send()


async def _test_single_connection(key: str, url: str, token: str) -> str:
    """Test a single connection and return a status message."""
    label = _DEFAULT_CONNECTIONS.get(key, {}).get("label", key)

    if key == "api":
        import httpx
        try:
            r = httpx.get(f"{url}/health", timeout=10)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "unknown")
                return (
                    f"✅ **{label} — Connected**\n\n"
                    f"- URL: `{mask_url(url)}`\n"
                    f"- Status: `{status}`\n"
                    f"- Response: HTTP {r.status_code}"
                )
            return f"⚠️ **{label} — Unexpected response**\n\nHTTP {r.status_code}"
        except Exception as e:
            return f"❌ **{label} — Connection failed**\n\n`{e}`"

    elif key == "qdrant":
        import httpx
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        if not url:
            return f"⚠️ **{label} — Not configured**\n\n`QDRANT_URL` is not set."
        try:
            headers = {"api-key": qdrant_api_key} if qdrant_api_key else {}
            r = httpx.get(f"{url}/collections", headers=headers, timeout=10)
            if r.status_code == 200:
                collections = r.json().get("result", {}).get("collections", [])
                names = [c.get("name") for c in collections]
                return (
                    f"✅ **{label} — Connected**\n\n"
                    f"- URL: `{mask_url(url)}`\n"
                    f"- Collections: {', '.join(names) if names else '*(none yet)*'}"
                )
            return f"⚠️ **{label} — HTTP {r.status_code}**"
        except Exception as e:
            return f"❌ **{label} — Connection failed**\n\n`{e}`"

    elif key == "redis":
        redis_url = url or os.getenv("REDIS_URL", "")
        if not redis_url:
            return f"⚠️ **{label} — Not configured**\n\n`REDIS_URL` is not set."
        try:
            import redis as redis_lib
            r = redis_lib.from_url(redis_url, socket_connect_timeout=5,
                                   socket_timeout=5, decode_responses=True)
            pong = r.ping()
            info = r.info("memory")
            used_mb = round(info.get("used_memory", 0) / 1024 / 1024, 2)
            return (
                f"✅ **{label} — Connected**\n\n"
                f"- PING: `{'PONG' if pong else 'no response'}`\n"
                f"- Memory used: `{used_mb} MB`"
            )
        except Exception as e:
            return f"❌ **{label} — Connection failed**\n\n`{e}`"

    elif key == "supabase":
        db_url = url or os.getenv("AUDIT_DB_URL", "")
        if not db_url:
            return f"⚠️ **{label} — Not configured**\n\n`AUDIT_DB_URL` is not set."
        try:
            import asyncpg, socket, re
            # Force IPv4 to avoid Windows IPv6 DNS issues
            host_match = re.search(r"@([^:/]+)", db_url)
            if host_match:
                hostname = host_match.group(1)
                try:
                    ipv4_results = socket.getaddrinfo(hostname, 5432, socket.AF_INET)
                    if ipv4_results:
                        ipv4 = ipv4_results[0][4][0]
                        db_url_ipv4 = db_url.replace(hostname, ipv4)
                    else:
                        db_url_ipv4 = db_url
                except Exception:
                    db_url_ipv4 = db_url
            else:
                db_url_ipv4 = db_url
            conn = await asyncpg.connect(db_url_ipv4, timeout=10)
            version = await conn.fetchval("SELECT version()")
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            table_names = [t["tablename"] for t in tables]
            await conn.close()
            return (
                f"✅ **{label} — Connected**\n\n"
                f"- Version: `{version[:60]}`\n"
                f"- Tables: {', '.join(table_names) if table_names else '*(none)*'}"
            )
        except Exception as e:
            # Some exceptions (e.g. asyncio.TimeoutError) stringify to "" —
            # fall back to the exception class name so the message isn't blank.
            err_detail = str(e) or type(e).__name__
            return (
                f"❌ **{label} — Connection failed**\n\n"
                f"`{err_detail}`\n\n"
                f"_Check if Supabase project is paused at supabase.com_"
            )

    return f"❓ **{key} — Unknown connection type**"


# ── Session sidebar (ElementSidebar) ────────────────────────────────────────

async def update_sidebar():
    """Update the ElementSidebar with current session state."""
    config = cl.user_session.get("config", {})
    indexed = cl.user_session.get("indexed_docs", [])
    conv_id = cl.user_session.get("conversation_id", "None")
    evaluations = cl.user_session.get("evaluations", [])

    docs_section = (
        "\n".join(f"- 📄 `{d}`" for d in indexed)
        if indexed else "_No documents indexed yet_"
    )
    evals_section = (
        f"{len(evaluations)} evaluation(s) stored"
        if evaluations else "_No evaluations yet_"
    )
    conv_section = f"`{str(conv_id)[:8]}...`" if conv_id and conv_id != "None" else "_New conversation_"
    rerank_icon = "✅" if config.get("rerank", True) else "❌"

    sidebar_text = f"""**📁 Indexed Documents**
{docs_section}

**💬 Conversation**
{conv_section}

**📊 Evaluations**
{evals_section}

---
**⚙️ Current Settings**

🤖 LLM: `{config.get("llm_provider", "openai")}` / `{config.get("llm_model", "gpt-4o-mini")}` (temp: `{config.get("temperature", 0.0)}`)
🔍 Embed: `{config.get("embedding_provider", "openai")}` / `{config.get("embedding_model", "text-embedding-3-small")}`
📚 Retrieval: `{config.get("retrieval_mode", "hybrid")}` · top-k `{config.get("top_k", 5)}` · rerank {rerank_icon}
✂️ Chunk: `{config.get("chunk_method", "recursive")}` · size `{config.get("chunk_size", 500)}` · overlap `{config.get("chunk_overlap", 50)}`
👁️ OCR: `{config.get("ocr_method", "easyocr")}`
🖼️ Vision Mode: {"✅ on" if config.get("vision_mode", False) else "⭕ off"}

_Type `/sidebar` to refresh_
"""
    try:
        await cl.ElementSidebar.set_title("RAG Platform")
        await cl.ElementSidebar.set_elements([
            cl.Text(name="session_info", content=sidebar_text, display="inline")
        ])
    except Exception:
        pass  # Sidebar not supported in all Chainlit versions


async def refresh_settings_sidebar():
    """Refresh the read-only session info fields in the settings sidebar."""
    role = cl.user_session.get("role", "user")
    if role not in ("admin", "engineer"):
        return  # only admin/engineer have this expanded settings panel

    indexed = cl.user_session.get("indexed_docs", [])
    conv_id = cl.user_session.get("conversation_id", "")
    evals_count = len(cl.user_session.get("evaluations", []))
    config = cl.user_session.get("config", {})

    indexed_str = ", ".join(indexed) if indexed else "None"
    conv_str = str(conv_id)[:16] + "..." if conv_id and len(str(conv_id)) > 16 else (str(conv_id) or "New")

    try:
        await cl.ChatSettings([
            cl.input_widget.TextInput(
                id="__indexed_docs", label="📁 Indexed Documents",
                initial=indexed_str,
            ),
            cl.input_widget.TextInput(
                id="__conversation_id", label="💬 Conversation ID",
                initial=conv_str,
            ),
            cl.input_widget.TextInput(
                id="__evaluations", label="📊 Evaluations",
                initial=f"{evals_count} evaluation(s)" if evals_count else "None",
            ),
            cl.input_widget.Select(
                id="llm_model_full", label="🤖 LLM Model (provider:model)",
                values=[
                    "openai:gpt-4o-mini", "openai:gpt-4o", "openai:gpt-4-turbo",
                    "google:gemini-2.0-flash", "google:gemini-1.5-pro",
                    "anthropic:claude-3-5-sonnet", "anthropic:claude-3-haiku",
                    "azure:gpt-4o-mini", "azure:gpt-4o",
                    "aws:amazon.titan-text-express-v1", "aws:meta.llama3-8b-instruct-v1:0",
                ],
                initial_value=f"{config.get('llm_provider','openai')}:{config.get('llm_model','gpt-4o-mini')}",
            ),
            cl.input_widget.Slider(
                id="temperature", label="Temperature",
                initial=config.get("temperature", 0.0),
                min=0.0, max=1.0, step=0.1,
            ),
            cl.input_widget.Select(
                id="embedding_model_full", label="🔍 Embedding Model (provider:model)",
                values=[
                    "openai:text-embedding-3-small", "openai:text-embedding-3-large",
                    "openai:text-embedding-ada-002",
                    "google:text-multilingual-embedding-002",
                    "huggingface:sentence-transformers/all-MiniLM-L6-v2",
                    "azure:text-embedding-3-small",
                ],
                initial_value=f"{config.get('embedding_provider','openai')}:{config.get('embedding_model','text-embedding-3-small')}",
            ),
            cl.input_widget.Select(
                id="retrieval_mode", label="Retrieval Mode",
                values=["hybrid", "vector", "bm25"],
                initial_value=config.get("retrieval_mode", "hybrid"),
            ),
            cl.input_widget.Slider(
                id="top_k", label="Top-K Results",
                initial=config.get("top_k", 5),
                min=1, max=20, step=1,
            ),
            cl.input_widget.Switch(
                id="rerank", label="Cross-encoder Reranking",
                initial=config.get("rerank", True),
            ),
            cl.input_widget.Select(
                id="chunk_method", label="Chunking Method",
                values=["recursive", "sentence", "fixed"],
                initial_value=config.get("chunk_method", "recursive"),
            ),
            cl.input_widget.Slider(
                id="chunk_size", label="Chunk Size (tokens)",
                initial=config.get("chunk_size", 500),
                min=100, max=2000, step=100,
            ),
            cl.input_widget.Slider(
                id="chunk_overlap", label="Chunk Overlap (tokens)",
                initial=config.get("chunk_overlap", 50),
                min=0, max=200, step=10,
            ),
            cl.input_widget.Select(
                id="ocr_method", label="OCR Method",
                values=["easyocr", "tesseract", "google_docai", "none"],
                initial_value=config.get("ocr_method", "easyocr"),
            ),
            cl.input_widget.Switch(
                id="vision_mode",
                label="🖼️ Vision Mode — read charts & images (uses AI, adds cost)",
                initial=config.get("vision_mode", False),
            ),
            cl.input_widget.Select(
                id="caption_model_full",
                label="🖼️ Vision/Caption Model (provider:model)",
                values=[
                    "openai:gpt-4o-mini", "openai:gpt-4o",
                    "google:gemini-2.5-flash", "google:gemini-2.0-flash",
                    "anthropic:claude-3-5-sonnet", "anthropic:claude-3-haiku",
                    "azure:gpt-4o-mini", "azure:gpt-4o",
                ],
                initial_value=(
                    f"{config.get('caption_provider', config.get('llm_provider', 'openai'))}:"
                    f"{config.get('caption_model', config.get('llm_model', 'gpt-4o-mini'))}"
                ),
            ),
            cl.input_widget.TextInput(
                id="api_url", label="🔌 API Endpoint URL",
                initial=config.get("api_url", os.getenv("RAG_API_URL", "")),
            ),
            cl.input_widget.TextInput(
                id="qdrant_url", label="🔌 Qdrant Vector DB URL",
                initial=config.get("qdrant_url", os.getenv("QDRANT_URL", "")),
            ),
        ]).send()
    except Exception:
        pass  # Don't fail if settings refresh fails


async def rebuild_settings_widget(config: dict):
    """
    Rebuild ONLY the ChatSettings widget with given config values.
    Does NOT open the sidebar panel (unlike refresh_settings_sidebar).
    Used by reset/clear actions to update settings panel values silently.
    Works for all roles — users also have a settings panel now.
    """
    indexed = cl.user_session.get("indexed_docs", [])
    conv_id = cl.user_session.get("conversation_id", "")
    evals_count = len(cl.user_session.get("evaluations", []))
    indexed_str = ", ".join(indexed) if indexed else "None"
    conv_str = str(conv_id)[:16] + "..." if conv_id and len(str(conv_id)) > 16 else (str(conv_id) or "New")

    widgets = [
        cl.input_widget.TextInput(id="__indexed_docs", label="📁 Indexed Documents",
            initial=indexed_str),
        cl.input_widget.TextInput(id="__conversation_id", label="💬 Conversation ID",
            initial=conv_str),
        cl.input_widget.TextInput(id="__evaluations", label="📊 Evaluations",
            initial=f"{evals_count} evaluation(s)" if evals_count else "No evaluations yet"),
        cl.input_widget.Select(
            id="llm_model_full", label="🤖 LLM Model (provider:model)",
            values=[
                "openai:gpt-4o-mini", "openai:gpt-4o", "openai:gpt-4-turbo",
                "google:gemini-2.0-flash", "google:gemini-1.5-pro",
                "anthropic:claude-3-5-sonnet", "anthropic:claude-3-haiku",
                "azure:gpt-4o-mini", "azure:gpt-4o",
                "aws:amazon.titan-text-express-v1", "aws:meta.llama3-8b-instruct-v1:0",
            ],
            initial_value=f"{config.get('llm_provider','openai')}:{config.get('llm_model','gpt-4o-mini')}",
        ),
        cl.input_widget.Slider(id="temperature", label="Temperature",
            initial=config.get("temperature", 0.0), min=0.0, max=1.0, step=0.1),
        cl.input_widget.Select(
            id="embedding_model_full", label="🔍 Embedding Model (provider:model)",
            values=[
                "openai:text-embedding-3-small", "openai:text-embedding-3-large",
                "openai:text-embedding-ada-002",
                "google:text-multilingual-embedding-002",
                "huggingface:sentence-transformers/all-MiniLM-L6-v2",
                "azure:text-embedding-3-small",
            ],
            initial_value=f"{config.get('embedding_provider','openai')}:{config.get('embedding_model','text-embedding-3-small')}",
        ),
        cl.input_widget.Select(id="retrieval_mode", label="📚 Retrieval Mode",
            values=["hybrid", "vector", "bm25"],
            initial_value=config.get("retrieval_mode", "hybrid")),
        cl.input_widget.Slider(id="top_k", label="Top-K Results",
            initial=config.get("top_k", 5), min=1, max=20, step=1),
        cl.input_widget.Switch(id="rerank", label="Cross-encoder Reranking",
            initial=config.get("rerank", True)),
        cl.input_widget.Select(id="chunk_method", label="✂️ Chunking Method",
            values=["recursive", "sentence", "fixed"],
            initial_value=config.get("chunk_method", "recursive")),
        cl.input_widget.Slider(id="chunk_size", label="Chunk Size (tokens)",
            initial=config.get("chunk_size", 500), min=100, max=2000, step=100),
        cl.input_widget.Slider(id="chunk_overlap", label="Chunk Overlap (tokens)",
            initial=config.get("chunk_overlap", 50), min=0, max=200, step=10),
        cl.input_widget.Select(id="ocr_method", label="👁️ OCR Method",
            values=["easyocr", "tesseract", "google_docai", "none"],
            initial_value=config.get("ocr_method", "easyocr")),
        cl.input_widget.Switch(
            id="vision_mode",
            label="🖼️ Vision Mode — read charts & images (uses AI, adds cost)",
            initial=config.get("vision_mode", False),
        ),
        cl.input_widget.Select(
            id="caption_model_full",
            label="🖼️ Vision/Caption Model (provider:model)",
            values=[
                "openai:gpt-4o-mini", "openai:gpt-4o",
                "google:gemini-2.5-flash", "google:gemini-2.0-flash",
                "anthropic:claude-3-5-sonnet", "anthropic:claude-3-haiku",
                "azure:gpt-4o-mini", "azure:gpt-4o",
            ],
            initial_value=(
                f"{config.get('caption_provider', config.get('llm_provider', 'openai'))}:"
                f"{config.get('caption_model', config.get('llm_model', 'gpt-4o-mini'))}"
            ),
        ),
        cl.input_widget.Select(
            id="timezone", label="🌍 Display Timezone",
            values=["Asia/Kolkata", "UTC", "America/New_York", "America/Los_Angeles",
                    "Europe/London", "Europe/Paris", "Asia/Singapore", "Asia/Tokyo",
                    "Australia/Sydney", "America/Chicago", "America/Sao_Paulo"],
            initial_value=config.get("timezone", "Asia/Kolkata"),
        ),
    ]
    try:
        await cl.ChatSettings(widgets).send()
    except Exception:
        pass


# ── Settings update handler ─────────────────────────────────────────────────

@cl.on_settings_update
async def on_settings_update(settings: dict):
    config = cl.user_session.get("config", {})
    role = cl.user_session.get("role", "user")
    settings = {k: v for k, v in settings.items() if not k.startswith("__")}

    was_vision_on = config.get("vision_mode", False)
    prior_ocr_method = config.get("ocr_method", "easyocr")

    # Parse combined provider:model dropdowns — auto-split on first ":"
    if settings.get("llm_model_full"):
        parts = settings["llm_model_full"].split(":", 1)
        if len(parts) == 2:
            config["llm_provider"] = parts[0]
            config["llm_model"] = parts[1]

    if settings.get("embedding_model_full"):
        parts = settings["embedding_model_full"].split(":", 1)
        if len(parts) == 2:
            config["embedding_provider"] = parts[0]
            config["embedding_model"] = parts[1]

    if settings.get("caption_model_full"):
        parts = settings["caption_model_full"].split(":", 1)
        if len(parts) == 2:
            config["caption_provider"] = parts[0]
            config["caption_model"] = parts[1]

    config.update({
        "temperature":    float(settings.get("temperature",   config.get("temperature", 0.0))),
        "retrieval_mode": settings.get("retrieval_mode",      config.get("retrieval_mode", "hybrid")),
        "top_k":          int(settings.get("top_k",           config.get("top_k", 5))),
        "rerank":         bool(settings.get("rerank",         config.get("rerank", True))),
        "chunk_method":   settings.get("chunk_method",        config.get("chunk_method", "recursive")),
        "chunk_size":     int(settings.get("chunk_size",      config.get("chunk_size", 500))),
        "chunk_overlap":  int(settings.get("chunk_overlap",   config.get("chunk_overlap", 50))),
        "ocr_method":     settings.get("ocr_method",          config.get("ocr_method", "easyocr")),
        "vision_mode":    bool(settings.get("vision_mode",    config.get("vision_mode", False))),
        "timezone":       settings.get("timezone",            config.get("timezone", "Asia/Kolkata")),
    })
    if role in ("admin", "engineer"):
        for key in ("api_url", "qdrant_url", "redis_url", "supabase_url"):
            val = settings.get(key)
            # Reject the sentinel placeholder AND any masked value (contains "***")
            # so a display-only masked URL can never overwrite the real one in config.
            if val and val not in ("(loaded from Secret Manager)", "(not configured)", "") and "***" not in val:
                config[key] = val

    cl.user_session.set("config", config)

    # Persist config to Redis so timezone and other settings survive profile switches
    try:
        from app.graph.nodes.ragas_node import _get_redis
        import json as _json
        _r = _get_redis()
        if _r:
            _username = cl.user_session.get("username", "default")
            _existing = _r.get(f"session_state:{_username}")
            _state = _json.loads(_existing) if _existing else {}
            _state["config"] = config
            _state["indexed_docs"] = cl.user_session.get("indexed_docs", [])
            _state["collection_name"] = cl.user_session.get("collection_name", f"user_{_username}")
            _state["conversation_id"] = cl.user_session.get("conversation_id")
            _r.set(f"session_state:{_username}", _json.dumps(_state), ex=28800)
    except Exception:
        pass

    await cl.Message(
        content=(
            f"⚙️ **Settings updated**\n"
            f"- LLM: `{config.get('llm_provider','openai')}:{config.get('llm_model','gpt-4o-mini')}` "
            f"(temp: `{config.get('temperature',0.0)}`)\n"
            f"- Embedding: `{config.get('embedding_provider','openai')}:{config.get('embedding_model','text-embedding-3-small')}`\n"
            f"- Retrieval: `{config.get('retrieval_mode','hybrid')}` · top-k `{config.get('top_k',5)}` · "
            f"rerank `{config.get('rerank',True)}`\n"
            f"- Chunking: `{config.get('chunk_method','recursive')}` · size `{config.get('chunk_size',500)}` · "
            f"overlap `{config.get('chunk_overlap',50)}` · OCR `{config.get('ocr_method','easyocr')}`"
        ),
        actions=[
            cl.Action(name="reset_settings", label="🔄 Reset to Defaults", payload={},
                      tooltip="Reset all settings back to defaults")
        ]
    ).send()

    # ── Proactive guidance when Vision Mode is newly on, or on with a
    # weak OCR method selected. This exists specifically because a
    # silently-hardcoded vision model once failed for an entire session
    # with no visible error -- every image quietly fell back to plain
    # OCR because the picked provider had no configured credentials.
    # Surfacing the choice AND its risk here, right when it matters,
    # is the fix -- not hiding the setting again.
    now_vision_on = config.get("vision_mode", False)
    just_enabled = now_vision_on and not was_vision_on
    current_ocr = config.get("ocr_method", "easyocr")

    if just_enabled or (now_vision_on and current_ocr == "easyocr"):
        tips = ["🖼️ **Vision Mode tips**\n"]

        if just_enabled:
            cap_provider = config.get("caption_provider", config.get("llm_provider", "openai"))
            cap_model = config.get("caption_model", config.get("llm_model", "gpt-4o-mini"))
            tips.append(
                f"- Your **Vision/Caption Model** is set to `{cap_provider}:{cap_model}`. "
                f"Make sure that provider's API key is actually configured — if it isn't, "
                f"every chart/image will silently fall back to plain OCR with no visible error. "
                f"Defaulting to whatever LLM you're already chatting with (proven to work) is usually safest."
            )

        if current_ocr == "easyocr":
            tips.append(
                "- Your **OCR Method** is set to `easyocr`, which tends to perform below "
                "average on complex layouts and multi-column documents. For better base "
                "text extraction (independent of Vision Mode), consider switching to "
                "`google_docai` in Settings."
            )

        await cl.Message(content="\n".join(tips)).send()


@cl.action_callback("open_sidebar")
async def on_open_sidebar(action: cl.Action):
    """Refresh and open the session info sidebar."""
    await update_sidebar()
    await cl.Message(content="📋 Session panel refreshed — check the right side →").send()


@cl.action_callback("reset_settings")
async def on_reset_settings(action: cl.Action):
    """Reset session config to defaults — rebuilds settings widget, no sidebar opens."""
    defaults = {
        "llm_provider":       "openai",
        "llm_model":          "gpt-4o-mini",
        "temperature":        0.0,
        "embedding_provider": "openai",
        "embedding_model":    "text-embedding-3-small",
        "retrieval_mode":     "hybrid",
        "top_k":              5,
        "rerank":             True,
        "chunk_size":         500,
        "chunk_overlap":      50,
        "chunk_method":       "recursive",
        "ocr_method":         "easyocr",
        "vision_mode":        False,
        "domain":             "uploaded documents",
        "assistant_name":     "Document Assistant",
    }
    cl.user_session.set("config", defaults)
    await rebuild_settings_widget(defaults)
    await cl.Message(content="🔄 Settings reset to defaults — open ⚙️ to confirm values.").send()