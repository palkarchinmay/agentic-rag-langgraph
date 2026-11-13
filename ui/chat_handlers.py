"""
app/chainlit_ui/chat_handlers.py — Chat lifecycle events and message dispatch.

Covers: starter suggestions, on_chat_start (session init + profile routing),
on_chat_end, on_message (dispatches to file upload / slash commands / query),
and the confirm_reindex / cancel_reindex action callbacks.

NOTE: on_message dispatches to file upload handling (ui.upload_handler),
slash commands (ui.commands), and the core RAG query flow
(ui.query_handler). All three are fully modularized as of this step — no
more deferred imports back into chainlit_app.py.
"""

import os
import logging

import chainlit as cl

from ui.helpers import load_evals
from ui.admin_actions import render_admin_panel
from ui.user_management_actions import render_user_management_panel
from ui.usage_actions import render_usage_panel
from ui.settings_actions import render_connections_panel

logger = logging.getLogger("financial_rag.chainlit")


@cl.set_starters
async def set_starters():
    """Show starter suggestions including session info button."""
    return [
        cl.Starter(
            label="📋 View Session Info",
            message="/sidebar",
            icon="📋",
        ),
        cl.Starter(
            label="📄 Index a document",
            message="Please upload a document using the 📎 button below",
            icon="📄",
        ),
        cl.Starter(
            label="❓ Ask a question",
            message="What would you like to know about your documents?",
            icon="💬",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():

    user = cl.context.session.user
    role = user.metadata.get("role", "user")
    # `or`, not .get(key, default) — an empty-but-present full_name would
    # otherwise pass through and render as "****" (empty string inside
    # markdown ** ** bold) in the welcome message below.
    full_name = user.metadata.get("full_name") or user.identifier
    token = user.metadata.get("api_token", "")
    chat_profile = cl.user_session.get("chat_profile", "💬 Chat")

    # Persist basic session state (always)
    cl.user_session.set("token", token)
    cl.user_session.set("role", role)
    cl.user_session.set("username", user.identifier)

    # Always restore session state from Redis on every profile switch
    # cl.user_session is reset on each profile switch in Chainlit
    try:
        from app.graph.nodes.ragas_node import _get_redis
        import json as _json
        r = _get_redis()
        if r:
            stored = r.get(f"session_state:{user.identifier}")
            if stored:
                state = _json.loads(stored)
                cl.user_session.set("indexed_docs", state.get("indexed_docs", []))
                cl.user_session.set("collection_name", state.get("collection_name",
                                                                  f"user_{user.identifier}"))
                cl.user_session.set("conversation_id", state.get("conversation_id"))
                # Restore config including timezone preference
                if state.get("config"):
                    cl.user_session.set("config", state["config"])
                # else: no config in Redis, default_config will be used below
            else:
                cl.user_session.set("indexed_docs", [])
                cl.user_session.set("collection_name", f"user_{user.identifier}")
                cl.user_session.set("conversation_id", None)
        else:
            cl.user_session.set("indexed_docs", [])
            cl.user_session.set("collection_name", f"user_{user.identifier}")
            cl.user_session.set("conversation_id", None)
    except Exception:
        cl.user_session.set("indexed_docs", [])
        cl.user_session.set("collection_name", f"user_{user.identifier}")
        cl.user_session.set("conversation_id", None)

    # Always load evaluations fresh from Redis
    cl.user_session.set("evaluations", load_evals(user.identifier))

    # ── Config initialization (BEFORE profile routing) ─────────────────────
    # Must happen here so all profiles have access to user's saved config
    default_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "temperature": 0.0,
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "rerank": True,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chunk_method": "recursive",
        "ocr_method": "easyocr",
        "domain": "uploaded documents",
        "assistant_name": "Document Assistant",
        "api_url": os.getenv("RAG_API_URL", "http://localhost:8000"),
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "timezone": "Asia/Kolkata",
    }
    # Use Redis-restored config if available, otherwise use defaults
    if not cl.user_session.get("config"):
        cl.user_session.set("config", default_config)
    config = cl.user_session.get("config", default_config)

    # ── Route to correct panel based on Chat Profile ───────────────────────
    if chat_profile == "🔌 Connections" and role in ("admin", "engineer"):
        await render_connections_panel()
        return

    if chat_profile == "⚙️ Admin Panel" and role == "admin":
        await render_admin_panel()
        return

    if chat_profile == "👥 User Management" and role == "admin":
        await render_user_management_panel()
        return

    if chat_profile == "💰 Usage & FinOps" and role == "admin":
        await render_usage_panel()
        return

    if chat_profile == "📊 Diagnostics" and role in ("admin", "engineer"):
        # Load evaluations from Redis (persists across profile switches)
        username = user.identifier
        evals = load_evals(username)
        # Sync to session so other functions can use it
        cl.user_session.set("evaluations", evals)

        if not evals:
            await cl.Message(
                content=(
                    "📊 **Diagnostics Panel**\n\n"
                    "No evaluations yet. Switch to 💬 Chat, run some RAG queries, "
                    "then click **📊 Evaluate (RAGAS)** on any response.\n\n"
                    "Come back here to see and export your results."
                )
            ).send()
        else:
            # Show summary table of all evaluations
            lines = [f"📊 **Diagnostics — {len(evals)} Evaluation(s)**\n",
                     "| # | Question | Faithfulness | Answer Relevancy | Status |",
                     "|---|---|---|---|---|"]
            for i, e in enumerate(evals, 1):
                q = e.get("question", e.get("label", f"Q{i}"))[:50]
                faith = f"{round(e['faithfulness']*100,1)}% 🟢" if e.get("faithfulness") is not None and e["faithfulness"] >= 0.8 else (f"{round(e['faithfulness']*100,1)}% 🟡" if e.get("faithfulness") is not None and e["faithfulness"] >= 0.6 else (f"{round(e['faithfulness']*100,1)}% 🔴" if e.get("faithfulness") is not None else "—"))
                rel = f"{round(e['answer_relevancy']*100,1)}% 🟢" if e.get("answer_relevancy") is not None and e["answer_relevancy"] >= 0.8 else (f"{round(e['answer_relevancy']*100,1)}% 🟡" if e.get("answer_relevancy") is not None and e["answer_relevancy"] >= 0.6 else (f"{round(e['answer_relevancy']*100,1)}% 🔴" if e.get("answer_relevancy") is not None else "—"))
                lines.append(f"| Q{i} | {q} | {faith} | {rel} | ✅ |")

            export_actions = [
                cl.Action(name="export_all", label=f"📊 Export All ({len(evals)})",
                          payload={"indices": list(range(len(evals)))}),
                cl.Action(name="run_golden_dataset", label="🎯 Golden Dataset Test",
                          payload={},
                          tooltip="Run batch evaluation against a test dataset"),
                cl.Action(name="clear_evaluations", label="🗑️ Clear All Evaluations",
                          payload={},
                          tooltip="Remove all stored evaluation results"),
            ]
            await cl.Message(content="\n".join(lines), actions=export_actions).send()
        return

    # NOTE: config already initialized above before profile routing
    # indexed_docs, collection_name, conversation_id, evaluations
    # are already restored from Redis above — do NOT overwrite them here

    # ── Settings panel (all roles) ──────────────────────────────────────────
    if role in ("admin", "engineer", "user"):
        indexed = cl.user_session.get("indexed_docs", [])
        conv_id = cl.user_session.get("conversation_id", "")
        evals_count = len(cl.user_session.get("evaluations", []))

        indexed_str = ", ".join(indexed) if indexed else "None"
        conv_str = str(conv_id)[:16] + "..." if conv_id and len(str(conv_id)) > 16 else (str(conv_id) or "New")

        widgets = [
            cl.input_widget.TextInput(id="__indexed_docs", label="📁 Indexed Documents",
                initial=indexed_str, placeholder="No documents indexed yet"),
            cl.input_widget.TextInput(id="__conversation_id", label="💬 Conversation ID",
                initial=conv_str, placeholder="New conversation"),
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
        ]

        widgets += [
            cl.input_widget.Select(
                id="timezone", label="🌍 Display Timezone",
                values=[
                    "Asia/Kolkata", "UTC", "America/New_York",
                    "America/Los_Angeles", "Europe/London", "Europe/Paris",
                    "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney",
                    "America/Chicago", "America/Sao_Paulo",
                ],
                initial_value=config.get("timezone", "Asia/Kolkata"),
            ),
        ]

        # Connection URL settings (admin + engineer only — not for regular users)
        if role in ("admin", "engineer"):
            widgets += [
                cl.input_widget.TextInput(
                    id="api_url", label="🔌 API Endpoint URL",
                    initial=default_config["api_url"],
                    placeholder="https://rag-api-prod-xxx.run.app",
                ),
                cl.input_widget.TextInput(
                    id="qdrant_url", label="🔌 Qdrant Vector DB URL",
                    initial=os.getenv("QDRANT_URL", ""),
                    placeholder="https://xxx.qdrant.io",
                ),
                cl.input_widget.TextInput(
                    id="redis_url", label="🔌 Redis/Upstash Cache URL",
                    initial="(loaded from Secret Manager)" if os.getenv("REDIS_URL") else "(not configured)",
                    placeholder="rediss://default:token@xxx.upstash.io:6379",
                ),
                cl.input_widget.TextInput(
                    id="supabase_url", label="🔌 Supabase PostgreSQL URL",
                    initial="(loaded from Secret Manager)" if os.getenv("AUDIT_DB_URL") else "(not configured)",
                    placeholder="postgresql://postgres:pass@db.xxx.supabase.co:5432/postgres",
                ),
            ]

        await cl.ChatSettings(widgets).send()

    # ── Welcome message ─────────────────────────────────────────────────────
    role_badge = {"admin": "🔴 Admin", "engineer": "🟡 Engineer"}.get(role, "🟢 User")
    extra_cmds = ""
    if role == "admin":
        extra_cmds = "\n\n**Admin commands:** `/stats` · `/logs` · `/clear` · `/help`"
    elif role == "engineer":
        extra_cmds = "\n\n**Engineer commands:** `/logs` · `/clear` · `/help`"

    indexed_docs = cl.user_session.get("indexed_docs", [])
    if indexed_docs:
        docs_status = (
            f"\n\n📁 **Indexed documents:** {', '.join(f'`{d}`' for d in indexed_docs)}\n"
            f"_Type `/clear` to remove indexed documents and start fresh._"
        )
    else:
        docs_status = "\n\n_No documents indexed yet — upload files to start document Q&A._"

    conv_id = cl.user_session.get("conversation_id")
    if conv_id and indexed_docs:
        session_note = (
            f"\n\n_👋 Welcome back! Your documents and conversation context are ready. "
            f"Previous messages aren't shown here, but I remember everything we discussed — "
            f"just continue where you left off._"
        )
    elif conv_id:
        session_note = (
            f"\n\n_👋 Welcome back! I remember our previous conversation — "
            f"feel free to continue where you left off._"
        )
    else:
        session_note = ""

    await cl.Message(content=f"""👋 Welcome, **{full_name}** {role_badge}

I'm your **Document Assistant** — powered by RAG.

**What you can do:**
- 💬 **Chat** — ask questions about your documents or general topics
- 📎 **Upload files** — attach documents in the chat (auto-indexed instantly)
- ⚙️ **Settings** — configure LLM, embedding, retrieval and chunking{extra_cmds}{docs_status}{session_note}
""").send()


@cl.action_callback("confirm_reindex")
async def on_confirm_reindex(action: cl.Action):
    """User confirmed re-indexing despite duplicates."""
    await cl.Message(content="✅ Re-indexing... note this will add duplicate chunks. Consider clearing the collection first.").send()


@cl.action_callback("cancel_reindex")
async def on_cancel_reindex(action: cl.Action):
    """User cancelled re-indexing."""
    await cl.Message(content="❌ Re-indexing cancelled. Use **🗑️ Clear My Collection** in Admin Panel to reset, then upload fresh.").send()


@cl.on_chat_end
async def on_chat_end():
    """Called when session ends (browser close/logout, NOT profile switches).

    Chainlit calls on_chat_end for both logout AND profile switches, so we
    do NOT clear session_state here — that would break profile switching
    (indexed_docs and conversation_id would be lost). Session state expires
    naturally via TTL (8 hours). Users can explicitly clear with /clear.
    """
    pass


@cl.on_message
async def on_message(message: cl.Message):
    from ui.upload_handler import handle_file_upload
    from ui.commands import handle_command
    from ui.query_handler import handle_query

    content = message.content.strip()
    role = cl.user_session.get("role", "user")

    # Handle file uploads first, then process any accompanying text query
    if message.elements:
        # Accept any element that has a name attribute (covers all file types)
        files = [e for e in message.elements
                 if hasattr(e, "name") and e.name]
        if files:
            for f in files:
                logger.info("File received: name=%s type=%s", f.name,
                            getattr(f, "mime", getattr(f, "type", "unknown")))
            indexed_ok = await handle_file_upload(files)
            # If there's also a text query, run it after indexing completes
            if content and not content.startswith("/"):
                if indexed_ok:
                    await cl.Message(
                        content=f"_Documents indexed. Now answering: **{content[:80]}**..._"
                    ).send()
                await handle_query(content)
            return

    if content.startswith("/"):
        await handle_command(content, role)
        return

    if content:
        await handle_query(content)