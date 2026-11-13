"""
chainlit_app.py — RAG Platform Chainlit UI. Entry point only.

All actual UI logic lives in ui/ (see app/README or the module docstrings
in each ui/*.py file). This file's only job is to import every module that
registers a @cl.* decorated callback, so Chainlit picks them up at startup
— NOT lazily on first message. Modules with no @cl decorators of their own
(upload_handler, commands, query_handler) don't need to be imported here;
they're pulled in on demand by chat_handlers.py / commands.py.

UX IMPROVEMENTS (latest):
- Main chat shows only response + sources (clean)
- Diagnostics shown in collapsible cl.Step (engineer/admin only)
- Meta (latency/tokens/cost) shown via cl.Message metadata footer
- HTML tags removed — uses Chainlit native formatting
- RAGAS evaluate action button on each response (engineer/admin)
"""

import logging

import chainlit as cl  # noqa: F401  (imported for side effect: Chainlit runtime init)

# Modules that register @cl.* decorated callbacks — import order doesn't
# matter, but all of them must be imported eagerly (not lazily) so their
# callbacks exist before Chainlit serves the first request.
from ui import auth               # noqa: F401 — password_auth_callback, set_chat_profiles
from ui import admin_actions      # noqa: F401 — Admin Panel action callbacks
from ui import user_management_actions  # noqa: F401 — User Management panel + user CRUD callbacks
from ui import usage_actions      # noqa: F401 — Usage & FinOps panel + quota callbacks
from ui import settings_actions   # noqa: F401 — on_settings_update, open_sidebar, reset_settings
from ui import chat_handlers      # noqa: F401 — set_starters, on_chat_start, on_chat_end, on_message, confirm/cancel_reindex
from ui import evaluation_actions  # noqa: F401 — evaluate_ragas, run_golden_dataset, clear/export_*

logger = logging.getLogger("financial_rag.chainlit")