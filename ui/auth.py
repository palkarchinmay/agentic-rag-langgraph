"""
ui/auth.py — Login and Chat Profile registration.

Registers @cl.password_auth_callback, @cl.oauth_callback, and
@cl.set_chat_profiles with the Chainlit runtime. This module must be
imported (even if nothing is called directly) so the decorators run and
register with Chainlit — see the import in chainlit_app.py.

Both login paths are backed by Firebase Auth (or whatever AUTH_PROVIDER is
configured on the API side — see app/providers/auth/factory.py), but this
file never talks to Firebase directly. It only calls the FastAPI backend:
  - password_auth_callback -> api_client.login()      -> POST /api/v1/auth/login
  - oauth_callback         -> api_client.login_sso()  -> POST /api/v1/auth/login/sso

Both callbacks can be registered simultaneously — Chainlit's login page
shows the password form and the Google button together automatically
once OAUTH_GOOGLE_CLIENT_ID/OAUTH_GOOGLE_CLIENT_SECRET env vars are set.
"""

import logging
from typing import Optional

import chainlit as cl

from app.services.api_client import login, login_sso, APIError

logger = logging.getLogger("financial_rag.chainlit")


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    # "username" here is really the user's email — kept as "username" only
    # because that's the field name Chainlit's decorator signature uses.
    try:
        result = login(username, password)
        return cl.User(
            identifier=username,
            metadata={
                "role": result.get("role", "user"),
                # .get(key, default) only falls back for a MISSING key — an
                # empty string "" (a real value the API can legitimately
                # return when no display name was ever set) passes through
                # unchanged and renders as "****" in the welcome message's
                # markdown bold. `or` catches both cases.
                "full_name": result.get("full_name") or username,
                "api_token": result.get("access_token", ""),
            },
        )
    except APIError as e:
        logger.warning("Login failed for %s: %s", username, e)
        return None


@cl.oauth_callback
async def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> Optional[cl.User]:
    """Fires after Chainlit completes the Google OAuth handshake itself —
    by the time this runs, the user's Google identity is already verified.
    This function's only job is to map that identity onto our own
    user/role system via the API's /login/sso endpoint (protected by
    INTERNAL_SERVICE_SECRET, see api_client.login_sso)."""
    if provider_id != "google":
        logger.warning("Unsupported OAuth provider: %s", provider_id)
        return None

    email     = raw_user_data.get("email", "")
    full_name = raw_user_data.get("name", "")

    if not email:
        logger.warning("Google OAuth callback missing email in raw_user_data: %s",
                        list(raw_user_data.keys()))
        return None

    try:
        result = login_sso(email, full_name)
        return cl.User(
            identifier=email,
            metadata={
                "role":      result.get("role", "user"),
                "full_name": result.get("full_name") or full_name or email,
                "api_token": result.get("access_token", ""),
            },
        )
    except APIError as e:
        logger.warning("SSO login failed for %s: %s", email, e)
        return None


@cl.set_chat_profiles
async def set_chat_profiles(current_user: cl.User):
    """Define Chat Profiles based on user role."""
    role = current_user.metadata.get("role", "user")

    profiles = [
        cl.ChatProfile(
            name="💬 Chat",
            markdown_description="Chat with your documents and ask general questions.",
            default=True,
        )
    ]

    if role in ("admin", "engineer"):
        profiles.append(
            cl.ChatProfile(
                name="📊 Diagnostics",
                markdown_description="View RAGAS evaluations and export results.",
            )
        )

    if role in ("admin", "engineer"):
        profiles.append(
            cl.ChatProfile(
                name="🔌 Connections",
                markdown_description="Test and update external service connections (API, vector DB, cache, database).",
            )
        )

    if role == "admin":
        profiles.append(
            cl.ChatProfile(
                name="👥 User Management",
                markdown_description="Create, list, promote, disable, and delete users.",
            )
        )
        profiles.append(
            cl.ChatProfile(
                name="💰 Usage & FinOps",
                markdown_description="View usage and estimated costs, manage per-user quotas.",
            )
        )
        profiles.append(
            cl.ChatProfile(
                name="⚙️ Admin Panel",
                markdown_description="System operations: cache, logs, index collections, conversation cleanup.",
            )
        )

    return profiles