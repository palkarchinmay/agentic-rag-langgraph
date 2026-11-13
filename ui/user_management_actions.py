"""
ui/user_management_actions.py — "👥 User Management" chat profile.

Split out from admin_actions.py so user-account operations (create,
list, promote/demote, disable/enable, delete) have their own dedicated
panel, separate from system-level admin operations (cache, logs,
collections, conversation cleanup — still in admin_actions.py).

All user CRUD goes through app.services.api_client, which calls the
provider-agnostic /api/v1/admin/users* endpoints (api/routers/admin.py),
themselves backed by app.providers.auth.factory.get_auth_provider() —
Firebase today, Cognito/Azure AD later without any change needed here.
"""

import logging

import chainlit as cl

from app.services.api_client import APIError

logger = logging.getLogger("financial_rag.chainlit")


async def render_user_management_panel():
    """Entry point for the '👥 User Management' chat profile."""
    await cl.Message(content=(
        "## 👥 User Management\n\n"
        "Click **👤 List Users** below to see everyone, promote/demote roles, "
        "disable/enable accounts, or delete users. To create a new user, use "
        "the `/create_user` command shown after the list loads."
    )).send()

    await cl.Message(
        content="",
        actions=[
            cl.Action(name="list_users", label="👤 List Users", payload={}),
        ]
    ).send()

    await cl.Message(content=(
        "**Creating a user?** Copy the line below into the message box, "
        "edit the values, and send it:\n\n"
        "```\n"
        "/create_user email=newuser@example.com password=ChangeMe123 role=user name=Jane_Doe\n"
        "```\n\n"
        "- `role` — one of `admin`, `engineer`, `user` (defaults to `user` if omitted)\n"
        "- `name` — use underscores for spaces (e.g. `name=Jane_Doe`), since this is a "
        "space-separated command\n"
        "- `password` — minimum 6 characters\n\n"
        "**Managing an existing user?** Click **👤 List Users** above, then click the "
        "`🔧 email` button next to whoever you want to promote, disable, or delete — "
        "no command needed for that part."
    )).send()


# ── User CRUD action callbacks (moved from admin_actions.py) ──────────────

@cl.action_callback("list_users")
async def on_list_users(action: cl.Action):
    """Show current users, live from the configured auth provider (Firebase
    by default — see app/providers/auth/factory.py). Replaces the old
    users.json/RAG_USERS approach, which is dead now that AUTH_PROVIDER is
    set — that file/secret is no longer read anywhere in the app."""
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    from app.services.api_client import list_users as api_list_users

    token = cl.user_session.get("token", "")
    page_token = action.payload.get("page_token")

    try:
        result = api_list_users(token, page_token=page_token, limit=50)
        users = result.get("users", [])
        next_token = result.get("next_page_token")

        if not users:
            await cl.Message(content="👥 **No users found.**").send()
            return

        lines = [f"👥 **Users** (showing {len(users)})\n",
                 "| Email | Role | Full Name | Status |",
                 "|---|---|---|---|"]
        for u in users:
            status_icon = "🚫 disabled" if u.get("disabled") else "✅ active"
            lines.append(
                f"| {u.get('email','?')} | {u.get('role','user')} | "
                f"{u.get('full_name','') or '—'} | {status_icon} |"
            )

        manage_actions = [
            cl.Action(
                name="manage_user",
                label=f"🔧 {u.get('email','?')}",
                payload={"uid": u.get("uid"), "email": u.get("email"),
                         "role": u.get("role"), "disabled": u.get("disabled")},
                tooltip=f"Manage {u.get('email','?')}",
            )
            for u in users
        ]
        if next_token:
            manage_actions.append(cl.Action(
                name="list_users", label="➡️ Next Page",
                payload={"page_token": next_token},
            ))

        await cl.Message(content="\n".join(lines), actions=manage_actions).send()

    except APIError as e:
        await cl.Message(content=f"❌ Could not load users: {e}").send()


@cl.action_callback("manage_user")
async def on_manage_user(action: cl.Action):
    """Show role-change / disable-enable / delete actions for one user."""
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    uid = action.payload.get("uid", "")
    email = action.payload.get("email", "?")
    role = action.payload.get("role", "user")
    disabled = action.payload.get("disabled", False)

    role_actions = [
        cl.Action(name="set_role", label=f"{'✓ ' if role == r else ''}{r}",
                   payload={"uid": uid, "email": email, "role": r})
        for r in ("admin", "engineer", "user")
    ]

    status_action = (
        cl.Action(name="enable_user", label="✅ Enable", payload={"uid": uid, "email": email})
        if disabled else
        cl.Action(name="disable_user", label="🚫 Disable", payload={"uid": uid, "email": email})
    )

    await cl.Message(
        content=f"🔧 **Managing `{email}`**\n\nCurrent role: `{role}` · Status: {'🚫 disabled' if disabled else '✅ active'}",
        actions=role_actions + [
            status_action,
            cl.Action(name="confirm_delete_user", label="🗑️ Delete User",
                       payload={"uid": uid, "email": email},
                       tooltip="This cannot be undone"),
        ],
    ).send()


@cl.action_callback("set_role")
async def on_set_role(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    from app.services.api_client import set_user_role as api_set_role

    token = cl.user_session.get("token", "")
    uid = action.payload.get("uid", "")
    email = action.payload.get("email", "?")
    role = action.payload.get("role", "user")

    try:
        api_set_role(token, uid, role)
        await cl.Message(content=(
            f"✅ **Role updated** — `{email}` is now `{role}`.\n\n"
            f"_Note: Firebase caches roles in the token for up to 1 hour — "
            f"the affected user needs to log out/in for this to take effect "
            f"immediately, otherwise it applies automatically on their next "
            f"token refresh._"
        )).send()
    except APIError as e:
        await cl.Message(content=f"❌ Could not update role: {e}").send()


@cl.action_callback("disable_user")
async def on_disable_user(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    from app.services.api_client import disable_user as api_disable_user

    token = cl.user_session.get("token", "")
    uid = action.payload.get("uid", "")
    email = action.payload.get("email", "?")

    try:
        api_disable_user(token, uid)
        await cl.Message(content=f"🚫 **`{email}` disabled** — they can no longer log in.").send()
    except APIError as e:
        await cl.Message(content=f"❌ Could not disable user: {e}").send()


@cl.action_callback("enable_user")
async def on_enable_user(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    from app.services.api_client import enable_user as api_enable_user

    token = cl.user_session.get("token", "")
    uid = action.payload.get("uid", "")
    email = action.payload.get("email", "?")

    try:
        api_enable_user(token, uid)
        await cl.Message(content=f"✅ **`{email}` re-enabled** — they can log in again.").send()
    except APIError as e:
        await cl.Message(content=f"❌ Could not enable user: {e}").send()


@cl.action_callback("confirm_delete_user")
async def on_confirm_delete_user(action: cl.Action):
    uid = action.payload.get("uid", "")
    email = action.payload.get("email", "?")
    await cl.Message(
        content=f"⚠️ **Permanently delete `{email}`?** This cannot be undone.",
        actions=[
            cl.Action(name="delete_user_confirmed", label="🗑️ Yes, Delete",
                       payload={"uid": uid, "email": email}),
            cl.Action(name="cancel_delete_user", label="❌ Cancel", payload={}),
        ],
    ).send()


@cl.action_callback("delete_user_confirmed")
async def on_delete_user_confirmed(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    from app.services.api_client import delete_user as api_delete_user

    token = cl.user_session.get("token", "")
    uid = action.payload.get("uid", "")
    email = action.payload.get("email", "?")

    try:
        api_delete_user(token, uid)
        await cl.Message(content=f"🗑️ **`{email}` deleted.**").send()
    except APIError as e:
        await cl.Message(content=f"❌ Could not delete user: {e}").send()


@cl.action_callback("cancel_delete_user")
async def on_cancel_delete_user(action: cl.Action):
    await cl.Message(content="Cancelled — no changes made.").send()