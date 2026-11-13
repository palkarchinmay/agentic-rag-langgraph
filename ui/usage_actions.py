"""
ui/usage_actions.py — "💰 Usage & FinOps" chat profile.

Shows per-user usage (requests/tokens/estimated cost) for the current
period, a time trend, a per-model cost breakdown, and quota management.

All numbers here are ESTIMATES, computed from token counts via
app/monitoring/cost_tracker.py's MODEL_PRICING table at request time —
not reconciled against actual provider invoices. Every place a cost
figure is shown, it's labeled "estimated" for exactly that reason.

Aggregation happens server-side in app/services/usage_service.py, queried
via the /api/v1/admin/usage* endpoints (api/routers/admin.py). Quota
ENFORCEMENT happens separately, in api/routers/query.py, before a query
even runs — this panel is for viewing and configuring, not gating.
"""

import logging

import chainlit as cl

from app.services.api_client import (
    get_usage, get_usage_timeseries, get_usage_by_model,
    get_user_quota, delete_user_quota, list_users, APIError,
)

logger = logging.getLogger("financial_rag.chainlit")


def _bar(value: float, max_value: float, width: int = 10) -> str:
    """Relative ASCII bar — scaled to the max value in the current result
    set, same visual language as the RAGAS score bars elsewhere in the app."""
    if max_value <= 0:
        return "░" * width
    filled = round((value / max_value) * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _fmt_cost(cost_usd: float) -> str:
    if cost_usd < 0.01:
        return f"${cost_usd:.5f}"
    return f"${cost_usd:.2f}"


async def _build_uid_lookup(token: str) -> dict:
    """uid -> {"email": ..., "full_name": ...}. Usage rows only ever carry
    a raw uid (that's what audit_logs.user_id stores) — this cross-
    references it against the Firebase user list for a human-readable
    display. Only fetches the first page (up to 200 users); fine for most
    orgs, but if you're running with more users than that, this silently
    won't find the rest — worth adding pagination here if that becomes a
    real problem."""
    try:
        result = list_users(token, limit=200)
    except APIError:
        return {}
    return {
        u["uid"]: {"email": u.get("email", ""), "full_name": u.get("full_name", "")}
        for u in result.get("users", [])
    }


def _display_name(uid: str, lookup: dict) -> str:
    info = lookup.get(uid)
    if not info:
        return uid  # unknown uid (e.g. user since deleted) — fall back to raw uid
    full_name = info.get("full_name", "").strip()
    email = info.get("email", "")
    return f"{full_name} ({email})" if full_name else (email or uid)


async def render_usage_panel():
    """Entry point for the '💰 Usage & FinOps' chat profile."""
    token = cl.user_session.get("token", "")

    await cl.Message(content=(
        "## 💰 Usage & FinOps\n\n"
        "_All costs below are **estimates** computed from token counts, "
        "not reconciled provider invoices._\n\n"
        "**Filters:** type `/usage from=2026-08-01 to=2026-08-31 user=<uid>` to change "
        "the date range or narrow to one user — all fields optional, defaults to the "
        "current calendar month, all users."
    )).send()

    try:
        result = get_usage(token)
    except APIError as e:
        await cl.Message(content=f"❌ Could not load usage data: {e}").send()
        return

    rows = result.get("rows", [])
    if not rows:
        await cl.Message(content=(
            f"No usage recorded for {result.get('period_start','?')[:10]} "
            f"to {result.get('period_end','?')[:10]} yet."
        )).send()
        return

    lookup = await _build_uid_lookup(token)

    total_cost = sum(r["cost_usd"] for r in rows)
    total_tokens = sum(r["tokens"] for r in rows)
    total_requests = sum(r["requests"] for r in rows)
    max_cost = max((r["cost_usd"] for r in rows), default=0)

    lines = [
        f"### Usage — {result['period_start'][:10]} to {result['period_end'][:10]}\n",
        f"**Totals:** {total_requests} requests · {total_tokens:,} tokens · "
        f"est. **{_fmt_cost(total_cost)}**\n",
        "| User | Requests | Tokens | Est. Cost |",
        "|---|---|---|---|",
    ]
    sorted_rows = sorted(rows, key=lambda x: x["cost_usd"], reverse=True)
    for r in sorted_rows:
        bar = _bar(r["cost_usd"], max_cost)
        err_note = f" ⚠️{r['errors']}" if r.get("errors") else ""
        name = _display_name(r["user_id"], lookup)
        lines.append(
            f"| {name} | {r['requests']}{err_note} | {r['tokens']:,} | "
            f"{_fmt_cost(r['cost_usd'])} `{bar}` |"
        )

    quota_actions = [
        cl.Action(
            name="manage_quota",
            label=f"💰 {_display_name(r['user_id'], lookup)[:30]}",
            payload={"user_id": r["user_id"], "display_name": _display_name(r["user_id"], lookup)},
            tooltip=f"View/set quota for {_display_name(r['user_id'], lookup)}",
        )
        for r in sorted_rows
    ]

    await cl.Message(
        content="\n".join(lines),
        actions=[
            cl.Action(name="view_usage_trend", label="📈 Trend (This Month)", payload={}),
            cl.Action(name="view_usage_by_model", label="🧮 By Model", payload={}),
        ] + quota_actions,
    ).send()


@cl.action_callback("view_usage_trend")
async def on_view_usage_trend(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    token = cl.user_session.get("token", "")
    try:
        result = get_usage_timeseries(token)
    except APIError as e:
        await cl.Message(content=f"❌ Could not load trend: {e}").send()
        return

    points = result.get("points", [])
    if not points:
        await cl.Message(content="No usage data to chart yet this period.").send()
        return

    max_cost = max((p["cost_usd"] for p in points), default=0)
    lines = ["### 📈 Cost Trend (by day)\n",
             "| Date | Requests | Tokens | Est. Cost |",
             "|---|---|---|---|"]
    for p in points:
        bar = _bar(p["cost_usd"], max_cost)
        date_label = p["bucket"][:10]
        lines.append(
            f"| {date_label} | {p['requests']} | {p['tokens']:,} | "
            f"{_fmt_cost(p['cost_usd'])} `{bar}` |"
        )
    await cl.Message(content="\n".join(lines)).send()


@cl.action_callback("view_usage_by_model")
async def on_view_usage_by_model(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    token = cl.user_session.get("token", "")
    try:
        result = get_usage_by_model(token)
    except APIError as e:
        await cl.Message(content=f"❌ Could not load model breakdown: {e}").send()
        return

    rows = result.get("rows", [])
    if not rows:
        await cl.Message(content="No usage data yet this period.").send()
        return

    max_cost = max((r["cost_usd"] for r in rows), default=0)
    lines = ["### 🧮 Cost by Model\n",
             "| Provider | Model | Requests | Tokens | Est. Cost |",
             "|---|---|---|---|---|"]
    for r in rows:
        bar = _bar(r["cost_usd"], max_cost)
        lines.append(
            f"| {r['llm_provider']} | {r['llm_model']} | {r['requests']} | "
            f"{r['tokens']:,} | {_fmt_cost(r['cost_usd'])} `{bar}` |"
        )
    await cl.Message(content="\n".join(lines)).send()


@cl.action_callback("manage_quota")
async def on_manage_quota(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    token = cl.user_session.get("token", "")
    uid = action.payload.get("user_id", "")
    display_name = action.payload.get("display_name", uid)

    try:
        quota = get_user_quota(token, uid)
    except APIError as e:
        await cl.Message(content=f"❌ Could not load quota: {e}").send()
        return

    tok_limit = quota.get("monthly_token_limit")
    cost_limit = quota.get("monthly_cost_limit_usd")
    req_limit = quota.get("monthly_request_limit")
    has_quota = any(v is not None for v in (tok_limit, cost_limit, req_limit))

    lines = [f"### 💰 Quota — {display_name}\n_(`{uid}`)_\n"]
    if has_quota:
        lines.append(f"- Monthly tokens: `{tok_limit if tok_limit is not None else 'unlimited'}`")
        lines.append(f"- Monthly cost: `{('$' + str(cost_limit)) if cost_limit is not None else 'unlimited'}`")
        lines.append(f"- Monthly requests: `{req_limit if req_limit is not None else 'unlimited'}`")
        if quota.get("updated_by"):
            lines.append(f"\n_Last set by {quota['updated_by']}_")
    else:
        lines.append("_No quota set — this user is unlimited._")

    lines.append(
        f"\n**To change this**, copy and edit:\n```\n"
        f"/set_quota uid={uid} tokens=500000 cost=50 requests=1000\n```\n"
        f"_Any field can be omitted or set to 0 for unlimited on that dimension._"
    )

    actions = []
    if has_quota:
        actions.append(cl.Action(name="remove_quota", label="🗑️ Remove Quota",
                                   payload={"user_id": uid, "display_name": display_name}))

    await cl.Message(content="\n".join(lines), actions=actions).send()


@cl.action_callback("remove_quota")
async def on_remove_quota(action: cl.Action):
    if cl.user_session.get("role") != "admin":
        await cl.Message(content="❌ Admin only.").send()
        return

    token = cl.user_session.get("token", "")
    uid = action.payload.get("user_id", "")
    display_name = action.payload.get("display_name", uid)

    try:
        delete_user_quota(token, uid)
        await cl.Message(content=f"✅ Quota removed for {display_name} — now unlimited.").send()
    except APIError as e:
        await cl.Message(content=f"❌ Could not remove quota: {e}").send()