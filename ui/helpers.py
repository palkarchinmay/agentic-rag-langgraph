"""
app/chainlit_ui/helpers.py — Shared utilities for the Chainlit UI.

Pure functions only — no @cl decorators, no side effects on import.
Safe to import from any other chainlit_ui module without circular-import risk.
"""

import os
import logging

logger = logging.getLogger("financial_rag.chainlit")

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv"]


# ── Redis-backed evaluation store ──────────────────────────────────────────
# cl.user_session is profile-scoped — switching Chat Profiles loses session
# state. Evaluations are stored in Redis under evals:{username} so they
# survive profile switches and can be read by the Diagnostics panel.

def _evals_redis_key(username: str) -> str:
    return f"chainlit_evals:{username}"


def load_evals(username: str) -> list:
    """Load evaluations from Redis (falls back to empty list if unavailable)."""
    if os.getenv("CACHE_BACKEND", "local").lower() != "redis":
        return []
    try:
        import redis as _redis
        r = _redis.from_url(
            os.getenv("REDIS_URL", ""),
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        import json as _json
        raw = r.get(_evals_redis_key(username))
        return _json.loads(raw) if raw else []
    except Exception:
        return []


def save_evals(username: str, evaluations: list) -> None:
    """Save evaluations to Redis (2 hour TTL, same as RAGAS results)."""
    if os.getenv("CACHE_BACKEND", "local").lower() != "redis":
        return
    try:
        import redis as _redis, json as _json
        r = _redis.from_url(
            os.getenv("REDIS_URL", ""),
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        r.set(_evals_redis_key(username), _json.dumps(evaluations), ex=7200)
    except Exception:
        pass


def convert_ts(ts_str: str, timezone_name: str = "Asia/Kolkata") -> str:
    """Convert a UTC timestamp string to the user's configured timezone."""
    try:
        from datetime import datetime, timezone as tz, timedelta
        import zoneinfo
        if not ts_str:
            return "—"
        dt_str = str(ts_str).replace(" ", "T")
        if "+" not in dt_str and "Z" not in dt_str:
            dt_str += "+00:00"
        dt_utc = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        try:
            user_tz = zoneinfo.ZoneInfo(timezone_name)
            dt_local = dt_utc.astimezone(user_tz)
            tz_abbr = dt_local.strftime("%Z")
        except Exception:
            # Fallback to fixed offsets if zoneinfo data is unavailable
            offsets = {
                "Asia/Kolkata": 330, "UTC": 0, "America/New_York": -300,
                "America/Los_Angeles": -480, "Europe/London": 0,
                "Europe/Paris": 60, "Asia/Singapore": 480,
                "Asia/Tokyo": 540, "Australia/Sydney": 600,
                "America/Chicago": -360, "America/Sao_Paulo": -180,
            }
            offset_mins = offsets.get(timezone_name, 330)
            dt_local = dt_utc.astimezone(tz(timedelta(minutes=offset_mins)))
            tz_abbr = timezone_name.split("/")[-1]
        return dt_local.strftime(f"%Y-%m-%d %H:%M {tz_abbr}")
    except Exception:
        return str(ts_str)[:16].replace("T", " ")


def mask_url(url: str) -> str:
    """Mask password/token in a connection URL for display."""
    if not url:
        return "*(not configured)*"
    import re
    masked = re.sub(r"(://[^:]+:)[^@]+(@)", r"\1***\2", url)
    masked = re.sub(r"(token=)[^&\s]+", r"\1***", masked)
    return masked


def guess_mime(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "txt": "text/plain", "md": "text/markdown", "csv": "text/csv",
    }.get(ext, "application/octet-stream")


def build_progress(bars: list, done: int, total: int, results: list, status: str) -> str:
    """Build a compact progress display for golden dataset evaluation."""
    bar_str = " ".join(bars)
    pct = round(done / total * 100) if total > 0 else 0
    filled = int(pct / 10)
    prog_bar = "█" * filled + "░" * (10 - filled)

    faith_vals = [r.get("faithfulness") for r in results if r.get("faithfulness") is not None]
    rel_vals = [r.get("answer_relevancy") for r in results if r.get("answer_relevancy") is not None]
    avg_faith = f"{round(sum(faith_vals)/len(faith_vals)*100,1)}%" if faith_vals else "—"
    avg_rel = f"{round(sum(rel_vals)/len(rel_vals)*100,1)}%" if rel_vals else "—"

    return (
        f"🎯 **Golden Dataset Progress**\n\n"
        f"`{prog_bar}` {pct}% — {done}/{total} complete\n\n"
        f"{bar_str}\n\n"
        f"📊 Running averages: Faith `{avg_faith}` · Relevancy `{avg_rel}`\n\n"
        f"_{status}_"
    )
