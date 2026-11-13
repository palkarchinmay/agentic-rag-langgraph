"""
app/services/api_client.py — HTTP client for Streamlit → FastAPI calls.

Single place for all API communication. If the API URL changes
(local → cloud), only RAG_API_URL in .env needs updating.

All methods raise APIError on failure with a user-friendly message.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("financial_rag.api_client")

# Read from env — same value works for local, Docker, and cloud
API_URL     = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
# Fail fast if the API is unreachable instead of hanging for API_TIMEOUT.
# (On Windows, "localhost" lookups from worker threads were seen to hang.)
CONNECT_TIMEOUT = float(os.getenv("API_CONNECT_TIMEOUT_SECONDS", "10"))
# Timeout for query/index calls — must be long enough for:
# reranker first load (~30s) + LLM call (~30s) + embedding (~5s) = ~65s
# Set high to avoid false timeouts; real hung requests will still timeout
API_TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "300"))


class APIError(Exception):
    """Raised when the API returns an error response."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _headers(token: Optional[str] = None) -> dict:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _handle(response: httpx.Response) -> dict:
    """Raise APIError for non-2xx responses."""
    if response.status_code == 401:
        raise APIError("Session expired — please log in again.", 401)
    if response.status_code == 403:
        raise APIError("Access denied — insufficient permissions.", 403)
    if response.status_code == 422:
        detail = response.json().get("detail", "Invalid request")
        raise APIError(f"Invalid request: {detail}", 422)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text[:200])
        except Exception:
            detail = response.text[:200]
        raise APIError(f"API error ({response.status_code}): {detail}",
                       response.status_code)
    return response.json()


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(username: str, password: str) -> dict:
    """
    POST /api/v1/auth/login
    Returns: {access_token, role, username, full_name, expires_in}
    """
    try:
        r = httpx.post(
            f"{API_URL}/api/v1/auth/login",
            json={"username": username, "password": password},
            timeout=httpx.Timeout(10, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Cannot reach API at {API_URL}. Is it running? ({e})")


def login_sso(email: str, full_name: str = "") -> dict:
    """
    POST /api/v1/auth/login/sso
    Called from ui/auth.py's @cl.oauth_callback after Chainlit has already
    completed the Google OAuth handshake and verified the user's identity.
    Requires INTERNAL_SERVICE_SECRET to be set in this service's own env —
    same value the API service has, used to prove this call is genuinely
    coming from our own Chainlit backend and not an arbitrary caller.
    Returns: {access_token, role, username, full_name, expires_in}
    """
    import os
    service_key = os.getenv("INTERNAL_SERVICE_SECRET", "")
    if not service_key:
        raise APIError(
            "INTERNAL_SERVICE_SECRET is not set in the Chainlit environment — "
            "SSO login cannot proceed."
        )
    try:
        r = httpx.post(
            f"{API_URL}/api/v1/auth/login/sso",
            json={"email": email, "full_name": full_name},
            headers={"X-Internal-Service-Key": service_key},
            timeout=httpx.Timeout(10, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Cannot reach API at {API_URL}. Is it running? ({e})")


# ── Query ─────────────────────────────────────────────────────────────────────

def query(
    query_text:        str,
    token:             str,
    collection_name:   str   = "default",
    llm_provider:      str   = "openai",
    llm_model:         str   = "gpt-4o-mini",
    temperature:       float = 0.0,
    top_k:             int   = 5,
    retrieval_mode:    str   = "hybrid",
    rerank:            bool  = True,
    embedding_provider:str   = "openai",
    embedding_model:   str   = "text-embedding-3-small",
    domain:            str   = "uploaded documents",
    assistant_name:    str   = "Document Assistant",
    user_role:         str   = "user",
    chat_history:      list  = None,
    conversation_id:   str   = None,
    skip_cache:        bool  = False,
) -> dict:
    """
    POST /api/v1/query
    Returns full QueryResponse dict.
    """
    try:
        r = httpx.post(
            f"{API_URL}/api/v1/query",
            json={
                "query":              query_text,
                "collection_name":    collection_name,
                "llm_provider":       llm_provider,
                "llm_model":          llm_model,
                "temperature":        temperature,
                "top_k":              top_k,
                "retrieval_mode":     retrieval_mode,
                "rerank":             rerank,
                "embedding_provider": embedding_provider,
                "embedding_model":    embedding_model,
                "domain":             domain,
                "assistant_name":     assistant_name,
                "user_role":          user_role,
                "chat_history":       chat_history or [],
                "conversation_id":    conversation_id,
                "skip_cache":         skip_cache,
            },
            headers=_headers(token),
            timeout=httpx.Timeout(API_TIMEOUT, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except httpx.TimeoutException:
        raise APIError(
            "Request timed out. The query is taking longer than expected. "
            f"Try increasing API_TIMEOUT_SECONDS (currently {API_TIMEOUT}s) in .env",
            status_code=408,
        )
    except Exception as e:
        raise APIError(f"Connection error: {e}")


# ── Index ─────────────────────────────────────────────────────────────────────

def index_files(
    files: list,           # list of (filename, bytes, mime_type)
    token: str,
    username:           str = "default",
    ocr_method:         str = "easyocr",
    use_vision:         bool = False,
    caption_images:     bool = False,
    caption_provider:   str  = "openai",
    caption_model:      str  = "gpt-4o-mini",
    chunk_size:         int  = 500,
    chunk_overlap:      int  = 50,
    chunk_method:       str  = "recursive",
    embedding_provider: str  = "openai",
    embedding_model:    str  = "text-embedding-3-small",
) -> dict:
    """
    POST /api/v1/index (multipart)
    Returns: {status, collection_name, chunks_total, elapsed_seconds,
              total_sections, scanned_sections, charts_detected,
              charts_interpreted, images_detected, images_captioned, ...}
    """
    try:
        file_tuples = [
            ("files", (name, data, mime))
            for name, data, mime in files
        ]
        r = httpx.post(
            f"{API_URL}/api/v1/index",
            files=file_tuples,
            data={
                "username":           username,
                "ocr_method":         ocr_method,
                "use_vision":         str(use_vision).lower(),
                "caption_images":     str(caption_images).lower(),
                "caption_provider":   caption_provider,
                "caption_model":      caption_model,
                "chunk_size":         str(chunk_size),
                "chunk_overlap":      str(chunk_overlap),
                "chunk_method":       chunk_method,
                "embedding_provider": embedding_provider,
                "embedding_model":    embedding_model,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(API_TIMEOUT, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Indexing failed: {e}")


# ── Admin ─────────────────────────────────────────────────────────────────────

def get_logs(
    token:       str,
    user_id:     str  = None,
    errors_only: bool = False,
    page:        int  = 1,
    limit:       int  = 50,
) -> dict:
    """GET /api/v1/admin/logs"""
    try:
        params = {"page": page, "limit": limit, "errors_only": errors_only}
        if user_id:
            params["user_id"] = user_id
        r = httpx.get(
            f"{API_URL}/api/v1/admin/logs",
            params=params,
            headers=_headers(token),
            timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not fetch logs: {e}")


def get_stats(token: str, hours: int = 24) -> dict:
    """GET /api/v1/admin/stats"""
    try:
        r = httpx.get(
            f"{API_URL}/api/v1/admin/stats",
            params={"hours": hours},
            headers=_headers(token),
            timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not fetch stats: {e}")


# ── User management (admin only) ────────────────────────────────────────────

def list_users(token: str, page_token: str = None, limit: int = 100) -> dict:
    """GET /api/v1/admin/users -> {users: [...], next_page_token}"""
    try:
        params = {"limit": limit}
        if page_token:
            params["page_token"] = page_token
        r = httpx.get(f"{API_URL}/api/v1/admin/users", params=params,
                       headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not list users: {e}")


def create_user(token: str, email: str, password: str, role: str = "user",
                 full_name: str = "") -> dict:
    """POST /api/v1/admin/users"""
    try:
        r = httpx.post(
            f"{API_URL}/api/v1/admin/users",
            json={"email": email, "password": password, "role": role, "full_name": full_name},
            headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not create user: {e}")


def set_user_role(token: str, uid: str, role: str) -> dict:
    """PATCH /api/v1/admin/users/{uid}/role"""
    try:
        r = httpx.patch(
            f"{API_URL}/api/v1/admin/users/{uid}/role",
            json={"role": role}, headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not set role: {e}")


def disable_user(token: str, uid: str) -> dict:
    """POST /api/v1/admin/users/{uid}/disable"""
    try:
        r = httpx.post(f"{API_URL}/api/v1/admin/users/{uid}/disable",
                        headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not disable user: {e}")


def enable_user(token: str, uid: str) -> dict:
    """POST /api/v1/admin/users/{uid}/enable"""
    try:
        r = httpx.post(f"{API_URL}/api/v1/admin/users/{uid}/enable",
                        headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not enable user: {e}")


def delete_user(token: str, uid: str) -> None:
    """DELETE /api/v1/admin/users/{uid}"""
    try:
        r = httpx.delete(f"{API_URL}/api/v1/admin/users/{uid}",
                          headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        if r.status_code != 204:
            _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not delete user: {e}")


def health() -> dict:
    """GET /health — check if API is reachable."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=httpx.Timeout(5, connect=CONNECT_TIMEOUT))
        return r.json()
    except Exception as e:
        raise APIError(f"API unreachable at {API_URL}: {e}")

# ── Usage / FinOps (admin only) ─────────────────────────────────────────────

def get_usage(token: str, user_id: str = None, from_: str = None, to: str = None) -> dict:
    """GET /api/v1/admin/usage -> {period_start, period_end, rows: [...], note}"""
    try:
        params = {}
        if user_id:
            params["user_id"] = user_id
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to
        r = httpx.get(f"{API_URL}/api/v1/admin/usage", params=params,
                       headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not fetch usage: {e}")


def get_usage_timeseries(token: str, user_id: str = None, from_: str = None,
                          to: str = None, group_by: str = "day") -> dict:
    """GET /api/v1/admin/usage/timeseries -> {period_start, period_end, points: [...]}"""
    try:
        params = {"group_by": group_by}
        if user_id:
            params["user_id"] = user_id
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to
        r = httpx.get(f"{API_URL}/api/v1/admin/usage/timeseries", params=params,
                       headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not fetch usage timeseries: {e}")


def get_usage_by_model(token: str, user_id: str = None, from_: str = None, to: str = None) -> dict:
    """GET /api/v1/admin/usage/by-model -> {period_start, period_end, rows: [...]}"""
    try:
        params = {}
        if user_id:
            params["user_id"] = user_id
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to
        r = httpx.get(f"{API_URL}/api/v1/admin/usage/by-model", params=params,
                       headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not fetch usage by model: {e}")


def get_user_quota(token: str, uid: str) -> dict:
    """GET /api/v1/admin/usage/{uid}/quota"""
    try:
        r = httpx.get(f"{API_URL}/api/v1/admin/usage/{uid}/quota",
                       headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not fetch quota: {e}")


def set_user_quota(token: str, uid: str, tokens: int = None,
                    cost: float = None, requests: int = None) -> dict:
    """PATCH /api/v1/admin/usage/{uid}/quota"""
    try:
        body = {
            "monthly_token_limit": tokens,
            "monthly_cost_limit_usd": cost,
            "monthly_request_limit": requests,
        }
        r = httpx.patch(f"{API_URL}/api/v1/admin/usage/{uid}/quota", json=body,
                         headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not set quota: {e}")


def delete_user_quota(token: str, uid: str) -> None:
    """DELETE /api/v1/admin/usage/{uid}/quota"""
    try:
        r = httpx.delete(f"{API_URL}/api/v1/admin/usage/{uid}/quota",
                          headers=_headers(token), timeout=httpx.Timeout(15, connect=CONNECT_TIMEOUT))
        if r.status_code != 204:
            _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not delete quota: {e}")

def delete_collection(token: str, collection_name: str) -> dict:
    """DELETE /api/v1/index/{collection_name}
    Actually removes the collection server-side (Qdrant + BM25 + cache) --
    NOT the same as Chainlit's /clear, which only resets client-side
    session state. See ui/commands.py's /clear handler, which calls this
    to actually purge server data too (confirmed missing before -- chunks
    from every prior indexing run this session were silently accumulating
    in the same collection, never actually deleted)."""
    try:
        r = httpx.delete(
            f"{API_URL}/api/v1/index/{collection_name}",
            headers=_headers(token),
            timeout=httpx.Timeout(30, connect=CONNECT_TIMEOUT),
        )
        return _handle(r)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Could not delete collection: {e}")