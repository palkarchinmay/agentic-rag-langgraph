"""
llm_guard.py — LLM retry, throttling, rate limiting and circuit breaker.

Three protection layers:
  1. Retry        — exponential backoff on transient errors (429, 500, 503)
  2. Rate limiter — per-user RPM/TPM caps to control cost
  3. Circuit breaker — stops calling LLM after N consecutive failures

Usage (wraps any LangChain LLM call):
    from app.services.llm_guard import guarded_invoke
    response = guarded_invoke(llm, prompt, user_id="user_admin")
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("financial_rag.llm_guard")


# ── Configuration (override via .env) ─────────────────────────────────────────
import os

MAX_RETRIES          = int(os.getenv("LLM_MAX_RETRIES",        "3"))
RETRY_BASE_DELAY     = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))  # seconds
RETRY_MAX_DELAY      = float(os.getenv("LLM_RETRY_MAX_DELAY",  "30.0")) # seconds
REQUEST_TIMEOUT      = float(os.getenv("LLM_REQUEST_TIMEOUT",  "60.0")) # seconds

# Rate limits per user (rolling 60s window)
RPM_LIMIT            = int(os.getenv("LLM_RPM_LIMIT",          "20"))   # requests/min
TPM_LIMIT            = int(os.getenv("LLM_TPM_LIMIT",          "50000"))# tokens/min

# Circuit breaker
CB_FAILURE_THRESHOLD = int(os.getenv("LLM_CB_FAILURES",        "5"))    # consecutive failures
CB_RECOVERY_TIMEOUT  = float(os.getenv("LLM_CB_TIMEOUT",       "60.0")) # seconds before retry

# Retryable HTTP status codes / error strings
RETRYABLE_STATUS     = {429, 500, 502, 503, 504}
RETRYABLE_PHRASES    = [
    "rate limit", "too many requests", "service unavailable",
    "internal server error", "gateway timeout", "overloaded",
    "connection error", "timeout", "retry",
]


# ── Per-user rate limiter ──────────────────────────────────────────────────────

@dataclass
class UserBucket:
    """Rolling window rate limiter for one user."""
    request_times: deque = field(default_factory=deque)  # timestamps of requests
    token_counts:  deque = field(default_factory=deque)  # (timestamp, tokens) pairs
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check_and_record(self, estimated_tokens: int = 500) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        Records the request if allowed.
        """
        now = time.time()
        window = 60.0

        with self.lock:
            # Evict old entries outside rolling window
            while self.request_times and now - self.request_times[0] > window:
                self.request_times.popleft()
            while self.token_counts and now - self.token_counts[0][0] > window:
                self.token_counts.popleft()

            rpm_used = len(self.request_times)
            tpm_used = sum(t for _, t in self.token_counts)

            if rpm_used >= RPM_LIMIT:
                wait = window - (now - self.request_times[0])
                return False, f"Rate limit: {rpm_used}/{RPM_LIMIT} RPM. Wait {wait:.0f}s."

            if tpm_used + estimated_tokens > TPM_LIMIT:
                wait = window - (now - self.token_counts[0][0])
                return False, f"Token limit: {tpm_used}/{TPM_LIMIT} TPM. Wait {wait:.0f}s."

            # Record this request
            self.request_times.append(now)
            self.token_counts.append((now, estimated_tokens))
            return True, "ok"

    def record_actual_tokens(self, tokens: int):
        """Update last entry with actual token count."""
        with self.lock:
            if self.token_counts:
                ts, _ = self.token_counts[-1]
                self.token_counts[-1] = (ts, tokens)


# ── Circuit breaker ────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Three states:
      CLOSED   → normal operation
      OPEN     → failing, reject requests immediately
      HALF_OPEN → testing recovery after timeout
    """
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self):
        self.state             = self.CLOSED
        self.failure_count     = 0
        self.last_failure_time = 0.0
        self.lock              = threading.Lock()

    def call_allowed(self) -> tuple[bool, str]:
        with self.lock:
            if self.state == self.CLOSED:
                return True, "ok"
            if self.state == self.OPEN:
                if time.time() - self.last_failure_time >= CB_RECOVERY_TIMEOUT:
                    self.state = self.HALF_OPEN
                    logger.info("Circuit breaker: HALF_OPEN — testing recovery")
                    return True, "half_open"
                wait = CB_RECOVERY_TIMEOUT - (time.time() - self.last_failure_time)
                return False, f"Circuit open — LLM failing. Retry in {wait:.0f}s."
            # HALF_OPEN — allow one test call
            return True, "half_open"

    def record_success(self):
        with self.lock:
            self.failure_count = 0
            if self.state != self.CLOSED:
                logger.info("Circuit breaker: CLOSED — recovery confirmed")
            self.state = self.CLOSED

    def record_failure(self):
        with self.lock:
            self.failure_count    += 1
            self.last_failure_time = time.time()
            if self.failure_count >= CB_FAILURE_THRESHOLD:
                if self.state != self.OPEN:
                    logger.warning(
                        "Circuit breaker: OPEN after %d failures", self.failure_count
                    )
                self.state = self.OPEN


# ── Global singletons ──────────────────────────────────────────────────────────

_user_buckets:  dict[str, UserBucket] = {}
_buckets_lock   = threading.Lock()
_circuit_breaker = CircuitBreaker()


def _get_bucket(user_id: str) -> UserBucket:
    with _buckets_lock:
        if user_id not in _user_buckets:
            _user_buckets[user_id] = UserBucket()
        return _user_buckets[user_id]


# ── Retry helper ───────────────────────────────────────────────────────────────

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(p in msg for p in RETRYABLE_PHRASES):
        return True
    # Check status code in exception
    for attr in ("status_code", "code", "http_status"):
        code = getattr(exc, attr, None)
        if code in RETRYABLE_STATUS:
            return True
    return False


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Parse Retry-After header value from exception if present."""
    msg = str(exc)
    import re
    m = re.search(r"retry.after[:\s]+(\d+)", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ── Main guarded invoke ────────────────────────────────────────────────────────

def guarded_invoke(
    llm,
    prompt:           str,
    user_id:          str = "default",
    estimated_tokens: int = 500,
) -> Any:
    """
    Invoke LLM with full protection:
      - Rate limiting (per user)
      - Circuit breaker (per provider)
      - Exponential backoff retry
      - Request timeout

    Returns LLM response object.
    Raises RuntimeError with clear message on permanent failure.
    """
    # ── 1. Rate limit check ───────────────────────────────────────────────────
    bucket  = _get_bucket(user_id)
    allowed, reason = bucket.check_and_record(estimated_tokens)
    if not allowed:
        logger.warning("Rate limited user '%s': %s", user_id, reason)
        raise RateLimitError(f"⚠️ Rate limit reached: {reason}")

    # ── 2. Circuit breaker check ──────────────────────────────────────────────
    cb_allowed, cb_state = _circuit_breaker.call_allowed()
    if not cb_allowed:
        logger.warning("Circuit open: %s", reason)
        raise CircuitOpenError(f"⚠️ LLM service unavailable: {reason}")

    # ── 3. Retry with exponential backoff ─────────────────────────────────────
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.debug("LLM call attempt %d/%d", attempt + 1, MAX_RETRIES + 1)

            # Timeout wrapper
            result    = [None]
            exc_holder= [None]

            def _invoke():
                try:
                    result[0] = llm.invoke(prompt)
                except Exception as e:
                    exc_holder[0] = e

            thread = threading.Thread(target=_invoke, daemon=True)
            thread.start()
            thread.join(timeout=REQUEST_TIMEOUT)

            if thread.is_alive():
                raise TimeoutError(
                    f"LLM request timed out after {REQUEST_TIMEOUT}s"
                )
            if exc_holder[0]:
                raise exc_holder[0]

            # ── Success ───────────────────────────────────────────────────────
            _circuit_breaker.record_success()

            # Update token count with actuals if available
            resp = result[0]
            usage = getattr(resp, "response_metadata", {}) or {}
            actual_tokens = (
                usage.get("token_usage", {}).get("total_tokens") or
                usage.get("usage", {}).get("total_tokens") or
                estimated_tokens
            )
            bucket.record_actual_tokens(actual_tokens)
            logger.debug("LLM call succeeded on attempt %d", attempt + 1)
            return resp

        except (RateLimitError, CircuitOpenError):
            raise  # don't retry our own guards

        except Exception as exc:
            last_exc = exc
            _circuit_breaker.record_failure()

            if not _is_retryable(exc):
                logger.error("Non-retryable LLM error: %s", exc)
                raise

            if attempt >= MAX_RETRIES:
                break

            # Exponential backoff: 1s, 2s, 4s, ... capped at RETRY_MAX_DELAY
            retry_after = _extract_retry_after(exc)
            delay = retry_after or min(
                RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY
            )
            logger.warning(
                "LLM error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_RETRIES, delay, str(exc)[:100]
            )
            time.sleep(delay)

    raise RuntimeError(
        f"LLM failed after {MAX_RETRIES} retries: {str(last_exc)[:200]}"
    )


# ── Custom exceptions ──────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised when user exceeds RPM or TPM limit."""

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""


# ── Status helpers (for diagnostics UI) ───────────────────────────────────────

def get_circuit_status() -> dict:
    return {
        "state":          _circuit_breaker.state,
        "failures":       _circuit_breaker.failure_count,
        "threshold":      CB_FAILURE_THRESHOLD,
        "recovery_in":    max(0, CB_RECOVERY_TIMEOUT -
                             (time.time() - _circuit_breaker.last_failure_time))
                          if _circuit_breaker.state == CircuitBreaker.OPEN else 0,
    }

def get_user_usage(user_id: str) -> dict:
    bucket = _get_bucket(user_id)
    now    = time.time()
    with bucket.lock:
        rpm  = sum(1 for t in bucket.request_times if now - t <= 60)
        tpm  = sum(tk for ts, tk in bucket.token_counts if now - ts <= 60)
    return {
        "rpm":       rpm,
        "rpm_limit": RPM_LIMIT,
        "tpm":       tpm,
        "tpm_limit": TPM_LIMIT,
    }

def reset_circuit_breaker():
    """Manual reset — useful when provider recovers."""
    _circuit_breaker.record_success()
    logger.info("Circuit breaker manually reset")