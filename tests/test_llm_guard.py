"""
test_llm_guard.py — Comprehensive tests for LLM retry, rate limiting, circuit breaker.

Run: python -m pytest tests/test_llm_guard.py -v
  or: python tests/test_llm_guard.py

Tests cover:
  1. Rate limiter — single user, multiple users simultaneously
  2. Circuit breaker — open/close/recover cycle
  3. Retry — exponential backoff, non-retryable errors
  4. Concurrent users — thread safety
  5. Integration — all layers together
"""

import os, sys, time, threading, unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Override limits for fast testing
os.environ["LLM_RPM_LIMIT"]        = "5"
os.environ["LLM_TPM_LIMIT"]        = "1000"
os.environ["LLM_MAX_RETRIES"]      = "2"
os.environ["LLM_RETRY_BASE_DELAY"] = "0.05"
os.environ["LLM_RETRY_MAX_DELAY"]  = "0.2"
os.environ["LLM_REQUEST_TIMEOUT"]  = "3.0"
os.environ["LLM_CB_FAILURES"]      = "3"
os.environ["LLM_CB_TIMEOUT"]       = "1.0"

# llm_resilience reads these env vars at import time. If another test module
# imported it first (with the defaults), reload it so the values above apply.
import importlib
import app.services.llm_resilience as _llm_resilience
importlib.reload(_llm_resilience)

from app.services.llm_resilience import (
    UserBucket, CircuitBreaker, guarded_invoke,
    RateLimitError, CircuitOpenError,
    get_circuit_status, reset_circuit_breaker,
    _user_buckets, _circuit_breaker,
    RPM_LIMIT, TPM_LIMIT, MAX_RETRIES,
    CB_FAILURE_THRESHOLD, CB_RECOVERY_TIMEOUT,
)


def _mock_llm(response="ok", fail_times=0, error_msg="503 service unavailable"):
    calls = [0]
    class MockLLM:
        def invoke(self, prompt):
            calls[0] += 1
            if calls[0] <= fail_times:
                raise Exception(error_msg)
            r = MagicMock()
            r.content = response
            r.response_metadata = {"token_usage": {"total_tokens": 50}}
            return r
    return MockLLM(), calls


def _reset():
    _user_buckets.clear()
    reset_circuit_breaker()


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────
class TestRateLimiter(unittest.TestCase):

    def setUp(self): _reset()

    def test_within_limit_allowed(self):
        """All requests within RPM_LIMIT should succeed."""
        llm, calls = _mock_llm()
        for _ in range(RPM_LIMIT):
            guarded_invoke(llm, "q", user_id="u1")
        self.assertEqual(calls[0], RPM_LIMIT)
        print(f"\n    ✅ {RPM_LIMIT}/{RPM_LIMIT} requests allowed")

    def test_exceeding_limit_blocked(self):
        """Request #RPM_LIMIT+1 must raise RateLimitError."""
        llm, _ = _mock_llm()
        for _ in range(RPM_LIMIT):
            guarded_invoke(llm, "q", user_id="u2")
        with self.assertRaises(RateLimitError) as ctx:
            guarded_invoke(llm, "q", user_id="u2")
        print(f"\n    ✅ Blocked: {ctx.exception}")

    def test_users_are_independent(self):
        """Exhausting userA limit must NOT affect userB."""
        llm, _ = _mock_llm()
        for _ in range(RPM_LIMIT):
            guarded_invoke(llm, "q", user_id="userA")
        with self.assertRaises(RateLimitError):
            guarded_invoke(llm, "q", user_id="userA")
        # userB still fine
        resp = guarded_invoke(llm, "q", user_id="userB")
        self.assertEqual(resp.content, "ok")
        print(f"\n    ✅ userA blocked, userB unaffected")

    def test_window_resets(self):
        """Old requests (>60s) should not count toward limit."""
        bucket = UserBucket()
        from collections import deque
        old = time.time() - 61
        bucket.request_times = deque([old] * RPM_LIMIT)
        allowed, _ = bucket.check_and_record()
        self.assertTrue(allowed)
        print(f"\n    ✅ Old requests expired from window")

    def test_token_limit(self):
        """Requests that would exceed TPM_LIMIT should be blocked."""
        bucket = UserBucket()
        bucket.check_and_record(estimated_tokens=TPM_LIMIT - 10)
        allowed, reason = bucket.check_and_record(estimated_tokens=100)
        self.assertFalse(allowed)
        self.assertIn("Token", reason)
        print(f"\n    ✅ Token limit enforced: {reason}")

    def test_20_concurrent_users(self):
        """20 simultaneous users should each succeed (independent buckets)."""
        ok, fail = [0], [0]
        lock = threading.Lock()

        def task(uid):
            try:
                llm, _ = _mock_llm()
                guarded_invoke(llm, "q", user_id=uid)
                with lock: ok[0] += 1
            except Exception:
                with lock: fail[0] += 1

        threads = [threading.Thread(target=task, args=(f"cu_{i}",)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(ok[0], 20)
        self.assertEqual(fail[0], 0)
        print(f"\n    ✅ 20 concurrent users: {ok[0]} success, {fail[0]} failed")

    def test_same_user_concurrent_safe(self):
        """Same user hammering concurrently — thread-safe bucket."""
        ok, blocked = [0], [0]
        lock = threading.Lock()

        def task():
            try:
                llm, _ = _mock_llm()
                guarded_invoke(llm, "q", user_id="power_user")
                with lock: ok[0] += 1
            except RateLimitError:
                with lock: blocked[0] += 1

        n = RPM_LIMIT * 2
        threads = [threading.Thread(target=task) for _ in range(n)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(ok[0] + blocked[0], n)
        self.assertLessEqual(ok[0], RPM_LIMIT)
        self.assertGreater(blocked[0], 0)
        print(f"\n    ✅ Thread-safe: {ok[0]} ok, {blocked[0]} blocked (no data race)")


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────
class TestCircuitBreaker(unittest.TestCase):

    def setUp(self): _reset()

    def test_starts_closed(self):
        cb = CircuitBreaker()
        ok, _ = cb.call_allowed()
        self.assertTrue(ok)
        print(f"\n    ✅ Circuit starts CLOSED")

    def test_opens_after_failures(self):
        cb = CircuitBreaker()
        for _ in range(CB_FAILURE_THRESHOLD): cb.record_failure()
        ok, reason = cb.call_allowed()
        self.assertFalse(ok)
        self.assertEqual(cb.state, "open")
        print(f"\n    ✅ Circuit OPEN after {CB_FAILURE_THRESHOLD} failures")

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker()
        for _ in range(CB_FAILURE_THRESHOLD): cb.record_failure()
        cb.last_failure_time = time.time() - CB_RECOVERY_TIMEOUT - 0.1
        ok, state = cb.call_allowed()
        self.assertTrue(ok)
        self.assertEqual(cb.state, "half_open")
        print(f"\n    ✅ Circuit HALF_OPEN after timeout")

    def test_closes_after_recovery(self):
        cb = CircuitBreaker()
        for _ in range(CB_FAILURE_THRESHOLD): cb.record_failure()
        cb.last_failure_time = time.time() - CB_RECOVERY_TIMEOUT - 0.1
        cb.call_allowed()
        cb.record_success()
        self.assertEqual(cb.state, "closed")
        print(f"\n    ✅ Circuit CLOSED after recovery")

    def test_open_rejects_without_calling_llm(self):
        """When OPEN, LLM must NOT be called — instant rejection."""
        for _ in range(CB_FAILURE_THRESHOLD): _circuit_breaker.record_failure()
        llm, calls = _mock_llm()
        with self.assertRaises(CircuitOpenError):
            guarded_invoke(llm, "q", user_id="cb_test")
        self.assertEqual(calls[0], 0)
        print(f"\n    ✅ Circuit OPEN: LLM not called (calls={calls[0]})")

    def test_open_rejection_is_instant(self):
        """Open circuit rejects in <50ms — no waiting."""
        for _ in range(CB_FAILURE_THRESHOLD): _circuit_breaker.record_failure()
        llm, _ = _mock_llm()
        t0 = time.time()
        try:
            guarded_invoke(llm, "q", user_id="speed_test")
        except CircuitOpenError:
            pass
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.05)
        print(f"\n    ✅ Rejected in {elapsed*1000:.1f}ms (instant)")


# ─────────────────────────────────────────────────────────────────────────────
# RETRY
# ─────────────────────────────────────────────────────────────────────────────
class TestRetry(unittest.TestCase):

    def setUp(self): _reset()

    def test_success_first_try(self):
        llm, calls = _mock_llm()
        resp = guarded_invoke(llm, "q", user_id="r1")
        self.assertEqual(calls[0], 1)
        print(f"\n    ✅ 1 call, succeeded immediately")

    def test_retry_on_503(self):
        """Fails once with 503, then succeeds on retry."""
        llm, calls = _mock_llm(fail_times=1, error_msg="503 service unavailable")
        resp = guarded_invoke(llm, "q", user_id="r2")
        self.assertEqual(resp.content, "ok")
        self.assertEqual(calls[0], 2)
        print(f"\n    ✅ Recovered after 1 retry ({calls[0]} calls total)")

    def test_gives_up_after_max_retries(self):
        """
        After MAX_RETRIES attempts the guard raises.
        Note: circuit breaker may open partway through — that's fine,
        either RuntimeError or CircuitOpenError means "gave up".
        What matters: LLM is not called indefinitely.
        """
        llm, calls = _mock_llm(fail_times=999, error_msg="500 internal error")
        with self.assertRaises((RuntimeError, Exception)):
            guarded_invoke(llm, "q", user_id="r3")
        # Should have stopped at MAX_RETRIES+1 OR earlier via circuit breaker
        self.assertLessEqual(calls[0], MAX_RETRIES + 1)
        self.assertGreater(calls[0], 0)
        print(f"\n    ✅ Gave up after {calls[0]} calls (max={MAX_RETRIES+1})")

    def test_no_retry_on_auth_error(self):
        """401 = permanent error, should not retry."""
        llm, calls = _mock_llm(fail_times=999, error_msg="401 Unauthorized")
        with self.assertRaises(Exception):
            guarded_invoke(llm, "q", user_id="r4")
        self.assertEqual(calls[0], 1)
        print(f"\n    ✅ No retry on auth error ({calls[0]} call only)")

    def test_timeout_kills_slow_llm(self):
        class SlowLLM:
            def invoke(self, _):
                time.sleep(10)
        t0 = time.time()
        with self.assertRaises(Exception):
            guarded_invoke(SlowLLM(), "q", user_id="r5")
        elapsed = time.time() - t0
        self.assertLess(elapsed, 5.0)  # killed well before 10s
        print(f"\n    ✅ Slow LLM killed after {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — full outage + recovery scenario
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration(unittest.TestCase):

    def setUp(self): _reset()

    def test_outage_and_recovery(self):
        """
        1. Normal requests work
        2. Provider goes down → circuit opens
        3. Requests rejected instantly
        4. Provider recovers → circuit closes
        5. Normal operation resumes
        """
        print(f"\n    🎬 Full outage + recovery scenario")

        llm_good, _ = _mock_llm("good")
        llm_bad,  _ = _mock_llm(fail_times=999, error_msg="503 unavailable")

        # 1. Normal
        guarded_invoke(llm_good, "q", user_id="int1")
        print(f"    ✅ Phase 1: Normal operation working")

        # 2. Trigger circuit open via failures
        for i in range(CB_FAILURE_THRESHOLD):
            try: guarded_invoke(llm_bad, "q", user_id=f"fail_{i}")
            except: pass
        self.assertEqual(get_circuit_status()["state"], "open")
        print(f"    ✅ Phase 2: Circuit OPEN after outage")

        # 3. Fast rejection
        t0 = time.time()
        try: guarded_invoke(llm_bad, "q", user_id="fast")
        except CircuitOpenError: pass
        self.assertLess(time.time() - t0, 0.05)
        print(f"    ✅ Phase 3: Requests rejected instantly")

        # 4. Simulate recovery timeout
        _circuit_breaker.last_failure_time = time.time() - CB_RECOVERY_TIMEOUT - 0.1
        print(f"    ✅ Phase 4: Recovery timeout elapsed")

        # 5. Normal operation resumes
        resp = guarded_invoke(llm_good, "q", user_id="recovered")
        self.assertEqual(resp.content, "good")
        self.assertEqual(get_circuit_status()["state"], "closed")
        print(f"    ✅ Phase 5: Circuit CLOSED — full recovery!")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestRateLimiter, TestCircuitBreaker, TestRetry, TestIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    print("\n" + "="*60)
    print("  LLM GUARD TEST SUITE")
    print(f"  RPM={RPM_LIMIT} | TPM={TPM_LIMIT} | Retries={MAX_RETRIES}")
    print(f"  CB threshold={CB_FAILURE_THRESHOLD} | Timeout={CB_RECOVERY_TIMEOUT}s")
    print("="*60)

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\n" + "="*60)
    print(f"  Run: {result.testsRun}  Fail: {len(result.failures)}  "
          f"Error: {len(result.errors)}")
    print("="*60)
    sys.exit(0 if result.wasSuccessful() else 1)
