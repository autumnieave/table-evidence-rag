"""Offline Redis sliding-window tests; live Redis E2E lives under eval/integration/."""

from __future__ import annotations

import fnmatch
import time

import pytest

from utils.api_security import (
    RedisSlidingWindowRateLimiter,
    SlidingWindowRateLimiter,
    create_rate_limiter,
)


class FakeRedis:
    """Returns preset eval results for wrapper-level tests."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def eval(self, *args, **kwargs):
        self.calls += 1
        value = self.results.pop(0) if self.results else [1, 0, 1]
        if isinstance(value, Exception):
            raise value
        return value


class SlidingWindowFakeRedis:
    """Deterministic local implementation of the Lua zset sliding-window contract."""

    def __init__(self) -> None:
        self.buckets: dict[str, dict[str, int]] = {}

    def eval(self, script, numkeys, key, limit, window_ms, member, now_ms):
        assert numkeys == 1
        now = int(now_ms) if str(now_ms) else int(time.time() * 1000)
        window = int(window_ms)
        bucket = self.buckets.setdefault(str(key), {})
        cutoff = now - window
        for old_member, score in list(bucket.items()):
            if int(score) <= cutoff:
                del bucket[old_member]
        if len(bucket) >= int(limit):
            oldest = min(bucket.values())
            retry_ms = max(0, int(oldest) + window - now)
            return [0, retry_ms, len(bucket)]
        bucket[str(member)] = now
        return [1, 0, len(bucket) + 1]

    def scan_iter(self, match: str):
        return (key for key in self.buckets if fnmatch.fnmatch(key, match))

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.buckets.pop(key, None)


def test_allow_then_reject_retry_after() -> None:
    client = FakeRedis([[1, 0, 1], [1, 0, 2], [0, 12500, 20]])
    limiter = RedisSlidingWindowRateLimiter(client, limit=2, window_seconds=60)

    assert limiter.allow("u1") == (True, 0.0)
    assert limiter.allow("u1") == (True, 0.0)
    allowed, retry_after = limiter.allow("u1")
    assert allowed is False
    assert retry_after == pytest.approx(12.5)
    assert client.calls == 3


def test_redis_failure_degrades_to_memory() -> None:
    client = FakeRedis([RuntimeError("redis down")])
    limiter = RedisSlidingWindowRateLimiter(client, limit=2, window_seconds=60)

    assert limiter.allow("u1") == (True, 0.0)
    assert limiter.degraded is True
    assert limiter.allow("u1") == (True, 0.0)
    allowed, retry_after = limiter.allow("u1")
    assert allowed is False
    assert retry_after > 0


def test_factory_falls_back_when_redis_unreachable() -> None:
    limiter = create_rate_limiter(
        limit=3,
        backend="redis",
        redis_url="redis://127.0.0.1:1/0",
        prefix="test:fallback",
    )
    assert isinstance(limiter, SlidingWindowRateLimiter)


def test_redis_sliding_window_lua_semantics() -> None:
    client = SlidingWindowFakeRedis()
    prefix = "test:rl:window"
    limiter = RedisSlidingWindowRateLimiter(
        client, limit=3, window_seconds=60, prefix=prefix
    )
    try:
        assert all(limiter.allow("u1", now=1000 + i)[0] for i in range(3))
        allowed, retry_after = limiter.allow("u1", now=1002)
        assert allowed is False
        assert retry_after == pytest.approx(58.0)

        # The oldest hit expires after 60 seconds, so the window opens again.
        assert limiter.allow("u1", now=1061)[0] is True
    finally:
        limiter.reset()


def test_redis_namespaces_are_isolated() -> None:
    client = SlidingWindowFakeRedis()
    prefix = "test:rl:namespace"
    limiter = RedisSlidingWindowRateLimiter(
        client, limit=1, window_seconds=60, prefix=prefix
    )
    try:
        assert limiter.allow("u1", now=2000)[0] is True
        assert limiter.allow("u1", now=2001)[0] is False
        assert limiter.allow("u2", now=2002)[0] is True
        assert len(client.buckets) == 2
    finally:
        limiter.reset()
