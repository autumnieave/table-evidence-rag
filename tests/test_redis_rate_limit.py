"""Redis 滑动窗口限流单测：假客户端 + 真实 Redis（可用时）。"""

from __future__ import annotations

import uuid

import pytest

from utils.api_security import (
    RedisSlidingWindowRateLimiter,
    SlidingWindowRateLimiter,
    create_rate_limiter,
)


class FakeRedis:
    """按预设序列返回 eval 结果。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def eval(self, *args, **kwargs):
        self.calls += 1
        value = self.results.pop(0) if self.results else [1, 0, 1]
        if isinstance(value, Exception):
            raise value
        return value


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


def _redis_client_or_none(url: str):
    try:
        import redis  # noqa: PLC0415

        client = redis.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=1.0
        )
        client.ping()
        return client
    except Exception:  # noqa: BLE001
        return None


REDIS_URL = "redis://127.0.0.1:6379/15"


@pytest.mark.skipif(_redis_client_or_none(REDIS_URL) is None, reason="Redis 不可用")
def test_real_redis_sliding_window() -> None:
    client = _redis_client_or_none(REDIS_URL)
    prefix = f"test:rl:{uuid.uuid4().hex[:8]}"
    limiter = RedisSlidingWindowRateLimiter(
        client, limit=3, window_seconds=60, prefix=prefix
    )
    try:
        assert all(limiter.allow("u1", now=1000 + i)[0] for i in range(3))
        allowed, retry_after = limiter.allow("u1", now=1002)
        assert allowed is False
        assert retry_after == pytest.approx(58.0)

        # 窗口滑过后恢复
        assert limiter.allow("u1", now=1061)[0] is True
    finally:
        limiter.reset()


@pytest.mark.skipif(_redis_client_or_none("redis://127.0.0.1:6379/15") is None, reason="Redis 不可用")
def test_real_redis_namespaces_are_isolated() -> None:
    client = _redis_client_or_none(REDIS_URL)
    prefix = f"test:rl:{uuid.uuid4().hex[:8]}"
    limiter = RedisSlidingWindowRateLimiter(
        client, limit=1, window_seconds=60, prefix=prefix
    )
    try:
        assert limiter.allow("u1", now=2000)[0] is True
        assert limiter.allow("u1", now=2001)[0] is False
        assert limiter.allow("u2", now=2002)[0] is True
    finally:
        limiter.reset()
