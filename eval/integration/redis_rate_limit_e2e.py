"""Live Redis sliding-window smoke test (not collected by pytest tests/)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import uuid

REDIS_URL = os.getenv("REDIS_E2E_URL", "redis://127.0.0.1:6379/15")


def run() -> int:
    try:
        import redis
        from utils.api_security import RedisSlidingWindowRateLimiter
    except Exception as exc:
        print(f"SKIP: missing dependency: {exc}")
        return 0

    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=1.0,
        )
        client.ping()
    except Exception as exc:
        print(f"SKIP: Redis is not reachable at {REDIS_URL}: {exc}")
        return 0

    prefix = f"e2e:rl:{uuid.uuid4().hex[:8]}"
    limiter = RedisSlidingWindowRateLimiter(
        client, limit=3, window_seconds=60, prefix=prefix
    )
    try:
        assert all(limiter.allow("u1", now=1000 + i)[0] for i in range(3))
        allowed, retry_after = limiter.allow("u1", now=1002)
        assert allowed is False
        assert retry_after == 58.0
        assert limiter.allow("u1", now=1061)[0] is True
    finally:
        limiter.reset()

    print(f"PASS: Redis Lua sliding window at {REDIS_URL}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
