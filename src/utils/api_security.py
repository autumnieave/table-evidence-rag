"""API key 解析、提取与滑动窗口限流（Redis 分布式 / 进程内降级）。

设计边界：

- key 只从环境变量 ``API_KEYS``（逗号分隔）读取，不在代码里硬编码；
- 限流优先走 **Redis + Lua 分布式滑动窗口**；Redis 不可用时自动降级单进程内存；
- 时间使用 ``time.monotonic()``，不受系统时钟回拨影响。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Mapping, Optional, Set, Tuple

logger = logging.getLogger(__name__)


#: Redis 滑动窗口 Lua：zset 存时间戳，先清理窗口外，再判断/写入。
#: ARGV = [limit, window_ms, member, now_ms_or_empty]；返回 {allowed, retry_ms, count}。
_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local member = ARGV[3]
local now_ms
if ARGV[4] ~= '' then
  now_ms = tonumber(ARGV[4])
else
  local t = redis.call('TIME')
  now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
end
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry = 0
  if oldest[2] then
    retry = tonumber(oldest[2]) + window_ms - now_ms
    if retry < 0 then retry = 0 end
  end
  return {0, retry, count}
end
redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms + 1000)
return {1, 0, count + 1}
"""


def parse_api_keys(raw: str) -> Set[str]:
    """把 ``k1, k2`` 解析为去空白、去空项的集合。"""
    if not raw:
        return set()
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def extract_api_key(headers: Mapping[str, str]) -> Optional[str]:
    """从 ``X-API-Key`` 或 ``Authorization: Bearer`` 提取 key。"""
    api_key = headers.get("x-api-key") or headers.get("X-API-Key")
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return None


class SlidingWindowRateLimiter:
    """按 key 的 60 秒滑动窗口限流（线程安全）。"""

    backend = "memory"

    def __init__(self, limit: int = 10, window_seconds: float = 60.0) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = float(window_seconds)
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: Optional[float] = None) -> Tuple[bool, float]:
        """返回 ``(是否允许, 需等待秒数)``；拒绝时等待秒数为建议 Retry-After。"""
        current = time.monotonic() if now is None else float(now)
        cutoff = current - self.window_seconds
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(0.0, bucket[0] + self.window_seconds - current)
                return False, round(retry_after, 3)
            bucket.append(current)
            return True, 0.0

    def reset(self, key: Optional[str] = None) -> None:
        """清空某个 key 或全部窗口（测试/运维用）。"""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


class RedisSlidingWindowRateLimiter:
    """基于 Redis zset + Lua 的分布式滑动窗口限流。

    所有节点共享同一份窗口；Lua 在 Redis 内原子执行“清理过期 → 计数 → 写入”，
    避免多 worker 之间的检查-写入竞态。Redis 调用失败时自动降级为进程内限流，
    并置 ``degraded=True`` 供监控。
    """

    backend = "redis"

    def __init__(
        self,
        redis_client: Any,
        limit: int = 20,
        window_seconds: float = 60.0,
        prefix: str = "ratelimit",
    ) -> None:
        self._redis = redis_client
        self.limit = max(1, int(limit))
        self.window_seconds = float(window_seconds)
        self.prefix = str(prefix or "ratelimit").rstrip(":")
        self.degraded = False
        self.last_error: Optional[str] = None
        self._fallback = SlidingWindowRateLimiter(
            limit=self.limit, window_seconds=self.window_seconds
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def allow(self, key: str, now: Optional[float] = None) -> Tuple[bool, float]:
        """返回 ``(是否允许, 需等待秒数)``；拒绝时等待秒数用于 Retry-After。"""
        if self.degraded:
            return self._fallback.allow(key, now)
        now_ms = "" if now is None else str(int(float(now) * 1000))
        member = f"{now_ms or int(time.time() * 1000)}:{uuid.uuid4().hex[:8]}"
        try:
            raw = self._redis.eval(
                _LUA_SLIDING_WINDOW,
                1,
                self._key(key),
                self.limit,
                int(self.window_seconds * 1000),
                member,
                now_ms,
            )
            allowed = int(raw[0]) == 1
            retry_after = float(raw[1]) / 1000.0
            return allowed, (0.0 if allowed else round(max(0.0, retry_after), 3))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rate_limit_redis_failed fallback=memory error=%s", exc)
            self.degraded = True
            self.last_error = str(exc)
            return self._fallback.allow(key, now)

    def reset(self, key: Optional[str] = None) -> None:
        """清空某个 key 或全部窗口（测试/运维用）。"""
        self._fallback.reset(key)
        if self.degraded:
            return
        try:
            if key is None:
                keys = list(self._redis.scan_iter(match=f"{self.prefix}:*"))
                if keys:
                    self._redis.delete(*keys)
            else:
                self._redis.delete(self._key(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rate_limit_redis_reset_failed error=%s", exc)


def create_rate_limiter(
    limit: int,
    window_seconds: float = 60.0,
    backend: str = "redis",
    redis_url: Optional[str] = None,
    prefix: str = "ratelimit",
    redis_client: Any = None,
) -> Any:
    """构建限流器：redis 可用则返回分布式实现，否则降级内存实现。"""
    if str(backend or "").strip().lower() == "redis":
        try:
            client = redis_client
            if client is None:
                if not redis_url:
                    raise ValueError("redis_url is required for redis backend")
                import redis  # noqa: PLC0415

                client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                )
                client.ping()
            return RedisSlidingWindowRateLimiter(
                client,
                limit=limit,
                window_seconds=window_seconds,
                prefix=prefix,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("rate_limiter_redis_unavailable fallback=memory error=%s", exc)
    return SlidingWindowRateLimiter(limit=limit, window_seconds=window_seconds)
