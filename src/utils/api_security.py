"""API key 解析、提取与进程内滑动窗口限流。

设计边界：

- key 只从环境变量 ``API_KEYS``（逗号分隔）读取，不在代码里硬编码；
- 限流是**单进程内存**实现；多 worker 部署时需要 Redis/网关级限流；
- 时间使用 ``time.monotonic()``，不受系统时钟回拨影响。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Mapping, Optional, Set, Tuple


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
