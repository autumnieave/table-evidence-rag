"""SSE 帧编码与 Redis 断点续传存储。

设计边界：
- 事件 id 从 1 开始自增，按 ``user_id`` 绑定该用户最新一条流；
- Redis 用 hash(meta) + list(events) 保存，TTL 自动过期；
- Redis 不可用时自动降级进程内内存，接口行为不变；
- 本模块不依赖 FastAPI，便于单测。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def format_sse(event: Dict[str, Any]) -> str:
    """把带内部 ``id`` 的事件编码为 ``id: N\ndata: {...}\n\n``。

    ``id`` 只出现在 SSE 的 id 行，不重复写入 JSON 载荷，保持原有
    ``{"type": ...}`` 契约不变。
    """
    event_id = event.get("id")
    payload = {key: value for key, value in event.items() if key != "id"}
    return f"id: {event_id}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def parse_last_event_id(raw: Optional[str]) -> Optional[int]:
    """解析 ``Last-Event-ID``；非法/缺失返回 None。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def question_fingerprint(question: str, mode: str) -> str:
    """同一 user_id 下区分“同一请求续传”与“新请求”。"""
    raw = f"{mode or 'rag'}\n{question or ''}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class SSEResumeStore:
    """按 user_id 保存最新一条 SSE 流；Redis 为主，内存镜像兜底。"""

    def __init__(
        self,
        redis_client: Any = None,
        prefix: str = "sse:resume",
        ttl_seconds: int = 1800,
    ) -> None:
        self._redis = redis_client
        self.prefix = str(prefix or "sse:resume").rstrip(":")
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.degraded = redis_client is None
        self.last_error: Optional[str] = None
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _meta_key(self, user_id: str) -> str:
        return f"{self.prefix}:{user_id}:meta"

    def _events_key(self, stream_id: str) -> str:
        return f"{self.prefix}:events:{stream_id}"

    @staticmethod
    def _redis_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stream_id": str(meta.get("stream_id") or ""),
            "fingerprint": str(meta.get("fingerprint") or ""),
            "mode": str(meta.get("mode") or "rag"),
            "done": 1 if meta.get("done") else 0,
            "error": str(meta.get("error") or ""),
            "last_id": int(meta.get("last_id") or 0),
            "updated": float(meta.get("updated") or time.time()),
        }

    def start(
        self,
        user_id: str,
        question: str,
        mode: str,
        stream_id: Optional[str] = None,
    ) -> str:
        """创建一条新流记录并返回 stream_id。"""
        stream_id = stream_id or uuid.uuid4().hex
        meta = {
            "stream_id": stream_id,
            "fingerprint": question_fingerprint(question, mode),
            "mode": mode or "rag",
            "done": False,
            "error": None,
            "last_id": 0,
            "updated": time.time(),
        }
        with self._lock:
            self._memory[user_id] = {"meta": dict(meta), "events": []}
        if self._redis is not None:
            try:
                old = self._redis.hgetall(self._meta_key(user_id)) or {}
                old_stream = old.get("stream_id")
                if old_stream and old_stream != stream_id:
                    self._redis.delete(self._events_key(old_stream))
                self._redis.hset(self._meta_key(user_id), mapping=self._redis_meta(meta))
                self._redis.expire(self._meta_key(user_id), self.ttl_seconds)
                self._redis.delete(self._events_key(stream_id))
                self._redis.expire(self._events_key(stream_id), self.ttl_seconds)
            except Exception as exc:  # noqa: BLE001
                self._degrade("start", exc)
        return stream_id

    def append(
        self,
        user_id: str,
        stream_id: str,
        event: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """追加事件并分配自增 id；记录不存在时返回 None。"""
        with self._lock:
            record = self._memory.get(user_id)
            if record is None or record["meta"].get("stream_id") != stream_id:
                return None
            event_id = int(record["meta"].get("last_id") or 0) + 1
            stored = dict(event)
            stored["id"] = event_id
            record["meta"]["last_id"] = event_id
            record["meta"]["updated"] = time.time()
            record["events"].append(stored)
        if self._redis is not None:
            try:
                meta_key = self._meta_key(user_id)
                self._redis.rpush(
                    self._events_key(stream_id),
                    json.dumps(stored, ensure_ascii=False),
                )
                self._redis.hset(
                    meta_key,
                    mapping={
                        "last_id": event_id,
                        "updated": time.time(),
                    },
                )
                self._redis.expire(meta_key, self.ttl_seconds)
                self._redis.expire(self._events_key(stream_id), self.ttl_seconds)
            except Exception as exc:  # noqa: BLE001
                self._degrade("append", exc)
        return stored

    def get_meta(self, user_id: str) -> Optional[Dict[str, Any]]:
        """读取该用户最新流的元信息。"""
        with self._lock:
            record = self._memory.get(user_id)
            if record is not None:
                return dict(record["meta"])
        if self._redis is not None:
            try:
                raw = self._redis.hgetall(self._meta_key(user_id))
                if raw:
                    return self._parse_meta(raw)
            except Exception as exc:  # noqa: BLE001
                self._degrade("get_meta", exc)
        return None

    def load_events(self, user_id: str, after_id: int = 0) -> List[Dict[str, Any]]:
        """读取 id > after_id 的事件（用于 Last-Event-ID 回放）。"""
        after_id = max(0, int(after_id))
        with self._lock:
            record = self._memory.get(user_id)
            if record is not None:
                return [
                    dict(event)
                    for event in record["events"]
                    if int(event.get("id") or 0) > after_id
                ]
        if self._redis is not None:
            try:
                meta = self.get_meta(user_id) or {}
                stream_id = meta.get("stream_id")
                if not stream_id:
                    return []
                raw_events = self._redis.lrange(
                    self._events_key(stream_id), after_id, -1
                )
                events: List[Dict[str, Any]] = []
                for raw in raw_events:
                    try:
                        event = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if int(event.get("id") or 0) > after_id:
                        events.append(event)
                return events
            except Exception as exc:  # noqa: BLE001
                self._degrade("load_events", exc)
        return []

    def mark_done(
        self,
        user_id: str,
        stream_id: str,
        error: Optional[str] = None,
    ) -> None:
        """标记流结束（正常或错误）。"""
        with self._lock:
            record = self._memory.get(user_id)
            if record is not None and record["meta"].get("stream_id") == stream_id:
                record["meta"]["done"] = True
                record["meta"]["error"] = error
                record["meta"]["updated"] = time.time()
        if self._redis is not None:
            try:
                key = self._meta_key(user_id)
                self._redis.hset(
                    key,
                    mapping={
                        "done": 1,
                        "error": error or "",
                        "updated": time.time(),
                    },
                )
                self._redis.expire(key, self.ttl_seconds)
            except Exception as exc:  # noqa: BLE001
                self._degrade("mark_done", exc)

    def clear(self, user_id: str) -> None:
        """删除指定 user_id 的续传记录（测试/运维用）。"""
        with self._lock:
            record = self._memory.pop(user_id, None)
        if self._redis is not None:
            try:
                meta = self._redis.hgetall(self._meta_key(user_id)) or {}
                stream_id = meta.get("stream_id")
                if stream_id:
                    self._redis.delete(self._events_key(stream_id))
                self._redis.delete(self._meta_key(user_id))
            except Exception as exc:  # noqa: BLE001
                self._degrade("clear", exc)

    def reset(self) -> None:
        """清空全部续传记录（测试/运维用）。"""
        with self._lock:
            self._memory.clear()
        if self._redis is not None:
            try:
                keys = list(self._redis.scan_iter(match=f"{self.prefix}:*"))
                if keys:
                    self._redis.delete(*keys)
            except Exception as exc:  # noqa: BLE001
                self._degrade("reset", exc)

    @staticmethod
    def _parse_meta(raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stream_id": str(raw.get("stream_id") or ""),
            "fingerprint": str(raw.get("fingerprint") or ""),
            "mode": str(raw.get("mode") or "rag"),
            "done": str(raw.get("done") or "0") in ("1", "true", "True"),
            "error": str(raw["error"]) if raw.get("error") else None,
            "last_id": int(raw.get("last_id") or 0),
            "updated": float(raw.get("updated") or 0),
        }

    def _degrade(self, op: str, exc: Exception) -> None:
        self.degraded = True
        self.last_error = str(exc)
        logger.warning(
            "sse_resume_redis_failed op=%s fallback=memory error=%s", op, exc
        )
