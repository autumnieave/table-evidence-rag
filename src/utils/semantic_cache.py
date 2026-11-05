"""utils/semantic_cache.py —— 语义缓存（Redis 存储 + 内存回退）。

用途：对 ``/chat`` 的相似问法做语义命中，命中后直接回放上次结果，跳过检索、
精排与生成，降低重复问题的端到端延迟和生成 token 成本。

实现要点：

- 查询向量由 ``EmbeddingClient.generate_embeddings(text_type="query")`` 生成；
- Redis 只做 KV + 索引集合，相似度在进程内用余弦计算（不依赖 Redis Stack 向量索引）；
- 每条缓存带 TTL，索引用 zset 按时间排序，超过 ``max_entries`` 淘汰最旧条目；
- Redis 不可用时降级为进程内字典，并在 ``stats()`` 的 ``backend`` 字段标注 ``memory``；
- 只缓存成功结果；payload 以 JSON 存储。

注意：语义缓存有“相似但不相同”的误命中风险（例如年份/指标不同但句式相近），
默认关闭（``SEMANTIC_CACHE_ENABLED=false``），需要显式开启并配合阈值验证。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """计算两个向量的余弦相似度；维度不一致或零向量返回 0.0。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += float(a) * float(b)
        norm_left += float(a) * float(a)
        norm_right += float(b) * float(b)
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    score = dot / (math.sqrt(norm_left) * math.sqrt(norm_right))
    return max(-1.0, min(1.0, score))


class SemanticCache:
    """基于查询向量的语义缓存。

    Args:
        embedding_client: 具备 ``generate_embeddings(texts, text_type="query")`` 的客户端。
        redis_url: Redis 连接串；为空或连接失败时降级内存。
        threshold: 余弦相似度命中阈值，默认 0.92。
        ttl: 条目存活秒数，默认 86400；<=0 表示不过期。
        max_entries: 单 namespace 最大条目数，超出淘汰最旧。
        version: 缓存版本号，变更提示词/检索口径时 bump。
        redis_client: 可注入的 Redis 客户端（单测用）。
    """

    def __init__(
        self,
        embedding_client: Any,
        redis_url: Optional[str] = None,
        threshold: float = 0.92,
        ttl: int = 86400,
        max_entries: int = 1000,
        version: str = "v1",
        redis_client: Any = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.threshold = float(threshold)
        self.ttl = int(ttl)
        self.max_entries = max(1, int(max_entries))
        self.version = str(version or "v1")

        self._lock = threading.RLock()
        self._memory: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._redis = redis_client
        self.backend = "redis" if redis_client is not None else "memory"

        if self._redis is None and redis_url:
            try:
                import redis  # noqa: PLC0415

                client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=3,
                )
                client.ping()
                self._redis = client
                self.backend = "redis"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "semantic_cache_redis_unavailable fallback=memory error=%s", exc
                )

        self._stats: Dict[str, Any] = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "embedding_errors": 0,
            "evictions": 0,
        }
        self.last_similarity: Optional[float] = None
        self.last_lookup_ms: Optional[float] = None
        self.last_question: Optional[str] = None

    # ── 内部工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _namespace_key(self, namespace: str) -> str:
        return self._digest(str(namespace or "default"))[:16]

    def _index_key(self, ns: str) -> str:
        return f"semcache:{self.version}:{ns}:index"

    def _entry_key(self, ns: str, entry_id: str) -> str:
        return f"semcache:{self.version}:{ns}:entry:{entry_id}"

    def _embed(self, question: str) -> Optional[List[float]]:
        try:
            vectors = self.embedding_client.generate_embeddings(
                [question], text_type="query"
            )
            if not vectors or not vectors[0]:
                raise ValueError("empty embedding")
            return [float(x) for x in vectors[0]]
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stats["embedding_errors"] += 1
            logger.warning("semantic_cache_embedding_failed error=%s", exc)
            return None

    def _now(self) -> float:
        return time.time()

    def _expired(self, entry: Dict[str, Any], now: float) -> bool:
        expires_at = float(entry.get("expires_at") or 0.0)
        return expires_at > 0 and expires_at <= now

    # ── 读写 ────────────────────────────────────────────────────────────
    def get(self, question: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """查找最相似的缓存条目；命中返回 payload 副本，否则返回 None。"""
        started = time.perf_counter()
        question = str(question or "").strip()
        if not question:
            return None
        ns = self._namespace_key(namespace)
        embedding = self._embed(question)
        if embedding is None:
            with self._lock:
                self._stats["lookups"] += 1
                self._stats["misses"] += 1
            return None

        now = self._now()
        best_score = -1.0
        best_entry: Optional[Dict[str, Any]] = None

        if self._redis is not None:
            try:
                index_key = self._index_key(ns)
                entry_ids = self._redis.zrange(index_key, 0, self.max_entries * 2)
                if entry_ids:
                    keys = [self._entry_key(ns, eid) for eid in entry_ids]
                    raw_items = self._redis.mget(keys)
                    for eid, raw in zip(entry_ids, raw_items):
                        if not raw:
                            continue
                        entry = json.loads(raw)
                        if self._expired(entry, now):
                            self._redis.zrem(index_key, eid)
                            self._redis.delete(self._entry_key(ns, eid))
                            continue
                        score = cosine_similarity(embedding, entry.get("embedding") or [])
                        if score > best_score:
                            best_score = score
                            best_entry = entry
            except Exception as exc:  # noqa: BLE001
                logger.warning("semantic_cache_redis_get_failed error=%s", exc)
                best_entry = None
                best_score = -1.0
        else:
            with self._lock:
                bucket = dict(self._memory.get(ns, {}))
                for eid, entry in bucket.items():
                    if self._expired(entry, now):
                        self._memory.get(ns, {}).pop(eid, None)
                        continue
                    score = cosine_similarity(embedding, entry.get("embedding") or [])
                    if score > best_score:
                        best_score = score
                        best_entry = entry

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._stats["lookups"] += 1
            self.last_similarity = round(best_score, 6) if best_score >= 0 else None
            self.last_lookup_ms = round(elapsed_ms, 3)
            self.last_question = question

        if best_entry is not None and best_score >= self.threshold:
            with self._lock:
                self._stats["hits"] += 1
            payload = best_entry.get("payload")
            return dict(payload) if isinstance(payload, dict) else None

        with self._lock:
            self._stats["misses"] += 1
        return None

    def set(
        self,
        question: str,
        payload: Dict[str, Any],
        namespace: str = "default",
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """写入一条缓存；payload 必须是可 JSON 序列化的成功结果。"""
        question = str(question or "").strip()
        if not question or not isinstance(payload, dict):
            return False
        vector = embedding if embedding is not None else self._embed(question)
        if vector is None:
            return False

        ns = self._namespace_key(namespace)
        now = self._now()
        entry_id = self._digest(question)
        entry = {
            "question": question,
            "embedding": vector,
            "payload": payload,
            "created_at": now,
            "expires_at": (now + self.ttl) if self.ttl > 0 else 0.0,
        }

        if self._redis is not None:
            try:
                index_key = self._index_key(ns)
                entry_key = self._entry_key(ns, entry_id)
                pipe = self._redis.pipeline()
                pipe.zadd(index_key, {entry_id: now})
                if self.ttl > 0:
                    pipe.setex(entry_key, self.ttl, json.dumps(entry, ensure_ascii=False))
                else:
                    pipe.set(entry_key, json.dumps(entry, ensure_ascii=False))
                pipe.execute()
                self._trim_redis(ns, index_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("semantic_cache_redis_set_failed error=%s", exc)
                return False
        else:
            with self._lock:
                bucket = self._memory.setdefault(ns, {})
                bucket[entry_id] = entry
                if len(bucket) > self.max_entries:
                    oldest = sorted(bucket.items(), key=lambda kv: kv[1].get("created_at", 0))
                    for key, _ in oldest[: len(bucket) - self.max_entries]:
                        bucket.pop(key, None)
                        self._stats["evictions"] += 1

        with self._lock:
            self._stats["sets"] += 1
        return True

    def _trim_redis(self, ns: str, index_key: str) -> None:
        try:
            size = int(self._redis.zcard(index_key))
            overflow = size - self.max_entries
            if overflow <= 0:
                return
            stale = self._redis.zrange(index_key, 0, overflow - 1)
            if not stale:
                return
            self._redis.zrem(index_key, *stale)
            self._redis.delete(*[self._entry_key(ns, eid) for eid in stale])
            with self._lock:
                self._stats["evictions"] += overflow
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_cache_trim_failed error=%s", exc)

    # ── 运维 ────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        """返回缓存运行统计（含最近一次相似度与查询耗时）。"""
        with self._lock:
            data = dict(self._stats)
            lookups = int(data.get("lookups") or 0)
            hits = int(data.get("hits") or 0)
            data["hit_rate"] = round(hits / lookups, 4) if lookups else 0.0
            data["backend"] = self.backend
            data["threshold"] = self.threshold
            data["entries"] = self._entry_count()
            data["last_similarity"] = self.last_similarity
            data["last_lookup_ms"] = self.last_lookup_ms
            data["last_question"] = self.last_question
            return data

    def _entry_count(self) -> int:
        if self._redis is not None:
            try:
                total = 0
                for key in self._redis.scan_iter(match=f"semcache:{self.version}:*:index"):
                    total += int(self._redis.zcard(key))
                return total
            except Exception:  # noqa: BLE001
                return 0
        return sum(len(bucket) for bucket in self._memory.values())

    def clear(self, namespace: Optional[str] = None) -> int:
        """清空指定 namespace 或全部缓存，返回清理的条目数。"""
        removed = 0
        if self._redis is not None:
            try:
                if namespace is None:
                    pattern = f"semcache:{self.version}:*"
                    keys = list(self._redis.scan_iter(match=pattern))
                else:
                    ns = self._namespace_key(namespace)
                    index_key = self._index_key(ns)
                    entry_ids = self._redis.zrange(index_key, 0, -1)
                    keys = [self._entry_key(ns, eid) for eid in entry_ids] + [index_key]
                    removed = len(entry_ids)
                if keys:
                    self._redis.delete(*keys)
                return removed
            except Exception as exc:  # noqa: BLE001
                logger.warning("semantic_cache_clear_failed error=%s", exc)
                return 0
        with self._lock:
            if namespace is None:
                removed = sum(len(bucket) for bucket in self._memory.values())
                self._memory.clear()
            else:
                ns = self._namespace_key(namespace)
                removed = len(self._memory.pop(ns, {}))
        return removed

    def reset_stats(self) -> None:
        """重置统计计数（不清缓存条目）。"""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0
            self.last_similarity = None
            self.last_lookup_ms = None
            self.last_question = None
