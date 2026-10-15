"""utils/query_cache.py —— 查询结果缓存（SQLite，线程安全）

为前端 / Agent 的重复查询提供 KV 缓存，避免重复调用财务查询
（原生 SQL 链路）与 RAG 检索-精排-生成全链路，缓解"回答慢"问题（路线 1）。

- SQLiteQueryCache：表 query_cache(key PK, payload TEXT, created_at REAL, expires_at REAL)
- 线程安全：每线程独立连接（与 memory/store.py 同模式）
- 开关与 TTL 由 RAGConfig 控制：QUERY_CACHE_ENABLED / QUERY_CACHE_TTL / QUERY_CACHE_DB
- 命中统计：进程内 hit / miss / set 计数（stats() / reset_stats()），用于量化缓存命中率

注意（证据链口径）：缓存命中不会重新生成 SQL/答案。做"修复后真实重跑"回归时，
必须设 QUERY_CACHE_ENABLED=false 或清空缓存库，否则会命中旧结果。

用法::

    from utils.query_cache import SQLiteQueryCache, make_cache_key
    cache = SQLiteQueryCache(db_path="database/query_cache.db", ttl=86400)
    data = cache.get(make_cache_key("fin-native", user_id, question))
    cache.set(make_cache_key("fin-native", user_id, question), {"content": "..."})
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS query_cache ("
    " key TEXT PRIMARY KEY, payload TEXT NOT NULL,"
    " created_at REAL NOT NULL, expires_at REAL)"
)


def make_cache_key(*parts: str) -> str:
    """生成缓存 key：对参与部分做 sha1 摘要，避免中文/长文本直接落库。"""
    raw = "|".join(str(p) for p in parts if p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class SQLiteQueryCache:
    """SQLite 查询缓存（零外部依赖，线程安全）。"""

    def __init__(self, db_path: str = "database/query_cache.db", ttl: int = 86400) -> None:
        self.db_path = str(db_path)
        self.default_ttl = int(ttl)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {"hit": 0, "miss": 0, "set": 0}
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        """获取当前线程的 SQLite 连接（惰性创建）。"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute(_TABLE_SQL)
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        """初始化表结构（首连时执行）。"""
        conn = self._conn()
        conn.execute(_TABLE_SQL)
        conn.commit()

    def _bump(self, name: str) -> None:
        """累加命中统计（线程安全；统计失败不影响主流程）。"""
        try:
            with self._stats_lock:
                self._stats[name] = self._stats.get(name, 0) + 1
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> Dict[str, Any]:
        """返回命中统计快照：hit / miss / set / total / hit_rate（四舍五入保留 4 位）。"""
        with self._stats_lock:
            snap: Dict[str, Any] = dict(self._stats)
        total = snap.get("hit", 0) + snap.get("miss", 0)
        snap["total"] = total
        snap["hit_rate"] = round(snap.get("hit", 0) / total, 4) if total else 0.0
        return snap

    def reset_stats(self) -> None:
        """清零命中统计（不影响缓存内容）。"""
        with self._stats_lock:
            for key in list(self._stats):
                self._stats[key] = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """读取缓存；不存在/已过期/解析失败返回 None（不抛异常）。"""
        now = time.time()
        try:
            conn = self._conn()
            row = conn.execute(
                "SELECT payload, expires_at FROM query_cache WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                self._bump("miss")
                return None
            payload, expires_at = row
            if expires_at is not None and now > expires_at:
                conn.execute("DELETE FROM query_cache WHERE key=?", (key,))
                conn.commit()
                self._bump("miss")
                return None
            self._bump("hit")
            return json.loads(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_cache.get 失败（按未命中处理）: %s", exc)
            self._bump("miss")
            return None

    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """写入缓存；ttl<=0 表示永不过期。失败只告警，不影响主流程。"""
        now = time.time()
        ttl_value = int(ttl if ttl is not None else self.default_ttl)
        expires_at = now + ttl_value if ttl_value > 0 else None
        try:
            payload = json.dumps(data, ensure_ascii=False)
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO query_cache(key, payload, created_at, expires_at)"
                " VALUES(?,?,?,?)",
                (key, payload, now, expires_at),
            )
            conn.commit()
            self._bump("set")
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_cache.set 失败（忽略，不影响主流程）: %s", exc)

    def delete(self, key: str) -> None:
        """删除单条缓存。"""
        try:
            conn = self._conn()
            conn.execute("DELETE FROM query_cache WHERE key=?", (key,))
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_cache.delete 失败: %s", exc)

    def clear(self) -> int:
        """清空缓存，返回删除条数。"""
        try:
            conn = self._conn()
            cur = conn.execute("DELETE FROM query_cache")
            conn.commit()
            return cur.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_cache.clear 失败: %s", exc)
            return 0

    def close(self) -> None:
        """释放当前线程连接。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
