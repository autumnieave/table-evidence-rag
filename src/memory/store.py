"""memory/store.py —— 会话记忆持久化存储层（2026-08-22 记忆持久化 #6）

按 user_id 存取 ConversationState 快照，解决纯内存状态"重启即失、多用户串话"问题：

- SQLiteMemoryStore：默认后端（stdlib sqlite3，零依赖），TTL 过期清理
- RedisMemoryStore：可选后端（需安装 redis 包），TTL 由 Redis 原生过期
- create_memory_store()：按配置创建后端；backend=none 或依赖缺失时返回 None（降级纯内存）

用法::

    store = create_memory_store(backend="sqlite", db_path="database/chat_memory.db", ttl=86400)
    store.set("user-1", {"sql": "SELECT 1"}, ttl=3600)
    data = store.get("user-1")
    store.delete("user-1")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """会话记忆存储抽象：get/set/delete/close"""

    @abstractmethod
    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        """读取用户会话快照；不存在/已过期返回 None"""

    @abstractmethod
    def set(self, user_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """写入用户会话快照（ttl 秒后过期）"""

    @abstractmethod
    def delete(self, user_id: str) -> None:
        """删除用户会话快照"""

    def close(self) -> None:
        """释放连接（子类按需覆盖）"""


class SQLiteMemoryStore(MemoryStore):
    """SQLite 会话存储（默认后端，零依赖）

    表结构：conversation_memory(user_id PK, payload TEXT, updated_at REAL, expires_at REAL)
    读取时惰性清理已过期行；线程安全（每线程独立连接）。
    """

    _TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS conversation_memory ("
        " user_id TEXT PRIMARY KEY, payload TEXT NOT NULL,"
        " updated_at REAL NOT NULL, expires_at REAL)"
    )

    def __init__(self, db_path: str = "database/chat_memory.db", ttl: int = 86400) -> None:
        self.db_path = str(db_path)
        self.default_ttl = int(ttl)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        """获取当前线程的 SQLite 连接（惰性创建）"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute(self._TABLE_SQL)
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        """初始化表结构（首连时执行）"""
        self._conn().commit()

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        """读取快照；已过期删除并返回 None"""
        now = time.time()
        try:
            row = self._conn().execute(
                "SELECT payload, expires_at FROM conversation_memory WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.warning("SQLite 读取失败(user_id=%s): %s", user_id, e)
            return None
        if not row:
            return None
        payload, expires_at = row
        if expires_at is not None and expires_at <= now:
            self.delete(user_id)
            return None
        try:
            return json.loads(payload)
        except (TypeError, ValueError) as e:
            logger.warning("SQLite 快照解析失败(user_id=%s): %s", user_id, e)
            return None

    def set(self, user_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """写入快照（ttl 缺省用 default_ttl）"""
        ttl_sec = self.default_ttl if ttl is None else int(ttl)
        now = time.time()
        expires_at = now + ttl_sec if ttl_sec > 0 else None
        try:
            self._conn().execute(
                "INSERT OR REPLACE INTO conversation_memory (user_id, payload, updated_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (user_id, json.dumps(data, ensure_ascii=False), now, expires_at),
            )
            self._conn().commit()
        except sqlite3.Error as e:
            logger.warning("SQLite 写入失败(user_id=%s): %s", user_id, e)

    def delete(self, user_id: str) -> None:
        """删除快照"""
        try:
            self._conn().execute("DELETE FROM conversation_memory WHERE user_id = ?", (user_id,))
            self._conn().commit()
        except sqlite3.Error as e:
            logger.warning("SQLite 删除失败(user_id=%s): %s", user_id, e)

    def close(self) -> None:
        """关闭当前线程连接"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class RedisMemoryStore(MemoryStore):
    """Redis 会话存储（可选后端，需 pip install redis）

    key 形如 conversation:{user_id}，TTL 由 Redis EXPIRE 原生管理。
    """

    def __init__(self, url: str = "redis://localhost:6379/0", ttl: int = 86400) -> None:
        try:
            import redis  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "Redis 后端需要安装 redis 包：pip install redis；" "或改用 MEMORY_BACKEND=sqlite/none"
            ) from e
        self.client = redis.from_url(url, decode_responses=True)
        self.client.ping()  # 启动即校验连通性
        self.default_ttl = int(ttl)

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            payload = self.client.get(f"conversation:{user_id}")
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 读取失败(user_id=%s): %s", user_id, e)
            return None
        if not payload:
            return None
        try:
            return json.loads(payload)
        except (TypeError, ValueError) as e:
            logger.warning("Redis 快照解析失败(user_id=%s): %s", user_id, e)
            return None

    def set(self, user_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        ttl_sec = self.default_ttl if ttl is None else int(ttl)
        try:
            if ttl_sec > 0:
                self.client.setex(f"conversation:{user_id}", ttl_sec, json.dumps(data, ensure_ascii=False))
            else:
                self.client.set(f"conversation:{user_id}", json.dumps(data, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 写入失败(user_id=%s): %s", user_id, e)

    def delete(self, user_id: str) -> None:
        try:
            self.client.delete(f"conversation:{user_id}")
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 删除失败(user_id=%s): %s", user_id, e)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass


def create_memory_store(
    backend: str = "sqlite",
    db_path: str = "database/chat_memory.db",
    redis_url: str = "redis://localhost:6379/0",
    ttl: int = 86400,
) -> Optional[MemoryStore]:
    """按配置创建记忆存储后端；不可用返回 None（调用方降级为纯内存）。

    Args:
        backend: sqlite（默认）/ redis / none

    Returns:
        MemoryStore 实例；backend=none 或后端初始化失败返回 None
    """
    b = (backend or "sqlite").strip().lower()
    if b == "none":
        return None
    if b == "redis":
        try:
            return RedisMemoryStore(url=redis_url, ttl=ttl)
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 记忆后端初始化失败，降级为纯内存: %s", e)
            return None
    if b == "sqlite":
        try:
            return SQLiteMemoryStore(db_path=db_path, ttl=ttl)
        except Exception as e:  # noqa: BLE001
            logger.warning("SQLite 记忆后端初始化失败，降级为纯内存: %s", e)
            return None
    logger.warning("未知记忆后端 %r，降级为纯内存", backend)
    return None
