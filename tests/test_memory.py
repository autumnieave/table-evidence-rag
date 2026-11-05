"""memory 包单元测试：ConversationState 序列化 + SQLite 存储层。"""

import time

from memory.conversation import ClarifyStatus, ConversationState
from memory.store import SQLiteMemoryStore, create_memory_store


def test_conversation_state_roundtrip():
    st = ConversationState(
        user_id="u1",
        history=[{"role": "user", "content": "你好"}],
        sql="SELECT 1",
        status=ClarifyStatus.NEED_CLARIFY,
        clarify_question="缺股票名",
    )
    st.filters.stock_name = "贵州茅台"
    st.rounds.append({"content": "ok"})
    st2 = ConversationState.from_dict(st.to_dict())
    assert st2.user_id == "u1"
    assert st2.sql == "SELECT 1"
    assert st2.status == ClarifyStatus.NEED_CLARIFY
    assert st2.clarify_question == "缺股票名"
    assert st2.filters.stock_name == "贵州茅台"
    assert st2.history == [{"role": "user", "content": "你好"}]
    assert st2.rounds == [{"content": "ok"}]


def test_conversation_state_from_empty():
    st = ConversationState.from_dict(None)
    assert st.user_id == "default" and st.status == ClarifyStatus.READY
    st2 = ConversationState.from_dict({"user_id": "u9", "status": "bad-value"})
    assert st2.user_id == "u9" and st2.status == ClarifyStatus.READY


def test_sqlite_store_roundtrip_and_overwrite(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"), ttl=3600)
    s.set("u1", {"sql": "SELECT 1"})
    assert s.get("u1") == {"sql": "SELECT 1"}
    s.set("u1", {"sql": "SELECT 2"})
    assert s.get("u1")["sql"] == "SELECT 2"
    s.delete("u1")
    assert s.get("u1") is None


def test_sqlite_store_ttl_expiry(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"), ttl=1)
    s.set("u3", {"sql": "SELECT 4"}, ttl=1)
    assert s.get("u3") is not None
    time.sleep(1.1)
    assert s.get("u3") is None  # 过期自动清理


def test_sqlite_store_no_expiry(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"), ttl=3600)
    s.set("u2", {"sql": "SELECT 3"}, ttl=-1)
    assert s.get("u2")["sql"] == "SELECT 3"


def test_create_store_backends(tmp_path):
    assert create_memory_store(backend="none") is None
    assert create_memory_store(backend="sqlite", db_path=str(tmp_path / "m.db"), ttl=60) is not None
    # redis 不可用时应优雅降级为 None（不抛异常）。
    # 用不可达端口，避免依赖“当前环境是否装了 redis / 是否启动了服务”。
    assert create_memory_store(backend="redis", redis_url="redis://127.0.0.1:1/0") is None
