"""utils.query_cache 命中统计单元测试（纯逻辑，SQLite 本地，不依赖外部服务）。"""

from utils.query_cache import SQLiteQueryCache, make_cache_key


def test_stats_miss_then_hit(tmp_path):
    cache = SQLiteQueryCache(db_path=str(tmp_path / "qc.db"), ttl=3600)
    key = make_cache_key("chat", "agent", "u1", "q1")

    assert cache.get(key) is None
    assert cache.stats()["miss"] == 1
    assert cache.stats()["hit"] == 0

    cache.set(key, {"content": "x"})
    assert cache.stats()["set"] == 1

    assert cache.get(key) == {"content": "x"}
    stats = cache.stats()
    assert stats["hit"] == 1 and stats["miss"] == 1
    assert stats["total"] == 2
    assert stats["hit_rate"] == 0.5


def test_stats_hit_rate_zero_when_no_calls(tmp_path):
    cache = SQLiteQueryCache(db_path=str(tmp_path / "qc.db"), ttl=3600)
    stats = cache.stats()
    assert stats == {"hit": 0, "miss": 0, "set": 0, "total": 0, "hit_rate": 0.0}


def test_reset_stats_clears_counters_only(tmp_path):
    cache = SQLiteQueryCache(db_path=str(tmp_path / "qc.db"), ttl=3600)
    key = make_cache_key("chat", "rag", "u2", "q2")
    cache.set(key, {"content": "y"})
    cache.get(key)
    assert cache.stats()["hit"] == 1

    cache.reset_stats()
    assert cache.stats()["hit"] == 0 and cache.stats()["set"] == 0
    assert cache.get(key) == {"content": "y"}


def test_expired_read_counts_as_miss(tmp_path):
    cache = SQLiteQueryCache(db_path=str(tmp_path / "qc.db"), ttl=3600)
    key = make_cache_key("chat", "rag", "u3", "q3")
    cache.set(key, {"content": "z"}, ttl=1)
    cache.get(key)
    import time

    time.sleep(1.1)
    assert cache.get(key) is None
    assert cache.stats()["hit"] == 1 and cache.stats()["miss"] == 1
