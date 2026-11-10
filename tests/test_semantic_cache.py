"""语义缓存单测：不依赖 Redis / Embedding API。"""

from __future__ import annotations

from typing import Dict, List

import pytest

from utils.semantic_cache import SemanticCache, cosine_similarity


class FakeEmbeddingClient:
    """按问题返回固定向量；未知问题返回零向量。"""

    def __init__(self, mapping: Dict[str, List[float]]) -> None:
        self.mapping = mapping
        self.calls = 0

    def generate_embeddings(self, texts: List[str], text_type: str = "document"):
        self.calls += 1
        return [self.mapping.get(texts[0], [0.0, 0.0, 0.0])]


def _cache(vectors: Dict[str, List[float]], **kwargs) -> SemanticCache:
    return SemanticCache(
        embedding_client=FakeEmbeddingClient(vectors),
        redis_url=None,
        threshold=kwargs.pop("threshold", 0.92),
        ttl=kwargs.pop("ttl", 3600),
        max_entries=kwargs.pop("max_entries", 100),
        version="test",
        **kwargs,
    )


def test_cosine_similarity_basic() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_miss_then_hit_for_paraphrase() -> None:
    cache = _cache(
        {
            "片仔癀的利润": [1.0, 0.0, 0.0],
            "片仔癀赚了多少钱": [0.99, 0.01, 0.0],
        }
    )
    assert cache.get("片仔癀的利润", "rag:u1") is None
    assert cache.set("片仔癀的利润", {"content": "利润 10 亿"}, "rag:u1") is True

    hit = cache.get("片仔癀赚了多少钱", "rag:u1")
    assert hit == {"content": "利润 10 亿"}
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["last_similarity"] >= 0.92


def test_below_threshold_is_miss() -> None:
    cache = _cache(
        {
            "营收": [1.0, 0.0, 0.0],
            "净利润": [0.0, 1.0, 0.0],
        }
    )
    cache.set("营收", {"content": "A"}, "rag:u1")
    assert cache.get("净利润", "rag:u1") is None
    assert cache.stats()["hits"] == 0


def test_namespace_isolation() -> None:
    cache = _cache({"同一个问题": [1.0, 0.0, 0.0]})
    cache.set("同一个问题", {"content": "A"}, "rag:u1")
    assert cache.get("同一个问题", "rag:u2") is None
    assert cache.get("同一个问题", "rag:u1") == {"content": "A"}


def test_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1000.0
    monkeypatch.setattr("utils.semantic_cache.time.time", lambda: now)
    cache = _cache({"q": [1.0, 0.0, 0.0]}, ttl=10)
    cache.set("q", {"content": "A"}, "ns")

    monkeypatch.setattr("utils.semantic_cache.time.time", lambda: now + 11)
    assert cache.get("q", "ns") is None
    assert cache.stats()["entries"] == 0


def test_max_entries_eviction() -> None:
    vectors = {f"q{i}": [1.0, float(i), 0.0] for i in range(5)}
    cache = _cache(vectors, max_entries=2)
    for i in range(4):
        cache.set(f"q{i}", {"content": str(i)}, "ns")
    assert cache.stats()["entries"] == 2
    assert cache.stats()["evictions"] >= 2


def test_clear_namespace() -> None:
    cache = _cache({"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]})
    cache.set("a", {"content": "A"}, "ns1")
    cache.set("b", {"content": "B"}, "ns2")
    assert cache.clear("ns1") == 1
    assert cache.get("a", "ns1") is None
    assert cache.get("b", "ns2") == {"content": "B"}


@pytest.mark.parametrize(
    ("stored_question", "probe_question"),
    [
        ("片仔癀2023年利润", "片仔癀2024年利润"),
        ("片仔癀营收", "片仔癀利润"),
        ("片仔癀毛利率", "云南白药毛利率"),
    ],
)
def test_financial_constraint_mismatch_never_hits(
    stored_question: str, probe_question: str
) -> None:
    """阈值以上但实体/期间/指标不同，必须被硬约束隔离。"""
    cache = _cache(
        {
            stored_question: [1.0, 0.0, 0.0],
            probe_question: [0.999, 0.01, 0.0],
        }
    )
    assert cache.set(stored_question, {"content": "must-not-replay"}, "rag:u1") is True

    assert cache.get(probe_question, "rag:u1") is None
    stats = cache.stats()
    assert stats["constraint_rejections"] == 1
    assert stats["last_similarity"] >= 0.92
