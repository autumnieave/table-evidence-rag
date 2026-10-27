# -*- coding: utf-8 -*-
"""表级 BM25 文档构建单测（假 Qdrant，不依赖外部服务）。"""
from __future__ import annotations

from types import SimpleNamespace

from core.table_bm25 import build_table_bm25_cached, scroll_table_docs, table_bm25_cache_target


class FakeQdrantClient:
    """按批次返回倒序切片，验证实现按 slice_index 重排。"""

    def __init__(self):
        self.pages = [
            [
                SimpleNamespace(id="t1#s1", payload={"chunk_type": "table", "table_id": "t1",
                                                    "table_slice_index": 1, "chunk_index": 1,
                                                    "content": "尾部", "stock_name": "片仔癀"}),
                SimpleNamespace(id="t2#s0", payload={"chunk_type": "table", "table_id": "t2",
                                                    "table_slice_index": 0, "chunk_index": 0,
                                                    "content": "另一张表"}),
            ],
            [
                SimpleNamespace(id="t1#s0", payload={"chunk_type": "table", "table_id": "t1",
                                                    "table_slice_index": 0, "chunk_index": 0,
                                                    "content": "表头"}),
            ],
        ]
        self.offset = None

    def scroll(self, **kwargs):
        if not self.pages:
            return [], None
        return self.pages.pop(0), "next" if self.pages else None


def test_scroll_table_docs_groups_slices_and_keeps_metadata():
    docs = scroll_table_docs(FakeQdrantClient(), "fake")
    by_id = {doc["id"]: doc for doc in docs}
    assert set(by_id) == {"t1", "t2"}
    assert by_id["t1"]["payload"]["content"] == "表头\n尾部"
    assert by_id["t1"]["payload"]["slice_count"] == 2
    assert by_id["t1"]["payload"]["stock_name"] == "片仔癀"
    assert by_id["t2"]["payload"]["content"] == "另一张表"


class FakeWrapper:
    """最小 QdrantClientWrapper：count + client.scroll。"""

    def __init__(self, pages, count=3, fail_scroll=False):
        self.client = FakeQdrantClient()
        self.client.pages = pages
        self._count = count
        self._fail = fail_scroll

    def count(self):
        return self._count

    def scroll(self, **kwargs):
        if self._fail:
            raise AssertionError("缓存命中时不应再 scroll")
        return self.client.scroll(**kwargs)


def test_build_table_bm25_cached_writes_and_reuses_cache(tmp_path):
    pages = [[
        SimpleNamespace(id="t1#s0", payload={"chunk_type": "table", "table_id": "t1",
                                            "table_slice_index": 0, "content": "表头"}),
        SimpleNamespace(id="t2#s0", payload={"chunk_type": "table", "table_id": "t2",
                                            "table_slice_index": 0, "content": "另一张表"}),
    ]]
    cache = tmp_path / "table_bm25.pkl"
    first = build_table_bm25_cached(FakeWrapper(pages), "fake", cache_path=str(cache))
    target = table_bm25_cache_target(str(cache), "fake", 3)
    assert target.exists()
    assert {doc["id"] for doc in first.docs} == {"t1", "t2"}

    second = build_table_bm25_cached(FakeWrapper([], fail_scroll=True), "fake", cache_path=str(cache))
    assert {doc["id"] for doc in second.docs} == {"t1", "t2"}
