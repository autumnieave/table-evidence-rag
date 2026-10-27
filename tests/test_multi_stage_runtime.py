# -*- coding: utf-8 -*-
"""多阶段检索运行时单测（假 Qdrant/Embedding，不依赖外部服务）。"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from core import multi_stage_runtime as runtime_module
from core.multi_stage_runtime import build_multi_stage_runtime, load_known_entities


class FakeQdrant:
    def __init__(self, names=None, fail=False):
        self._names = names or []
        self._fail = fail

    def distinct_stock_names(self):
        if self._fail:
            raise RuntimeError("qdrant down")
        return set(self._names)


class FakeEmbedding:
    def generate_embeddings(self, texts, text_type="query"):
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.retrieve_calls = []
        self.last_trace = None

    def retrieve(self, *args, **kwargs):
        self.retrieve_calls.append((args, kwargs))
        return [{"id": "t1", "score": 1.0, "payload": {"content": "x"}}]

    def retrieve_with_iteration(self, query, **kwargs):
        self.retrieve_calls.append((query, kwargs))
        return [{"id": "t1", "score": 1.0, "payload": {"content": "x"}}]


def _config():
    return SimpleNamespace(
        QDRANT_COLLECTION_NAME="fake",
        TABLE_BM25_INDEX_PATH="unused.pkl",
        EXCEL_METADATA_PATH="unused.xlsx",
    )


def test_load_known_entities_from_qdrant():
    entities, source = load_known_entities(_config(), FakeQdrant(["华润三九", "白云山"]))
    assert entities == ("华润三九", "白云山")
    assert source == "qdrant"


def test_load_known_entities_falls_back_to_excel(monkeypatch):
    monkeypatch.setattr(runtime_module, "load_stock_vocabulary", lambda path: {"片仔癀"})
    entities, source = load_known_entities(_config(), FakeQdrant(fail=True))
    assert entities == ("片仔癀",)
    assert source == "excel"


def test_build_runtime_logs_bm25_fallback(monkeypatch, caplog):
    monkeypatch.setattr(runtime_module, "MultiStageRetriever", FakeRetriever)
    monkeypatch.setattr(runtime_module, "load_stock_vocabulary", lambda path: {"片仔癀"})
    monkeypatch.setattr(
        runtime_module,
        "build_table_bm25_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no table chunks")),
    )
    caplog.set_level(logging.WARNING)
    runtime = build_multi_stage_runtime(
        _config(),
        qdrant_client=FakeQdrant(fail=True),
        embedding_client=FakeEmbedding(),
    )
    assert runtime.bm25_status == "unavailable"
    assert runtime.known_entities == ("片仔癀",)
    assert "bm25_unavailable_fallback" in caplog.text
    assert runtime.retriever.kwargs["bm25_retriever"] is None


def test_runtime_warmup(monkeypatch):
    monkeypatch.setattr(runtime_module, "MultiStageRetriever", FakeRetriever)
    monkeypatch.setattr(runtime_module, "build_table_bm25_cached", lambda *a, **k: object())
    runtime = build_multi_stage_runtime(
        _config(),
        qdrant_client=FakeQdrant(["华润三九"]),
        embedding_client=FakeEmbedding(),
    )
    info = runtime.warmup()
    assert info["status"] == "ok"
    assert info["bm25_status"] == "ready"
    assert info["known_entities"] == 1
    calls = [kwargs for _args, kwargs in runtime.retriever.retrieve_calls]
    assert len(calls) == 9
    assert all(call["row_k"] == 200 and call["top_k"] == 10 for call in calls)


def test_runtime_retrieve_passes_context(monkeypatch):
    monkeypatch.setattr(runtime_module, "MultiStageRetriever", FakeRetriever)
    monkeypatch.setattr(runtime_module, "build_table_bm25_cached", lambda *a, **k: object())
    runtime = build_multi_stage_runtime(
        _config(),
        qdrant_client=FakeQdrant(["华润三九"]),
        embedding_client=FakeEmbedding(),
    )
    runtime.retrieve("差异原因", context_queries=["999与白云山相比"])
    query, kwargs = runtime.retriever.retrieve_calls[-1]
    assert query == "差异原因"
    assert kwargs["context_queries"] == ["999与白云山相比"]
    assert kwargs["rerank"] is False
