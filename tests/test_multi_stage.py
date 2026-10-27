# -*- coding: utf-8 -*-
"""多阶段检索（阶段 3）离线单测：假 embedding / 假 Qdrant，不依赖外部服务。"""
from __future__ import annotations

from types import SimpleNamespace

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from core.multi_stage import (
    MultiStageRetriever,
    combine_score,
    dedupe_table_hits,
    group_row_hits,
    rrf_fuse,
)
from core.retrievers import BM25Retriever


class FakeEmbedding:
    """固定向量，记录调用次数。"""

    def __init__(self):
        self.calls = 0

    def generate_embeddings(self, texts, text_type="query"):
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


def _payload(**kwargs):
    return dict(kwargs)


def _table_hit(tid, score, content="", file_path=None, slice_idx=0):
    return {
        "id": "%s#s%d" % (tid, slice_idx),
        "score": score,
        "payload": _payload(
            chunk_type="table",
            table_id=tid,
            table_slice_index=slice_idx,
            content=content or ("%s-content-%d" % (tid, slice_idx)),
            file_path=file_path or (tid + ".md"),
        ),
    }


def _row_hit(tid, score, content, metric=None, period=None):
    payload = _payload(
        chunk_type="row_fact",
        table_id=tid,
        content=content,
        anchor_cells=[{"row_index": 1, "col_index": 2, "value": "1"}],
    )
    if metric is not None:
        payload["metric"] = metric
    if period is not None:
        payload["period"] = period
    return {
        "id": "%s#row" % tid,
        "score": score,
        "payload": payload,
    }


def _bm25_docs():
    return [
        {"id": "t1", "payload": {"table_id": "t1", "stockName": "华润三九", "content": "华润三九 营业收入 100"}},
        {"id": "t2", "payload": {"table_id": "t2", "stockName": "白云山", "content": "白云山 营业收入 200"}},
        {"id": "t3", "payload": {"table_id": "t3", "stockName": "片仔癀", "content": "片仔癀 营业收入 300"}},
    ]


def test_bm25_candidate_indices_match_full_scan_scores():
    bm25 = BM25Retriever(_bm25_docs())
    full = bm25.retrieve("华润三九", top_k=10)
    subset = bm25.retrieve("华润三九", top_k=10, candidate_indices=[0])
    assert [row["id"] for row in subset] == ["t1"]
    assert subset[0]["score"] == next(row["score"] for row in full if row["id"] == "t1")
    assert bm25.retrieve("华润三九", top_k=10, candidate_indices=[]) == []


def test_entity_index_limits_targeted_bm25_to_named_stock():
    bm25 = BM25Retriever(_bm25_docs())
    retriever = MultiStageRetriever(
        FakeEmbedding(),
        SimpleNamespace(config=SimpleNamespace(QDRANT_COLLECTION_NAME="fake")),
        bm25_retriever=bm25,
        known_entities=["华润三九", "白云山", "片仔癀"],
    )
    assert retriever._entity_candidate_indices(["华润三九"]) == [0]
    hits = retriever._bm25_entity_hits(["华润三九"], top_k=10)
    assert [hit["id"] for hit in hits] == ["t1"]


class FakeQdrant:
    """按过滤条件返回固定候选；scroll 负责整表回取。"""

    def __init__(self, table_hits, row_hits=None, slices=None):
        self.table_hits = list(table_hits)
        self.row_hits = list(row_hits or [])
        self.slices = dict(slices or {})
        self.filters = []
        self.limits = []
        self.table_limits = []
        self.config = SimpleNamespace(QDRANT_COLLECTION_NAME="fake")
        self.client = self

    @staticmethod
    def _chunk_type(flt):
        for cond in (flt or {}).get("must", []):
            if isinstance(cond, dict) and cond.get("key") == "chunk_type":
                return (cond.get("match") or {}).get("value")
        return None

    @staticmethod
    def _table_ids(flt):
        for cond in (flt or {}).get("must", []):
            if isinstance(cond, dict) and cond.get("key") == "table_id":
                return set((cond.get("match") or {}).get("any") or [])
        return None

    @staticmethod
    def _period(flt):
        for cond in (flt or {}).get("must", []):
            if isinstance(cond, dict) and cond.get("key") == "period":
                return (cond.get("match") or {}).get("value")
        return None

    def query_points(self, collection_name, query, limit, with_payload, query_filter):
        self.filters.append(query_filter)
        self.limits.append(limit)
        ctype = self._chunk_type(query_filter)
        if ctype == "table":
            self.table_limits.append(limit)
        if ctype == "table":
            return SimpleNamespace(points=[SimpleNamespace(**h) for h in self.table_hits[:limit]])
        if ctype == "row_fact":
            ids = self._table_ids(query_filter)
            period = self._period(query_filter)
            rows = [
                h for h in self.row_hits
                if (ids is None or h["payload"].get("table_id") in ids)
                and (period is None or h["payload"].get("period") == period)
            ]
            return SimpleNamespace(points=[SimpleNamespace(**h) for h in rows[:limit]])
        raise AssertionError("缺少 chunk_type 过滤：%r" % (query_filter,))

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors, scroll_filter):
        ids = self._table_ids(scroll_filter) or set()
        points = []
        for tid in ids:
            for idx, content in enumerate(self.slices.get(tid, [])):
                points.append(SimpleNamespace(id="%s#s%d" % (tid, idx), payload=_payload(
                    chunk_type="table",
                    table_id=tid,
                    table_slice_index=idx,
                    chunk_index=idx,
                    content=content,
                    file_path=tid + ".md",
                )))
        # 故意倒序返回，验证实现按 table_slice_index 重排
        points.reverse()
        return points, None


class FakeBM25:
    """固定 BM25 候选，记录调用参数。"""

    def __init__(self, hits):
        self.hits = list(hits)
        self.calls = []

    def retrieve(self, query, query_filter=None, top_k=None):
        self.calls.append({"query": query, "top_k": top_k})
        return self.hits[: top_k or len(self.hits)]


def _retriever(qdrant, **kwargs):
    return MultiStageRetriever(FakeEmbedding(), qdrant, reranker=None, **kwargs)


def test_dedupe_table_hits_keeps_best_slice_per_table():
    hits = [_table_hit("t1", 0.2, slice_idx=0), _table_hit("t1", 0.9, slice_idx=1), _table_hit("t2", 0.5)]
    best = dedupe_table_hits(hits)
    assert set(best) == {"t1", "t2"}
    assert best["t1"]["score"] == 0.9


def test_group_row_hits_sorts_within_table_and_drops_missing_id():
    hits = [_row_hit("t1", 0.2, "a"), _row_hit("t1", 0.8, "b"), _row_hit("t2", 0.1, "c")]
    hits.append({"id": "x", "score": 1.0, "payload": {}})
    groups = group_row_hits(hits)
    assert [r["score"] for r in groups["t1"]] == [0.8, 0.2]
    assert set(groups) == {"t1", "t2"}


def test_combine_score_with_and_without_row_evidence():
    assert combine_score(0.8, [{"score": 0.6}], coarse_weight=0.3) == 0.3 * 0.8 + 0.7 * 0.6
    assert combine_score(0.8, [], coarse_weight=0.3) == 0.3 * 0.8


def test_rrf_fuse_prefers_tables_hit_by_both_legs():
    vector = {"a": {"score": 0.9}, "b": {"score": 0.8}}
    bm25 = {"b": {"score": 12.0}, "c": {"score": 9.0}}
    fused = rrf_fuse(vector, bm25)
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_merge_filter_keeps_chunk_type_with_dict_and_qdrant_filter():
    merged = MultiStageRetriever._merge_filter({"must": [{"key": "source_file", "match": {"value": "x.md"}}]}, "table")
    assert {"key": "chunk_type", "match": {"value": "table"}} in merged["must"]
    assert len(merged["must"]) == 2

    flt = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value="x.md"))])
    merged_obj = MultiStageRetriever._merge_filter(flt, "row_fact")
    assert isinstance(merged_obj, dict)
    assert {"key": "chunk_type", "match": {"value": "row_fact"}} in merged_obj["must"]


def test_table_first_returns_whole_table_and_row_evidence_in_slice_order():
    q = FakeQdrant(
        table_hits=[_table_hit("t1", 0.8, slice_idx=0), _table_hit("t1", 0.7, slice_idx=1), _table_hit("t2", 0.6)],
        row_hits=[_row_hit("t1", 0.95, "命中行 A"), _row_hit("t1", 0.9, "命中行 B")],
        slices={"t1": ["片头\n裁剪", "主体\n尾部"], "t2": ["另一张表"]},
    )
    r = _retriever(q, final_k=2, rerank_pool=2, coarse_weight=0.3)
    out = r.retrieve("测试问题", multi_stage=True, rerank=False)
    trace = r.last_trace

    assert len(out) == 2
    assert out[0]["payload"]["table_id"] == "t1"
    assert out[0]["payload"]["content"] == "片头\n裁剪\n主体\n尾部"
    assert out[0]["payload"]["slice_count"] == 2
    assert len(out[0]["payload"]["row_hits"]) == 2
    assert out[0]["payload"]["row_hits"][0]["content"] == "命中行 A"
    assert out[0]["payload"]["stage_scores"]["row"] == 0.95
    assert trace.coarse_hits == 3 and trace.coarse_tables == 2
    assert trace.row_hits == 2 and trace.row_tables == 1
    assert trace.selected_tables == 2
    assert trace.timings_ms["total"] >= 0

    ctypes = [FakeQdrant._chunk_type(f) for f in q.filters]
    assert ctypes == ["table", "row_fact"]


def test_single_stage_does_not_query_row_facts():
    q = FakeQdrant(
        table_hits=[_table_hit("t1", 0.8), _table_hit("t1", 0.5, slice_idx=1), _table_hit("t2", 0.7)],
        row_hits=[_row_hit("t2", 0.99, "不应被查到")],
        slices={"t1": ["A"], "t2": ["B"]},
    )
    r = _retriever(q, final_k=2, rerank_pool=2)
    out = r.retrieve("测试问题", multi_stage=False, rerank=False)

    assert [x["payload"]["table_id"] for x in out] == ["t1", "t2"]
    assert [FakeQdrant._chunk_type(f) for f in q.filters] == ["table"]
    assert r.last_trace.row_hits == 0


def test_row_first_diagnostic_mode_uses_row_recall_then_fetches_tables():
    q = FakeQdrant(
        table_hits=[_table_hit("never", 1.0)],
        row_hits=[_row_hit("t2", 0.9, "行证据"), _row_hit("t1", 0.8, "另一行")],
        slices={"t1": ["A"], "t2": ["B"]},
    )
    r = _retriever(q, final_k=2, rerank_pool=2)
    out = r.retrieve("测试问题", multi_stage=True, mode="row_first", rerank=False)

    assert [x["payload"]["table_id"] for x in out] == ["t2", "t1"]
    assert [FakeQdrant._chunk_type(f) for f in q.filters] == ["row_fact"]
    assert r.last_trace.coarse_hits == 0
    assert r.last_trace.row_tables == 2

def test_row_hit_falls_back_to_anchor_cells_for_legacy_payload():
    legacy_row = _row_hit("t1", 0.95, "row")
    legacy_row["payload"].pop("metric", None)
    legacy_row["payload"].pop("period", None)
    legacy_row["payload"]["anchor_cells"] = [{
        "row_index": 1,
        "col_index": 2,
        "value": "1",
        "metric": "\u7814\u53d1\u8d39\u7528",
        "period": "2025Q3",
        "unit": "\u4ebf\u5143",
        "unit_source": "table_fallback",
    }]
    q = FakeQdrant(
        table_hits=[_table_hit("t1", 0.9)],
        row_hits=[legacy_row],
        slices={"t1": ["A"]},
    )
    r = _retriever(q, final_k=1, rerank_pool=1)
    out = r.retrieve("\u7814\u53d1\u8d39\u7528\u662f\u591a\u5c11", multi_stage=True, rerank=False)

    row = out[0]["payload"]["row_hits"][0]
    assert row["metric"] == "\u7814\u53d1\u8d39\u7528"
    assert row["period"] == "2025Q3"
    assert row["unit"] == "\u4ebf\u5143"
    assert row["unit_source"] == "table_fallback"


def test_query_metric_and_period_hard_match_filters_row_candidates():
    q = FakeQdrant(
        table_hits=[_table_hit("t1", 0.9), _table_hit("t2", 0.8)],
        row_hits=[
            _row_hit("t1", 0.99, "研发费用 2025Q3", metric="研发费用", period="2025Q3"),
            _row_hit("t1", 0.95, "营业收入 2025Q3", metric="营业收入", period="2025Q3"),
            _row_hit("t1", 0.90, "研发费用 2025A", metric="研发费用", period="2025A"),
            _row_hit("t2", 0.80, "研发费用 2025Q3", metric="研发费用", period="2025Q3"),
        ],
        slices={"t1": ["A"], "t2": ["B"]},
    )
    r = _retriever(q, final_k=2, rerank_pool=2)
    out = r.retrieve(
        "佐力药业2025年第三季度研发费用占比同比提升，分析原因",
        multi_stage=True,
        rerank=False,
    )

    assert len(out) == 2
    assert r.last_trace.row_candidates == 3  # Qdrant 期间过滤后、指标过滤前
    assert r.last_trace.row_hits == 2
    assert r.last_trace.query_period == "2025Q3"
    assert "研发费用" in r.last_trace.query_metrics
    for item in out:
        for row in item["payload"]["row_hits"]:
            assert row["metric"] == "研发费用"
            assert row["period"] == "2025Q3"


# ── 阶段5：证据审计迭代（扩召回 / 转 SQL / 拒绝）──────────────────────────


def _schema_table(tid, score, metrics, period, content=None):
    hit = _table_hit(tid, score, content=content)
    hit["payload"]["metric_names"] = list(metrics)
    hit["payload"]["report_period"] = period
    return hit


def test_retrieve_with_iteration_expands_recall_then_accepts():
    """首轮只召回无关表 → retry_table 扩大 coarse_k；第二轮命中指标表 → accept。"""
    q = FakeQdrant(
        table_hits=[
            _table_hit("t0", 0.99, "无关表"),
            _schema_table("t1", 0.80, ["研发费用"], "2025Q3", content="研发费用 3.21 亿元"),
        ],
        slices={"t0": ["无关表"], "t1": ["研发费用 3.21 亿元"]},
    )
    r = _retriever(q, coarse_k=1, row_k=1, final_k=2, rerank_pool=2)
    out = r.retrieve_with_iteration("2025Q3研发费用", max_rounds=2, rerank=False)

    assert q.table_limits[:2] == [1, 51]  # max(2, 1+50)
    assert r.last_trace.final_action == "accept"
    assert len(r.last_trace.iterations) == 2
    assert r.last_trace.iterations[0]["action"] == "retry_table"
    assert r.last_trace.iterations[-1]["action"] == "accept"
    assert all(hit["payload"]["evidence_decision"]["action"] == "accept" for hit in out)
    trace = r.last_trace.as_dict()
    assert trace["final_action"] == "accept"
    assert trace["final_mismatch"] == "none"


def test_retrieve_with_iteration_routes_to_sql_on_value_mismatch():
    """表里有指标 schema 但期间不匹配 → 不重试，直接 route_to_sql。"""
    q = FakeQdrant(
        table_hits=[_schema_table("t1", 0.9, ["研发费用"], "2025A")],
        slices={"t1": ["研发费用 2025A"]},
    )
    r = _retriever(q, coarse_k=100, row_k=200, final_k=1, rerank_pool=1)
    out = r.retrieve_with_iteration("2025Q3研发费用", max_rounds=2, rerank=False)

    assert len(q.table_limits) == 1  # 不重试
    assert r.last_trace.final_action == "route_to_sql"
    assert r.last_trace.final_mismatch == "value_mismatch"
    assert out[0]["payload"]["evidence_decision"]["action"] == "route_to_sql"


def test_retrieve_with_iteration_rejects_empty_context():
    """无候选表且已到最大轮数 → reject_with_reason，trace 保留决策。"""
    q = FakeQdrant(table_hits=[])
    r = _retriever(q, coarse_k=10, row_k=10, final_k=1, rerank_pool=1)
    out = r.retrieve_with_iteration("2025Q3研发费用", max_rounds=0, rerank=False)

    assert out == []
    assert r.last_trace.final_action == "reject_with_reason"
    assert r.last_trace.final_mismatch == "empty_context"
    assert r.last_trace.iterations[0]["action"] == "reject_with_reason"


def test_bm25_leg_merges_vector_candidates_and_returns_bm25_only_table():
    """BM25-only 表也应进入候选并返回整表；trace 记录两路合并数。"""
    q = FakeQdrant(
        table_hits=[_table_hit("t1", 0.9, "向量表")],
        row_hits=[_row_hit("t2", 0.95, "BM25 表命中行")],
        slices={"t1": ["向量表"], "t2": ["BM25 表"]},
    )
    bm25 = FakeBM25([_table_hit("t2", 12.0, "BM25 表")])
    r = MultiStageRetriever(
        FakeEmbedding(),
        q,
        reranker=None,
        bm25_retriever=bm25,
        rerank_pool=2,
        final_k=2,
    )
    out = r.retrieve("中药材价格波动", rerank=False)

    assert {item["payload"]["table_id"] for item in out} == {"t1", "t2"}
    assert r.last_trace.bm25_hits == 1
    assert r.last_trace.bm25_tables == 1
    assert r.last_trace.merged_tables == 2
    assert bm25.calls and bm25.calls[0]["top_k"] == 60


def test_bm25_gate_skips_industry_unconstrained_query():
    """行业级、无实体/指标约束的问题不开 BM25，避免同词不同主体的表涌入。"""
    q = FakeQdrant(
        table_hits=[_table_hit("t1", 0.9, "行业表")],
        slices={"t1": ["行业表"], "t2": ["BM25 表"]},
    )
    bm25 = FakeBM25([_table_hit("t2", 12.0, "BM25 表")])
    r = MultiStageRetriever(
        FakeEmbedding(),
        q,
        reranker=None,
        bm25_retriever=bm25,
        known_entities=["片仔癀"],
        rerank_pool=2,
        final_k=2,
    )
    r.retrieve("中药材价格波动对不同规模中药企业的利润影响差异有多大？", rerank=False)

    assert bm25.calls == []
    assert r.last_trace.bm25_enabled is False
    assert r.last_trace.bm25_gate == "gate_off"


def test_bm25_gate_enables_with_entity_and_metric():
    """实体 + 指标/期间可定位时开 BM25（B2006|2 场景）。"""
    q = FakeQdrant(
        table_hits=[_table_hit("t1", 0.9, "向量表")],
        slices={"t1": ["向量表"], "t2": ["BM25 表"]},
    )
    bm25 = FakeBM25([_table_hit("t2", 12.0, "BM25 表")])
    r = MultiStageRetriever(
        FakeEmbedding(),
        q,
        reranker=None,
        bm25_retriever=bm25,
        known_entities=["片仔癀"],
        rerank_pool=2,
        final_k=2,
    )
    r.retrieve("片仔癀2025Q3研发费用是多少", rerank=False)

    assert bm25.calls
    assert r.last_trace.bm25_enabled is True
    assert r.last_trace.bm25_gate == "entity+metric_or_period"


class FakeDocBM25:
    """只提供 docs 索引的 BM25 替身（兄弟表召回用）。"""

    def __init__(self, docs):
        self.docs = list(docs)

    def retrieve(self, query, query_filter=None, top_k=None):
        return []


class SiblingQdrant(FakeQdrant):
    """整表回取时返回完整 payload 的假 Qdrant。"""

    def __init__(self, table_hits, payloads):
        super().__init__(table_hits=table_hits)
        self.payloads = dict(payloads)

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors, scroll_filter):
        ids = self._table_ids(scroll_filter) or set()
        points = []
        for tid in ids:
            if tid not in self.payloads:
                continue
            payload = dict(self.payloads[tid])
            payload.update(
                chunk_type="table",
                table_id=tid,
                table_slice_index=0,
                chunk_index=0,
            )
            points.append(SimpleNamespace(id=tid, payload=payload))
        return points, None


def _full_table_hit(tid, score, *, file_path, stock, table_type, metrics, anchors, content):
    return {
        "id": tid,
        "score": score,
        "payload": {
            "chunk_type": "table",
            "table_id": tid,
            "file_path": file_path,
            "stockName": stock,
            "stock_name": stock,
            "table_type": table_type,
            "metric_names": list(metrics),
            "anchor_cells": list(anchors),
            "content": content,
        },
    }


def _sibling_setup(metric="营业收入(百万元)", period="2025E", unit="百万元", include_t2=False):
    """构造“同文件已选估值表、兄弟预测表未选”的最小场景。"""
    t1 = _full_table_hit(
        "f.md#t1", 0.9, file_path="f.md", stock="桂林三金", table_type="valuation",
        metrics=["市盈率"],
        anchors=[{"metric": "市盈率", "period": "当前", "value": "17", "unit": None}],
        content="市盈率: 17",
    )
    t2 = _full_table_hit(
        "f.md#t2", 0.8, file_path="f.md", stock="桂林三金", table_type="forecast",
        metrics=[metric],
        anchors=[{"metric": metric, "period": period, "value": "2307", "unit": unit}],
        content="营业收入(百万元): 2307",
    )
    t9 = _full_table_hit(
        "g.md#t9", 0.7, file_path="g.md", stock="华润三九", table_type="ranking",
        metrics=[], anchors=[], content="行业排名",
    )
    t8 = _full_table_hit(
        "h.md#t8", 0.6, file_path="h.md", stock="同仁堂", table_type="ranking",
        metrics=[], anchors=[], content="另一张行业表",
    )
    hits = [t1, t9, t8] + ([t2] if include_t2 else [])
    payloads = {hit["payload"]["table_id"]: hit["payload"] for hit in [t1, t2, t9, t8]}
    docs = [
        {"id": tid, "payload": dict(payload)}
        for tid, payload in payloads.items()
    ]
    return hits, payloads, docs


def _sibling_retriever(hits, payloads, docs, *, enabled, final_k=3):
    q = SiblingQdrant(table_hits=hits, payloads=payloads)
    return q, MultiStageRetriever(
        FakeEmbedding(),
        q,
        reranker=None,
        bm25_retriever=FakeDocBM25(docs),
        known_entities=["桂林三金", "华润三九", "同仁堂"],
        rerank_pool=final_k,
        final_k=final_k,
        sibling_table_recall=enabled,
    )


def test_sibling_recall_disabled_keeps_old_output():
    hits, payloads, docs = _sibling_setup()
    q, r = _sibling_retriever(hits, payloads, docs, enabled=False)
    out = r.retrieve("桂林三金2025年营业收入预计为多少百万元？", rerank=False)

    assert [item["payload"]["table_id"] for item in out] == ["f.md#t1", "g.md#t9", "h.md#t8"]
    assert r.last_trace.sibling_recall_enabled is False
    assert r.last_trace.sibling_replaced == 0


def test_sibling_recall_enabled_replaces_lowest_with_matching_sibling():
    hits, payloads, docs = _sibling_setup()
    q, r = _sibling_retriever(hits, payloads, docs, enabled=True)
    out = r.retrieve("桂林三金2025年营业收入预计为多少百万元？", rerank=False)

    ids = [item["payload"]["table_id"] for item in out]
    assert len(ids) == 3
    assert "f.md#t2" in ids
    assert ids.count("f.md#t2") == 1
    assert r.last_trace.sibling_recall_enabled is True
    assert r.last_trace.sibling_replaced == 1
    assert r.last_trace.sibling_table_ids == ["f.md#t2"]


def test_sibling_recall_skips_unrelated_metric_or_period():
    hits, payloads, docs = _sibling_setup(metric="净利润(百万元)")
    q, r = _sibling_retriever(hits, payloads, docs, enabled=True)
    out = r.retrieve("桂林三金2025年营业收入预计为多少百万元？", rerank=False)

    ids = [item["payload"]["table_id"] for item in out]
    assert "f.md#t2" not in ids
    assert r.last_trace.sibling_replaced == 0


def test_sibling_recall_does_not_duplicate_when_sibling_already_selected():
    hits, payloads, docs = _sibling_setup(include_t2=True)
    q, r = _sibling_retriever(hits, payloads, docs, enabled=True, final_k=4)
    out = r.retrieve("桂林三金2025年营业收入预计为多少百万元？", rerank=False)

    ids = [item["payload"]["table_id"] for item in out]
    assert len(ids) == len(set(ids)) == 4
    assert ids.count("f.md#t2") == 1
    assert r.last_trace.sibling_replaced == 0
