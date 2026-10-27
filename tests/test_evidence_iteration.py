# -*- coding: utf-8 -*-
"""证据审计迭代决策离线单测（不依赖 Qdrant/LLM）。"""
from __future__ import annotations

from core.evidence_iteration import (
    ACCEPT,
    CLARIFY_QUERY,
    MISMATCH_EMPTY,
    MISMATCH_ENTITY,
    MISMATCH_QUERY,
    MISMATCH_SCHEMA,
    MISMATCH_VALUE,
    REJECT_WITH_REASON,
    RETRY_ENTITY,
    RETRY_TABLE,
    ROUTE_TO_SQL,
    audit_evidence,
    table_metrics,
    table_periods,
    table_supports_constraints,
)
from filters.row_match import extract_query_constraints


def _hit(**payload):
    return {"id": "t1#s0", "score": 0.9, "payload": payload}


def test_table_metrics_and_periods_include_anchor_cells():
    """指标/期间既读顶层字段，也读 anchor_cells（旧索引兼容）。"""
    payload = {
        "metric_names": ["研发费用"],
        "metric_name": "营业收入,归母净利润",
        "anchor_cells": [{"metric": "毛利率", "period": "2024A"}],
        "report_period": "2025Q3",
    }
    assert "研发费用" in table_metrics(payload)
    assert "营业收入" in table_metrics(payload)
    assert "毛利率" in table_metrics(payload)
    assert "2025Q3" in table_periods(payload)
    assert "2024A" in table_periods(payload)


def test_table_supports_constraints_metric_and_period():
    """指标别名与期间精确匹配同口径。"""
    c = extract_query_constraints("2025年第三季度研发费用是多少")
    ok, period_ok = table_supports_constraints(
        {"metric_names": ["研发费用"], "report_period": "2025Q3"}, c
    )
    assert ok is True and period_ok is True
    ok2, period_ok2 = table_supports_constraints(
        {"metric_names": ["研发费用"], "report_period": "2025A"}, c
    )
    assert ok2 is True and period_ok2 is False


def test_empty_context_retries_then_rejects():
    """无候选表：未超轮数 retry；超过轮数 reject_with_reason。"""
    d1 = audit_evidence("2025Q3研发费用", [], round_no=0, max_rounds=2)
    assert d1.action == RETRY_TABLE
    assert d1.mismatch_type == MISMATCH_EMPTY
    assert d1.next_coarse_k > 100 and d1.next_row_k > 200

    d2 = audit_evidence("2025Q3研发费用", [], round_no=2, max_rounds=2)
    assert d2.action == REJECT_WITH_REASON
    assert d2.next_coarse_k is None and d2.next_row_k is None


def test_schema_mismatch_retries_then_routes_to_sql():
    """候选表没有查询指标 schema：先扩召回，仍没有则转 SQL。"""
    hits = [_hit(metric_names=["销售费用"], report_period="2025Q3", content="...")]
    d1 = audit_evidence("2025Q3研发费用", hits, round_no=0, max_rounds=1)
    assert d1.action == RETRY_TABLE
    assert d1.mismatch_type == MISMATCH_SCHEMA
    assert d1.schema_tables == 0

    d2 = audit_evidence("2025Q3研发费用", hits, round_no=1, max_rounds=1)
    assert d2.action == ROUTE_TO_SQL
    assert d2.mismatch_type == MISMATCH_SCHEMA


def test_value_mismatch_routes_to_sql_immediately():
    """表里有指标 schema，但没有匹配期间 → 典型季度题，转 SQL/财报库。"""
    hits = [_hit(metric_names=["研发费用"], report_period="2025A")]
    d = audit_evidence("2025Q3研发费用", hits, round_no=0, max_rounds=2)
    assert d.action == ROUTE_TO_SQL
    assert d.mismatch_type == MISMATCH_VALUE
    assert d.schema_tables == 1 and d.value_tables == 0


def test_accept_when_metric_and_period_both_match():
    """至少一张表同时满足指标与期间 → accept。"""
    hits = [
        _hit(metric_names=["销售费用"], report_period="2025Q3"),
        _hit(metric_names=["研发费用"], report_period="2025Q3"),
    ]
    d = audit_evidence("2025Q3研发费用", hits, round_no=0, max_rounds=2)
    assert d.action == ACCEPT
    assert d.schema_tables == 1 and d.value_tables == 1


def test_no_constraints_accepts_without_hard_filter():
    """抽不到指标/期间时不做硬判，直接 accept。"""
    d = audit_evidence("帮我看看这家公司", [_hit(content="...")], round_no=0)
    assert d.action == ACCEPT
    assert d.table_count == 1


def test_missing_entity_returns_clarify_not_retry():
    """B2004：市场级问题没有公司/指数实体，应澄清而不是扩大召回。"""
    query = "北向资金占A股每天成交金额的15%，外资集体撤退大A股还顶得住吗？"
    d = audit_evidence(
        query,
        [_hit(content="...")],
        round_no=0,
        known_entities=["云南白药", "片仔癀"],
    )
    assert d.action == CLARIFY_QUERY
    assert d.mismatch_type == MISMATCH_QUERY
    assert d.missing_fields == ("entity",)
    assert d.clarify_question


def test_industry_question_does_not_clarify():
    """行业级问题允许没有单一公司实体，不误判成澄清。"""
    d = audit_evidence(
        "中药材价格波动对不同规模中药企业的利润影响差异有多大？",
        [_hit(metric_names=["毛利率"])],
        round_no=0,
        known_entities=["片仔癀"],
    )
    assert d.action != CLARIFY_QUERY


def test_context_entity_prevents_clarify():
    """追问可从前文补实体时，不触发澄清。"""
    d = audit_evidence(
        "涨跌的原因是什么",
        [_hit()],
        round_no=0,
        known_entities=["片仔癀"],
        context_queries=["片仔癀近几年的利润总额变化趋势是什么样的"],
    )
    assert d.action != CLARIFY_QUERY
    assert d.entities == ("片仔癀",)


def test_entity_mismatch_retries_then_rejects():
    """B2007|2：点名主体不在 top-10 时先按主体定向重查，仍无则拒答。"""
    query = "999与白云山相比，去年收益率最高的是"
    hits = [_hit(stockName="云南白药", stockCode=538, content="...")]
    known = ["华润三九", "白云山", "云南白药"]
    d1 = audit_evidence(query, hits, round_no=0, max_rounds=2, coarse_k=100, row_k=200, known_entities=known)
    assert d1.action == RETRY_ENTITY
    assert d1.mismatch_type == MISMATCH_ENTITY
    assert d1.missing_entities == ("华润三九", "白云山")
    assert d1.next_coarse_k is None and d1.next_row_k is None

    d2 = audit_evidence(query, hits, round_no=2, max_rounds=2, coarse_k=400, row_k=400, known_entities=known)
    assert d2.action == REJECT_WITH_REASON
    assert d2.mismatch_type == MISMATCH_ENTITY
    assert d2.next_coarse_k is None and d2.next_row_k is None


def test_entity_partial_coverage_still_retries():
    """多主体题只覆盖一个主体仍不算通过。"""
    query = "999与白云山相比，去年收益率最高的是"
    hits = [
        _hit(stockName="华润三九", stockCode=999),
        _hit(stockName="云南白药", stockCode=538),
    ]
    d = audit_evidence(query, hits, round_no=1, max_rounds=2, known_entities=["华润三九", "白云山", "云南白药"])
    assert d.action == RETRY_ENTITY
    assert d.missing_entities == ("白云山",)


def test_entity_mismatch_rejects_after_targeted_retry():
    """按主体定向重查后仍缺实体：不再二次重试，直接拒答。"""
    query = "999与白云山相比，去年收益率最高的是"
    hits = [_hit(stockName="云南白药", stockCode=538)]
    d = audit_evidence(
        query,
        hits,
        round_no=1,
        max_rounds=2,
        known_entities=["华润三九", "白云山", "云南白药"],
        entity_retry_used=True,
    )
    assert d.action == REJECT_WITH_REASON
    assert d.mismatch_type == MISMATCH_ENTITY
    assert d.missing_entities == ("华润三九", "白云山")


def test_entity_full_coverage_accepts_without_metric_period():
    """两个点名主体都覆盖时，即使没有指标/期间约束也 accept。"""
    query = "999与白云山相比，去年收益率最高的是"
    hits = [
        _hit(stockName="华润三九", stockCode=999),
        _hit(stockName="白云山", stockCode=600332),
    ]
    d = audit_evidence(query, hits, round_no=0, max_rounds=2, known_entities=["华润三九", "白云山", "云南白药"])
    assert d.action == ACCEPT
    assert d.missing_entities == ()


def test_entity_code_matches_payload_int_without_leading_zero():
    """000538 应命中 payload stockCode=538。"""
    query = "云南白药000538的收益率如何"
    hits = [_hit(stockName="云南白药", stockCode=538)]
    d = audit_evidence(query, hits, round_no=0, max_rounds=2, known_entities=["云南白药"])
    assert d.action == ACCEPT
