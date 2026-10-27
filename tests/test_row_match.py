# -*- coding: utf-8 -*-
"""查询 metric/period 硬匹配单测（纯逻辑，零外部依赖）。"""

from filters.row_match import (
    QueryConstraints,
    extract_query_constraints,
    extract_query_metrics,
    row_fact_matches,
)


def test_b2065_extracts_rnd_and_quarter():
    q = "佐力药业2025年第三季度研发费用占比同比提升5个百分点，请结合核心业绩指标表分析原因。"
    c = extract_query_constraints(q)
    assert "研发费用" in c.metrics
    assert c.period == "2025Q3"


def test_b2015_maps_revenue_alias_and_quarter():
    q = "2025第三季度方盛制药营收多少？同比增长了多少？"
    c = extract_query_constraints(q)
    assert "营业收入" in c.metrics
    assert c.period == "2025Q3"
    assert row_fact_matches({"metric": "营业收入", "period": "2025Q3"}, c)
    assert not row_fact_matches({"metric": "营业收入", "period": "2025A"}, c)


def test_b2069_maps_investing_cashflow_alias():
    q = "佐力药业2025年第三季度投资性现金流量净额为正且金额较大，分析原因。"
    c = extract_query_constraints(q)
    assert c.period == "2025Q3"
    assert row_fact_matches({"metric": "投资活动产生的现金流量净额", "period": "2025Q3"}, c)
    assert not row_fact_matches({"metric": "经营活动产生的现金流量净额", "period": "2025Q3"}, c)


def test_b2005_matches_total_profit_without_period():
    q = "片仔癀利润总额是多少"
    c = extract_query_constraints(q)
    assert c.metrics == ("利润总额",)
    assert c.period is None
    assert row_fact_matches({"metric": "利润总额", "period": "2024A"}, c)


def test_hard_mismatch_rejects_wrong_metric_and_missing_period():
    c = QueryConstraints(metrics=("研发费用",), period="2025Q3")
    assert row_fact_matches({"metric": "研发费用", "period": "2025Q3"}, c)
    assert not row_fact_matches({"metric": "营业收入", "period": "2025Q3"}, c)
    assert not row_fact_matches({"metric": "研发费用", "period": "2025A"}, c)
    assert not row_fact_matches({"metric": "研发费用"}, c)


def test_empty_constraints_do_not_filter():
    c = QueryConstraints()
    assert row_fact_matches({"metric": "营业收入", "period": "2025A"}, c)
    assert extract_query_metrics("请分析原因") == ()


def test_anchor_cells_fallback_for_old_payload():
    c = QueryConstraints(metrics=("研发费用",), period="2025Q3")
    payload = {"anchor_cells": [{"metric": "研发费用", "period": "2025Q3"}]}
    assert row_fact_matches(payload, c)

def test_metric_alias_with_unit_suffix():
    assert row_fact_matches(
        {"metric": "市盈率(P/E)", "period": "2025E"},
        QueryConstraints(metrics=("PE",)),
    )
    assert row_fact_matches(
        {"metric": "总市值（百万元）", "period": "2025E"},
        QueryConstraints(metrics=("总市值",)),
    )

