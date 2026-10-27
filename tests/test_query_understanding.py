# -*- coding: utf-8 -*-
"""查询理解纯规则单测（不依赖 Qdrant/LLM）。"""
from __future__ import annotations

from filters.query_understanding import (
    detect_scope,
    extract_query_entities,
    understand_query,
)

VOCAB = ["云南白药", "华润三九", "片仔癀", "白云山", "江中药业"]


def test_extract_entities_longest_match_and_code():
    entities = extract_query_entities("华润三九与云南白药000538的对比", VOCAB)
    assert "华润三九" in entities
    assert "云南白药" in entities
    assert "000538" in entities


def test_b2004_market_question_requires_entity():
    query = "北向资金占A股每天成交金额的15%，影响力不可谓不大，外资集体撤退大A股还顶得住吗？"
    result = understand_query(query, known_entities=VOCAB)
    assert result.scope == "market"
    assert result.missing_fields == ("entity",)
    assert result.needs_clarification is True
    assert result.clarify_question and "公司" in result.clarify_question


def test_industry_question_does_not_force_clarify():
    query = "中药材价格波动对不同规模中药企业的利润影响差异有多大？"
    result = understand_query(query, known_entities=VOCAB)
    assert result.scope == "industry"
    assert result.missing_fields == ()
    assert result.needs_clarification is False


def test_deictic_followup_can_be_resolved_by_context():
    query = "涨跌的原因是什么"
    alone = understand_query(query, known_entities=VOCAB)
    assert alone.needs_clarification is True

    resolved = understand_query(
        query,
        known_entities=VOCAB,
        context_queries=["片仔癀近几年的利润总额变化趋势是什么样的"],
    )
    assert "片仔癀" in resolved.entities
    assert resolved.needs_clarification is False


def test_metric_without_entity_requires_clarify():
    result = understand_query("2025Q3研发费用是多少", known_entities=VOCAB)
    assert result.metrics
    assert result.period == "2025Q3"
    assert result.missing_fields == ("entity",)


def test_scope_company_when_entity_present():
    assert detect_scope("片仔癀近几年的利润总额", ("片仔癀",)) == "company"
