# -*- coding: utf-8 -*-
"""答案实体覆盖检查单测：零外部依赖。"""
from __future__ import annotations

from core.entity_coverage import audit_entity_coverage, build_entity_constraint

KNOWN = ("华润三九", "白云山", "云南白药", "同仁堂", "片仔癀")


def test_detects_subject_substitution() -> None:
    """问题点名 999/白云山，答案换成云南白药/同仁堂 → 主体漂移。"""
    result = audit_entity_coverage(
        "999与白云山相比，去年收益率最高的是",
        "云南白药（000538.SZ）与同仁堂的收益率差异主要来自……",
        known_entities=KNOWN,
        entity_aliases={"000538": "云南白药"},
    )
    assert result.query_entities == ("华润三九", "白云山")
    assert result.answer_entities == ("云南白药", "同仁堂")
    assert result.unexpected_entities == ("云南白药", "同仁堂")
    assert result.missing_query_entities == ("华润三九", "白云山")
    assert result.drift is True
    assert result.needs_repair is True
    assert "华润三九" in build_entity_constraint(result)


def test_industry_question_skips_entity_gate() -> None:
    """行业级问题没有点名主体，答案提到公司不判漂移。"""
    result = audit_entity_coverage(
        "中药材价格波动对不同规模中药企业的利润影响差异有多大？",
        "片仔癀、同仁堂等企业的毛利率存在差异……",
        known_entities=KNOWN,
    )
    assert result.query_entities == ()
    assert result.drift is False
    assert result.needs_repair is False


def test_same_entity_passes() -> None:
    """问题与答案主体一致 → 通过。"""
    result = audit_entity_coverage(
        "华润三九去年净资产收益率是多少？",
        "华润三九 2024 年 ROE 为 12.23%……",
        known_entities=KNOWN,
    )
    assert result.drift is False
    assert result.needs_repair is False
    assert result.missing_query_entities == ()


def test_peer_comparison_is_not_drift() -> None:
    """点名主体已覆盖，额外公司只是对照 → 不算主体漂移。"""
    result = audit_entity_coverage(
        "华润三九与同行相比表现如何？",
        "华润三九毛利率高于同仁堂……",
        known_entities=KNOWN,
    )
    assert result.drift is False
    assert result.needs_repair is False
    assert result.unexpected_entities == ("同仁堂",)


def test_answer_without_named_entity_needs_repair() -> None:
    """问题点名公司，答案完全没提任何公司 → 需要修复。"""
    result = audit_entity_coverage(
        "华润三九去年业绩如何？",
        "中药行业整体承压，营业收入增速放缓……",
        known_entities=KNOWN,
    )
    assert result.answer_entities == ()
    assert result.missing_query_entities == ("华润三九",)
    assert result.drift is False
    assert result.needs_repair is True
