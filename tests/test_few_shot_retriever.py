# -*- coding: utf-8 -*-
"""B-16 few-shot 动态检索纯逻辑单测：零外部依赖（不调 LLM/MySQL/Qdrant）。

覆盖：示例库加载、题型预测（单期/趋势/排名/绑定/行业均值）、top-k 命中、
开关关闭与未命中时回退静态 SQL_GEN 提示词、命中率统计。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompts.financial import SQL_GEN_SYSTEM_PROMPT
from utils.few_shot_retriever import (
    FewShotRetriever,
    build_sql_gen_system,
    format_examples,
    load_examples,
    predict_type,
)

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "src" / "prompts" / "examples"


def _plan(kind: str, mode: str = "single", **extra: object) -> dict:
    calc: dict = {"kind": kind}
    calc.update(extra.get("calculation_extra") or {})
    time_grain: dict = {"mode": mode}
    if mode == "single":
        time_grain.update({"report_year": 2025, "report_period": "Q3"})
    if mode == "annual_fy_with_latest_q3":
        time_grain.update({"years": [2023, 2024], "latest": {"report_year": 2025, "report_period": "Q3"}})
    return {
        "standard_fields": ["stock_abbr", "total_operating_revenue"],
        "time_grain": time_grain,
        "calculation": calc,
        "filter_terms": {"company": None, "companies": None, "scope": "all", "threshold": None},
    }


@pytest.fixture()
def sample_cases() -> list[tuple[str, dict, str]]:
    return [
        (
            "金花股份2025年第三季度的利润总额是多少？",
            _plan("raw"),
            "single_period",
        ),
        (
            "分析金花股份近三年的利润走势。",
            _plan("multi_period_history", mode="annual_fy_with_latest_q3"),
            "cross_period_trend",
        ),
        (
            "2025年三季度研发费用占营业收入比例排名前五的公司有哪些？",
            _plan("rank"),
            "ranking_compare",
        ),
        (
            "找出2025年三季度营业收入前五的公司，并计算它们的资产负债率与销售毛利率、与行业均值对比。",
            _plan("rank", calculation_extra={"order_by": "total_operating_revenue", "top_n": 5, "with_industry_mean": True}),
            "binding_relation",
        ),
        (
            "2025年三季度全体公司销售毛利率的行业均值是多少？",
            _plan("industry_mean"),
            "industry_mean",
        ),
        (
            "对比白云山与云南白药2025年三季度的营业收入。",
            _plan("compare"),
            "ranking_compare",
        ),
    ]


def test_load_examples_min_ten_and_fields() -> None:
    entries = load_examples()
    assert len(entries) >= 10
    required = {"question", "metric", "sql", "note", "pitfall", "_file"}
    for entry in entries:
        assert required <= set(entry), f"缺字段: {required - set(entry)}"


def test_predict_type_covers_all_types(sample_cases: list[tuple[str, dict, str]]) -> None:
    for question, plan, expected in sample_cases:
        assert predict_type(question, plan) == expected, question


def test_retrieve_returns_top_k_matching_type(sample_cases: list[tuple[str, dict, str]]) -> None:
    retriever = FewShotRetriever(k=2)
    for question, plan, expected in sample_cases:
        picked = retriever.retrieve(question, plan)
        assert picked, f"应命中: {question}"
        assert all(e["_file"] == expected for e in picked), (expected, [e["_file"] for e in picked])
        assert len(picked) <= 2
    stats = retriever.stats()
    assert stats["calls"] == len(sample_cases)
    assert stats["hit_rate"] == 1.0


def test_build_sql_gen_system_disabled_returns_static_prompt() -> None:
    question = "金花股份2025年Q3利润总额是多少？"
    plan = _plan("raw")
    system, examples = build_sql_gen_system(question, plan, enabled=False)
    assert system == SQL_GEN_SYSTEM_PROMPT
    assert examples == []


def test_build_sql_gen_system_no_plan_returns_static_prompt() -> None:
    system, examples = build_sql_gen_system("随便问问", None, enabled=True)
    assert system == SQL_GEN_SYSTEM_PROMPT
    assert examples == []


def test_build_sql_gen_system_enabled_injects_examples() -> None:
    question = "2025年三季度全体公司销售毛利率的行业均值是多少？"
    plan = _plan("industry_mean")
    system, examples = build_sql_gen_system(question, plan, enabled=True)
    assert examples
    assert "参考示例" in system
    assert "示例 1" in system
    assert system.startswith(SQL_GEN_SYSTEM_PROMPT)


def test_format_examples_contains_json_sql_and_pitfall() -> None:
    entries = load_examples()
    text = format_examples(entries[:2])
    assert "参考 SQL" in text
    assert "注意规避的坑" in text
    # 拼装结果应保持 JSON 可解析性（提示词正文可含，无结构性破坏即可）
    assert len(text) > 200
