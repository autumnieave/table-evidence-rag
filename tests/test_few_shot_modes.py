# -*- coding: utf-8 -*-
"""B-39 few-shot 三态开关纯逻辑单测：零外部依赖（不调 LLM/MySQL/Qdrant）。

覆盖：resolve_mode 归一（none/static/dynamic + 旧布尔回退 + 非法值）、
load_static_examples 固定示例集（head / per_type 与问题无关）、
build_sql_gen_system 三态分支（none 原样 / static 固定 / dynamic 检索）、
config 的 AGENT_FEWSHOT_MODE 解析、_gen_stats 记录器。
"""

from __future__ import annotations

import os

import pytest

from prompts.financial import SQL_GEN_SYSTEM_PROMPT
from utils.few_shot_retriever import (
    VALID_MODES,
    build_sql_gen_system,
    load_static_examples,
    resolve_mode,
)


def _plan(kind: str = "raw", mode: str = "single", **extra: object) -> dict:
    calc: dict = {"kind": kind}
    calc.update(extra.get("calculation_extra") or {})
    return {
        "standard_fields": ["stock_abbr", "total_operating_revenue"],
        "time_grain": {"mode": mode},
        "calculation": calc,
        "filter_terms": {"company": None, "companies": None, "scope": "all", "threshold": None},
    }


# ---------------------------------------------------------------- resolve_mode
def test_resolve_mode_explicit_values() -> None:
    for m in VALID_MODES:
        assert resolve_mode(enabled=True, mode=m) == m
        assert resolve_mode(enabled=False, mode=m) == m


def test_resolve_mode_legacy_bool_fallback() -> None:
    assert resolve_mode(enabled=True, mode=None) == "dynamic"
    assert resolve_mode(enabled=False, mode=None) == "none"
    assert resolve_mode(enabled=True, mode="") == "dynamic"
    assert resolve_mode(enabled=False, mode="") == "none"


def test_resolve_mode_invalid_falls_back_to_enabled() -> None:
    assert resolve_mode(enabled=True, mode="wild") == "dynamic"
    assert resolve_mode(enabled=False, mode="wild") == "none"


def test_resolve_mode_case_and_space_tolerant() -> None:
    assert resolve_mode(enabled=False, mode=" STATIC ") == "static"


# -------------------------------------------------------- load_static_examples
def test_load_static_examples_head_is_deterministic() -> None:
    first = load_static_examples(k=2, strategy="head")
    second = load_static_examples(k=2, strategy="head")
    assert [e["question"] for e in first] == [e["question"] for e in second]
    assert len(first) == 2


def test_load_static_examples_per_type_covers_distinct_files() -> None:
    picked = load_static_examples(k=3, strategy="per_type")
    assert len(picked) == 3
    assert len({e["_file"] for e in picked}) == 3


def test_load_static_examples_default_strategy_is_per_type() -> None:
    picked = load_static_examples(k=2)
    assert len({e["_file"] for e in picked}) == 2, "默认固定示例应覆盖不同类型"


def test_load_static_examples_zero_k() -> None:
    assert load_static_examples(k=0) == []


# ------------------------------------------------------- build_sql_gen_system
def test_mode_none_returns_prompt_verbatim() -> None:
    system, examples = build_sql_gen_system("任意问题", _plan(), mode="none")
    assert system == SQL_GEN_SYSTEM_PROMPT
    assert examples == []


def test_mode_static_is_question_independent() -> None:
    plan = _plan()
    s1, e1 = build_sql_gen_system("金花股份2025年Q3利润总额是多少？", plan, mode="static")
    s2, e2 = build_sql_gen_system("完全无关的另一个问题：行业均值", plan, mode="static")
    assert e1 and e2
    assert s1 == s2, "static 组必须与问题无关（固定示例）"
    assert s1.startswith(SQL_GEN_SYSTEM_PROMPT)
    assert "参考示例" in s1


def test_mode_dynamic_is_question_dependent() -> None:
    trend_plan = _plan("multi_period_history", "full_history")
    rank_plan = _plan("rank", "single", calculation_extra={"top_n": 10})
    _, e_trend = build_sql_gen_system("片仔癀近几年的利润总额变化趋势", trend_plan, mode="dynamic")
    _, e_rank = build_sql_gen_system("净利润最高的top10企业", rank_plan, mode="dynamic")
    assert e_trend and e_rank
    assert all(e["_file"] == "cross_period_trend" for e in e_trend), [e["_file"] for e in e_trend]
    assert all(e["_file"] == "ranking_compare" for e in e_rank), [e["_file"] for e in e_rank]
    assert [e["question"] for e in e_trend] != [e["question"] for e in e_rank]


def test_mode_dynamic_and_static_differ_for_trend_question() -> None:
    plan = _plan("multi_period_history", "full_history")
    q = "片仔癀近几年的利润总额变化趋势"
    _, e_dyn = build_sql_gen_system(q, plan, mode="dynamic")
    _, e_stat = build_sql_gen_system(q, plan, mode="static")
    assert [e["question"] for e in e_dyn] != [e["question"] for e in e_stat]


def test_no_metric_plan_never_injects_examples() -> None:
    for m in VALID_MODES:
        system, examples = build_sql_gen_system("随便问问", None, mode=m)
        assert system == SQL_GEN_SYSTEM_PROMPT
        assert examples == []


def test_legacy_enabled_kwarg_still_works() -> None:
    assert build_sql_gen_system("问题", _plan("raw"), enabled=False) == (SQL_GEN_SYSTEM_PROMPT, [])
    system, examples = build_sql_gen_system(
        "2025年三季度全体公司销售毛利率的行业均值是多少？", _plan("industry_mean"), enabled=True
    )
    assert examples and system.startswith(SQL_GEN_SYSTEM_PROMPT)


# ------------------------------------------------------------------- config
def test_config_fewshot_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.rag_config import RAGConfig

    monkeypatch.setenv("AGENT_FEWSHOT_MODE", "static")
    cfg = RAGConfig()
    assert cfg.AGENT_FEWSHOT_MODE == "static"
    monkeypatch.delenv("AGENT_FEWSHOT_MODE", raising=False)
    assert RAGConfig().AGENT_FEWSHOT_MODE == ""


def test_config_fewshot_static_k_default() -> None:
    from config.rag_config import RAGConfig

    assert isinstance(RAGConfig().AGENT_FEWSHOT_STATIC_K, int)


# ---------------------------------------------------------------- _gen_stats
def test_gen_stats_recorder_roundtrip() -> None:
    from tools.native_financial import _gen_stats

    stats = _gen_stats()
    stats.record(attempts=2, first_ok=False, mode="dynamic", injected=2, ok=True)
    last = stats.last()
    assert last["attempts"] == 2
    assert last["first_ok"] is False
    assert last["mode"] == "dynamic"
    assert last["injected"] == 2
    assert last["ok"] is True
    stats.record(attempts=1, first_ok=True, mode="static", injected=2, ok=True)
    assert stats.last()["first_ok"] is True
    assert stats.last()["mode"] == "static"
    assert stats.last()["injected"] == 2


def test_resolve_few_shot_mode_reads_config() -> None:
    from tools.native_financial import _resolve_few_shot_mode

    class _Cfg:
        AGENT_FEWSHOT_MODE = "static"
        AGENT_DYNAMIC_FEWSHOT = False

    class _Rag:
        config = _Cfg()

    assert _resolve_few_shot_mode(_Rag()) == "static"

    class _Cfg2:
        AGENT_FEWSHOT_MODE = ""
        AGENT_DYNAMIC_FEWSHOT = True

    class _Rag2:
        config = _Cfg2()

    assert _resolve_few_shot_mode(_Rag2()) == "dynamic"
    assert _resolve_few_shot_mode(object()) == "none"


# --------------------------------------------------- check_sql_structure（B-39）
def _trend_plan() -> dict:
    return {
        "standard_fields": ["stock_abbr", "report_year", "report_period", "total_profit"],
        "time_grain": {"mode": "full_history"},
        "calculation": {"kind": "multi_period_history"},
        "filter_terms": {"company": "片仔癀", "companies": None, "scope": "named", "threshold": None},
    }


def test_check_sql_structure_passes_wellformed_trend_sql() -> None:
    from tools.data_scripts.few_shot_value_experiment import check_sql_structure

    sql = (
        "SELECT stock_abbr, report_year, report_period, total_profit FROM income_sheet "
        "WHERE stock_abbr LIKE '%片仔癀%' ORDER BY report_year ASC"
    )
    res = check_sql_structure(sql, _trend_plan())
    assert res["ok"] is True, res


def test_check_sql_structure_flags_unit_scaling_and_division_in_select() -> None:
    from tools.data_scripts.few_shot_value_experiment import check_sql_structure

    sql = "SELECT net_profit_10k_yuan * 10000 / net_profit FROM core_performance_indicators_sheet"
    res = check_sql_structure(sql, _trend_plan())
    assert res["ok"] is False
    assert "no_10000_scaling" in res["failed"]
    assert "no_division_in_select" in res["failed"]
    assert "trend_has_year_label" in res["failed"]


def test_check_sql_structure_requires_order_by_limit_for_rank() -> None:
    from tools.data_scripts.few_shot_value_experiment import check_sql_structure

    plan = {
        "standard_fields": ["stock_abbr", "net_profit_10k_yuan"],
        "time_grain": {"mode": "single"},
        "calculation": {"kind": "rank", "top_n": 10},
        "filter_terms": {"company": None, "companies": None, "scope": "all", "threshold": None},
    }
    no_limit = "SELECT stock_abbr, net_profit_10k_yuan FROM core_performance_indicators_sheet ORDER BY net_profit_10k_yuan DESC"
    res = check_sql_structure(no_limit, plan)
    assert res["checks"]["rank_has_order_by"] is True
    assert res["checks"]["rank_has_limit"] is False
    assert res["ok"] is False


def test_check_sql_structure_empty_sql_fails_cleanly() -> None:
    from tools.data_scripts.few_shot_value_experiment import check_sql_structure

    res = check_sql_structure("", _trend_plan())
    assert res["ok"] is False
    assert "non_empty" in res["failed"]
