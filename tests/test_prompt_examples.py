# -*- coding: utf-8 -*-
"""B-15 few-shot 示例库 schema 校验单测：零外部依赖（不调 LLM/MySQL）。

校验 prompts/examples/*.jsonl 的字段结构与指标标准化 JSON 契约一致，
作为示例库入库的门禁；示例库与 golden 同规则版本化。
"""

from __future__ import annotations

import json
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "src" / "prompts" / "examples"

_REQUIRED_ENTRY_KEYS = {"question", "metric", "sql", "note", "pitfall"}
_REQUIRED_METRIC_KEYS = {"standard_fields", "time_grain", "calculation", "filter_terms"}
_REQUIRED_FILTER_KEYS = {"company", "companies", "scope", "threshold"}
_EXPECTED_TYPE_FILES = {
    "single_period",
    "cross_period_trend",
    "ranking_compare",
    "binding_relation",
    "industry_mean",
}
_VALID_CALC_KINDS = {"raw", "rank", "industry_mean", "compare", "multi_period_history"}


def _load_all_examples() -> list[dict]:
    entries = []
    for path in sorted(_EXAMPLES_DIR.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            entries.append((path.name, line_no, json.loads(line)))
    return entries


def test_example_files_cover_five_types_and_minimum_count() -> None:
    """5 类题型文件齐全且总数 >= 10 条。"""
    files = {p.stem for p in _EXAMPLES_DIR.glob("*.jsonl")}
    assert files == _EXPECTED_TYPE_FILES, f"示例题型文件与期望不一致: {files ^ _EXPECTED_TYPE_FILES}"
    entries = _load_all_examples()
    assert len(entries) >= 10, f"示例总数不足: {len(entries)}"


def test_example_entry_schema() -> None:
    """每条示例字段齐全，metric 契约键与 standard_fields 非空、SQL 可执行形状。"""
    for file_name, line_no, entry in _load_all_examples():
        missing = _REQUIRED_ENTRY_KEYS - set(entry)
        assert not missing, f"{file_name}:{line_no} 缺字段 {missing}"
        metric = entry["metric"]
        missing = _REQUIRED_METRIC_KEYS - set(metric)
        assert not missing, f"{file_name}:{line_no} metric 缺键 {missing}"
        missing = _REQUIRED_FILTER_KEYS - set(metric["filter_terms"])
        assert not missing, f"{file_name}:{line_no} filter_terms 缺键 {missing}"
        assert metric["standard_fields"], f"{file_name}:{line_no} standard_fields 为空"
        assert metric["calculation"]["kind"] in _VALID_CALC_KINDS, f"{file_name}:{line_no} kind 非法"
        assert entry["sql"].strip().upper().startswith("SELECT"), f"{file_name}:{line_no} sql 必须以 SELECT 开头"
        assert entry["question"] and entry["note"] and entry["pitfall"]


def test_example_metric_time_grain_consistency() -> None:
    """time_grain.mode 合法且与 JSON 结构配套。"""
    allowed_modes = {"single", "annual_fy", "annual_fy_with_latest_q3", "full_history", "none"}
    for file_name, line_no, entry in _load_all_examples():
        tg = entry["metric"]["time_grain"]
        assert tg["mode"] in allowed_modes, f"{file_name}:{line_no} mode 非法: {tg}"
        if tg["mode"] == "single":
            assert "report_year" in tg and "report_period" in tg, f"{file_name}:{line_no}"
        if tg["mode"] == "annual_fy_with_latest_q3":
            assert "years" in tg and "latest" in tg, f"{file_name}:{line_no}"
