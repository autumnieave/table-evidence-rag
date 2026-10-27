# -*- coding: utf-8 -*-
"""行级事实与 anchor_cells 单元测试（纯逻辑、零外部依赖）。"""

from data.row_facts import build_anchor_cells, build_row_facts, classify_columns
from data.table_struct import TableGrid, build_table


def _table(headers, rows, caption="", title="", doc_type="stock", is_key_value=False):
    grid = TableGrid(
        headers=list(headers), rows=[list(r) for r in rows],
        caption="", is_key_value=is_key_value,
    )
    return build_table(
        0, "a.md", grid, title=title or "某公司2024年报点评",
        caption=caption, doc_type=doc_type,
    )


def test_row_facts_financial_table_core_fields():
    table = _table(
        ["项目", "2024A", "2025E"],
        [["营业收入", "100", "120"], ["归母净利润", "20", "24"]],
        caption="主要财务指标（万元）",
    )
    facts = build_row_facts(table, {"stockName": "贵州茅台", "stockCode": "600519"})
    assert len(facts) == 4
    first = facts[0]
    assert first["text"].startswith("贵州茅台(600519)｜主要财务指标（万元）｜营业收入 > 2024A: 100 万元")
    assert first["anchor"]["source"] == table.table_id
    assert first["anchor"]["row_index"] == 0
    assert first["anchor"]["col_index"] == 1
    assert first["anchor"]["metric"] == "营业收入"
    assert first["anchor"]["period"] == "2024A"
    assert first["anchor"]["value"] == "100"
    assert first["unit"] == "万元" and first["unit_source"] == "caption"
    assert first["anchor"]["chunk_id"].endswith("#r0c1")


def test_row_facts_forecast_period_preserved():
    table = _table(
        ["项目", "2024", "2025E"],
        [["营业收入", "100", "120"]],
        caption="盈利预测与估值",
    )
    facts = build_row_facts(table)
    periods = {f["period"] for f in facts}
    assert periods == {"2024A", "2025E"}
    assert any(f["period_type"] == "forecast" for f in facts)


def test_row_facts_key_value_card_and_unit():
    table = _table(
        ["项目", "列2"], [["总市值（百万元）", "11,349.65"]],
        caption="基本数据", doc_type="stock", is_key_value=True,
    )
    facts = build_row_facts(table, {"stockName": "某公司", "stockCode": "600000"})
    assert len(facts) == 1
    fact = facts[0]
    assert fact["metric"] == "总市值（百万元）"
    assert fact["value"] == "11,349.65"
    assert fact["unit"] == "百万元" and fact["unit_source"] == "cell"
    assert "百万元" in fact["text"]


def test_row_facts_compound_unit_per_share():
    table = _table(
        ["项目", "2024A"], [["每股收益（元/股）", "1.23"]],
        caption="主要财务指标",
    )
    facts = build_row_facts(table)
    assert facts and facts[0]["unit"] == "元/股"
    assert facts[0]["anchor"]["unit_kind"] == "per_share"


def test_row_facts_entity_column_and_percent_value():
    table = _table(
        ["排名", "公司", "涨跌幅"],
        [["1", "A公司", "5.2%"]],
        caption="个股涨幅排名", doc_type="industry",
    )
    facts = build_row_facts(table)
    assert facts
    fact = facts[0]
    assert fact["entity"] == "A公司"
    assert fact["metric"] == "涨跌幅"
    assert fact["value"] == "5.2%"


def test_row_facts_skip_section_rows_and_text_values():
    table = _table(
        ["项目", "2024A"],
        [["利润表", ""], ["营业收入", "100"], ["备注", "资料来源：Wind"]],
        caption="主要财务指标",
    )
    table.section_rows = [0]
    facts = build_row_facts(table)
    assert len(facts) == 1
    assert facts[0]["metric"] == "营业收入"


def test_row_facts_column_plan_for_financial_table():
    table = _table(
        ["项目", "2024A", "2025E"],
        [["营业收入", "100", "120"]],
    )
    plan = classify_columns(table)
    assert plan.metric_col == 0
    assert plan.period_cols == (1, 2)
    assert plan.period_by_col[1] == "2024A"


def test_build_anchor_cells_matches_facts():
    table = _table(
        ["项目", "2024A"], [["营业收入", "100"]], caption="主要财务指标"
    )
    facts = build_row_facts(table)
    anchors = build_anchor_cells(table)
    assert len(anchors) == len(facts) == 1
    assert anchors[0] == facts[0]["anchor"]

def test_table_unit_fallback_is_explicit():
    table = _table(
        ["项目", "2024A"], [["营业收入", "100"]], caption="主要财务指标"
    )
    # 模拟表级单位来自正文/单元格、但行级三来源都取不到的场景。
    table.unit = "万元"
    table.unit_source = "cell"
    table.unit_candidates = ["万元"]
    facts = build_row_facts(table)
    assert facts and facts[0]["unit"] == "万元"
    assert facts[0]["unit_source"] == "table_fallback"


def test_ratio_metric_does_not_inherit_amount_unit():
    table = _table(
        ["项目", "2025E"], [["市盈率(P/E)", "17.62"]],
        caption="盈利预测与估值",
    )
    table.unit = "百万元"
    table.unit_source = "cell"
    table.unit_candidates = ["百万元"]
    facts = build_row_facts(table)
    assert facts
    assert facts[0]["metric"] == "市盈率(P/E)"
    assert facts[0]["unit"] is None
    assert facts[0]["unit_source"] == "none"
    assert facts[0]["anchor"]["unit_kind"] == "ratio"


def test_multi_unit_table_does_not_inherit_cross_metric_unit():
    table = _table(
        ["项目", "2024A"], [["营业收入", "100"]], caption="主要财务指标"
    )
    table.unit = "万元"
    table.unit_source = "cell"
    table.unit_candidates = ["万元", "元/股"]
    facts = build_row_facts(table)
    assert facts
    assert facts[0]["unit"] is None
    assert facts[0]["unit_source"] == "none"
