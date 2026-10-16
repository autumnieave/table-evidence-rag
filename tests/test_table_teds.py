"""table_teds（TEDS 简化版）单元测试：含**注入破坏**，证明分数能区分好坏（纯逻辑、零外部依赖）。"""

import copy

import pytest

from data.table_struct import Table
from tools.data_scripts.table_teds import (
    STRUCTURE_DIMENSIONS,
    cell_similarity,
    f1_similarity,
    gold_cells_missing_in_raw,
    pair_similarity,
    parse_number,
    row_cell_cost,
    sequence_cost,
    stack_rows,
    structure_similarity,
    table_to_structure_view,
    teds,
    view_cols,
)

GOLD = {
    "headers": ["项目", "2024A", "2025E"],
    "rows": [["营业收入", "100", "120"], ["净利润", "10", "12"]],
    "has_header": True,
}


def _pred(**overrides):
    """构造解析结果视图（默认与 GOLD 完全一致）。"""
    view = copy.deepcopy(GOLD)
    view.update(overrides)
    return view


def test_identical_tables_score_one():
    assert teds(GOLD, _pred()) == 1.0


def test_empty_tables_score_one():
    assert teds({"headers": [], "rows": []}, {"headers": [], "rows": []}) == 1.0


def test_missing_row_lowers_score():
    pred = _pred(rows=[["营业收入", "100", "120"]])
    # 少一行：代价 = 行节点 + 3 格 = 4；分母 = max(1+4+4, 1+4+4) …… 逐值核对
    score = teds(GOLD, pred)
    assert 0.0 < score < 1.0
    gold_rows = stack_rows(GOLD)
    pred_rows = stack_rows(pred)
    assert len(gold_rows) == 3 and len(pred_rows) == 2


def test_extra_row_lowers_score():
    pred = _pred(rows=[["营业收入", "100", "120"], ["净利润", "10", "12"], ["毛利率", "30%", "31%"]])
    assert teds(GOLD, pred) < 1.0


def test_wrong_number_is_penalised():
    pred = _pred(rows=[["营业收入", "100", "120"], ["净利润", "10", "99"]])
    assert teds(GOLD, pred) < 1.0


def test_formatted_number_is_accepted():
    """``1,234`` 与 ``1234`` 视为同一个数（展示位差异不该扣分）。"""
    gold = {"headers": ["项目", "值"], "rows": [["营收", "1,234.50"]], "has_header": True}
    pred = {"headers": ["项目", "值"], "rows": [["营收", "1234.5"]], "has_header": True}
    assert teds(gold, pred) == 1.0


def test_percent_suffix_must_match():
    """``18.85%`` 与 ``18.85`` 不算相同（单位丢失是真实错误）。"""
    gold = {"headers": ["项目", "值"], "rows": [["毛利率", "18.85%"]], "has_header": True}
    pred = {"headers": ["项目", "值"], "rows": [["毛利率", "18.85"]], "has_header": True}
    assert teds(gold, pred) < 1.0


def test_gold_empty_header_matches_placeholder():
    """Gold 该格为空、解析给占位名（``项目`` / ``列N``）视为相同。"""
    assert pair_similarity("", "项目") == 1.0
    assert pair_similarity("", "列3") == 1.0
    assert pair_similarity("", "毛利率") < 1.0
    gold = {"headers": ["", "2024A"], "rows": [["营收", "1"]], "has_header": True}
    pred = {"headers": ["项目", "2024A"], "rows": [["营收", "1"]], "has_header": True}
    assert teds(gold, pred) == 1.0


def test_header_row_excluded_when_no_header():
    """``has_header=False`` 的表，表头行不参与比对（不存在表头行就不该扣分）。"""
    gold = {"headers": ["项目", "列2"], "rows": [["当前PE", "30.35"]], "has_header": False}
    same = copy.deepcopy(gold)
    assert teds(gold, same) == 1.0
    assert len(stack_rows(gold)) == 1
    with_header = {"headers": ["项目", "列2"], "rows": [["当前PE", "30.35"]], "has_header": True}
    assert teds(gold, with_header) < 1.0  # 解析多出一条表头行 → 扣分


def test_parse_number_handles_cells():
    assert parse_number("1,234.50") == (1234.5, "")
    assert parse_number("18.85%") == (18.85, "%")
    assert parse_number("（x）") is None
    assert parse_number("营业收入") is None
    assert parse_number("") is None


def test_parenthesised_negative():
    assert parse_number("(1,234.5)") == (-1234.5, "")


def test_cell_similarity_bounds():
    assert cell_similarity("", "") == 1.0
    assert cell_similarity("", "x") == 0.0
    assert cell_similarity("毛利率", "毛利率") == 1.0
    assert 0.0 < cell_similarity("毛利率", "净利率") < 1.0


def test_sequence_cost_identity_is_zero():
    items = ["a", "b", "c"]
    cost = sequence_cost(items, items, lambda g, p: 1.0 - float(g == p), lambda _p: 1.0, lambda _g: 1.0)
    assert cost == 0.0


def test_sequence_cost_insert_delete():
    cost = sequence_cost(["a"], [], lambda g, p: 0.0, lambda _p: 1.0, lambda _g: 1.0)
    assert cost == 1.0


def test_row_cell_cost_zero_for_identical_rows():
    assert row_cell_cost(["a", "1"], ["a", "1"]) == 0.0


def test_teds_is_bounded():
    gold = {"headers": ["项目"], "rows": [["a"]], "has_header": True}
    pred = {"headers": ["完全不同"], "rows": [["zzz"]], "has_header": True}
    score = teds(gold, pred)
    assert 0.0 <= score <= 1.0


RAW_HTML = (
    "<table><tr><td>项目</td><td>2024A</td></tr>"
    "<tr><td>营业收入</td><td>1,234.5</td></tr></table>"
)


def test_gold_check_passes_for_faithful_cells():
    assert gold_cells_missing_in_raw(["项目", "2024A"], [["营业收入", "1,234.5"]], RAW_HTML) == []


def test_gold_check_catches_typo():
    """转写错字（原文是 1,234.5，Gold 写成 1,324.5）必须被自检抓到。"""
    missing = gold_cells_missing_in_raw(["项目", "2024A"], [["营业收入", "1,324.5"]], RAW_HTML)
    assert len(missing) == 1
    assert missing[0]["文本"] == "1,324.5"


def test_gold_check_skips_placeholder_header_when_no_header():
    """无表头表：Gold 的表头是占位名，不应被当成「凭空生成」。"""
    assert gold_cells_missing_in_raw(
        ["项目", "列2"], [["营业收入", "1,234.5"]], RAW_HTML, has_header=False
    ) == []
    # 反过来：has_header=True 时占位名不在原文里，就要被查出来
    assert gold_cells_missing_in_raw(
        ["列9"], [["营业收入"]], RAW_HTML, has_header=True
    ) != []


@pytest.mark.parametrize("score_check", [lambda s: 0.0 <= s <= 1.0])
def test_score_range(score_check):
    assert score_check(teds(GOLD, _pred()))

# ---------------------------------------------------------------- 结构分（语义维度）

STRUCT = {
    "section_rows": [0, 26],
    "sub_header_rows": [3],
    "merged_cells": [{"row": 0, "col": 0, "rowspan": 7, "colspan": 1}],
}


def _struct(**overrides):
    view = copy.deepcopy(STRUCT)
    view.update(overrides)
    return view


def test_f1_similarity_empty_vs_empty_is_one():
    """两边都没有分节行 = 这张表本来就没有，算对。"""
    assert f1_similarity(set(), set()) == 1.0


def test_f1_similarity_bounds():
    assert f1_similarity({1}, {1}) == 1.0
    assert f1_similarity({1}, {2}) == 0.0
    assert f1_similarity({1, 2}, {1}) == pytest.approx(2 / 3)


def test_structure_similarity_identical_is_one():
    parts = structure_similarity(STRUCT, _struct())
    assert parts["structure_score"] == 1.0
    assert all(parts[d] == 1.0 for d in STRUCTURE_DIMENSIONS)


def test_structure_score_drops_when_section_rows_lost():
    """**注入破坏**：解析器丢掉分节行 → 结构分必须掉下来（证明指标不是恒 1 的装饰）。"""
    parts = structure_similarity(STRUCT, _struct(section_rows=[]))
    assert parts["section_rows"] == 0.0
    assert parts["sub_header_rows"] == 1.0 and parts["merged_cells"] == 1.0
    assert parts["structure_score"] == pytest.approx(2 / 3)
    assert parts["structure_score"] < 1.0


def test_structure_score_drops_when_extra_section_rows_added():
    parts = structure_similarity(STRUCT, _struct(section_rows=[0, 26, 40]))
    assert parts["section_rows"] == pytest.approx(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))
    assert parts["structure_score"] < 1.0


def test_structure_score_drops_when_merged_geometry_wrong():
    """行/列/rowspan/colspan 任一维错都算不一致。"""
    parts = structure_similarity(
        STRUCT, _struct(merged_cells=[{"row": 0, "col": 1, "rowspan": 7, "colspan": 1}])
    )
    assert parts["merged_cells"] == 0.0
    assert parts["structure_score"] == pytest.approx(2 / 3)


def test_structure_score_drops_when_sub_header_lost():
    parts = structure_similarity(STRUCT, _struct(sub_header_rows=[]))
    assert parts["sub_header_rows"] == 0.0
    assert parts["structure_score"] == pytest.approx(2 / 3)


def test_combined_score_penalises_structure_loss_even_when_teds_is_one():
    """TEDS 满分但结构丢了 → 综合分必须 < 1：这正是加结构分的意义。"""
    struct = structure_similarity(STRUCT, _struct(section_rows=[]))
    assert teds(GOLD, _pred()) == 1.0
    assert teds(GOLD, _pred()) * struct["structure_score"] < 1.0


def test_table_to_structure_view_reads_table_fields():
    table = Table(
        table_id="a.md#t1",
        section_rows=[1],
        sub_header_rows=[2],
        merged_cells=[{"row": 0, "col": 0, "rowspan": 2, "colspan": 1}],
    )
    view = table_to_structure_view(table)
    assert view["section_rows"] == [1]
    assert view["sub_header_rows"] == [2]
    assert view["merged_cells"] == [{"row": 0, "col": 0, "rowspan": 2, "colspan": 1}]
    assert structure_similarity(view, view)["structure_score"] == 1.0


def test_view_cols_uses_rows_width_when_headers_empty():
    """键值卡片（headers 记空表）的列数须看 rows 宽度，否则报告出「gold 0 列 vs pred N 列」假差异。"""
    kv_gold = {"headers": [], "rows": [["项目A", "1.0"], ["项目B", "2.0"]], "has_header": False}
    assert view_cols(kv_gold) == 2
    assert view_cols({"headers": ["a", "b", "c"], "rows": [["1", "2"]], "has_header": True}) == 3
    assert view_cols({"headers": [], "rows": []}) == 0
