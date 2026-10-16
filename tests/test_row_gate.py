# -*- coding: utf-8 -*-
"""B-59 A：低信息「裸表行」闸门单测（纯逻辑、零外部依赖）

覆盖：真实劫持样式的排除、业绩对比表/财务预测表的豁免、长正文的豁免、无公司引用不误杀。
"""
from __future__ import annotations

from filters.row_gate import (
    MAX_ROW_CHARS,
    apply_row_gate,
    count_numbers,
    cjk_ratio,
    has_stock_reference,
    is_lowinfo_row,
    is_rowlike,
)

VOCAB = ["佐力药业", "方盛制药", "片仔癀", "昆药集团", "贵州百灵"]

#: 实测劫持样本（B2065 top-1 chunk，54 字，精排 0.6988）
HARMFUL_ROW = "300181. SZ  佐力药业  0.72  2.73  3.71  5.08  2.53  130.81"

#: 实测劫持样本（B2065，135 字，待分配利润/货币资金数据行）
HARMFUL_KV_ROW = (
    "证券代码: 300181.SZ; 证券简称: 佐力药业; 待分配利润\n(2023年报): 7; 待分配利润\n"
    "(2024年报): 8; 待分配利润\n(2025半年报): 8; 货币资金\n(23年报): 9; 货币资金\n"
    "(24年报): 8; 货币资金 (25半年报): 7"
)

#: 业绩对比表（含指标词）——必须保留
PERF_TABLE = (
    "项目 | 2023A | 2024A | 2025E\n"
    "营业收入（百万元） | 1942 | 2580 | 3200\n"
    "归母净利润（百万元） | 383 | 505 | 640\n"
    "毛利率（%） | 66.2 | 67.1 | 68.0\n"
)


def test_count_numbers_and_cjk_ratio() -> None:
    assert count_numbers("1.2 3% 4,567") == 3
    assert count_numbers("无数字") == 0
    assert cjk_ratio("佐力药业") == 1.0
    assert cjk_ratio("300181.SZ") == 0.0


def test_rowlike_detects_table_rows() -> None:
    assert is_rowlike(HARMFUL_ROW) is True
    assert is_rowlike(HARMFUL_KV_ROW) is True
    assert is_rowlike("公司发布2024年业绩快报，实现营业收入25.80亿元，同比增长32.81%。") is False


def test_has_stock_reference_by_name_or_code() -> None:
    assert has_stock_reference("佐力药业2025年三季度", VOCAB) is True
    assert has_stock_reference("603998.SH 方盛制药 9.1%", VOCAB) is True
    assert has_stock_reference("医药行业整体承压", VOCAB) is False


def test_drop_bare_stock_row() -> None:
    """实测劫持样本必须被排除"""
    assert is_lowinfo_row(HARMFUL_ROW, VOCAB) is True


def test_drop_key_value_data_row() -> None:
    """「待分配利润/货币资金」键值数据行（135 字）也要排除（不含指标词）"""
    assert is_lowinfo_row(HARMFUL_KV_ROW, VOCAB) is True


def test_keep_perf_comparison_table() -> None:
    """业绩对比表含指标词（营业收入/归母净利润/毛利率）⇒ 绝不误杀"""
    assert is_rowlike(PERF_TABLE) is True  # 形态上确实是表格
    assert is_lowinfo_row(PERF_TABLE, VOCAB) is False  # 但含指标词 ⇒ 豁免


def test_keep_financial_forecast_table() -> None:
    table = "佐力药业三大财务预测表\n营业收入 2580 3200 3900\n归母净利润 505 640 780"
    assert is_lowinfo_row(table, VOCAB) is False


def test_keep_long_prose_even_without_metric_terms() -> None:
    """长正文（> max_chars）一律保留，即使没有指标词"""
    long_text = "佐力药业" + "公司围绕核心产品持续做渠道下沉与终端覆盖。" * 20
    assert len(long_text) > MAX_ROW_CHARS
    assert is_lowinfo_row(long_text, VOCAB) is False


def test_keep_short_text_without_stock_reference() -> None:
    """没有公司引用的短片段不属于本闸门职责（交其他环节处理）"""
    assert is_lowinfo_row("2025 年 1-9 月行业整体收入增长 3.2%，利润增长 1.8%", VOCAB) is False


def test_keep_short_text_with_metric_term() -> None:
    assert is_lowinfo_row("佐力药业 2025Q3 毛利率 68.2%", VOCAB) is False


def test_keep_when_below_number_threshold() -> None:
    assert is_lowinfo_row("佐力药业 300181", VOCAB) is False


def test_empty_content_is_kept() -> None:
    assert is_lowinfo_row("", VOCAB) is False
    assert is_lowinfo_row(None, VOCAB) is False


def test_vocab_empty_falls_back_to_code_pattern() -> None:
    """词表为空时仍能靠证券代码样式识别"""
    assert is_lowinfo_row(HARMFUL_ROW, None) is True
    assert is_lowinfo_row("佐力药业 0.72  2.73  3.71  5.08", None) is False


def test_apply_row_gate_keeps_order_and_reports_meta() -> None:
    cands = [
        {"payload": {"content": HARMFUL_ROW}, "tag": "bad0"},
        {"payload": {"content": PERF_TABLE}, "tag": "good0"},
        {"payload": {"content": HARMFUL_KV_ROW}, "tag": "bad1"},
        {"payload": {"content": "佐力药业 2025Q3 毛利率 68.2%"}, "tag": "good1"},
    ]
    kept, meta = apply_row_gate(cands, VOCAB)
    assert [c["tag"] for c in kept] == ["good0", "good1"]
    assert meta == {"applied": True, "dropped": 2, "n_in": 4, "n_kept": 2, "max_chars": MAX_ROW_CHARS}


def test_apply_row_gate_noop_when_nothing_dropped() -> None:
    cands = [{"payload": {"content": PERF_TABLE}}]
    kept, meta = apply_row_gate(cands, VOCAB)
    assert kept == cands
    assert meta["applied"] is False and meta["dropped"] == 0


def test_apply_row_gate_empty_is_safe() -> None:
    kept, meta = apply_row_gate([], VOCAB)
    assert kept == []
    assert meta["applied"] is False and meta["n_in"] == 0