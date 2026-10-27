# -*- coding: utf-8 -*-
"""证据上下文元数据前缀单测：零外部依赖。"""
from __future__ import annotations

from core.evidence_context import build_contexts_from_hits, build_evidence_context


def test_prefixes_metadata() -> None:
    """公司/代码/报告期/表名都前置，正文保持不变。"""
    text = build_evidence_context(
        "| 营业收入 | 10058 | 10788 |",
        {
            "stockName": "片仔癀",
            "stockCode": 600436,
            "report_period": "2024A",
            "table_id": "首次覆盖报告.md#t9",
        },
    )
    assert text.startswith("【公司：片仔癀｜代码：600436｜报告期：2024A｜表：首次覆盖报告.md#t9】\n")
    assert text.endswith("| 营业收入 | 10058 | 10788 |")


def test_snake_case_fields() -> None:
    """兼容 stock_name / stock_code 的 payload 口径。"""
    text = build_evidence_context(
        "正文",
        {"stock_name": "太极集团", "stock_code": "600129"},
    )
    assert text.startswith("【公司：太极集团｜代码：600129】\n正文")


def test_no_metadata_keeps_content() -> None:
    """没有元数据时不添加空前缀。"""
    assert build_evidence_context("正文", {}) == "正文"


def test_build_contexts_from_hits() -> None:
    """批量转换 hits，逐条带上前缀。"""
    contexts = build_contexts_from_hits(
        [
            {"payload": {"content": "A", "stockName": "片仔癀"}},
            {"payload": {"content": "B"}},
        ]
    )
    assert contexts[0].startswith("【公司：片仔癀】\nA")
    assert contexts[1] == "B"
