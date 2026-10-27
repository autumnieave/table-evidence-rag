# -*- coding: utf-8 -*-
"""splitter 行级事实开关测试（纯逻辑、零外部依赖）。"""

import pytest

from data.splitter import split_documents


DOC = {
    "content": "| 项目 | 2024A（万元） |\n| --- | --- |\n| 营业收入 | 100 |\n",
    "metadata": {
        "filename": "a.md",
        "title": "某公司2024年报点评",
        "stockCode": "600519",
        "stockName": "贵州茅台",
    },
}


def test_structured_table_chunks_carry_anchor_cells():
    chunks = split_documents([DOC], chunk_size=256, chunk_overlap=0, table_mode="structured")
    tables = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
    assert tables
    meta = tables[0]["metadata"]
    assert meta["anchor_cells"]
    assert meta["anchor_cells"][0]["metric"] == "营业收入"
    assert meta["anchor_cells"][0]["period"] == "2024A"
    assert meta["row_fact_count"] == 1
    assert meta["chunk_id"].endswith("#s0")


def test_row_fact_mode_adds_child_chunks():
    chunks = split_documents(
        [DOC], chunk_size=256, chunk_overlap=0,
        table_mode="structured", row_fact_mode="chunks",
    )
    tables = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
    facts = [c for c in chunks if c["metadata"].get("chunk_type") == "row_fact"]
    assert len(tables) == 1 and len(facts) == 1
    table, fact = tables[0]["metadata"], facts[0]["metadata"]
    assert fact["parent_id"] == table["table_id"]
    assert fact["is_row_fact"] is True
    assert fact["is_table_row"] is False
    assert fact["anchor_cells"][0]["metric"] == "营业收入"
    assert fact["metric"] == "营业收入"
    assert fact["metric_name"] == "营业收入"
    assert fact["unit"] == "万元"
    assert fact["chunk_id"].endswith("#r0c1")
    assert facts[0]["content"].startswith("贵州茅台(600519)｜")


def test_row_fact_mode_requires_structured():
    with pytest.raises(ValueError):
        split_documents([DOC], table_mode="legacy", row_fact_mode="chunks")
