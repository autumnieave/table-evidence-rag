# -*- coding: utf-8 -*-
"""坐标迁移契约测试（纯逻辑、零外部依赖）。"""

from eval.label_migration import migrate_anchor, migrate_labels


ANCHORS = [
    {
        "source": "a.md#t1", "table_id": "a.md#t1", "source_file": "a.md",
        "table_caption": "主要财务指标", "row_index": 0, "col_index": 1,
        "metric": "营业收入", "period": "2024A", "value": "100",
    },
    {
        "source": "a.md#t1", "table_id": "a.md#t1", "source_file": "a.md",
        "table_caption": "主要财务指标", "row_index": 1, "col_index": 1,
        "metric": "归母净利润", "period": "2024A", "value": "20",
    },
]


def test_migrate_exact():
    result = migrate_anchor(
        {"source_file": "a.md", "table_id": "a.md#t1", "row_index": 0, "col_index": 1},
        ANCHORS,
    )
    assert result["status"] == "mapped" and result["rule"] == "exact"
    assert result["new"]["metric"] == "营业收入"


def test_migrate_caption_fallback_when_table_id_changed():
    result = migrate_anchor(
        {"source_file": "a.md", "table_id": "a.md#t9", "table_caption": "主要财务指标",
         "row_index": 1, "col_index": 1},
        ANCHORS,
    )
    assert result["status"] == "mapped" and result["rule"] == "caption"
    assert result["new"]["metric"] == "归母净利润"


def test_migrate_range_ambiguous():
    anchors = ANCHORS + [{
        "source": "a.md#t1", "table_id": "a.md#t1", "source_file": "a.md",
        "table_caption": "主要财务指标", "row_index": 2, "col_index": 1,
        "metric": "总资产", "period": "2024A", "value": "300",
    }]
    result = migrate_anchor(
        {"source_file": "a.md", "table_id": "a.md#t1",
         "row_start": 0, "row_end": 3, "col_index": 1},
        anchors,
    )
    assert result["status"] == "ambiguous" and result["rule"] == "range"
    assert len(result["matches"]) == 3


def test_migrate_unmapped_without_coordinates():
    result = migrate_anchor({"file": "a.md", "rank": 1}, ANCHORS)
    assert result["status"] == "unmapped"


def test_migrate_labels_summary():
    result = migrate_labels(
        [
            {"source_file": "a.md", "table_id": "a.md#t1", "row_index": 0, "col_index": 1},
            {"source_file": "a.md", "table_id": "a.md#t1", "row_index": 9, "col_index": 1},
        ],
        ANCHORS,
    )
    assert result["summary"] == {"total": 2, "mapped": 1, "ambiguous": 0, "unmapped": 1}
