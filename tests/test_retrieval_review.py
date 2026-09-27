# -*- coding: utf-8 -*-
"""B-24B：retrieval 人工审核工作单单测（纯逻辑，零外部依赖）

覆盖：全文匹配键、人工判定四态归一、「同」沿用模型判定、判据 2 的 0.5 派生、
判据 3 的 Precision 同文件最多 2 条、CSV 摊平、--apply 的全部校验与写回。
不依赖 Qdrant / LLM / 网络。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from tools.data_scripts.retrieval_review import (
    CSV_FIELDS,
    JUDGMENT_FULL,
    JUDGMENT_NONE,
    JUDGMENT_PARTIAL,
    apply_csv,
    build_question_rows,
    derive_renjian_xiuzheng,
    entry_key,
    file_ordinals,
    lookup_fulltext,
    match_key,
    normalize_judgment,
    precision_flags,
    read_text_auto,
)


# --------------------------------------------------------------------------
# 全文匹配键
# --------------------------------------------------------------------------
def test_match_key_truncates_400_char_head_to_100():
    head = "甲" * 400
    assert match_key(head) == "甲" * 100


def test_match_key_short_text_returns_whole_text():
    assert match_key("状") == "状"
    assert match_key("") == ""


def test_match_key_handles_none():
    assert match_key(None) == ""


# --------------------------------------------------------------------------
# 人工判定归一（判据 1 + 判据 2）
# --------------------------------------------------------------------------
def test_normalize_empty_is_unfilled():
    assert normalize_judgment("", True) is None
    assert normalize_judgment(None, False) is None
    assert normalize_judgment("   ", True) is None


def test_normalize_tong_follows_model_true():
    assert normalize_judgment("同", True) == JUDGMENT_FULL


def test_normalize_tong_follows_model_false():
    assert normalize_judgment("同", False) == JUDGMENT_NONE


def test_normalize_tong_without_model_raises():
    with pytest.raises(ValueError):
        normalize_judgment("同", None)


def test_normalize_related_aliases():
    assert normalize_judgment("相关", False) == JUDGMENT_FULL
    assert normalize_judgment("完全相关", False) == JUDGMENT_FULL


def test_normalize_partial_and_irrelevant():
    assert normalize_judgment("部分相关", False) == JUDGMENT_PARTIAL
    assert normalize_judgment("不相关", True) == JUDGMENT_NONE


def test_normalize_invalid_raises():
    with pytest.raises(ValueError):
        normalize_judgment("大概相关", True)


def test_normalize_strips_whitespace():
    assert normalize_judgment("  同  ", True) == JUDGMENT_FULL


# --------------------------------------------------------------------------
# 判据 2：部分相关进分母不进分子（JSON 侧保持 bool 兼容）
# --------------------------------------------------------------------------
def test_derive_bool_mapping():
    assert derive_renjian_xiuzheng(JUDGMENT_FULL) is True
    assert derive_renjian_xiuzheng(JUDGMENT_NONE) is False
    assert derive_renjian_xiuzheng(JUDGMENT_PARTIAL) is None
    assert derive_renjian_xiuzheng(None) is None


# --------------------------------------------------------------------------
# 判据 3：Precision 同文件最多 2 条
# --------------------------------------------------------------------------
def test_precision_flags_keeps_first_two_per_file():
    assert precision_flags(["a.md", "a.md", "a.md"]) == [True, True, False]


def test_precision_flags_interleaved_files_are_counted_separately():
    assert precision_flags(["a.md", "b.md", "a.md", "b.md", "a.md"]) == [True, True, True, True, False]


def test_precision_flags_empty():
    assert precision_flags([]) == []


def test_precision_flags_max_is_configurable():
    assert precision_flags(["a.md", "a.md"], max_per_file=1) == [True, False]


def test_file_ordinals():
    assert file_ordinals(["a.md", "a.md", "b.md", "a.md"]) == [1, 2, 1, 3]


# --------------------------------------------------------------------------
# 摊平成 CSV 行
# --------------------------------------------------------------------------
def _question(bid="B2001", sub_idx=1, file_paths=None, flags=None):
    paths = file_paths or ["a.md"] * 3
    rel = flags if flags is not None else [True, False, None]
    return {
        "bid": bid,
        "问题类型": "融合",
        "sub_idx": sub_idx,
        "子问题": "问题文本",
        "chunks": [
            {
                "rank": i + 1,
                "file_path": paths[i],
                "score": 0.1 * (i + 1),
                "text_head": "头部" * 10,
                "_text_full_chars": 500,
                "judge_relevant": rel[i],
                "judge_reason": "理由 %d" % (i + 1),
            }
            for i in range(len(paths))
        ],
    }


def test_build_question_rows_marks_precision_by_rank_order():
    rows = build_question_rows(_question(file_paths=["a.md", "a.md", "a.md"]))
    assert [r["是否计入 Precision"] for r in rows] == ["是", "是", "否"]
    assert [r["该文件内序次"] for r in rows] == [1, 2, 3]


def test_build_question_rows_does_not_prejudge_relevance():
    rows = build_question_rows(_question())
    assert all(r["人工判定"] == "" for r in rows)
    assert all(r["人工备注"] == "" for r in rows)


def test_build_question_rows_renders_model_judgment():
    rows = build_question_rows(_question(flags=[True, False, None]))
    assert [r["模型判定"] for r in rows] == ["相关", "不相关", ""]


def test_build_question_rows_keeps_private_keys_for_fetch():
    rows = build_question_rows(_question())
    assert rows[0]["_text_head"] == "头部" * 10
    assert rows[0]["_model_relevant"] is True
    assert rows[0]["_问"] == "问题文本"


def test_csv_fields_exclude_private_keys():
    assert all(not f.startswith("_") for f in CSV_FIELDS)
    for col in ("是否计入 Precision", "人工判定", "人工备注", "片段全文"):
        assert col in CSV_FIELDS


# --------------------------------------------------------------------------
# 全文取用（同键多条按长度挑）
# --------------------------------------------------------------------------
def _index(file_path, entries):
    """entries: [(content, chunk_index)]"""
    bucket = {}
    for content, chunk_index in entries:
        bucket.setdefault(match_key(content), []).append(
            {"content": content, "chunk_index": chunk_index, "content_hash": "h"}
        )
    return {file_path: bucket}


def test_lookup_fulltext_exact_head_match():
    index = _index("a.md", [("甲" * 400, 0)])
    hit = lookup_fulltext(index, "a.md", "甲" * 300, 400)
    assert hit is not None and len(hit["content"]) == 400


def test_lookup_fulltext_picks_by_full_length_when_key_collides():
    index = _index("a.md", [("乙" * 400, 0), ("乙" * 900, 1)])
    hit = lookup_fulltext(index, "a.md", "乙" * 400, 900)
    assert hit is not None and len(hit["content"]) == 900


def test_lookup_fulltext_missing_returns_none():
    assert lookup_fulltext({}, "a.md", "头部", 500) is None
    assert lookup_fulltext(_index("a.md", [("x" * 400, 0)]), "a.md", "y" * 100, 400) is None


# --------------------------------------------------------------------------
# --apply
# --------------------------------------------------------------------------
def _write_labels(path: Path, questions) -> None:
    path.write_text(
        json.dumps({"questions": questions, "口径": "模型预标注"}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _csv_rows_from(questions, judgments, notes=None, precision=None):
    """按预标注生成待填写 CSV 行；judgments 为 {(bid,sub_idx,rank): 填写值}"""
    rows = []
    for question in questions:
        built = build_question_rows(question)
        for row in built:
            key = (row["bid"], row["sub_idx"], row["排名"])
            row = dict(row)
            row["人工判定"] = judgments.get(key, "")
            if notes and key in notes:
                row["人工备注"] = notes[key]
            if precision and key in precision:
                row["是否计入 Precision"] = precision[key]
            rows.append(row)
    return rows


def _args(labels: Path, csv_path: Path, dry_run=False, reviewer="", review_date=""):
    return argparse.Namespace(
        labels=str(labels),
        apply=str(csv_path),
        dry_run=dry_run,
        reviewer=reviewer,
        review_date=review_date,
    )


def test_apply_dry_run_reports_stats_and_does_not_write(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    before = labels.read_text(encoding="utf-8")
    _write_csv(csv_path, _csv_rows_from(questions, {(  "B2001", 1, 1): "同", ("B2001", 1, 2): "相关", ("B2001", 1, 3): "部分相关"}))
    stats = apply_csv(_args(labels, csv_path, dry_run=True))
    assert stats["完全相关"] == 2
    assert stats["部分相关"] == 1
    assert stats["不相关"] == 0
    assert stats["总条数"] == 3
    assert stats["与模型判定不同"] == 1
    assert labels.read_text(encoding="utf-8") == before


def test_apply_rejects_row_count_mismatch(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    rows = _csv_rows_from(questions, {})
    _write_csv(csv_path, rows[:-1])
    with pytest.raises(ValueError, match="CSV 数据行"):
        apply_csv(_args(labels, csv_path, dry_run=True))


def test_apply_rejects_unknown_row(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    rows = _csv_rows_from(questions, {})
    rows[0]["bid"] = "B9999"
    _write_csv(csv_path, rows)
    with pytest.raises(ValueError, match="没有的行"):
        apply_csv(_args(labels, csv_path, dry_run=True))


def test_apply_rejects_duplicate_row(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    rows = _csv_rows_from(questions, {})
    rows[1]["排名"] = rows[0]["排名"]
    _write_csv(csv_path, rows)
    with pytest.raises(ValueError, match="重复行"):
        apply_csv(_args(labels, csv_path, dry_run=True))


def test_apply_rejects_unfilled_rows(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    _write_csv(csv_path, _csv_rows_from(questions, {("B2001", 1, 1): "同"}))
    with pytest.raises(ValueError, match="未填"):
        apply_csv(_args(labels, csv_path, dry_run=True))


def test_apply_rejects_tong_when_model_judgment_missing(tmp_path):
    questions = [_question(flags=[None, None, None])]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    _write_csv(csv_path, _csv_rows_from(questions, {("B2001", 1, 1): "同", ("B2001", 1, 2): "同", ("B2001", 1, 3): "同"}))
    with pytest.raises(ValueError, match="不能填"):
        apply_csv(_args(labels, csv_path, dry_run=True))


def test_apply_writes_back_fields_and_backup(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    _write_csv(
        csv_path,
        _csv_rows_from(
            questions,
            {("B2001", 1, 1): "同", ("B2001", 1, 2): "不相关", ("B2001", 1, 3): "部分相关"},
            notes={("B2001", 1, 3): "只给了一半指标"},
        ),
    )
    stats = apply_csv(_args(labels, csv_path, reviewer="autumnieave", review_date="2026-09-27"))
    assert stats["完全相关"] == 1 and stats["不相关"] == 1 and stats["部分相关"] == 1

    data = json.loads(labels.read_text(encoding="utf-8"))
    chunks = data["questions"][0]["chunks"]
    # 判据 2：完全相关 -> bool True；不相关 -> False；部分相关 -> None（权威值在「人工判定」）
    assert [c["人工判定"] for c in chunks] == [JUDGMENT_FULL, JUDGMENT_NONE, JUDGMENT_PARTIAL]
    assert [c["人工修正"] for c in chunks] == [True, False, None]
    # 判据 3：CSV 的「是否计入 Precision」原样写回
    assert [c["计入Precision"] for c in chunks] == [True, True, False]
    assert chunks[2]["人工备注"] == "只给了一半指标"
    assert data["人工审核"]["审核人"] == "autumnieave"
    assert data["人工审核"]["审核日期"] == "2026-09-27"
    assert data["人工审核"]["统计"]["总条数"] == 3

    backups = list(tmp_path.glob("labels.backup-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["questions"][0]["chunks"][0].get("人工判定") is None


def test_apply_defaults_review_date_to_today(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    _write_csv(csv_path, _csv_rows_from(questions, {("B2001", 1, 1): "同", ("B2001", 1, 2): "同", ("B2001", 1, 3): "不相关"}))
    apply_csv(_args(labels, csv_path))
    data = json.loads(labels.read_text(encoding="utf-8"))
    assert len(data["人工审核"]["审核日期"]) == 10


def test_entry_key_is_stable_across_types():
    assert entry_key("B2001", 1, 3) == "B2001|1|3"

# --------------------------------------------------------------------------
# 编码自适应（Excel 在中文 Windows 下把 CSV 存成 GBK）
# --------------------------------------------------------------------------
def test_read_text_auto_decodes_utf8_and_bom(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text("人工判定", encoding="utf-8")
    assert read_text_auto(plain) == "人工判定"
    bom = tmp_path / "bom.txt"
    bom.write_text("人工判定", encoding="utf-8-sig")
    assert read_text_auto(bom) == "人工判定"


def test_read_text_auto_decodes_gbk(tmp_path):
    gbk = tmp_path / "gbk.txt"
    gbk.write_text("部分相关", encoding="gbk")
    assert read_text_auto(gbk) == "部分相关"


def _write_csv_gbk(path: Path, rows) -> None:
    """模拟 Excel 另存：GBK 编码 + CRLF"""
    with path.open("w", encoding="gbk", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_apply_accepts_gbk_csv_from_excel(tmp_path):
    questions = [_question()]
    labels, csv_path = tmp_path / "labels.json", tmp_path / "w.csv"
    _write_labels(labels, questions)
    _write_csv_gbk(
        csv_path,
        _csv_rows_from(questions, {("B2001", 1, 1): "同", ("B2001", 1, 2): "部分相关", ("B2001", 1, 3): "不相关"}),
    )
    stats = apply_csv(_args(labels, csv_path))
    assert stats["完全相关"] == 1
    assert stats["部分相关"] == 1
    assert stats["不相关"] == 1
    data = json.loads(labels.read_text(encoding="utf-8"))
    chunks = data["questions"][0]["chunks"]
    assert [c["人工判定"] for c in chunks] == [JUDGMENT_FULL, JUDGMENT_PARTIAL, JUDGMENT_NONE]
    assert chunks[1]["人工修正"] is None
