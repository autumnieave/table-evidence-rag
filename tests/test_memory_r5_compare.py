# -*- coding: utf-8 -*-
"""Memory R@5 对照的纯函数单测（不依赖 Qdrant）。"""

from __future__ import annotations

from eval.memory_r5_compare import FOLLOWUPS, build_query, render_markdown, score_hits, summarize


def test_build_query_injects_history_only_in_memory_mode() -> None:
    item = FOLLOWUPS[0]
    assert build_query(item, with_memory=False) == item.followup
    memory_query = build_query(item, with_memory=True)
    assert item.history in memory_query
    assert item.followup in memory_query


def test_score_hits_requires_entity_metric_and_period() -> None:
    item = FOLLOWUPS[0]
    hits = [
        {"payload": {"content": "云南白药 2025年三季度 净利润 10.2 亿元", "file_path": "云南白药.md"}},
        {"payload": {"content": "云南白药 2025年三季度 营业收入 100 亿元", "file_path": "云南白药.md"}},
    ]
    result = score_hits(hits, item, k=5)
    assert result["entity_hit"] is True
    assert result["metric_hit"] is True
    assert result["period_hit"] is True
    assert result["strict_hit"] is True


def test_score_hits_rejects_entity_only() -> None:
    item = FOLLOWUPS[0]
    hits = [{"payload": {"content": "云南白药 2025年三季度 营业收入 100 亿元"}}]
    result = score_hits(hits, item, k=5)
    assert result["entity_hit"] is True
    assert result["metric_hit"] is False
    assert result["strict_hit"] is False


def test_summarize_counts_errors_in_denominator() -> None:
    rows = [
        {"strict_hit": True, "entity_hit": True, "metric_hit": True, "error": None},
        {"strict_hit": False, "entity_hit": False, "metric_hit": False, "error": "timeout"},
    ]
    summary = summarize(rows, "memory")
    assert summary["n"] == 2
    assert summary["strict_hits"] == 1
    assert summary["r_at_5"] == 0.5
    assert summary["errors"] == 1


def test_render_dry_run_is_explicit() -> None:
    rows = [{"qid": "M01", "mode": "memory", "followup": "它的净利润呢？", "top_k": 0}]
    report = render_markdown(rows_no_memory=rows, rows_memory=rows, dry_run=True, errors=[])
    assert "[DRY-RUN]" in report
    assert "无真实 R@5" in report
