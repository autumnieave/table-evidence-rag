# -*- coding: utf-8 -*-
"""B-24A：检索排序质量指标单测（纯逻辑，零外部依赖）

覆盖：Hit Rate@K / Precision@K / MRR 的边界、人工修正覆盖模型判定、汇总跳过逻辑。
"""
from __future__ import annotations

from eval.retrieval_metrics import evaluate, evaluate_question, precision_at_k, recall_at_k, reciprocal_rank


def _chunks(flags):
    """把 bool 序列转成 chunks 结构（模型判定）"""
    return [{"judge_relevant": f} for f in flags]


def test_recall_at_k_basic():
    ranked = [False, True, False, True]
    assert recall_at_k(ranked, k=1) == 0.0
    assert recall_at_k(ranked, k=2) == 0.25
    assert recall_at_k(ranked, k=4) == 0.5
    assert recall_at_k(ranked, k=10) == 0.5


def test_recall_at_k_with_total_relevant_denominator():
    """显式分母：top-K 只覆盖一部分相关片段时的标准 Recall 口径"""
    ranked = [True, False, True]
    assert recall_at_k(ranked, k=3, total_relevant=5) == 0.4


def test_recall_at_k_empty():
    assert recall_at_k([], k=10) == 0.0


def test_precision_at_k_basic():
    ranked = [True, True, False, False]
    assert precision_at_k(ranked, k=2) == 1.0
    assert precision_at_k(ranked, k=4) == 0.5
    # K 大于实际条数时按实际条数算
    assert precision_at_k(ranked, k=10) == 0.5
    assert precision_at_k([], k=10) == 0.0


def test_mrr_first_relevant_rank():
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_evaluate_question_prefers_human_correction():
    """人工修正优先于模型判定（B-24B 审核后重算时不需要改脚本）"""
    chunks = [
        {"judge_relevant": True, "人工修正": False},
        {"judge_relevant": False, "人工修正": True},
        {"judge_relevant": False, "人工修正": None},
    ]
    m = evaluate_question(chunks, k_values=(1, 10))
    assert m["相关片段数"] == 1
    assert m["首相关排名"] == 2
    # 分母 = 标注集内相关数（3 条标注里 1 条相关）：k=1 未覆盖 -> 0；k=10 全覆盖 -> 1
    assert m["Hit Rate@1"] == 0.0
    assert m["Hit Rate@10"] == 1.0
    assert m["MRR"] == 0.5


def test_evaluate_aggregate_and_skip():
    rows = [
        {"bid": "Q1", "chunks": _chunks([True, False, False])},
        {"bid": "Q2", "chunks": _chunks([False, True, False])},
        {"bid": "Q3", "chunks": []},                       # 无片段 → 跳过
        {"bid": "Q4", "chunks": _chunks([None, None])},     # 未标注 → 跳过
    ]
    res = evaluate(rows, k_values=(2,))
    s = res["summary"]
    assert s["题数"] == 2
    assert s["跳过题数"] == 2
    assert s["跳过编号"] == ["Q3", "Q4"]
    assert s["Precision@2"] == 0.5
    assert s["MRR"] == 0.75
    assert s["零相关题数"] == 0