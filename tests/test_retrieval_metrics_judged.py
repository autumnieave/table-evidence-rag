# -*- coding: utf-8 -*-
"""B-24B：四态人工判定口径单测（纯逻辑，零外部依赖）

覆盖：相关性四态归一、部分相关 0.5 权重、Precision 只数「计入」条数、
MRR 的部分相关命中、整题跳过、legacy 口径未被改动。
"""
from __future__ import annotations

from eval.retrieval_metrics import (
    JUDGMENT_FULL,
    JUDGMENT_NONE,
    JUDGMENT_PARTIAL,
    _credit,
    evaluate,
    evaluate_judged,
    evaluate_judged_question,
    has_judged_labels,
    relevance_state,
)


def _c(judged=None, fixed=None, model=None, included=None):
    """构造片段：四态判定 / 人工修正 / 模型判定 / 是否计入 Precision"""
    chunk = {"人工判定": judged, "人工修正": fixed, "judge_relevant": model}
    if included is not None:
        chunk["计入Precision"] = included
    return chunk


# --------------------------------------------------------------------------
# 四态归一
# --------------------------------------------------------------------------
def test_relevance_state_reads_four_states():
    assert relevance_state(_c(judged=JUDGMENT_FULL)) == JUDGMENT_FULL
    assert relevance_state(_c(judged=JUDGMENT_PARTIAL)) == JUDGMENT_PARTIAL
    assert relevance_state(_c(judged=JUDGMENT_NONE)) == JUDGMENT_NONE
    assert relevance_state(_c(judged="")) is None
    assert relevance_state(_c(judged=None)) is None


def test_relevance_state_priority_judged_over_fixed_over_model():
    # 人工判定优先于 人工修正 与 模型判定
    assert relevance_state(_c(judged=JUDGMENT_PARTIAL, fixed=True, model=True)) == JUDGMENT_PARTIAL
    # 无四态时回退到 人工修正
    assert relevance_state(_c(judged=None, fixed=True, model=False)) == JUDGMENT_FULL
    assert relevance_state(_c(judged=None, fixed=False, model=True)) == JUDGMENT_NONE
    # 再回退到模型判定
    assert relevance_state(_c(judged=None, fixed=None, model=True)) == JUDGMENT_FULL
    assert relevance_state(_c(judged=None, fixed=None, model=False)) == JUDGMENT_NONE
    # 全无 → 未判定
    assert relevance_state({}) is None


def test_unknown_judgment_string_is_treated_as_unlabeled():
    assert relevance_state(_c(judged="大概相关", model=True)) == JUDGMENT_FULL


def test_credit_weights():
    assert _credit(JUDGMENT_FULL) == 1.0
    assert _credit(JUDGMENT_PARTIAL) == 0.5
    assert _credit(JUDGMENT_NONE) == 0.0
    assert _credit(None) == 0.0


# --------------------------------------------------------------------------
# 判据 2：部分相关 0.5，且进 Recall 分母
# --------------------------------------------------------------------------
def test_recall_gives_half_credit_and_partial_counts_in_denominator():
    chunks = [_c(judged=JUDGMENT_FULL), _c(judged=JUDGMENT_PARTIAL), _c(judged=JUDGMENT_NONE)]
    m = evaluate_judged_question(chunks)
    # 分子 = 1.0 + 0.5 = 1.5；分母 = 完全相关 1 + 部分相关 1 = 2
    assert m["相关片段数"] == 2
    assert m["完全相关片段数"] == 1
    assert m["部分相关片段数"] == 1
    assert m["Recall@10"] == 0.75


def test_recall_is_one_when_only_full_relevant():
    chunks = [_c(judged=JUDGMENT_FULL), _c(judged=JUDGMENT_NONE)]
    assert evaluate_judged_question(chunks)["Recall@10"] == 1.0


def test_recall_is_zero_when_no_relevant():
    chunks = [_c(judged=JUDGMENT_NONE), _c(judged=None, model=False)]
    m = evaluate_judged_question(chunks)
    assert m["相关片段数"] == 0
    assert m["Recall@10"] == 0.0


def test_partial_weight_is_configurable():
    chunks = [_c(judged=JUDGMENT_FULL), _c(judged=JUDGMENT_PARTIAL)]
    assert evaluate_judged_question(chunks, partial_weight=0.0)["Recall@10"] == 0.5
    assert evaluate_judged_question(chunks, partial_weight=1.0)["Recall@10"] == 1.0


# --------------------------------------------------------------------------
# 判据 3：Precision 只数「计入」条数（分子分母都排除）
# --------------------------------------------------------------------------
def test_precision_denominator_counts_only_included_chunks():
    chunks = [
        _c(judged=JUDGMENT_FULL, included=True),
        _c(judged=JUDGMENT_NONE, included=False),
        _c(judged=JUDGMENT_NONE, included=False),
        _c(judged=JUDGMENT_PARTIAL, included=True),
    ]
    m = evaluate_judged_question(chunks)
    assert m["计入Precision片段数"] == 2
    # 分子 = 1.0 + 0.5 = 1.5；分母 = 2（两条被排除）
    assert m["Precision@10"] == 0.75
    # 被排除的片段仍进 Recall：分子 1.5 / 分母 2
    assert m["Recall@10"] == 0.75


def test_precision_counts_all_chunks_when_field_absent():
    chunks = [_c(judged=JUDGMENT_FULL), _c(judged=JUDGMENT_NONE)]
    m = evaluate_judged_question(chunks)
    assert m["计入Precision片段数"] == 2
    assert m["Precision@10"] == 0.5


def test_explicit_false_excludes_but_true_and_absent_do_not():
    chunks = [_c(judged=JUDGMENT_FULL, included=False), _c(judged=JUDGMENT_FULL)]
    assert evaluate_judged_question(chunks)["Precision@10"] == 1.0


# --------------------------------------------------------------------------
# MRR：部分相关算命中
# --------------------------------------------------------------------------
def test_mrr_partial_counts_as_hit():
    chunks = [_c(judged=JUDGMENT_NONE), _c(judged=JUDGMENT_PARTIAL)]
    m = evaluate_judged_question(chunks)
    assert m["首相关排名"] == 2
    assert m["MRR"] == 0.5


def test_mrr_prefers_earliest_full_or_partial():
    chunks = [_c(judged=JUDGMENT_PARTIAL), _c(judged=JUDGMENT_FULL)]
    m = evaluate_judged_question(chunks)
    assert m["首相关排名"] == 1
    assert m["MRR"] == 1.0


def test_mrr_zero_when_nothing_relevant():
    assert evaluate_judged_question([_c(judged=JUDGMENT_NONE)])["MRR"] == 0.0


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def _row(bid, states, included=None):
    chunks = []
    for i, s in enumerate(states):
        inc = None if included is None else included[i]
        chunks.append({"rank": i + 1, "人工判定": s, "计入Precision": inc})
    return {"bid": bid, "问题类型": "融合", "chunks": chunks}


def test_evaluate_judged_summary_and_zero_relevant_count():
    rows = [
        _row("B1", [JUDGMENT_FULL, JUDGMENT_NONE]),
        _row("B2", [JUDGMENT_PARTIAL, JUDGMENT_PARTIAL]),
        _row("B3", [JUDGMENT_NONE, JUDGMENT_NONE]),
    ]
    result = evaluate_judged(rows)
    s = result["summary"]
    assert s["题数"] == 3
    assert s["跳过题数"] == 0
    assert s["零相关题数"] == 1
    assert s["Recall@10"] == round((1.0 + 0.5 + 0.0) / 3, 4)
    assert [q["bid"] for q in result["per_question"]] == ["B1", "B2", "B3"]


def test_evaluate_judged_skips_rows_without_any_label():
    rows = [_row("B1", [JUDGMENT_FULL]), {"bid": "B2", "chunks": [{}]}]
    result = evaluate_judged(rows)
    assert result["summary"]["题数"] == 1
    assert result["summary"]["跳过题数"] == 1
    assert result["summary"]["跳过编号"] == ["B2"]


def test_evaluate_judged_accepts_model_only_labels_via_fallback():
    row = {"bid": "B9", "chunks": [{"judge_relevant": True}, {"judge_relevant": False}]}
    result = evaluate_judged([row])
    assert result["summary"]["Recall@10"] == 1.0
    assert result["summary"]["零相关题数"] == 0


def test_k_larger_than_chunk_count_is_safe():
    chunks = [_c(judged=JUDGMENT_FULL)]
    m = evaluate_judged_question(chunks, k_values=(10, 20))
    assert m["Recall@10"] == m["Recall@20"] == 1.0


def test_has_judged_labels_detects_four_state_marks():
    assert has_judged_labels([_row("B1", [JUDGMENT_PARTIAL])]) is True
    assert has_judged_labels([{"bid": "B2", "chunks": [{"judge_relevant": True}]}]) is False


# --------------------------------------------------------------------------
# legacy 口径未被改动（回归护栏）
# --------------------------------------------------------------------------
def test_legacy_evaluate_still_uses_binary_model_judgment():
    rows = [{"bid": "B1", "chunks": [{"judge_relevant": True}, {"judge_relevant": False}]}]
    result = evaluate(rows, [10])
    # legacy 退化口径：top-K 内有相关 → Recall@10 = 1.0（保持原行为）
    assert result["summary"]["Recall@10"] == 1.0
    assert result["summary"]["Precision@10"] == 0.5
    assert result["summary"]["MRR"] == 1.0


def test_legacy_ignores_four_state_field():
    """legacy 不认识 人工判定，只看 人工修正 / judge_relevant（口径隔离）"""
    rows = [{"bid": "B1", "chunks": [{"人工判定": JUDGMENT_FULL, "judge_relevant": False}]}]
    result = evaluate(rows, [10])
    assert result["summary"]["Recall@10"] == 0.0