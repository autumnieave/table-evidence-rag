# -*- coding: utf-8 -*-
"""原子化 claim faithfulness 离线单测（不依赖 Qdrant/LLM）。"""
from __future__ import annotations

from core.claim_faithfulness import (
    aggregate_faithfulness,
    evaluate_claim_faithfulness,
    hits_to_materials,
    sanitize_answer_for_claims,
)


def _hit(table_id: str, content: str, row_hits=None):
    return {
        "id": table_id + "#s0",
        "score": 0.9,
        "payload": {
            "chunk_type": "table",
            "table_id": table_id,
            "chunk_index": 0,
            "file_path": table_id + ".md",
            "content": content,
            "row_hits": row_hits or [],
        },
    }


def test_hits_to_materials_skips_empty_and_dedupes():
    """空 content 不进材料；同表同文本只保留一次。"""
    hits = [
        _hit("t1", "营业收入 12.34 亿元"),
        _hit("t1", "营业收入 12.34 亿元"),
        _hit("t2", "   "),
    ]
    materials = hits_to_materials(hits)
    assert len(materials) == 1
    assert materials[0]["table_id"] == "t1"
    assert materials[0]["full_text"] == "营业收入 12.34 亿元"


def test_number_claim_supported_when_all_numbers_match():
    """数字 claim 的全部数字都在整表全文命中 → supported。"""
    out = evaluate_claim_faithfulness(
        "2025Q3营业收入12.34亿元，同比增长20%。",
        [_hit("t1", "2025Q3 营业收入 12.34 亿元，同比增长 20%。")],
    )
    assert out["summary"]["n_number_claims"] == 1
    assert out["claims"][0]["supported"] is True
    assert out["claims"][0]["missing_numbers"] == []
    assert out["claims"][0]["source_tables"] == ["t1"]


def test_stock_code_and_topn_are_masked_without_offset_shift():
    """股票代码 / Top N 不是财务数值，遮罩后偏移不变、claim 仍可判 supported。"""
    answer = "片仔癀(600436.SH)营业收入12.34亿元，位列Top10。"
    s = sanitize_answer_for_claims(answer)
    assert len(s["text"]) == len(answer)
    assert s["n_masked"] >= 2
    assert "600436" not in s["text"]

    out = evaluate_claim_faithfulness(
        answer,
        [_hit("t1", "片仔癀 2025Q3 营业收入 12.34 亿元。")],
    )
    assert out["summary"]["n_noise_tokens_masked"] >= 2
    assert out["claims"][0]["supported"] is True
    assert out["claims"][0]["text"] == answer  # 展示仍回原文


def test_quarter_and_code_context_are_masked_without_offset_shift():
    """2025Q3 的 3、代码：423、PZH2108/t2 都不应作为财务数字 claim，且偏移不变。"""
    answer = "2025Q3、2025H1 营业收入 12.34 亿元，公司代码：423，PZH2108 处于 II 期，来源 t2。"
    s = sanitize_answer_for_claims(answer)
    assert len(s["text"]) == len(answer)
    assert "Q3" not in s["text"] and "423" not in s["text"]
    assert "PZH2108" not in s["text"] and "t2" not in s["text"]
    out = evaluate_claim_faithfulness(
        answer,
        [_hit("t1", "2025Q3 营业收入 12.34 亿元。")],
    )
    assert out["claims"][0]["numbers"] == ["12.34"]
    assert out["claims"][0]["supported"] is True


def test_number_claim_unsupported_when_any_number_missing():
    """同一条 claim 里只要有一个数字找不到，就判 unsupported。"""
    out = evaluate_claim_faithfulness(
        "营业收入12.34亿元，净利润5.67亿元。",
        [_hit("t1", "2025Q3 营业收入 12.34 亿元。")],
    )
    assert out["claims"][0]["supported"] is False
    assert "5.67" in out["claims"][0]["missing_numbers"]


def test_faithfulness_uses_table_content_not_row_hits():
    """独立性：row_hits 里没有数字，但整表 content 有 → 仍判 supported。"""
    out = evaluate_claim_faithfulness(
        "研发费用为3.21亿元。",
        [_hit("t1", "研发费用 3.21 亿元", row_hits=[{"content": "无关行"}])],
    )
    assert out["claims"][0]["supported"] is True

    out2 = evaluate_claim_faithfulness(
        "研发费用为3.21亿元。",
        [_hit("t2", "销售费用 1.00 亿元", row_hits=[{"content": "研发费用 3.21 亿元"}])],
    )
    assert out2["claims"][0]["supported"] is False


def test_sql_material_uses_answer_precision_tolerance():
    """SQL 原值 248506.00 万元 ≈ 答案 24.85 亿元（展示位舍入）→ supported。"""
    out = evaluate_claim_faithfulness(
        "利润总额24.85亿元。",
        [],
        sql_rows=[{"total_profit": "248506.00"}],
        sql_text="SELECT total_profit FROM income",
    )
    assert out["summary"]["n_sql_materials"] == 1
    assert out["claims"][0]["supported"] is True
    assert out["claims"][0]["source_tables"] == ["sql"]


def test_no_claims_yields_zero_rate_and_aggregate():
    """无数字答案（未注入观点抽取）→ 0 claims，聚合不除零。"""
    out = evaluate_claim_faithfulness("公司经营稳健。", [_hit("t1", "公司经营稳健。")])
    assert out["summary"]["n_claims"] == 0
    assert out["summary"]["support_rate"] == 0.0
    agg = aggregate_faithfulness([out])
    assert agg["n_questions"] == 1
    assert agg["mean_support_rate"] == 0.0


def test_aggregate_is_question_weighted():
    """多题聚合按题等权，不按 claim 数加权。"""
    a = {"summary": {"support_rate": 1.0, "n_claims": 10, "n_supported": 10}}
    b = {"summary": {"support_rate": 0.0, "n_claims": 1, "n_supported": 0}}
    agg = aggregate_faithfulness([a, b])
    assert agg["mean_support_rate"] == 0.5
    assert agg["n_claims"] == 11
    assert agg["n_supported"] == 10
