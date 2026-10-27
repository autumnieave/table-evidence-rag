# -*- coding: utf-8 -*-
"""原子化 claim faithfulness：逐条 claim 检查是否被源表/SQL 结果支持。

定位口径**不复用 row_hits**：输入是多阶段检索返回的整表 ``payload.content``（以及可选的
SQL 结果行），由 :mod:`tools.claim_evidence_locator` 在整表全文里定位证据。这样即使行级
精定位失败（季度题缺研报实际值），也能独立回答“答案里的每个断言有没有材料支撑”。

支持的 claim 类型
-----------------
- **数字 claim**：规则抽取（复用 L1 数字口径），全部数字都能在材料中命中才算 supported；
- **观点 claim**：需注入 ``llm`` 做抽取、注入 ``embed`` 做语义定位；默认不开启。
  观点证据本质是语义线索，``support_basis`` 会标为 ``semantic_similarity``，需人工复核，
  不能当作 NLI 结论。

输出为纯字典，可直接落盘或作为评测报告的一节。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from tools.claim_evidence_locator import (
    DEFAULT_MAX_EVIDENCE_PER_CLAIM,
    DEFAULT_MAX_EVIDENCE_PER_REF,
    DEFAULT_SENTENCE_WINDOW,
    IDENTIFIER_TOKEN_RE,
    locate_answer,
    sql_rows_to_material,
)


#: faithfulness 专用数字噪声：股票代码 / 季度 / Top N / 第 N 名不是可核验的财务数值。
#: 用**等长空格**替换，保证 claim 的字符偏移与原文一致。
_NOISE_NUMBER_PATTERNS = (
    re.compile(r"代码[:：]\s*\d{3,6}", re.IGNORECASE),
    re.compile(
        r"(?<!\d)(?:20\d{2}\s*年?\s*)?(?:[Qq][1-4]|[Hh][12])(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\d)\d{6}(?:\.(?:SH|SZ|BJ))?(?!\d)", re.IGNORECASE),
    re.compile(r"(?i)(?<![A-Za-z])(?:top|前)\s*\d{1,3}"),
    re.compile(r"第\s*\d{1,3}(?:名|位|大|强)?"),
    IDENTIFIER_TOKEN_RE,
)


def sanitize_answer_for_claims(answer: str) -> Dict[str, Any]:
    """遮罩 faithfulness 不应核验的“伪数字”（股票代码 / 季度 / Top N / 第 N 名）。

    替换为等长空格，因此 claim 偏移可直接用于原始答案高亮。年份噪声仍由 locator
    的 ``drop_year_like`` 处理；季度 token 必须在这里先遮罩，否则 ``2025Q3`` 的 ``3``
    会作为独立数字 claim 进入证据配额。

    Args:
        answer: 原始答案

    Returns:
        ``{"text": 遮罩后文本, "n_masked": 遮罩片段数}``
    """
    text = answer or ""
    n_masked = 0
    for pattern in _NOISE_NUMBER_PATTERNS:
        def _blank(match: Any) -> str:
            return " " * (match.end() - match.start())

        text, count = pattern.subn(_blank, text)
        n_masked += count
    return {"text": text, "n_masked": n_masked}


def hits_to_materials(hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把多阶段检索的整表 hits 转成 locator 可用的材料结构。

    Args:
        hits: ``[{"id", "score", "payload"}]``，payload 需含 ``content``

    Returns:
        材料列表：``paper_path`` / ``chunk_index`` / ``full_text`` / ``table_id``
    """
    materials: List[Dict[str, Any]] = []
    seen = set()
    for hit in hits or []:
        payload = (hit or {}).get("payload") or {}
        text = str(payload.get("content") or "")
        if not text.strip():
            continue
        table_id = str(payload.get("table_id") or payload.get("parent_id") or "")
        paper_path = str(payload.get("file_path") or payload.get("paper_path") or "")
        chunk_index = payload.get("chunk_index")
        key = (table_id, paper_path, chunk_index, text[:120])
        if key in seen:
            continue
        seen.add(key)
        materials.append(
            {
                "paper_path": paper_path,
                "chunk_index": chunk_index,
                "full_text": text,
                "table_id": table_id,
                "is_table_material": True,
            }
        )
    return materials


def _source_tables(
    evidence: Sequence[Dict[str, Any]],
    materials: Sequence[Dict[str, Any]],
) -> List[str]:
    """把证据片段映射回来源 table_id（SQL 材料记为 ``sql``）。"""
    index: Dict[tuple, str] = {}
    for material in materials:
        key = (str(material.get("paper_path") or ""), material.get("chunk_index"))
        if material.get("is_sql_material"):
            index[key] = "sql"
        elif material.get("table_id"):
            index[key] = str(material["table_id"])
    out: List[str] = []
    for item in evidence:
        key = (str(item.get("paper_path") or ""), item.get("chunk_index"))
        table_id = index.get(key) or (str(item.get("paper_path") or "") or "unknown")
        if table_id and table_id not in out:
            out.append(table_id)
    return out


def evaluate_claim_faithfulness(
    answer: str,
    hits: Sequence[Dict[str, Any]],
    *,
    sql_rows: Optional[Sequence[Any]] = None,
    sql_text: str = "",
    llm: Optional[Any] = None,
    embed: Optional[Any] = None,
    window: int = DEFAULT_SENTENCE_WINDOW,
    drop_structure_noise: bool = True,
    max_evidence_per_claim: int = DEFAULT_MAX_EVIDENCE_PER_CLAIM,
    max_evidence_per_ref: int = DEFAULT_MAX_EVIDENCE_PER_REF,
) -> Dict[str, Any]:
    """逐 claim 评估答案是否被整表内容 / SQL 结果支持。

    Args:
        answer: 待评估答案
        hits: 多阶段检索返回的整表 hits（使用 payload.content，**不读 row_hits**）
        sql_rows: 可选 SQL 结果行（dict 列表）；S 层/mixed 题应传入
        sql_text: 可选 SQL 原文，写入 SQL 材料首行
        llm: 可选观点 claim 抽取函数；默认 None（只核数字 claim）
        embed: 可选观点 claim 向量化函数；默认 None（观点证据不定位）
        window: 数字命中句前后扩展句数
        drop_structure_noise: 是否剔除标题/序号数字噪声
        max_evidence_per_claim: 单 claim 证据上限
        max_evidence_per_ref: 单材料证据上限

    Returns:
        ``{"claims": [...], "summary": {...}, "materials": {...}}``
    """
    materials = hits_to_materials(hits)
    extra: List[Dict[str, Any]] = []
    if sql_rows:
        material = sql_rows_to_material(sql_rows, sql_text=sql_text)
        material["table_id"] = ""
        extra.append(material)

    sanitized = sanitize_answer_for_claims(answer or "")
    located = locate_answer(
        sanitized["text"],
        materials,
        llm=llm,
        embed=embed,
        window=window,
        drop_structure_noise=drop_structure_noise,
        max_evidence_per_claim=max_evidence_per_claim,
        max_evidence_per_ref=max_evidence_per_ref,
        extra_materials=extra,
    )

    claims: List[Dict[str, Any]] = []
    for item in located.get("claims") or []:
        claim = item.get("claim") or {}
        evidence = item.get("evidence") or []
        kind = str(claim.get("kind") or "")
        missing = list(item.get("missing_numbers") or [])
        numbers = list(claim.get("numbers") or [])
        start = claim.get("start")
        end = claim.get("end")
        display_text = str(claim.get("text") or "")
        if isinstance(start, int) and isinstance(end, int) and start >= 0 and end >= start:
            display_text = (answer or "")[start:end]

        if kind == "number":
            supported = bool(numbers) and not missing and bool(evidence)
            support_basis = "number_match"
        else:
            strong = any(str(e.get("tier") or "") == "高" for e in evidence)
            supported = bool(strong)
            support_basis = "semantic_similarity" if evidence else "none"

        claims.append(
            {
                "text": display_text,
                "kind": kind,
                "start": start,
                "end": end,
                "numbers": numbers,
                "matched_numbers": list(item.get("matched_numbers") or []),
                "missing_numbers": missing,
                "supported": supported,
                "support_basis": support_basis,
                "needs_review": kind != "number" and bool(evidence),
                "source_tables": _source_tables(evidence, list(materials) + extra),
                "evidence": evidence,
                "note": str(claim.get("note") or ""),
            }
        )

    n_claims = len(claims)
    n_supported = sum(1 for c in claims if c["supported"])
    number_claims = [c for c in claims if c["kind"] == "number"]
    opinion_claims = [c for c in claims if c["kind"] == "opinion"]
    summary = {
        "n_claims": n_claims,
        "n_supported": n_supported,
        "n_unsupported": n_claims - n_supported,
        "support_rate": round(n_supported / n_claims, 4) if n_claims else 0.0,
        "n_number_claims": len(number_claims),
        "n_number_supported": sum(1 for c in number_claims if c["supported"]),
        "n_opinion_claims": len(opinion_claims),
        "n_opinion_supported": sum(1 for c in opinion_claims if c["supported"]),
        "n_claims_without_evidence": sum(1 for c in claims if not c["evidence"]),
        "n_claims_with_missing_numbers": sum(1 for c in claims if c["missing_numbers"]),
        "n_missing_numbers": sum(len(c["missing_numbers"]) for c in claims),
        "n_table_materials": len(materials),
        "n_sql_materials": len(extra),
        "n_noise_tokens_masked": sanitized["n_masked"],
        "llm_injected": llm is not None,
        "embed_injected": embed is not None,
        "locator_meta": located.get("meta") or {},
    }
    return {
        "claims": claims,
        "summary": summary,
        "materials": {
            "tables": [
                {"table_id": m.get("table_id"), "paper_path": m.get("paper_path")}
                for m in materials
            ],
            "sql": bool(extra),
        },
    }


def aggregate_faithfulness(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把多题结果聚合成评测汇总（按题等权，不按 claim 数加权）。"""
    rows = list(rows or [])
    n = len(rows)
    if not n:
        return {
            "n_questions": 0,
            "mean_support_rate": 0.0,
            "n_claims": 0,
            "n_supported": 0,
            "n_number_claims": 0,
            "n_number_supported": 0,
        }
    summaries = [r.get("summary") or r for r in rows]
    return {
        "n_questions": n,
        "mean_support_rate": round(
            sum(float(s.get("support_rate") or 0.0) for s in summaries) / n, 4
        ),
        "n_claims": sum(int(s.get("n_claims") or 0) for s in summaries),
        "n_supported": sum(int(s.get("n_supported") or 0) for s in summaries),
        "n_number_claims": sum(int(s.get("n_number_claims") or 0) for s in summaries),
        "n_number_supported": sum(int(s.get("n_number_supported") or 0) for s in summaries),
        "n_claims_without_evidence": sum(
            int(s.get("n_claims_without_evidence") or 0) for s in summaries
        ),
        "n_missing_numbers": sum(int(s.get("n_missing_numbers") or 0) for s in summaries),
    }


__all__ = [
    "aggregate_faithfulness",
    "evaluate_claim_faithfulness",
    "hits_to_materials",
    "sanitize_answer_for_claims",
]
