# -*- coding: utf-8 -*-
"""把检索证据的元数据与正文拼成生成上下文。

只喂 ``payload.content`` 时，表行里的数字没有主体锚点，模型只能用“某企业”指代；
把 ``stockName / stockCode / report_period / table_id`` 前置后，模型可以点名公司、
写清报告期，人工也更容易回溯。产品链路与评测脚本共用本模块，避免两套口径。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


def build_evidence_context(content: str, payload: Optional[Mapping[str, Any]] = None) -> str:
    """给单条证据正文加一行元数据前缀；没有元数据时原样返回。

    Args:
        content: 证据正文（整表全文 / chunk 文本）
        payload: Qdrant payload（可含 stockName/stockCode/report_period/table_id）

    Returns:
        形如 ``【公司：片仔癀｜代码：600436｜报告期：2024A｜表：xxx.md#t9】`` 的上下文
    """
    payload = payload or {}
    text = str(content or "")
    meta = []
    name = str(payload.get("stock_name") or payload.get("stockName") or "").strip()
    code = str(payload.get("stock_code") or payload.get("stockCode") or "").strip()
    period = str(payload.get("report_period") or payload.get("period") or "").strip()
    table_id = str(payload.get("table_id") or "").strip()
    if name:
        meta.append("公司：" + name)
    if code:
        meta.append("代码：" + code)
    if period:
        meta.append("报告期：" + period)
    if table_id:
        meta.append("表：" + table_id)
    if not meta:
        return text
    return "【" + "｜".join(meta) + "】\n" + text


def build_contexts_from_hits(hits: Sequence[Dict[str, Any]]) -> list[str]:
    """把多阶段检索 hits 转成带元数据前缀的生成上下文列表。"""
    contexts: list[str] = []
    for hit in hits or []:
        payload = (hit or {}).get("payload") or {}
        contexts.append(build_evidence_context(str(payload.get("content") or ""), payload))
    return contexts
