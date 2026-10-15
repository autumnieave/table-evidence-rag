# -*- coding: utf-8 -*-
"""实体硬闸门（L5-B）——用查询解析出的公司实体对候选做**确定性硬过滤**。

为什么必须是「硬」过滤
----------------------
项目原有的「软约束」（``RAGPipeline.query()`` 里 ``alpha*vector + (1-alpha)*match`` 重排）对**最终结果无效**：

- 检索器返回 ``RETRIEVAL_K``（默认 50）条候选；
- ``query()`` 把这 **50 条全部**送精排（``reranker.rerank(documents=candidate_docs)``），
  再按相关性排序 + 每文件上限收敛；
- 精排对输入**顺序不敏感**，而软约束只是置换同一集合 ⇒ 集合不变 ⇒ 最终 top-10 不变。

L5 标注终稿里「16 条缺公司主体 / 13 条期间不符」正是软约束无法移除任何候选的直接后果。
⇒ 只有**硬过滤**才能改变线上结果。

实测（2026-10-16，27 条点名公司的子问题，`训练结果数据/l5b_measure/`）：
最终 top-10 里 **25.9% 是他公司**、仅 **13.7%** 是所问公司；开启闸门后他公司占比 **0.0**。

两条安全设计
------------
1. **``keep_unknown_entity=True``（默认）**：保留**没有任何实体标注**的候选（如行业研报）。
   必须开：27 条里有 7 条的公司**根本不在语料内**，严格模式会把 7/27 题的上下文**清空**。
2. **``min_keep``（可选安全闸门，**默认 0 = 不启用**）**：**命中实体的候选数**（不是保留总数）低于
   ``min_keep`` 时**不过滤**。实测（2026-10-16，27 条点名公司题）该闸门**不划算**：``min_keep=3`` 会让
   11 道题跳过过滤、他公司占比停在 0.1333，而 ``min_keep=0`` 的他公司占比 0.0、本主体命中率相同；
   且它**防不住**多主体对比题（它只数「所问公司」的命中数，解析出 A 时 B 的候选照样被移除）。
   真正的安全兜底是 ``keep_unknown_entity=True``（不会清空上下文，实测空上下文 0/27）。

匹配规则（确定性、可单测、可离线）
----------------------------------
- **闭词表归一**（``resolve_names``）：若给出 ``known_names``（语料规范简称全集），
  且查询简称在词表内**唯一子串命中**，则归一到该规范名（``三金`` → ``桂林三金``）；
  命中 0 个或多个（歧义）时不归一，退回原字符串。
  —— 对**闭词表**做子串匹配，比对任意字符串做子串匹配安全得多（不会把「生物」「药业」这类通用词误伤）。
- **代码**：两侧归一化为去空白字符串后相等（payload 里 ``stockCode`` 是 int）。
- **简称**：相等；或较短一侧是较长一侧的**前缀**且长度 ≥ ``MIN_CONTAIN_NAME_LEN``
  （用**前缀**而非子串：``康缘`` → ``康缘药业`` 命中，``生物`` ✗ ``华兰生物`` 不命中）。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: 简称「前缀匹配」允许的最短长度；低于该长度只接受完全相等（防「药」这类单字通用词误伤）
MIN_CONTAIN_NAME_LEN: int = 2


def _text(value: Any) -> str:
    """任意标量 → 去空白字符串（None → ""）"""
    return "" if value is None else str(value).strip()


def resolve_names(query_name: Any, known_names: Optional[Iterable[str]] = None) -> List[str]:
    """把查询简称按**闭词表**归一成规范简称列表。

    Args:
        query_name: 查询/LLM 抽取出的股票简称
        known_names: 语料内的规范简称全集（闭词表）；为空则不归一

    Returns:
        归一后的简称列表；无法唯一归一（0 个或多个命中）时返回 ``[原字符串]``。
        输入为空时返回 ``[]``。
    """
    query = _text(query_name)
    if not query:
        return []
    vocab = {_text(n) for n in (known_names or []) if _text(n)}
    if not vocab:
        return [query]
    if query in vocab:
        return [query]
    hits = sorted(n for n in vocab if query in n)
    return [hits[0]] if len(hits) == 1 else [query]


def name_matches(payload_name: Any, query_name: Any) -> bool:
    """股票简称是否匹配（相等；或较短一侧是较长一侧的前缀且长度 ≥ MIN_CONTAIN_NAME_LEN）"""
    a, b = _text(payload_name), _text(query_name)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= MIN_CONTAIN_NAME_LEN and long_.startswith(short)


def code_matches(payload_code: Any, query_code: Any) -> bool:
    """股票代码是否匹配（归一化为去空白字符串后相等；payload 侧通常是 int）"""
    a, b = _text(payload_code), _text(query_code)
    return bool(a) and a == b


def entity_matches(
    payload: Dict[str, Any],
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    keep_unknown_entity: bool = True,
    known_names: Optional[Iterable[str]] = None,
) -> bool:
    """候选 payload 是否命中查询实体。

    Args:
        payload: Qdrant 点的 payload（读 ``stockName`` / ``stockCode``）
        stock_name: 查询解析出的股票简称（会先经 :func:`resolve_names` 归一）
        stock_code: 查询解析出的股票代码
        keep_unknown_entity: 候选自身**没有任何实体标注**（如行业研报）时是否保留（默认 True，与 ``apply_entity_gate`` 一致）
        known_names: 语料规范简称闭词表（用于简称归一）

    Returns:
        是否保留该候选；未提供任何查询实体时一律返回 True（闸门不生效）
    """
    if not _text(stock_name) and not _text(stock_code):
        return True
    payload = payload or {}
    payload_name, payload_code = _text(payload.get("stockName")), _text(payload.get("stockCode"))
    if not payload_name and not payload_code:
        return bool(keep_unknown_entity)
    return entity_hit(payload, stock_name, stock_code, known_names)


def entity_hit(
    payload: Dict[str, Any],
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    known_names: Optional[Iterable[str]] = None,
) -> bool:
    """候选是否**命中实体本体**（忽略 ``keep_unknown_entity``，用于闸门计数与离线指标）"""
    payload = payload or {}
    payload_name, payload_code = _text(payload.get("stockName")), _text(payload.get("stockCode"))
    if not payload_name and not payload_code:
        return False
    if code_matches(payload_code, stock_code):
        return True
    return any(name_matches(payload_name, r) for r in resolve_names(stock_name, known_names))


def apply_entity_gate(
    candidates: List[Dict[str, Any]],
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    min_keep: int = 0,
    keep_unknown_entity: bool = True,
    known_names: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """对候选做实体硬过滤（带安全闸门），保持输入顺序。

    Args:
        candidates: 检索结果列表，每项需含 ``payload`` 字段
        stock_name: 查询解析出的股票简称（可为空）
        stock_code: 查询解析出的股票代码（可为空）
        min_keep: 安全闸门阈值（默认 0 = 不启用），比较对象是**命中实体的候选数**；低于该值时**不过滤**
        keep_unknown_entity: 是否保留无实体标注的候选（默认 True，见模块 docstring）
        known_names: 语料规范简称闭词表

    Returns:
        ``(kept, meta)``。``meta`` 记录是否真正启用与原因，供日志/留痕：
        ``{"applied", "reason", "n_in", "n_kept", "n_matched", "min_keep"}``
    """
    items = list(candidates or [])
    meta: Dict[str, Any] = {
        "applied": False,
        "reason": "",
        "n_in": len(items),
        "n_kept": len(items),
        "n_matched": 0,
        "min_keep": int(min_keep),
    }
    if not items:
        meta["reason"] = "空候选"
        return items, meta
    if not _text(stock_name) and not _text(stock_code):
        meta["reason"] = "未解析出实体"
        return items, meta

    n_matched = sum(
        1 for c in items if entity_hit(c.get("payload") or {}, stock_name, stock_code, known_names)
    )
    meta["n_matched"] = n_matched
    if n_matched < max(0, int(min_keep)):
        meta["reason"] = "实体命中 %d < min_keep %d，跳过（疑多主体/行业题或池内无该公司）" % (n_matched, min_keep)
        return items, meta

    kept = [
        c for c in items
        if entity_matches(c.get("payload") or {}, stock_name, stock_code, keep_unknown_entity, known_names)
    ]
    meta["applied"] = True
    meta["reason"] = "实体命中 %d/%d → 保留 %d" % (n_matched, len(items), len(kept))
    meta["n_kept"] = len(kept)
    return kept, meta
