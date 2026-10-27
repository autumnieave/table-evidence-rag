# -*- coding: utf-8 -*-
"""查询理解：实体 / 指标 / 期间抽取 + 缺实体澄清判定。

边界
----
- 纯规则，不调用 LLM、不访问 Qdrant；
- 公司词表由调用方注入（本项目用语料内公司名），模块内不硬编码公司；
- 指标 / 期间复用 ``filters.row_match`` 的同一口径；
- 行业级 / 板块级问题不强制澄清；只有“公司级指代、市场级问题、有指标但无实体”
  才要求补实体，避免把所有开放式问题都判成澄清。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from filters.row_match import extract_query_constraints

#: 6 位股票代码；前后不能紧邻数字，避免命中财务数字。
_ENTITY_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

#: 明确指代“前文某个主体”的词；没有实体时这些词是澄清信号。
_DEICTIC_TERMS: Tuple[str, ...] = (
    "这家公司", "该公司", "该股", "这只股票", "这个标的",
    "某公司", "某企业", "两家企业", "这些企业", "这些公司",
    "上述公司", "上述企业", "这些数据", "该数据", "涨跌",
)

#: 市场 / 指数级范围。它们不是公司实体，但需要用户补“公司/指数/行业”。
_MARKET_TERMS: Tuple[str, ...] = (
    "A股", "大A股", "大盘", "上证指数", "深证成指", "创业板指",
    "北向资金", "外资", "股市",
)

#: 行业 / 板块级范围。行业级问题允许没有单一公司实体。
_INDUSTRY_TERMS: Tuple[str, ...] = (
    "中药材", "中药", "医药", "创新药", "疫苗", "医疗器械", "CXO",
    "行业", "板块", "赛道", "产业链", "药企", "中药企业", "龙头企业", "OTC",
)

#: 语料内常见简称 → 标准公司名；只放确定的一对一别名。
_ENTITY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("999", "华润三九"),
)

_CLARIFY_ENTITY = (
    "请补充要分析的公司（名称或代码）、指数或行业范围，例如“云南白药”“000538”"
    "或“中药行业”；当前问题无法定位到具体检索单元。"
)


@dataclass(frozen=True)
class QueryUnderstanding:
    """一次查询理解的结果。"""

    query: str
    entities: Tuple[str, ...] = ()
    metrics: Tuple[str, ...] = ()
    period: Optional[str] = None
    scope: str = "unknown"  # company / industry / market / unknown
    deictic_terms: Tuple[str, ...] = ()
    missing_fields: Tuple[str, ...] = ()
    clarify_question: Optional[str] = None

    @property
    def needs_clarification(self) -> bool:
        """是否应先向用户澄清，而不是扩大召回。"""
        return bool(self.missing_fields)

    def as_dict(self) -> Dict[str, Any]:
        """转为可写入 trace / payload 的字典。"""
        return {
            "query": self.query,
            "entities": list(self.entities),
            "metrics": list(self.metrics),
            "period": self.period,
            "scope": self.scope,
            "deictic_terms": list(self.deictic_terms),
            "missing_fields": list(self.missing_fields),
            "clarify_question": self.clarify_question,
            "needs_clarification": self.needs_clarification,
        }


def _normalize_vocab(known_entities: Optional[Any]) -> Tuple[str, ...]:
    """把词表统一成“按长度倒序”的元组；dict 取其 keys。"""
    if not known_entities:
        return ()
    values: Iterable[Any]
    if isinstance(known_entities, Mapping):
        values = known_entities.keys()
    else:
        values = known_entities
    cleaned = {
        str(value).strip()
        for value in values
        if str(value or "").strip() and len(str(value).strip()) >= 2
    }
    return tuple(sorted(cleaned, key=lambda value: (-len(value), value)))


def extract_query_entities(
    query: str,
    known_entities: Optional[Any] = None,
) -> Tuple[str, ...]:
    """从问题中抽公司名 / 6 位代码，按最长匹配去重叠。

    Args:
        query: 用户问题（可含上下文拼接文本）
        known_entities: 公司名词表（dict 或 sequence）；为空时只抽 6 位代码

    Returns:
        按出现顺序去重的实体列表。
    """
    text = str(query or "")
    if not text:
        return ()
    found: list[Tuple[int, str]] = []
    spans: list[Tuple[int, int]] = []

    def _free(start: int, end: int) -> bool:
        return all(end <= left or start >= right for left, right in spans)

    for name in _normalize_vocab(known_entities):
        start = 0
        while True:
            idx = text.find(name, start)
            if idx < 0:
                break
            end = idx + len(name)
            if _free(idx, end):
                found.append((idx, name))
                spans.append((idx, end))
            start = idx + 1

    for alias, canonical in _ENTITY_ALIASES:
        idx = text.find(alias)
        if idx >= 0 and _free(idx, idx + len(alias)):
            found.append((idx, canonical))
            spans.append((idx, idx + len(alias)))

    for match in _ENTITY_CODE_RE.finditer(text):
        if _free(match.start(), match.end()):
            found.append((match.start(), match.group(1)))
            spans.append((match.start(), match.end()))

    found.sort(key=lambda item: item[0])
    out: list[str] = []
    for _, entity in found:
        if entity not in out:
            out.append(entity)
    return tuple(out)


def detect_scope(query: str, entities: Sequence[str] = ()) -> str:
    """判断问题范围：company / industry / market / unknown。"""
    if entities:
        return "company"
    text = str(query or "")
    if any(term in text for term in _MARKET_TERMS):
        return "market"
    if any(term in text for term in _INDUSTRY_TERMS):
        return "industry"
    return "unknown"


def detect_deictic_terms(query: str) -> Tuple[str, ...]:
    """抽取指代词；用于判断“前文主体未带过来”的追问。"""
    text = str(query or "")
    return tuple(term for term in _DEICTIC_TERMS if term in text)


def understand_query(
    query: str,
    *,
    known_entities: Optional[Any] = None,
    context_queries: Optional[Sequence[str]] = None,
) -> QueryUnderstanding:
    """查询理解入口。

    Args:
        query: 当前用户问题
        known_entities: 公司名词表
        context_queries: 同一会话中更早的子问题；只用于实体/指标/期间补全，
            不改变当前问题的展示文本

    Returns:
        ``QueryUnderstanding``；``missing_fields=("entity",)`` 表示应先澄清。
    """
    context = " ".join(str(item or "") for item in (context_queries or []))
    combined = (str(query or "") + " " + context).strip()
    entities = extract_query_entities(combined, known_entities)
    constraints = extract_query_constraints(combined)
    scope = detect_scope(query, entities)
    deictic = detect_deictic_terms(query)

    missing: list[str] = []
    # 行业级问题允许没有单一公司；市场级 / 指代 / 有指标无实体才要求补实体。
    if not entities and (
        deictic or scope == "market" or (scope != "industry" and constraints.metrics)
    ):
        missing.append("entity")

    return QueryUnderstanding(
        query=str(query or ""),
        entities=entities,
        metrics=constraints.metrics,
        period=constraints.period,
        scope=scope,
        deictic_terms=deictic,
        missing_fields=tuple(missing),
        clarify_question=_CLARIFY_ENTITY if missing else None,
    )


def with_entities(
    understanding: QueryUnderstanding,
    entities: Sequence[str],
) -> QueryUnderstanding:
    """补入外部已解析的实体（例如 Router 传入的会话主体）。"""
    values = tuple(str(item).strip() for item in entities if str(item or "").strip())
    if not values:
        return understanding
    merged: list[str] = []
    for value in values + list(understanding.entities):
        if value not in merged:
            merged.append(value)
    return replace(
        understanding,
        entities=tuple(merged),
        scope="company" if merged else understanding.scope,
        missing_fields=() if merged else understanding.missing_fields,
        clarify_question=None if merged else understanding.clarify_question,
    )
