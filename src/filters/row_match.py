# -*- coding: utf-8 -*-
"""行级事实的查询约束与硬匹配（阶段 3 修复）。

本模块只做三件事：

1. 从问题里抽**可确定的**期间（``2025Q3`` / ``2025H1`` / ``2024A``）；
2. 从问题里抽**指标家族**（营收 / 研发费用 / 投资活动现金流 / 利润总额……），
   统一别名后再匹配；
3. 判断一条 ``row_fact`` 的 ``metric`` / ``period`` 是否满足约束。

边界：
- 只做规则，不调用 LLM；
- 抽不到指标或期间时返回空约束，调用方**不过滤**（不猜）；
- 指标别名只做“同义/缩写”归一，不做语义扩张；
- 期间匹配是精确匹配，``2025Q3`` 不会匹配 ``2025A`` / ``2025E``。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from data.table_struct import extract_report_period_strict

from .row_gate import METRIC_TERMS

#: 这些词是修饰语，不作为独立指标；否则“营收同比”会被拆成“营收 + 同比”两条。
_IGNORED_METRIC_TERMS = {"同比", "环比", "增速", "增长率", "占比"}

#: 查询侧指标词表：B-59 A 的指标表 + 现金流/估值/同比口径的常用写法。
#: 按长度倒序，保证“投资活动现金流量净额”优先于“现金流”。
_QUERY_METRIC_TERMS: Tuple[str, ...] = tuple(sorted(
    (set(METRIC_TERMS) | {
        "营业总收入", "营收", "归母净利", "扣非净利", "利润总额",
        "投资性现金流量净额", "投资活动产生的现金流量净额", "投资活动现金流量净额",
        "投资性现金流", "投资活动现金流", "投资现金流",
        "经营活动产生的现金流量净额", "经营活动现金流量净额", "经营活动现金流",
        "筹资活动产生的现金流量净额", "筹资活动现金流量净额", "筹资活动现金流",
        "研发费用占比", "研发费用率", "研发投入占比", "销售毛利率", "销售净利率",
        "净资产收益率", "每股收益", "每股净资产", "市盈率", "市净率",
        "EV/EBITDA", "P/E", "P/B", "ROE", "ROA", "EPS", "PE", "PB",
    }) - _IGNORED_METRIC_TERMS,
    key=lambda value: (-len(value), value),
))

#: 指标别名（按顺序替换；长的/具体的放前面）。
_METRIC_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("投资活动产生的", "投资活动"),
    ("投资活动产生", "投资活动"),
    ("投资性", "投资活动"),
    ("现金流净额", "现金流量净额"),
    ("营业总收入", "营业收入"),
    ("营收", "营业收入"),
    ("归母净利", "归母净利润"),
    ("扣非净利", "扣非净利润"),
    ("研发费用占比", "研发费用"),
    ("研发费用率", "研发费用"),
    ("研发投入占比", "研发投入"),
    ("销售毛利率", "毛利率"),
    ("销售净利率", "净利率"),
    ("净资产收益率", "ROE"),
    ("每股收益", "EPS"),
    ("每股净资产", "每股净资产"),
    ("市盈率", "PE"),
    ("市净率", "PB"),
    ("同比", ""),
    ("环比", ""),
    ("增速", ""),
    ("增长率", ""),
    ("占比", ""),
)

_PUNCT_RE = re.compile(r"[\s\u3000（）()【】\[\]，,。．.？?：:；;、/\\%％\-_]+")


@dataclass(frozen=True)
class QueryConstraints:
    """从问题里抽出的硬约束；空字段 = 该维度不过滤。"""

    metrics: Tuple[str, ...] = ()
    period: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """转为可落盘的字典。"""
        return {"metrics": list(self.metrics), "period": self.period}


def _normalize_metric(text: str) -> str:
    """指标名归一：去标点、统一别名、去修饰语。"""
    value = _PUNCT_RE.sub("", str(text or "").upper())
    for old, new in _METRIC_ALIASES:
        value = value.replace(old, new)
    value = value.replace("的", "")
    # ``市盈率(P/E)`` 先被去标点成 ``市盈率PE``，再替换成 ``PEPE``；折叠重复 token。
    for duplicated, token in (("PEPE", "PE"), ("PBPB", "PB"), ("PSPS", "PS"),
                              ("EPSEPS", "EPS"), ("ROEROE", "ROE"), ("ROAROA", "ROA")):
        value = value.replace(duplicated, token)
    return value


def _metric_matches(row_metric: str, query_metric: str) -> bool:
    """归一后的指标是否匹配（允许较长的具体指标包含较短指标）。"""
    left = _normalize_metric(row_metric)
    right = _normalize_metric(query_metric)
    if not left or not right:
        return False
    if left == right:
        return True
    # 允许 ``总市值`` / ``EPS`` 这类短指标被“总市值（百万元）”/“EPS 元/股”包含；
    # 2 字符英文缩写（PE/PB）只接受精确匹配，避免误命中。
    if len(left) >= 3 and len(right) >= 3:
        return left in right or right in left
    return False


def metric_matches(row_metric: str, query_metric: str) -> bool:
    """公开的指标匹配入口（供表级 schema 审计复用同一别名口径）。

    Args:
        row_metric: 表/行上的指标名
        query_metric: 查询侧指标名

    Returns:
        归一后是否匹配
    """
    return _metric_matches(row_metric, query_metric)


def extract_query_metrics(query: str) -> Tuple[str, ...]:
    """从问题中抽指标家族；返回**归一后**、去掉上位词重复的列表。"""
    upper = str(query or "").upper()
    selected = []
    for term in _QUERY_METRIC_TERMS:
        if term.upper() not in upper:
            continue
        normalized = _normalize_metric(term)
        if normalized and normalized not in selected:
            selected.append(normalized)
    # “投资活动现金流”命中时，不再保留更短的“现金流”；避免宽词把别的指标放进来。
    return tuple(
        item for item in selected
        if not any(item != other and item in other for other in selected)
    )


def _normalize_period(value: Any) -> Optional[str]:
    """期间归一为 ``2025Q3`` / ``2025H1`` / ``2025A``；识别不到返回原串（大写、去空格）。"""
    text = str(value or "").strip()
    if not text:
        return None
    compact = re.sub(r"\s+", "", text).upper()
    parsed = extract_report_period_strict(compact)
    if parsed:
        return parsed
    return compact or None


def period_matches(row_period: Any, query_period: Any) -> bool:
    """公开的期间精确匹配入口（表级 schema 审计与行级硬匹配同口径）。

    Args:
        row_period: 表/行上的期间
        query_period: 查询侧期间

    Returns:
        归一后是否完全相等；任一为空返回 False
    """
    left = _normalize_period(row_period)
    right = _normalize_period(query_period)
    return bool(left and right and left == right)


def extract_query_constraints(query: str) -> QueryConstraints:
    """从问题中抽 (指标, 期间) 约束。"""
    return QueryConstraints(
        metrics=extract_query_metrics(query),
        period=_normalize_period(extract_report_period_strict(str(query or ""))),
    )


def _payload_metric(payload: Dict[str, Any]) -> str:
    """取行事实指标：优先顶层 ``metric``，兼容旧索引的 ``anchor_cells[0].metric``。"""
    value = payload.get("metric") or payload.get("metric_name")
    if not value:
        anchors = payload.get("anchor_cells") or []
        if anchors and isinstance(anchors[0], dict):
            value = anchors[0].get("metric")
    return str(value or "")


def _payload_period(payload: Dict[str, Any]) -> str:
    """取行事实期间：优先顶层 ``period``，兼容旧索引的 ``anchor_cells[0].period``。"""
    value = payload.get("period")
    if not value:
        anchors = payload.get("anchor_cells") or []
        if anchors and isinstance(anchors[0], dict):
            value = anchors[0].get("period")
    return str(value or "")


def row_fact_matches(payload: Dict[str, Any], constraints: QueryConstraints) -> bool:
    """行事实是否满足查询硬约束。

    注意：期间约束存在时，**缺期间也判不匹配**——这是“硬匹配”，不是“缺字段放行”。
    """
    if constraints.period:
        row_period = _normalize_period(_payload_period(payload))
        if row_period != constraints.period:
            return False
    if constraints.metrics:
        row_metric = _payload_metric(payload)
        if not row_metric:
            return False
        if not any(_metric_matches(row_metric, metric) for metric in constraints.metrics):
            return False
    return True
