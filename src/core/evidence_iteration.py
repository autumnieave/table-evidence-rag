# -*- coding: utf-8 -*-
"""Agent 迭代证据定位：检查 table schema / value mismatch 后给出下一步动作。

本模块只做**决策**，不直接查 Qdrant、不调 LLM、不执行 SQL：

- ``clarify_query``：问题缺少可定位的公司/指数/行业实体，先向用户澄清，不扩大召回；
- ``retry_entity``：候选表未覆盖点名主体时，按主体定向重查（不扩大全局召回）；
- ``reject_with_reason``：定向重查后仍无点名主体时拒答；
- ``retry_table``：当前候选表没有查询指标，扩大表级召回再试一轮；
- ``route_to_sql``：表里有指标 schema，但没有匹配期间/值（典型是研报表缺季度实际值）；
- ``reject_with_reason``：重试后仍没有可用表，明确拒答并保留原因。

动作只覆盖“有 mismatch 时怎么办”；完全匹配时返回 ``accept``，不进入上述三分支。
这样调用方可以先用多阶段检索省下的时间做 1–2 轮迭代，而不会把“没有约束”误判成失败。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from filters.query_understanding import QueryUnderstanding, understand_query
from filters.entity_gate import code_matches, entity_hit
from filters.row_match import (
    QueryConstraints,
    extract_query_constraints,
    metric_matches,
    period_matches,
)

ACCEPT: str = "accept"
CLARIFY_QUERY: str = "clarify_query"
RETRY_TABLE: str = "retry_table"
RETRY_ENTITY: str = "retry_entity"
ROUTE_TO_SQL: str = "route_to_sql"
REJECT_WITH_REASON: str = "reject_with_reason"

MISMATCH_NONE: str = "none"
MISMATCH_QUERY: str = "query_under_specified"
MISMATCH_ENTITY: str = "entity_mismatch"
MISMATCH_SCHEMA: str = "schema_mismatch"
MISMATCH_VALUE: str = "value_mismatch"
MISMATCH_EMPTY: str = "empty_context"


@dataclass(frozen=True)
class EvidenceDecision:
    """一次证据审计的结论。"""

    action: str
    mismatch_type: str
    reason: str
    round_no: int
    metrics: Tuple[str, ...] = ()
    period: Optional[str] = None
    table_count: int = 0
    schema_tables: int = 0
    value_tables: int = 0
    next_coarse_k: Optional[int] = None
    next_row_k: Optional[int] = None
    entities: Tuple[str, ...] = ()
    scope: str = ""
    missing_fields: Tuple[str, ...] = ()
    missing_entities: Tuple[str, ...] = ()
    clarify_question: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """转为可写入 trace / payload 的字典。"""
        return asdict(self)


def _dedup(values: Sequence[str]) -> List[str]:
    """保持顺序去重。"""
    out: List[str] = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in out:
            out.append(value)
    return out


def hit_covers_entity(
    hit: Dict[str, Any],
    entity: str,
    known_entities: Optional[Sequence[str]] = None,
) -> bool:
    """候选表是否覆盖点名实体（stockName / stockCode）。"""
    payload = (hit or {}).get("payload") or {}
    if entity_hit(payload, stock_name=entity, known_names=known_entities):
        return True
    # payload 的 stockCode 是 int，前导零会丢；6 位查询代码按去零形式再比一次。
    if entity.isdigit() and len(entity) == 6:
        return code_matches(payload.get("stockCode"), entity.lstrip("0") or "0")
    return False


def uncovered_entities(
    entities: Sequence[str],
    hits: Sequence[Dict[str, Any]],
    known_entities: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """返回没有被任何候选表覆盖的点名实体（保持查询顺序）。"""
    missing: List[str] = []
    for entity in entities:
        entity = str(entity or "").strip()
        if not entity:
            continue
        if not any(hit_covers_entity(hit, entity, known_entities) for hit in hits):
            missing.append(entity)
    return tuple(missing)


def table_metrics(payload: Dict[str, Any]) -> List[str]:
    """读取表 payload 中的指标 schema（metric_names / metric_name / anchors）。"""
    payload = payload or {}
    names: List[str] = []
    for value in payload.get("metric_names") or []:
        names.append(str(value))
    metric_name = str(payload.get("metric_name") or "")
    if metric_name:
        names.extend(part.strip() for part in metric_name.split(","))
    for anchor in payload.get("anchor_cells") or []:
        if isinstance(anchor, dict):
            names.append(str(anchor.get("metric") or ""))
    return _dedup(names)


def table_periods(payload: Dict[str, Any]) -> List[str]:
    """读取表 payload 中的期间值（report_period / period / anchors）。"""
    payload = payload or {}
    periods: List[str] = []
    for key in ("report_period", "period"):
        if payload.get(key):
            periods.append(str(payload[key]))
    for anchor in payload.get("anchor_cells") or []:
        if isinstance(anchor, dict) and anchor.get("period"):
            periods.append(str(anchor["period"]))
    return _dedup(periods)


def table_supports_constraints(payload: Dict[str, Any], constraints: QueryConstraints) -> Tuple[bool, bool]:
    """判断表是否满足指标 schema 与期间值约束。

    Returns:
        ``(metric_ok, period_ok)``；无对应约束时该维度为 True。
    """
    metrics = table_metrics(payload)
    periods = table_periods(payload)
    metric_ok = not constraints.metrics or any(
        metric_matches(metric, query_metric)
        for metric in metrics
        for query_metric in constraints.metrics
    )
    period_ok = not constraints.period or any(
        period_matches(period, constraints.period) for period in periods
    )
    return metric_ok, period_ok


def _finalize(
    decision: EvidenceDecision,
    understanding: Optional[QueryUnderstanding],
) -> EvidenceDecision:
    """把查询理解结果并入决策（不改动作本身）。"""
    if understanding is None:
        return decision
    return replace(
        decision,
        entities=understanding.entities,
        scope=understanding.scope,
    )


def audit_evidence(
    query: str,
    hits: Sequence[Dict[str, Any]],
    *,
    round_no: int = 0,
    max_rounds: int = 2,
    coarse_k: int = 100,
    row_k: int = 200,
    known_entities: Optional[Sequence[str]] = None,
    context_queries: Optional[Sequence[str]] = None,
    entity_retry_used: bool = False,
) -> EvidenceDecision:
    """检查候选表与查询约束的 schema/value mismatch，给出下一步动作。

    Args:
        query: 用户问题
        hits: 多阶段检索返回的整表结果（每项含 ``payload``）
        round_no: 当前轮次（0 基）
        max_rounds: 允许的最大重试轮数；超过后不再 retry
        coarse_k: 当前表级召回深度（用于计算下一轮）
        row_k: 当前行级召回深度（用于计算下一轮）
        known_entities: 公司名词表；提供时启用“缺实体 → clarify_query”与
            “候选表未覆盖点名主体 → retry_entity/reject”
        context_queries: 同一会话更早的子问题，用于补全实体/指标/期间
        entity_retry_used: 本轮之前是否已经做过“按主体定向重查”；做过仍缺实体则拒答

    Returns:
        ``EvidenceDecision``；没有 mismatch 时 action=``accept``。
    """
    understanding: Optional[QueryUnderstanding] = None
    if known_entities:
        understanding = understand_query(
            query,
            known_entities=known_entities,
            context_queries=context_queries,
        )
    constraints = extract_query_constraints(
        " ".join([query, *(context_queries or [])]).strip()
    )
    tables = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if understanding is not None and understanding.needs_clarification:
        return EvidenceDecision(
            action=CLARIFY_QUERY,
            mismatch_type=MISMATCH_QUERY,
            reason="问题缺少可定位的公司/指数/行业实体；先澄清，不扩大召回",
            round_no=round_no,
            metrics=constraints.metrics,
            period=constraints.period,
            table_count=len(tables),
            entities=understanding.entities,
            scope=understanding.scope,
            missing_fields=understanding.missing_fields,
            clarify_question=understanding.clarify_question,
        )
    if not tables:
        action = RETRY_TABLE if round_no < max_rounds else REJECT_WITH_REASON
        return _finalize(EvidenceDecision(
            action=action,
            mismatch_type=MISMATCH_EMPTY,
            reason="多阶段检索没有返回任何整表候选",
            round_no=round_no,
            metrics=constraints.metrics,
            period=constraints.period,
            next_coarse_k=min(max(coarse_k * 2, coarse_k + 50), 400) if action == RETRY_TABLE else None,
            next_row_k=min(max(row_k * 2, row_k + 50), 400) if action == RETRY_TABLE else None,
        ), understanding)

    if understanding is not None and understanding.entities:
        missing_entities = uncovered_entities(
            understanding.entities, tables, known_entities
        )
        if missing_entities:
            if round_no < max_rounds and not entity_retry_used:
                return _finalize(EvidenceDecision(
                    action=RETRY_ENTITY,
                    mismatch_type=MISMATCH_ENTITY,
                    reason="候选表未覆盖点名主体：%s；按主体定向重查" % "、".join(missing_entities),
                    round_no=round_no,
                    table_count=len(tables),
                    missing_entities=missing_entities,
                ), understanding)
            return _finalize(EvidenceDecision(
                action=REJECT_WITH_REASON,
                mismatch_type=MISMATCH_ENTITY,
                reason="按主体定向重查后仍无点名主体：%s；资料未覆盖，应澄清/拒答或转 SQL"
                % "、".join(missing_entities),
                round_no=round_no,
                table_count=len(tables),
                missing_entities=missing_entities,
            ), understanding)

    if not constraints.metrics and not constraints.period:
        return _finalize(EvidenceDecision(
            action=ACCEPT,
            mismatch_type=MISMATCH_NONE,
            reason="问题没有可确定的指标/期间约束，不做硬判",
            round_no=round_no,
            table_count=len(tables),
        ), understanding)

    schema_tables = 0
    value_tables = 0
    for hit in tables:
        payload = hit.get("payload") or {}
        metric_ok, period_ok = table_supports_constraints(payload, constraints)
        if metric_ok:
            schema_tables += 1
            if period_ok:
                value_tables += 1

    missing_metric = bool(constraints.metrics) and schema_tables == 0
    missing_value = not missing_metric and bool(constraints.period) and value_tables == 0
    if missing_metric:
        if round_no < max_rounds:
            return _finalize(EvidenceDecision(
                action=RETRY_TABLE,
                mismatch_type=MISMATCH_SCHEMA,
                reason="候选表没有查询指标的 schema；先扩大表级召回",
                round_no=round_no,
                metrics=constraints.metrics,
                period=constraints.period,
                table_count=len(tables),
                schema_tables=schema_tables,
                value_tables=value_tables,
                next_coarse_k=min(max(coarse_k * 2, coarse_k + 50), 400),
                next_row_k=min(max(row_k * 2, row_k + 50), 400),
            ), understanding)
        return _finalize(EvidenceDecision(
            action=ROUTE_TO_SQL,
            mismatch_type=MISMATCH_SCHEMA,
            reason="扩大表级召回后仍无查询指标 schema；转 SQL/财报库核验",
            round_no=round_no,
            metrics=constraints.metrics,
            period=constraints.period,
            table_count=len(tables),
            schema_tables=schema_tables,
            value_tables=value_tables,
        ), understanding)
    if missing_value:
        return _finalize(EvidenceDecision(
            action=ROUTE_TO_SQL,
            mismatch_type=MISMATCH_VALUE,
            reason="表里有指标 schema，但没有匹配期间/值；季度实际值应转 SQL/财报库",
            round_no=round_no,
            metrics=constraints.metrics,
            period=constraints.period,
            table_count=len(tables),
            schema_tables=schema_tables,
            value_tables=value_tables,
        ), understanding)
    return _finalize(EvidenceDecision(
        action=ACCEPT,
        mismatch_type=MISMATCH_NONE,
        reason="至少一张候选表同时满足指标与期间约束",
        round_no=round_no,
        metrics=constraints.metrics,
        period=constraints.period,
        table_count=len(tables),
        schema_tables=schema_tables,
        value_tables=value_tables,
    ), understanding)
