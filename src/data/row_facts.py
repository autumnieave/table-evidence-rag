# -*- coding: utf-8 -*-
"""行级自包含事实与 ``anchor_cells``（数据管道重构 · 阶段 2）。

设计边界
--------
- 一行数据可能包含多个「指标 × 期间」单元格；本模块按**数据单元格**生成事实，
  而不是把整行拼成一句（否则 ``营业收入 100 120 130`` 无法回答具体期间）。
- 事实文本格式：``{entity}｜{table_caption}｜{metric} > {period}: {value}{unit}``；
  例：``贵州茅台(600519)｜主要财务指标｜归母净利润 > 2024A: 3,367.89 百万元``。
- ``anchor_cells`` 的六项核心字段固定为
  ``source / row_index / col_index / metric / period / value``；
  另带 ``table_id / chunk_id / header_path / unit / unit_source / currency_raw``
  等定位辅助字段。
- 不在行级猜单位：单元格/表头 → caption/title → footnote → 表级单值单位；
  多来源冲突置 ``multi``，单位缺失保留 ``None``。表级回退显式标 ``table_fallback``；
  比率/估值指标（PE/PB/毛利率/ROE…）不继承金额单位。
- 只对**含数字的值**生成事实；纯文本管线/描述表不生成，避免把整张展示表搬进索引。
  该边界写入 ``docs/评估报告/行级事实与anchor_cells_20261017.md``。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .table_struct import (
    Table,
    _clean_cell,
    _is_footnote_row,
    detect_units,
    extract_report_period,
)

#: 空值 / 缺省值：不生成事实
_MISSING_VALUES = {"", "-", "--", "—", "–", "/", "不适用", "无", "暂无", "N/A", "NA", "na"}
#: 纯数字、千分位、百分数、括号负数、数值区间（``31.18/21.33``）
_NUM_TOKEN = r"[（(]?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[)）]?%?"
_NUMBER_RE = re.compile(r"^{}(?:\s*[/~\-–—]\s*{})?$".format(_NUM_TOKEN, _NUM_TOKEN))
_YEAR_RE = re.compile(r"^(20\d{2})\s*([AEae])?$")
_YEAR_CN_RE = re.compile(r"^(20\d{2})\s*年$")
_Q_RE = re.compile(r"^(20\d{2})\s*[Qq]\s*([1-4])$")
_Q_PRE_RE = re.compile(r"^[Qq]\s*([1-4])\s*(20\d{2})$")
_H_RE = re.compile(r"^(20\d{2})\s*[Hh]\s*([12])$")
_H_PRE_RE = re.compile(r"^([12])\s*[Hh]\s*(20\d{2}|\d{2})$")

#: 复合单位（表级 unit 不回退到这些；行级需要保留 ``元/股``、``%`` 等语义）
_COMPOUND_UNIT_RE = re.compile(
    r"(?:人民币|港币|港元|美元|RMB|CNY|HKD|USD)?\s*"
    r"(?:元|港元|美元)\s*/\s*(?:股|吨|千瓦时|千瓦|平方米|件|台)"
    r"|百分点|pp|倍"
)
_PERCENT_RE = re.compile(r"%|％")
_CURRENCY_RE = re.compile(r"人民币|港币|港元|美元|RMB|CNY|HKD|USD")

_METRIC_HEADERS = ("指标", "项目", "科目", "指标名称", "财务指标")
_ENTITY_HEADERS = ("公司", "证券简称", "股票简称", "标的", "行业", "公司名称", "名称")
_NON_VALUE_HEADERS = ("备注", "说明", "来源", "资料来源", "注", "序号", "排名")

#: 不应继承金额单位的指标（PE/PB/比率/每股类）。命中后除非单元格自带 ``%``/``倍``/``元/股``，
#: 否则不借用表级“万元/百万元”等单位，避免“市盈率 17.62 百万元”这类语义错配。
_RATIO_METRIC_TERMS = (
    "率", "占比", "PE", "P/E", "PB", "P/B", "EV/EBITDA", "PS", "P/S",
    "ROE", "ROA", "EPS", "每股", "倍", "同比", "环比", "增速", "增长率",
)

#: 行级数值事实的适用表类型；pipeline/other 以文本/日期/标签为主，不进入事实索引
FACT_TABLE_TYPES = ("financial_statement", "forecast", "valuation", "industry_summary", "ranking")


@dataclass(frozen=True)
class ColumnPlan:
    """一张表的列角色划分。"""

    metric_col: Optional[int]
    entity_col: Optional[int]
    period_cols: Tuple[int, ...]
    value_cols: Tuple[int, ...]
    period_by_col: Dict[int, str]


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    """大小写不敏感的任一词命中。"""
    upper = (text or "").upper()
    return any(term.upper() in upper for term in terms)


def _row_cells(table: Table, row_index: int) -> List[str]:
    """取一行并按表宽补齐，避免短行/合并单元格导致下标越界。"""
    width = max(1, len(table.headers), table.col_count)
    row = list(table.rows[row_index]) if 0 <= row_index < len(table.rows) else []
    row.extend([""] * (width - len(row)))
    return [_clean_cell(cell) for cell in row[:width]]


def _period_from_header(header: str, table_type: str = "") -> Optional[str]:
    """把列头归一为期间标签；识别不到返回 ``None``。"""
    text = _clean_cell(header)
    if not text:
        return None
    # 去掉单位括号，但保留期间本体：``2024A（万元）`` -> ``2024A``
    text = re.sub(r"[（(][^)）]*(?:元|%|％|股|吨)[^)）]*[)）]", "", text).strip()
    if not text:
        return None
    if any(term in text for term in ("同比", "环比", "占比", "增速", "变动", "增长率")):
        return None
    match = _YEAR_RE.match(text)
    if match:
        year, suffix = match.group(1), (match.group(2) or "").upper()
        if suffix:
            return year + suffix
        # 财务/预测/估值表的裸年份通常是实际期；行业汇总表保留裸年份，避免把发布年当报告期。
        return year + "A" if table_type in ("financial_statement", "forecast", "valuation") else year
    match = _YEAR_CN_RE.match(text)
    if match:
        return match.group(1) + "A"
    match = _Q_RE.match(text)
    if match:
        return "{}Q{}".format(match.group(1), match.group(2))
    match = _Q_PRE_RE.match(text)
    if match:
        return "{}Q{}".format(match.group(2), match.group(1))
    match = _H_RE.match(text)
    if match:
        return "{}H{}".format(match.group(1), match.group(2))
    match = _H_PRE_RE.match(text)
    if match:
        year = match.group(2)
        if len(year) == 2:
            year = "20" + year
        return "{}H{}".format(year, match.group(1))
    parsed = extract_report_period(text)
    if parsed:
        return parsed
    return None


def _period_type(period: str, table_type: str = "") -> str:
    """期间类型：actual / forecast / unknown（不根据数量级猜）。"""
    text = (period or "").upper()
    if "E" in text or "预测" in text:
        return "forecast"
    if "A" in text or re.search(r"20\d{2}[QH][1-4]", text):
        return "actual"
    if table_type == "financial_statement" and re.match(r"20\d{2}", text):
        return "actual"
    return "unknown"


def _is_numeric_value(value: str) -> bool:
    """值是否为数值/数值区间；纯文本（如「中药」「H3K27M」）不生成数值事实。"""
    return bool(_NUMBER_RE.match(_clean_cell(value)))


def _is_missing(value: str) -> bool:
    """空值 / 占位符。"""
    return _clean_cell(value).lower() in _MISSING_VALUES


def classify_columns(table: Table) -> ColumnPlan:
    """划分指标列、实体列、期间列与值列。"""
    headers = [_clean_cell(h) for h in table.headers]
    width = max(1, len(headers))

    period_by_col: Dict[int, str] = {}
    for col, header in enumerate(headers):
        period = _period_from_header(header, table.table_type)
        if period:
            period_by_col[col] = period

    metric_col: Optional[int] = None
    for col, header in enumerate(headers):
        if _contains_any(header, _METRIC_HEADERS):
            metric_col = col
            break

    entity_col: Optional[int] = None
    for col, header in enumerate(headers):
        if _contains_any(header, _ENTITY_HEADERS):
            entity_col = col
            break
    if entity_col is None and table.table_scope in ("industry", "mixed"):
        # 行业/多主体表：首列常是行业名、公司名或分区标签；若它不是指标列，则作为实体列。
        first = _clean_cell(headers[0]) if headers else ""
        if first and not _contains_any(first, _METRIC_HEADERS) and not _period_from_header(first, table.table_type):
            entity_col = 0

    if metric_col is None:
        # 有实体列（公司/行业）时，指标名来自列头（PE/PB/涨跌幅…）；
        # 没有实体列时才把首列当指标列（营业收入/归母净利润…）。
        metric_col = None if entity_col is not None else 0
    if entity_col == metric_col:
        entity_col = None

    value_cols: List[int] = []
    for col, header in enumerate(headers):
        if col == metric_col or col == entity_col:
            continue
        if _contains_any(header, _NON_VALUE_HEADERS):
            continue
        value_cols.append(col)

    # 2 列键值卡片：``键 | 值``；首列是键，第二列是值。
    if table.is_key_value:
        metric_col = 0
        entity_col = None
        value_cols = [1] if width > 1 else []

    return ColumnPlan(
        metric_col=metric_col,
        entity_col=entity_col,
        period_cols=tuple(sorted(period_by_col)),
        value_cols=tuple(value_cols),
        period_by_col=period_by_col,
    )


def _doc_entity(table: Table, metadata: Mapping[str, Any]) -> str:
    """文档级实体：股票名(代码) → 代码 → 文档标题 → 文件名。"""
    name = _clean_cell(str(metadata.get("stockName") or ""))
    code = _clean_cell(str(metadata.get("stockCode") or ""))
    if name and code:
        return "{}({})".format(name, code)
    if name:
        return name
    if code:
        return code
    title = _clean_cell(table.title or metadata.get("title") or "")
    return title or _clean_cell(table.source_file or "未知主体")


def _entity_for_row(
    table: Table,
    row: Sequence[str],
    plan: ColumnPlan,
    metadata: Mapping[str, Any],
) -> str:
    """行级实体：实体列优先，缺失时回退文档级实体。"""
    if plan.entity_col is not None and plan.entity_col < len(row):
        value = _clean_cell(row[plan.entity_col])
        if value and not _is_missing(value) and not _NUMBER_RE.match(value):
            return value
    return _doc_entity(table, metadata)


def _metric_for_cell(table: Table, row: Sequence[str], col: int, plan: ColumnPlan) -> str:
    """单元格对应的指标名（表头路径的最后一段）。"""
    if table.is_key_value:
        return _clean_cell(row[0]) if row else ""
    header = _clean_cell(table.headers[col]) if col < len(table.headers) else ""
    base = ""
    if plan.metric_col is not None and plan.metric_col != col and plan.metric_col < len(row):
        base = _clean_cell(row[plan.metric_col])
    if col in plan.period_by_col:
        return base or header
    if base and header:
        if header in base:
            return base
        # ``同比`` / ``PE`` 等附加列：拼成「营业收入同比」/「PE」。
        return "{}{}".format(base, header)
    return base or header


def _period_for_cell(table: Table, col: int, plan: ColumnPlan) -> str:
    """单元格期间：期间列优先，其次表/文档报告期，最后「当前」。"""
    if col in plan.period_by_col:
        return plan.period_by_col[col]
    if table.is_key_value:
        return table.report_period or "当前"
    if table.report_period:
        return table.report_period
    actual = [plan.period_by_col[c] for c in plan.period_cols
              if _period_type(plan.period_by_col[c], table.table_type) == "actual"]
    if actual:
        return actual[-1]
    if plan.period_by_col:
        return plan.period_by_col[plan.period_cols[-1]]
    return "当前"


def _compound_unit(text: str) -> Optional[str]:
    """识别 ``元/股`` / ``%`` / ``倍`` 等复合/比率单位。"""
    match = _COMPOUND_UNIT_RE.search(text or "")
    if match:
        return re.sub(r"\s+", "", match.group(0))
    if _PERCENT_RE.search(text or ""):
        return "%"
    return None


def _is_ratio_metric(metric: str) -> bool:
    """指标是否为比率/估值/每股类；这类指标不继承表级金额单位。"""
    upper = str(metric or "").upper()
    return any(term.upper() in upper for term in _RATIO_METRIC_TERMS)


def _unit_for_cell(
    table: Table,
    row: Sequence[str],
    col: int,
    plan: ColumnPlan,
    metric: str,
    value: str,
) -> Tuple[Optional[str], str, Optional[str], str]:
    """解析单元格单位，返回 ``(unit, unit_source, currency_raw, unit_kind)``。"""
    header = _clean_cell(table.headers[col]) if col < len(table.headers) else ""
    metric_cell = _clean_cell(row[plan.metric_col]) if plan.metric_col is not None and plan.metric_col < len(row) else ""
    cell_text = " ".join(part for part in (metric_cell, header, metric, value) if part)
    caption_text = " ".join(part for part in (_clean_cell(table.caption), _clean_cell(table.title)) if part)
    footnote_text = " ".join(
        " ".join(_clean_cell(c) for c in r)
        for r in table.rows
        if _is_footnote_row(r)
    )

    # 1) 复合/比率单位优先：``元/股``、``%`` 不能被表级「万元」覆盖。
    compound = _compound_unit(cell_text)
    if compound:
        currency = _CURRENCY_RE.search(cell_text)
        kind = "percent" if compound == "%" else ("per_share" if "/股" in compound else "ratio")
        return compound, "cell", currency.group(0) if currency else None, kind

    # 比率/估值/每股类指标不继承表级金额单位：P/E 不是“百万元”。
    if _is_ratio_metric(metric):
        return None, "none", None, "ratio"

    # 2) 单元格/表头 → caption/title → footnote。
    sources = [
        ("cell", detect_units(cell_text, limit=10)),
        ("caption", detect_units(caption_text, limit=10)),
        ("footnote", detect_units(footnote_text, limit=10)),
    ]
    candidates: List[str] = []
    for _name, units in sources:
        for unit in units:
            if unit not in candidates:
                candidates.append(unit)
    if len(candidates) > 1:
        return None, "multi", None, "none"
    if len(candidates) == 1:
        unit = candidates[0]
        source = next((name for name, units in sources if unit in units), "none")
        currency = _CURRENCY_RE.search(cell_text + " " + caption_text + " " + footnote_text)
        return unit, source, currency.group(0) if currency else None, "currency"

    # 3) 表级单值单位回退（阶段 1 已解析；不猜数量级）。
    # 多单位表（unit_candidates > 1）禁止跨指标继承；回退来源单独标 table_fallback。
    if table.unit and len(table.unit_candidates) <= 1:
        currency = _CURRENCY_RE.search(caption_text + " " + footnote_text)
        return table.unit, "table_fallback", currency.group(0) if currency else None, "currency"
    return None, "none", None, "none"


def _format_value_unit(value: str, unit: Optional[str]) -> str:
    """值 + 单位；值本身已带 % 或单位时不重复追加。"""
    value = _clean_cell(value)
    if not unit:
        return value
    if value.endswith(unit) or (unit == "%" and _PERCENT_RE.search(value)):
        return value
    return "{} {}".format(value, unit)


def build_row_facts(
    table: Table,
    metadata: Optional[Mapping[str, Any]] = None,
    row_start: int = 0,
    row_end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """生成一张表在 ``[row_start, row_end)`` 内的行级事实。

    Args:
        table: 结构化表格
        metadata: 文档级元数据（``stockName`` / ``stockCode`` / ``title``）
        row_start: 起始行下标（含）
        row_end: 结束行下标（不含）；``None`` = 到表尾

    Returns:
        ``[{"text", "anchor", "entity", "metric", "period", "value", ...}]``；
        分节标题行、次级表头、脚注行不生成事实。
    """
    metadata = metadata or {}
    if table.table_type not in FACT_TABLE_TYPES:
        return []
    plan = classify_columns(table)
    end = len(table.rows) if row_end is None else min(row_end, len(table.rows))
    start = max(0, row_start)
    skip_rows = set(table.section_rows) | set(table.sub_header_rows)

    caption = _clean_cell(table.caption) or _clean_cell(table.title) or "表格"
    facts: List[Dict[str, Any]] = []
    seen: set = set()

    for row_index in range(start, end):
        if row_index in skip_rows:
            continue
        row = _row_cells(table, row_index)
        if not any(row) or _is_footnote_row(row):
            continue
        entity = _entity_for_row(table, row, plan, metadata)
        for col in plan.value_cols:
            if col >= len(row):
                continue
            value = _clean_cell(row[col])
            if _is_missing(value) or not _is_numeric_value(value):
                continue
            metric_row = row[plan.metric_col] if plan.metric_col is not None and plan.metric_col < len(row) else ""
            # 日期/年份行的数字不是指标值；避免把「报告期 2024」生成成事实。
            if re.fullmatch(r"20\d{2}", value) and _contains_any(metric_row, ("日期", "时间", "年份", "报告期", "年度")):
                continue
            metric = _metric_for_cell(table, row, col, plan)
            if not metric:
                continue
            period = _period_for_cell(table, col, plan)
            unit, unit_source, currency_raw, unit_kind = _unit_for_cell(
                table, row, col, plan, metric, value
            )
            formatted = _format_value_unit(value, unit)
            text = "{}｜{}｜{} > {}: {}".format(entity, caption, metric, period, formatted)
            dedup_key = (entity, metric, period, value, unit)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            anchor = {
                "source": table.table_id,
                "row_index": row_index,
                "col_index": col,
                "metric": metric,
                "period": period,
                "value": value,
                "table_id": table.table_id,
                "source_file": table.source_file,
                "chunk_id": "{}#r{}c{}".format(table.table_id, row_index, col),
                "header_path": [metric, period],
                "unit": unit,
                "unit_source": unit_source,
                "currency_raw": currency_raw,
                "unit_kind": unit_kind,
                "period_type": _period_type(period, table.table_type),
                "entity": entity,
                "table_caption": caption,
                "table_type": table.table_type,
                "table_scope": table.table_scope,
            }
            facts.append({
                "text": text,
                "anchor": anchor,
                "entity": entity,
                "metric": metric,
                "period": period,
                "value": value,
                "unit": unit,
                "unit_source": unit_source,
                "currency_raw": currency_raw,
                "unit_kind": unit_kind,
                "period_type": anchor["period_type"],
                "row_index": row_index,
                "col_index": col,
                "table_id": table.table_id,
                "chunk_id": anchor["chunk_id"],
                "table_caption": caption,
            })
    return facts


def build_anchor_cells(
    table: Table,
    metadata: Optional[Mapping[str, Any]] = None,
    row_start: int = 0,
    row_end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """只取 anchor（不返回事实文本），供整表父块 payload 使用。"""
    return [fact["anchor"] for fact in build_row_facts(table, metadata, row_start, row_end)]
