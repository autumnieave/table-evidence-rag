# -*- coding: utf-8 -*-
"""SQL 生成守卫：对 SQL 生成链路（原生 finance 链路 / 任何 ask 提供方）返回的 SQL
做静态校验 + MySQL 编译终审，失败自动带错误提示重问。

背景：SQL 生成 LLM 偶发产出坏 SQL（字段不存在 / 未定义别名 / 全角标点 / 混入非 SELECT 文本），
导致语句级编译通过率被拉低。本模块提供三层防线中的「校验 + 重问编排」：

- ``sql_errors(sql, schema, conn)``：返回校验错误列表（空 = 通过）；
- ``call_with_guard(ask, question, user_id, schema, conn, retries)``：带校验重问的
  调用编排，失败时把错误提示拼回问题再问一次，最终返回最后一次的
  ``(analysis, image_path, sql, 剩余错误)``。

设计：纯逻辑 + 依赖注入（``ask`` 为可调用对象），不直接依赖 MySQL/网络，
便于单测（tests/test_sql_guard.py，零外部依赖）。
"""

from __future__ import annotations

import logging
import re
import difflib
from typing import Callable, Dict, List, Optional, Tuple

from tools.sql_validator import compile_check, validate_sql

logger = logging.getLogger(__name__)

#: SQL 中的全角标点（MySQL 不识别，B2011 即因全角逗号分隔列而编译失败）
_FULLWIDTH_RE = re.compile(r"[，；：（）、【】]")
#: 静态校验的字段不存在错误（如 "字段 `x` 不存在于表 `y`（别名 t2）"）
_FIELD_ERR_RE = re.compile(r"字段 `([^`]+)` 不存在于表 `([^`]+)`")


def _field_suggestions(
    schema: Optional[Dict[str, Dict[str, str]]],
    errors: List[str],
    max_items: int = 8,
) -> List[str]:
    """从字段不存在错误中提取相近/可用的替代字段（供重问提示使用）。"""
    if not schema:
        return []
    out: List[str] = []
    for e in errors:
        m = _FIELD_ERR_RE.search(e)
        if not m:
            continue
        col, table = m.group(1), m.group(2)
        cols = schema.get(table)
        if not cols:
            continue
        candidates: List[str] = []
        if "yoy" in col.lower():
            candidates += [c for c in cols if "yoy" in c.lower()]
        candidates += difflib.get_close_matches(col, list(cols), n=3, cutoff=0.45)
        for c in dict.fromkeys(candidates):
            if c != col:
                out.append(f"{table}.{c}")
    return out[:max_items]


def _yoy_whitelist(schema: Optional[Dict[str, Dict[str, str]]]) -> Optional[str]:
    """生成全库 yoy 字段白名单提示（字段不存在错误含 yoy 时追加，效果强于模糊建议）。"""
    if not schema:
        return None
    lines: List[str] = []
    for table in ("income_sheet", "core_performance_indicators_sheet", "balance_sheet", "cash_flow_sheet"):
        cols = schema.get(table)
        if not cols:
            continue
        yoy = [c for c in cols if "yoy" in c.lower() or c.endswith("_growth")]
        if yoy:
            lines.append(f"{table}: {', '.join(yoy)}")
    if not lines:
        return None
    return (
        "同比/环比(yoy/qoq)字段仅存在以下白名单："
        + "；".join(lines)
        + "。营业成本/销售费用/管理费用/财务费用/营业总支出等科目无现成 yoy 字段，"
        "禁止编造任何 *_yoy_growth / *_yoy / *_growth 字段，需要时改用跨年原始值计算或说明不可直接查询"
    )


def sql_errors(
    sql: str,
    schema: Optional[Dict[str, Dict[str, str]]] = None,
    conn=None,
    max_hint: int = 120,
) -> List[str]:
    """校验生成的 SQL，返回错误列表（空 = 通过）。

    Args:
        sql: 生成链路提取出的 SQL（可为空或多条）
        schema: validate_sql 所需的字段映射（load_schema 产物）；None 时跳过静态校验
        conn: pymysql 连接；None 时跳过 MySQL 编译终审
        max_hint: 编译错误文案截断长度

    Returns:
        错误信息列表（去重、保序）
    """
    if not sql or not sql.strip():
        return []
    errors: List[str] = []
    if schema is not None:
        ok, serrs = validate_sql(sql, schema)
        if not ok:
            errors.extend(serrs)
    if _FULLWIDTH_RE.search(sql):
        errors.append("SQL 包含全角标点（如 ，；：（），必须使用半角 ASCII")
    if conn is not None:
        cerr = compile_check(conn, sql)
        if cerr:
            errors.append(f"MySQL 编译失败：{cerr[:max_hint]}")
    seen: set = set()
    out: List[str] = []
    for e in errors:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _with_hint(
    question: str,
    errors: List[str],
    schema: Optional[Dict[str, Dict[str, str]]] = None,
    max_hint: int = 600,
) -> str:
    """把校验错误拼回问题，作为重问提示。"""
    parts: List[str] = []
    if schema is not None and any("yoy" in e.lower() or "growth" in e.lower() for e in errors):
        whitelist = _yoy_whitelist(schema)
        if whitelist:
            parts.append(whitelist)
    parts.extend(errors)
    suggestions = _field_suggestions(schema, errors)
    if suggestions:
        parts.append("可用字段参考：" + "、".join(suggestions))
    hint = "；".join(parts)[:max_hint]
    return f"{question}（注意：上次生成的 SQL 有误：{hint}。请重新生成正确的 SQL，避免同类错误）"


def call_with_guard(
    ask: Callable[[str, str], Tuple[str, Optional[str], str]],
    question: str,
    user_id: str,
    schema: Optional[Dict[str, Dict[str, str]]] = None,
    conn=None,
    retries: int = 1,
) -> Tuple[str, Optional[str], str, List[str]]:
    """带校验重问地调用 SQL 生成提供方（ask 返回 analysis, image_path, sql）。

    Args:
        ask: 实际调用函数 ``ask(question, user_id) -> (analysis, image_path, sql)``
        question: 原始用户问题
        user_id: 会话用户标识
        schema: SQL 静态校验 schema（None = 跳过静态校验）
        conn: MySQL 连接（None = 跳过编译终审）
        retries: 校验失败后的重问次数（默认 1）

    Returns:
        ``(analysis, image_path, sql, 剩余错误)``；
        剩余错误仅当所有重试耗尽后仍校验失败时才非空（用于诚实上报/日志）。
    """
    analysis: str = ""
    img_path: Optional[str] = None
    sql_text: str = ""
    last_errors: List[str] = []
    for attempt in range(retries + 1):
        q = (
            _with_hint(question, last_errors, schema)
            if attempt > 0 and last_errors
            else question
        )
        analysis, img_path, sql_text = ask(q, user_id)
        if not sql_text or not sql_text.strip():
            last_errors = []
            break
        errs = sql_errors(sql_text, schema, conn)
        last_errors = errs
        if not errs:
            break
        logger.warning("SQL 校验未通过（第 %d/%d 次尝试）：%s",
                       attempt + 1, retries + 1, errs[:2])
    return analysis, img_path, sql_text, last_errors
