# -*- coding: utf-8 -*-
"""SQL 字段-表归属校验器（SQL Validator）。

在 SQL 真正执行前做静态校验，拦截 LLM 生成 SQL 的常见错误：
1. 表名不在 4 张白名单表内（如 JOIN stock_info）；
2. SELECT 引用了 FROM/JOIN 中未定义的别名（如 FROM 只定义 t1/t2，SELECT 却引用 t3）；
3. 字段不存在于其引用表（字段-表归属错误，如 net_profit 误挂 core_performance_indicators_sheet）；
4. 裸字段在多表 JOIN 下歧义（同名字段未加表前缀，如 total_operating_revenue 同时存在于
   income_sheet 与 core_performance_indicators_sheet）。

配合重试/回退机制使用：生成 SQL → validate_sql 静态校验 → compile_check 在 MySQL 上终审。

用法::

    import pymysql
    from tools.sql_validator import compile_check, load_schema, validate_sql

    conn = pymysql.connect(host="127.0.0.1", user="root", password="***",
                           database="financial_database", charset="utf8mb4")
    schema = load_schema(conn)
    ok, errors = validate_sql(sql_text, schema)
    if not ok:
        print("\\n".join(errors))
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pymysql
import sqlparse
from sqlparse.sql import Function, Identifier, IdentifierList, Parenthesis

#: 4 张白名单表（与原生 SQL 生成提示词一致）
WHITELIST_TABLES: Tuple[str, ...] = (
    "core_performance_indicators_sheet",
    "balance_sheet",
    "cash_flow_sheet",
    "income_sheet",
)

_FROM_KEYWORDS = ("FROM", "JOIN")
_STOP_KEYWORDS = (
    "WHERE", "ON", "USING", "GROUP", "HAVING", "ORDER", "LIMIT",
    "UNION", "INTO", "SET", "VALUES", "RETURNING", "FOR",
)

#: 匹配 `alias`.`col` / alias.col（反引号可选）
_QUALIFIED_RE = re.compile(
    r"^`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?$"
)
#: 匹配裸字段名 col / `col`
_BARE_RE = re.compile(r"^`?([A-Za-z_][A-Za-z0-9_]*)$")


def load_schema(conn) -> Dict[str, Dict[str, str]]:
    """从 MySQL 读取 4 张白名单表的字段映射。

    参数:
        conn: pymysql 连接对象（已连接 financial_database）

    返回:
        形如 ``{"income_sheet": {"net_profit": "decimal(10,4)", ...}, ...}`` 的字典。
    """
    schema: Dict[str, Dict[str, str]] = {}
    with conn.cursor() as cur:
        for table in WHITELIST_TABLES:
            cur.execute("SHOW COLUMNS FROM `%s`" % table)
            schema[table] = {row[0]: row[1] for row in cur.fetchall()}
    return schema


def _parse_select_item(ident: Identifier) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """解析 SELECT 列表项，返回 (别名, 列名)。

    支持 ``alias.col``、``col``、``col AS c``、``alias.col AS c``（反引号可选）。
    函数调用、字符串常量、星号等返回 None（跳过内容校验）。
    """
    if isinstance(ident, Function):
        return None
    value = ident.value.strip()
    if not value or value == "*" or value.endswith(".*"):
        return None
    if "(" in value or "'" in value or '"' in value:
        return None
    match_as = re.search(r"\s+AS\s+", value, flags=re.IGNORECASE)
    if match_as:
        value = value[: match_as.start()].strip()
    match_q = _QUALIFIED_RE.match(value)
    if match_q:
        return match_q.group(1), match_q.group(2)
    match_b = _BARE_RE.match(value)
    if match_b:
        return None, match_b.group(1)
    return None


def _parse_table_identifier(ident: Identifier) -> Optional[Tuple[str, str]]:
    """解析 FROM/JOIN 中的表标识，返回 (表名, 别名)；子查询返回 None。"""
    value = ident.value.strip()
    if not value or value.startswith("("):
        return None
    parts = re.split(r"\s+AS\s+|\s+", value, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return None
    table = parts[0].strip().strip("`")
    alias = parts[1].strip().strip("`") if len(parts) >= 2 else table
    return table, alias


def _tables_at_level(tokens) -> List[str]:
    """提取单层 token 序列中 FROM/JOIN 引用的表名（不递归进子查询内部）。"""
    tables: List[str] = []
    state: Optional[str] = None
    skip_next = False
    for token in tokens:
        if token.is_whitespace:
            continue
        upper = token.value.upper()
        if upper == "SELECT":
            state = "select"
            continue
        if upper in _FROM_KEYWORDS:
            state = "from"
            continue
        if upper in _STOP_KEYWORDS:
            state = None
            continue
        if state == "from":
            if isinstance(token, Parenthesis):
                skip_next = True  # 子查询，下一个 Identifier 是其别名，跳过
                continue
            if skip_next:
                skip_next = False
                continue
            if token.ttype in (sqlparse.tokens.Keyword,):
                state = None
                continue
            items = []
            if isinstance(token, IdentifierList):
                items = list(token.get_identifiers())
            elif isinstance(token, Identifier):
                items = [token]
            for item in items:
                parsed_table = _parse_table_identifier(item)
                if parsed_table:
                    tables.append(parsed_table[0])
    return tables


def _collect_subquery_tables(stmt) -> List[str]:
    """递归收集语句内所有子查询(Parenthesis)中引用的表名（含嵌套子查询）。

    用于补足校验器覆盖边界：FROM/JOIN 派生表、WHERE IN (...) 等子查询内部的
    表名此前未校验，B2035 即因在派生表中 JOIN 了库中不存在的 research_reports
    表而漏检（由 MySQL 编译兜底）。
    """
    result: List[str] = []
    stack = list(stmt.tokens)
    while stack:
        token = stack.pop()
        if isinstance(token, Parenthesis):
            result.extend(_tables_at_level(token.tokens))
            stack.extend(token.tokens)
        else:
            stack.extend(getattr(token, "tokens", []))
    return result


def parse_sql(sql: str) -> Optional[Dict[str, object]]:
    """提取 SQL 的表/别名映射与 SELECT 列列表。

    参数:
        sql: 单条 SELECT 语句

    返回:
        ``{"tables": {别名: 表名}, "select_cols": [(别名或None, 列名或None), ...]}``；
        解析失败返回 None。
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return None
    stmt = parsed[0]
    tables: Dict[str, str] = {}
    select_cols: List[Tuple[Optional[str], Optional[str]]] = []
    state: Optional[str] = None
    skip_next_from_item = False
    for token in stmt.tokens:
        if token.is_whitespace:
            continue
        upper = token.value.upper()
        if upper == "SELECT":
            state = "select"
            continue
        if upper in _FROM_KEYWORDS:
            state = "from"
            continue
        if upper in _STOP_KEYWORDS:
            state = None
            continue
        if upper in ("DISTINCT", "ALL"):
            continue
        if state == "select":
            items: List[Identifier] = []
            if isinstance(token, IdentifierList):
                items = list(token.get_identifiers())
            elif isinstance(token, Identifier):
                items = [token]
            for item in items:
                parsed_item = _parse_select_item(item)
                if parsed_item:
                    select_cols.append(parsed_item)
        elif state == "from":
            if token.ttype in (sqlparse.tokens.Keyword,):
                state = None  # WHERE/ORDER/ON 等关键字结束表列表
                continue
            if isinstance(token, Parenthesis):
                skip_next_from_item = True  # 子查询，下一个 Identifier 是其别名，跳过
                continue
            if skip_next_from_item:
                skip_next_from_item = False
                continue
            items = []
            if isinstance(token, IdentifierList):
                items = list(token.get_identifiers())
            elif isinstance(token, Identifier):
                items = [token]
            for item in items:
                parsed_table = _parse_table_identifier(item)
                if parsed_table:
                    table, alias = parsed_table
                    tables[alias] = table
    subquery_tables = _collect_subquery_tables(stmt)
    if not tables and not select_cols:
        return None
    return {"tables": tables, "select_cols": select_cols, "subquery_tables": subquery_tables}


def _validate_structure(
    structure: Dict[str, object], schema: Dict[str, Dict[str, str]]
) -> List[str]:
    """对解析出的表/列结构执行静态校验，返回错误列表。"""
    errors: List[str] = []
    tables: Dict[str, str] = structure["tables"]  # type: ignore[assignment]
    select_cols: List[Tuple[Optional[str], Optional[str]]] = structure["select_cols"]  # type: ignore[assignment]
    if not tables:
        return ["FROM 中未解析到任何表（请检查表名/JOIN 写法）"]
    subquery_tables: List[str] = structure.get("subquery_tables") or []  # type: ignore[assignment]
    for sub_table in dict.fromkeys(subquery_tables):
        if sub_table not in WHITELIST_TABLES:
            errors.append(
                f"子查询中使用了未定义表 `{sub_table}`"
                f"（白名单仅 4 张表：{', '.join(WHITELIST_TABLES)}）"
            )

    for alias, table in tables.items():
        if table not in WHITELIST_TABLES:
            errors.append(
                f"使用了未定义表 `{table}`（白名单仅 4 张表：{', '.join(WHITELIST_TABLES)}）"
            )
        elif table not in schema:
            errors.append(f"表 `{table}` 不在已加载的 schema 中（请确认 load_schema 已连接正确库）")
    defined_aliases = set(tables)
    for alias, col in select_cols:
        if alias is not None:
            if alias not in defined_aliases:
                errors.append(
                    f"SELECT 引用了未定义别名 `{alias}`"
                    f"（FROM/JOIN 已定义：{', '.join(sorted(defined_aliases)) or '无'}）"
                )
            elif col is not None:
                table = tables[alias]
                if table in schema and col not in schema[table]:
                    errors.append(f"字段 `{col}` 不存在于表 `{table}`（别名 {alias}）")
        elif col is not None:
            owners = [t for t in tables.values() if t in schema and col in schema[t]]
            if not owners:
                errors.append(f"字段 `{col}` 不存在于 FROM 中的任何表")
            elif len(owners) > 1:
                errors.append(
                    f"字段 `{col}` 在表（{', '.join(owners)}）中同时存在，需加表别名消除歧义"
                )
    return errors


def validate_sql(sql: str, schema: Dict[str, Dict[str, str]]) -> Tuple[bool, List[str]]:
    """静态校验 SQL 的表/字段/别名合法性（不依赖 MySQL 执行）。

    参数:
        sql: 一条或多条 SQL（多条按分号拆分，逐条校验）
        schema: load_schema() 返回的字段映射

    返回:
        ``(是否全部通过, 错误信息列表)``
    """
    errors: List[str] = []
    statements = [s.strip() for s in sqlparse.split(sql) if s.strip()]
    if not statements:
        return False, ["未解析到 SQL 语句"]
    for stmt_text in statements:
        stmt_text = stmt_text.rstrip(";").strip()
        if not stmt_text:
            continue
        if not stmt_text.upper().startswith("SELECT"):
            errors.append(f"仅支持 SELECT 语句：{stmt_text[:60]}")
            continue
        structure = parse_sql(stmt_text)
        if structure is None:
            errors.append(f"SQL 结构解析失败：{stmt_text[:80]}")
            continue
        errors.extend(_validate_structure(structure, schema))
    return (len(errors) == 0), errors


def compile_check(conn, sql: str, max_execution_time: int = 15000) -> str:
    """在 MySQL 上真实编译执行校验（最终裁决）。

    参数:
        conn: pymysql 连接对象
        sql: 一条或多条 SELECT 语句
        max_execution_time: MySQL 单语句最大执行时间（毫秒）

    返回:
        空串表示全部通过；否则返回第一条错误信息。
    """
    statements = [s.strip() for s in sqlparse.split(sql) if s.strip()]
    for stmt in statements:
        try:
            with conn.cursor() as cur:
                cur.execute("SET SESSION MAX_EXECUTION_TIME = %s" % max_execution_time)
                cur.execute(stmt)
                cur.fetchall()
        except Exception as exc:  # noqa: BLE001 - 需要捕获任意 MySQL 错误并返回文案
            return str(exc)
    return ""