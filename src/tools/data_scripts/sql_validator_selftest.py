# -*- coding: utf-8 -*-
"""SQL 校验器自测：原始 9 条失败 SQL 应全部拦截；修复后 9 条通过 SQL 应零误报。

用法::

    .\\.venv\\Scripts\\python tools\\data_scripts\\sql_validator_selftest.py
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pymysql

from tools.sql_validator import load_schema, validate_sql

DB_CONFIG = dict(host="127.0.0.1", port=3306, user="root", password="123456",
                 database="financial_database", charset="utf8mb4", connect_timeout=10)


def main() -> None:
    """执行自测并打印拦截率/误报率汇总。"""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    conn = pymysql.connect(**DB_CONFIG)
    schema = load_schema(conn)
    conn.close()

    bad_cases: List[Tuple[str, str]] = []
    fails = Path("训练结果数据/badcase_sql_failures.json")
    for item in json.loads(fails.read_text(encoding="utf-8")):
        sql = (item.get("SQL") or "").strip()
        if sql:
            bad_cases.append((item["编号"], sql))

    good_cases: List[Tuple[str, str]] = []
    diag = Path("训练结果数据/e2e_diag_final.json")
    for row in json.loads(diag.read_text(encoding="utf-8")):
        for detail in row["明细"]:
            sql = (detail.get("sql") or "").strip()
            if sql:
                good_cases.append((row["编号"], sql))

    caught: List[str] = []
    missed: List[str] = []
    for bid, sql in bad_cases:
        ok, errors = validate_sql(sql, schema)
        if ok:
            missed.append(bid)
        else:
            caught.append(bid)

    false_positive: List[str] = []
    for bid, sql in good_cases:
        ok, errors = validate_sql(sql, schema)
        if not ok:
            false_positive.append(f"{bid}: {errors[:1]}")

    # 新增用例：子查询内表名白名单检查（B2035 类虚构表应被拦截；白名单子查询应通过）
    subquery_bad = [
        ("B2035-subquery",
         "SELECT t1.stock_abbr FROM core_performance_indicators_sheet t1 "
         "JOIN (SELECT stock_code FROM research_reports WHERE report_content LIKE '%行业龙头%') t2 "
         "ON t1.stock_code = t2.stock_code;"),
        ("B2035-where-subquery",
         "SELECT stock_abbr FROM core_performance_indicators_sheet "
         "WHERE stock_code IN (SELECT stock_code FROM stock_info WHERE stock_abbr LIKE '%x%');"),
    ]
    subquery_good = [
        ("B20XX-ok-derived",
         "SELECT t1.stock_abbr FROM core_performance_indicators_sheet t1 "
         "JOIN (SELECT stock_code FROM income_sheet WHERE report_year = 2025) t2 "
         "ON t1.stock_code = t2.stock_code;"),
        ("B20XX-ok-where-sub",
         "SELECT stock_abbr FROM core_performance_indicators_sheet "
         "WHERE stock_code IN (SELECT stock_code FROM balance_sheet WHERE report_year = 2025);"),
    ]
    for bid, sql in subquery_bad:
        ok, errors = validate_sql(sql, schema)
        if ok:
            missed.append(bid)
        else:
            caught.append(bid)
            print(f"  [子查询拦截] {bid}: {errors[0]}")
    for bid, sql in subquery_good:
        ok, errors = validate_sql(sql, schema)
        if not ok:
            false_positive.append(f"{bid}: {errors[:1]}")

    print("== 自测结果 ==")
    print(f"坏 SQL（原始 9 条）: {len(bad_cases)} 条，拦截 {len(caught)} 条，漏检 {missed or '无'}")
    print(f"好 SQL（修复后 e2e）: {len(good_cases)} 条，误报 {false_positive or '无'}")
    print()
    print("== 拦截示例（原始失败 SQL 的校验错误）==")
    for bid, sql in bad_cases[:9]:
        ok, errors = validate_sql(sql, schema)
        print(f"--- {bid} ---")
        for e in errors[:3]:
            print("  ✗", e)


if __name__ == "__main__":
    main()
