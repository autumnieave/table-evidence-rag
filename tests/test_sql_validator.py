"""tools.sql_validator 静态校验器单元测试（纯逻辑，不依赖 MySQL）。

覆盖四类拦截：白名单表 / 未定义别名 / 字段-表归属 / 裸字段歧义 / 子查询表检查。
样例取自 9 条 badcase（B2049/B2075/B2063 等）的原始失败 SQL。
"""

import pytest

from tools.sql_validator import (
    MAX_SELECT_ROWS,
    enforce_select_limit,
    ensure_readonly_mysql_user,
    parse_sql,
    validate_sql,
)


def _schema() -> dict:
    """最小 schema：4 张白名单表的常用字段"""
    core = ["stock_code", "stock_abbr", "report_year", "report_period",
            "net_profit_10k_yuan", "total_operating_revenue", "eps", "roe"]
    income = ["stock_code", "report_year", "report_period",
              "operating_expense_cost_of_sales", "operating_expense_selling_expenses"]
    balance = ["stock_code", "report_year", "report_period", "asset_total_assets", "liability_total_liabilities"]
    cash = ["stock_code", "report_year", "report_period", "cash_flow_net_operating"]
    return {
        "core_performance_indicators_sheet": {c: "varchar(20)" for c in core},
        "income_sheet": {c: "varchar(20)" for c in income},
        "balance_sheet": {c: "varchar(20)" for c in balance},
        "cash_flow_sheet": {c: "varchar(20)" for c in cash},
    }


def test_valid_sql_passes():
    sql = "SELECT stock_abbr, net_profit_10k_yuan FROM core_performance_indicators_sheet WHERE report_year = 2024"
    ok, errors = validate_sql(sql, _schema())
    assert ok, errors
    assert errors == []


def test_unknown_table_blocked():
    sql = "SELECT * FROM stock_info"
    ok, errors = validate_sql(sql, _schema())
    assert not ok
    assert any("stock_info" in e for e in errors)


def test_undefined_alias_blocked():
    # B2053 类：SELECT 引用 t1 但 FROM/JOIN 未定义
    sql = (
        "SELECT t1.stock_code, t2.net_profit_10k_yuan FROM core_performance_indicators_sheet t2 "
        "JOIN income_sheet t3 ON t2.stock_code = t3.stock_code"
    )
    ok, errors = validate_sql(sql, _schema())
    assert not ok
    assert any("未定义别名" in e and "t1" in e for e in errors)


def test_field_not_in_table_blocked():
    # B2049 类：net_profit 误挂 core 表
    sql = "SELECT t1.net_profit FROM core_performance_indicators_sheet t1"
    ok, errors = validate_sql(sql, _schema())
    assert not ok
    assert any("net_profit" in e and "不存在" in e for e in errors)


def test_bare_ambiguous_field_blocked():
    # B2075 类：多表 JOIN 下裸字段歧义
    sql = (
        "SELECT stock_code FROM core_performance_indicators_sheet t1 "
        "JOIN income_sheet t2 ON t1.stock_code = t2.stock_code"
    )
    ok, errors = validate_sql(sql, _schema())
    assert not ok
    assert any("歧义" in e for e in errors)


def test_subquery_unknown_table_blocked():
    # B2035 类：子查询中使用白名单外表
    sql = "SELECT stock_abbr FROM core_performance_indicators_sheet WHERE stock_code IN (SELECT stock_code FROM stock_info)"
    ok, errors = validate_sql(sql, _schema())
    assert not ok
    assert any("子查询" in e and "stock_info" in e for e in errors)


def test_parse_sql_structure():
    sql = "SELECT stock_abbr FROM core_performance_indicators_sheet WHERE report_year = 2024"
    parsed = parse_sql(sql)
    assert parsed is not None
    assert "core_performance_indicators_sheet" in parsed.get("tables", [])


def test_enforce_select_limit_appends_default():
    sql = "SELECT stock_abbr FROM core_performance_indicators_sheet WHERE report_year = 2024"
    limited = enforce_select_limit(sql)
    assert limited.endswith(f"LIMIT {MAX_SELECT_ROWS}")


def test_enforce_select_limit_clamps_large_limit():
    sql = "SELECT stock_abbr FROM core_performance_indicators_sheet LIMIT 500"
    assert enforce_select_limit(sql).endswith(f"LIMIT {MAX_SELECT_ROWS}")


def test_enforce_select_limit_keeps_smaller_limit():
    sql = "SELECT stock_abbr FROM core_performance_indicators_sheet LIMIT 5"
    assert enforce_select_limit(sql).endswith("LIMIT 5")


def test_enforce_select_limit_preserves_offset_forms():
    assert enforce_select_limit(
        "SELECT stock_abbr FROM core_performance_indicators_sheet LIMIT 10, 500"
    ).endswith("LIMIT 10, 100")
    assert enforce_select_limit(
        "SELECT stock_abbr FROM core_performance_indicators_sheet LIMIT 500 OFFSET 20"
    ).endswith("LIMIT 100 OFFSET 20")


def test_enforce_select_limit_handles_multiple_statements():
    sql = (
        "SELECT stock_abbr FROM core_performance_indicators_sheet;"
        "SELECT stock_abbr FROM income_sheet LIMIT 500;"
    )
    limited = enforce_select_limit(sql)
    assert limited.count("LIMIT 100") == 2


def test_enforce_select_limit_rejects_non_select():
    with pytest.raises(ValueError, match="仅允许执行 SELECT"):
        enforce_select_limit("DELETE FROM income_sheet")


def test_readonly_mysql_user_guard():
    assert ensure_readonly_mysql_user("rag_reader") == "rag_reader"
    with pytest.raises(ValueError, match="禁止使用 root"):
        ensure_readonly_mysql_user("root")
    with pytest.raises(ValueError, match="禁止使用 root"):
        ensure_readonly_mysql_user("root@localhost")


def test_enforce_select_limit_strips_trailing_line_comment():
    sql = "SELECT stock_abbr FROM core_performance_indicators_sheet LIMIT 50000 -- 全量导出"
    limited = enforce_select_limit(sql)
    assert limited.endswith("LIMIT 100")
    assert "--" not in limited


def test_enforce_select_limit_keeps_double_dash_inside_string():
    sql = "SELECT stock_abbr FROM core_performance_indicators_sheet WHERE note = 'a -- b'"
    limited = enforce_select_limit(sql)
    assert "'a -- b'" in limited
    assert limited.endswith("LIMIT 100")


def test_enforce_select_limit_handles_leading_comment_and_block_comment():
    sql = "-- 统计口径\nSELECT stock_abbr FROM core_performance_indicators_sheet /* 全量 */ LIMIT 500"
    limited = enforce_select_limit(sql)
    assert limited.lstrip().startswith("SELECT")
    assert "/*" not in limited
    assert "--" not in limited
    assert limited.endswith("LIMIT 100")


def test_enforce_select_limit_rejects_comment_only_statement():
    with pytest.raises(ValueError, match="未解析到 SQL 语句"):
        enforce_select_limit("-- 只有注释")
