"""L5-C 期间过滤（``src/filters/period_gate.py``）纯函数单测：零外部依赖。

口径：**个股研报用 ``report_period``、行业研报用 ``publish_year``**；键取不到 ⇒ 不过滤（``None`` / ``True``）。
"""
from filters.period_gate import (
    PERIOD_KEY_BY_DOC_TYPE,
    matches_period,
    period_key,
    period_key_name,
    period_quarter,
    period_year,
)


def test_period_key_name_by_doc_type():
    """按 ``doc_type`` 选键：个股 → ``report_period``、行业 → ``publish_year``、未登记 → 回退。"""
    assert period_key_name({"doc_type": "stock"}) == "report_period"
    assert period_key_name({"doc_type": "industry"}) == "publish_year"
    assert period_key_name({"doc_type": ""}) == "report_period"
    assert period_key_name({}) == "report_period"
    assert PERIOD_KEY_BY_DOC_TYPE == {"stock": "report_period", "industry": "publish_year"}


def test_period_key_reads_selected_field_only():
    """个股只读 ``report_period``：即使有 ``publish_year`` 也不当期间键。"""
    stock = {"doc_type": "stock", "report_period": "2025Q3", "publish_year": 2025}
    assert period_key(stock) == "2025Q3"

    industry = {"doc_type": "industry", "report_period": "2025Q3", "publish_year": 2025}
    assert period_key(industry) == "2025"

    # 个股缺 report_period 时不回退到 publish_year（那是行业的口径）
    assert period_key({"doc_type": "stock", "publish_year": 2025}) is None


def test_period_key_empty_values_are_none():
    for empty in (None, "", [], {}):
        assert period_key({"doc_type": "stock", "report_period": empty}) is None


def test_period_year_and_quarter():
    assert period_year({"doc_type": "stock", "report_period": "2025Q3"}) == 2025
    assert period_quarter({"doc_type": "stock", "report_period": "2025Q3"}) == 3
    assert period_quarter({"doc_type": "stock", "report_period": "25H1"}) == 1
    assert period_quarter({"doc_type": "stock", "report_period": "2024A"}) is None
    assert period_year({"doc_type": "industry", "publish_year": 2025}) == 2025
    assert period_quarter({"doc_type": "industry", "publish_year": 2025}) is None
    assert period_year({"doc_type": "stock", "report_period": ""}) is None


def test_matches_period_stock_month_and_year():
    q3 = {"doc_type": "stock", "report_period": "2025Q3"}
    assert matches_period(q3, 2025) is True
    assert matches_period(q3, 2025, 3) is True
    assert matches_period(q3, 2025, 2) is False
    assert matches_period(q3, 2024) is False


def test_matches_period_industry_year_only_never_blocks_on_quarter():
    """行业研报只有发布年 ⇒ 给了季度也**放行**（不误杀）。"""
    ind = {"doc_type": "industry", "publish_year": 2025}
    assert matches_period(ind, 2025) is True
    assert matches_period(ind, 2025, 3) is True
    assert matches_period(ind, 2025, 1) is True
    assert matches_period(ind, 2024) is False


def test_matches_period_without_key_returns_none():
    """取不到期间键 ⇒ ``None``（调用方应放行、不是判不匹配）。"""
    assert matches_period({"doc_type": "stock"}, 2025) is None
    assert matches_period({"doc_type": "industry"}, 2025, 3) is None
    assert matches_period({}, 2025) is None


def test_stock_year_only_period_still_matches_quarter_query():
    """个股只有年份（``2025A``）时，给了季度也应放行（季度信息缺失 ≠ 不匹配）。"""
    a = {"doc_type": "stock", "report_period": "2025A"}
    assert matches_period(a, 2025) is True
    assert matches_period(a, 2025, 2) is True
    assert matches_period(a, 2026, 2) is False
