"""metadata coverage helper tests (pure logic, no external services)."""

from tools.data_scripts.measure_metadata_coverage import (
    coverage,
    layered_table_coverage,
    unit_candidate_distribution,
)


def test_coverage_counts_non_empty_only():
    rows = [{"unit": "百万元"}, {"unit": None}, {"unit": ""}, {"unit": []}]
    assert coverage(rows, "unit") == {"覆盖": 1, "分母": 4, "覆盖率": 0.25}
    assert coverage([], "unit") == {"覆盖": 0, "分母": 0, "覆盖率": 0.0}


def test_layered_table_coverage_groups_by_doc_type_type_and_scope():
    rows = [
        {"chunk_type": "table", "doc_type": "stock", "table_type": "financial_statement",
         "table_scope": "stock", "unit": "百万元", "unit_source": "cell"},
        {"chunk_type": "table", "doc_type": "stock", "table_type": "financial_statement",
         "table_scope": "stock", "unit": None, "unit_source": "multi"},
        {"chunk_type": "table", "doc_type": "industry", "table_type": "valuation",
         "table_scope": "industry", "unit": "亿元", "unit_source": "caption"},
        {"chunk_type": "prose", "doc_type": "stock", "table_type": "other",
         "table_scope": "unknown", "unit": None, "unit_source": "none"},
    ]
    out = layered_table_coverage(rows, ["unit"])
    assert set(out) == {"stock|financial_statement|stock", "industry|valuation|industry"}
    stock = out["stock|financial_statement|stock"]
    assert stock["n"] == 2
    assert stock["unit"] == {"覆盖": 1, "分母": 2, "覆盖率": 0.5}
    assert stock["unit_source 分布"] == {"cell": 1, "multi": 1}
    assert out["industry|valuation|industry"]["unit"]["覆盖率"] == 1.0


def test_unit_candidate_distribution_buckets_zero_one_and_many():
    rows = [
        {"unit_candidates": []},
        {"unit_candidates": None},
        {"unit_candidates": ["百万元"]},
        {"unit_candidates": ["元", "百万元"]},
        {"unit_candidates": ["元", "百万元", "亿元"]},
    ]
    assert unit_candidate_distribution(rows) == {"0": 2, "1": 1, ">=2": 2}
