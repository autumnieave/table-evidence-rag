"""compare_table_parse 原文 vs 解析结果比对单元测试（纯逻辑、零外部依赖）。"""

import copy

from data.table_struct import build_table, parse_html_table_grid
from tools.data_scripts.compare_table_parse import (
    _is_span_title_row,
    _row_matches_headers,
    _sample_indices,
    align_table,
    build_markdown_source_grid,
    build_source_grid,
    clean_cell_text,
    compare_table,
    normalize_raw_text,
)

PLAIN_HTML = (
    "<table>"
    "<tr><td>项目</td><td>2024A</td><td>2025E</td></tr>"
    "<tr><td>营业收入</td><td>100</td><td>120</td></tr>"
    "<tr><td>净利润</td><td>10</td><td>12</td></tr>"
    "<tr><td>毛利率</td><td>30%</td><td>31%</td></tr>"
    "</table>"
)
SPAN_TITLE_HTML = (
    "<table>"
    "<tr><td colspan='3'>盈利预测与估值</td></tr>"
    "<tr><td>项目</td><td>2024A</td><td>2025E</td></tr>"
    "<tr><td>营收</td><td>1</td><td>2</td></tr>"
    "</table>"
)
KEY_VALUE_HTML = (
    "<table>"
    "<tr><td>总市值</td><td>100</td></tr>"
    "<tr><td>PE</td><td>20</td></tr>"
    "</table>"
)
SECTION_HTML = (
    "<table>"
    "<tr><td>资本结构</td><td></td><td></td></tr>"
    "<tr><td>资产负债率</td><td>15.40%</td><td>14.43%</td></tr>"
    "</table>"
)
ROWMAP_HTML = (
    "<table>"
    "<tr><td>板块</td><td>公司</td><td>占比</td></tr>"
    "<tr><td rowspan='2'>医药</td><td>A</td><td>10%</td></tr>"
    "<tr><td>B</td><td>20%</td></tr>"
    "</table>"
)


def _table(html, table_format="html"):
    """用解析器把 HTML 解析为 Table（测试夹具）。"""
    grid = parse_html_table_grid(html)
    assert grid is not None
    return build_table(0, "f.md", grid, table_format=table_format, raw=html)


def test_clean_cell_text_strips_zero_width_and_collapses_space():
    assert clean_cell_text("  a\u00a0\u200b b \n c ") == "a b c"


def test_normalize_raw_text_unescapes_entities():
    assert normalize_raw_text("<td>a&amp;b</td>\n<td>c</td>") == "<td>a&b</td> <td>c</td>"


def test_build_source_grid_expands_colspan_and_rowspan():
    grid = build_source_grid(ROWMAP_HTML, "html")
    assert grid == [
        ["板块", "公司", "占比"],
        ["医药", "A", "10%"],
        ["医药", "B", "20%"],
    ]


def test_build_source_grid_drops_empty_rows():
    html = "<table><tr><td></td><td> </td></tr><tr><td>a</td><td>b</td></tr></table>"
    assert build_source_grid(html, "html") == [["a", "b"]]


def test_build_source_grid_sniffs_html_even_if_label_is_markdown():
    """格式标签写错时按原文内容嗅探，不应产出空网格。"""
    assert len(build_source_grid(PLAIN_HTML, "markdown")) == 4


def test_build_markdown_source_grid_drops_separator():
    raw = "| 项目 | 2024A |\n| --- | --- |\n| 营收 | 100 |"
    assert build_markdown_source_grid(raw) == [["项目", "2024A"], ["营收", "100"]]


def test_row_matches_headers_accepts_real_project_header():
    """「项目」既是占位名也是真实表头文本，源格非空时应按真实表头匹配。"""
    assert _row_matches_headers(["项目", "2024A", "2025E"], ["项目", "2024A", "2025E"])


def test_row_matches_headers_accepts_placeholder_for_empty_cell():
    assert _row_matches_headers(["", "2024A"], ["项目", "2024A"])


def test_row_matches_headers_rejects_mismatch():
    assert not _row_matches_headers(["营收", "2024A"], ["项目", "2024A"])


def test_is_span_title_row():
    assert _is_span_title_row(["盈利预测与估值", "盈利预测与估值", "盈利预测与估值"])
    assert not _is_span_title_row(["项目", "2024A", "2025E"])
    assert not _is_span_title_row(["标题", "标题"])  # 列数 < 3 不算


def test_align_table_plain_header_consumes_one_row():
    align = align_table(build_source_grid(PLAIN_HTML, "html"), _table(PLAIN_HTML))
    assert align["消费行数"] == 1
    assert align["表头行"] == 0
    assert align["数据起点"] == 1
    assert align["通过"]


def test_align_table_span_title_consumes_two_rows():
    table = _table(SPAN_TITLE_HTML)
    align = align_table(build_source_grid(SPAN_TITLE_HTML, "html"), table)
    assert table.caption == "盈利预测与估值"
    assert align["消费行数"] == 2
    assert align["数据起点"] == 2
    assert align["未解释消费行"] == []
    assert align["通过"]


def test_align_table_key_value_consumes_nothing():
    table = _table(KEY_VALUE_HTML)
    assert table.is_key_value
    align = align_table(build_source_grid(KEY_VALUE_HTML, "html"), table)
    assert align["消费行数"] == 0
    assert align["通过"]


def test_align_table_section_label_table_keeps_all_rows():
    """分节标签表（首行仅 1 个非空单元格）→ 判无表头、全部源行保留为数据行。"""
    table = _table(SECTION_HTML)
    assert not table.has_header
    align = align_table(build_source_grid(SECTION_HTML, "html"), table)
    assert align["源行数"] == 2
    assert align["消费行数"] == 0
    assert align["通过"]


def test_compare_table_passes_for_correct_parse():
    for html in (PLAIN_HTML, SPAN_TITLE_HTML, KEY_VALUE_HTML, SECTION_HTML, ROWMAP_HTML):
        record = compare_table(html, _table(html))
        assert record["是否通过"], (html, record["差异类型"], record["差异详情"])


def test_compare_table_detects_dropped_row():
    table = _table(PLAIN_HTML)
    broken = copy.deepcopy(table)
    broken.rows = broken.rows[:-1]
    record = compare_table(PLAIN_HTML, broken)
    assert not record["是否通过"]
    assert "行数差异" in record["差异类型"]
    assert record["比对明细"]["行数"]["未解释消费行"]


def test_compare_table_detects_duplicated_header_row():
    table = _table(PLAIN_HTML)
    broken = copy.deepcopy(table)
    broken.rows = [list(broken.headers)] + list(broken.rows)
    record = compare_table(PLAIN_HTML, broken)
    assert not record["是否通过"]
    assert "行数差异" in record["差异类型"]


def test_compare_table_detects_tampered_cell():
    table = _table(PLAIN_HTML)
    broken = copy.deepcopy(table)
    broken.rows[-1][1] = "99%"
    record = compare_table(PLAIN_HTML, broken)
    assert not record["是否通过"]
    assert "单元格差异" in record["差异类型"]


def test_compare_table_detects_fabricated_header():
    table = _table(PLAIN_HTML)
    broken = copy.deepcopy(table)
    broken.headers[0] = "凭空表头"
    record = compare_table(PLAIN_HTML, broken)
    assert not record["是否通过"]
    assert "表头差异" in record["差异类型"]
    assert "凭空表头" in "".join(record["比对明细"]["表头"]["不在原文"])


def test_compare_table_detects_horizontal_truncation():
    table = _table(PLAIN_HTML)
    broken = copy.deepcopy(table)
    broken.headers = broken.headers[:2]
    broken.rows = [row[:2] for row in broken.rows]
    record = compare_table(PLAIN_HTML, broken)
    assert not record["是否通过"]
    assert "列数差异" in record["差异类型"]


def test_compare_table_detects_key_value_first_row_loss():
    table = _table(KEY_VALUE_HTML)
    broken = copy.deepcopy(table)
    broken.rows = broken.rows[1:]
    record = compare_table(KEY_VALUE_HTML, broken)
    assert not record["是否通过"]
    assert "行数差异" in record["差异类型"]


def test_sample_indices_first_and_last_three():
    assert _sample_indices(10) == [0, 1, 2, 7, 8, 9]
    assert _sample_indices(4) == [0, 1, 2, 3]
    assert _sample_indices(0) == []