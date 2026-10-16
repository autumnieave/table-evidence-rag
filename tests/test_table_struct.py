"""data.table_struct 表格结构化单元测试（纯逻辑、零外部依赖）。"""

import json

from data.splitter import split_documents
from data.table_struct import (
    TABLE_JSON_MAX_CHARS,
    TABLE_SLICE_ROWS,
    Table,
    TableGrid,
    build_table,
    detect_unit,
    detect_units,
    extract_metric_names,
    extract_report_period,
    extract_stock_codes,
    header_placeholder,
    normalize_unit,
    parse_html_table_grid,
    parse_markdown_table_grid,
    scan_tables,
    unit_factor,
)

MD_TABLE = (
    "| 指标 | 2025Q3 | 2024Q3 |\n"
    "| --- | --- | --- |\n"
    "| 营业收入 | 1,234 | 1,000 |\n"
    "| 归母净利润 | 321 | 300 |"
)

#: 语料里最常见的「2 列键值卡片」（无 <th>、首行即数据）
KEY_VALUE_HTML = (
    "<table>"
    "<tr><td>基本数据</td><td>2025年04月03日</td></tr>"
    "<tr><td>总市值（百万元）</td><td>11,349.65</td></tr>"
    "</table>"
)

#: 语料里最常见的「多列表」（无 <th>、首行即表头）
GRID_HTML = (
    "<table>"
    "<tr><td>会计年度</td><td>2024</td><td>2025E</td></tr>"
    "<tr><td>营业收入</td><td>3,728</td><td>4,411</td></tr>"
    "</table>"
)


def _grid(headers, rows, caption="", has_header=True, is_key_value=False):
    return TableGrid(
        headers=list(headers), rows=[list(r) for r in rows],
        caption=caption, has_header=has_header, is_key_value=is_key_value,
    )


# ---------------------------------------------------------------- 单位

def test_detect_unit_common_writes():
    assert detect_unit("表1：主要指标（万元）") == ("万元", 10000)
    assert detect_unit("表2：营业收入(单位:亿元)") == ("亿元", 100000000)
    assert detect_unit("单位：百万元") == ("百万元", 1000000)
    assert detect_unit("表3：主要指标") == (None, 1)


def test_detect_unit_ignores_compound_units():
    # 「（元/股）」「(%)」不是整表单位，不能当成「元」
    assert detect_unit("每股净资产（元/股）") == (None, 1)
    assert detect_unit("资产负债率 (%)") == (None, 1)


def test_detect_units_lists_all_distinct():
    assert detect_units("收盘价（元） 总市值（百万元）") == ["元", "百万元"]
    assert detect_units("收盘价（元） 市净率（元）") == ["元"]
    assert detect_units("没有单位") == []


def test_normalize_unit_and_factor():
    assert normalize_unit("万") == "万元"
    assert normalize_unit("亿") == "亿元"
    assert normalize_unit(" 万元 ") == "万元"
    assert normalize_unit("") is None
    assert normalize_unit("个") is None
    assert unit_factor("万元") == 10000
    assert unit_factor(None) == 1


# ---------------------------------------------------------------- 主体 / 报告期 / 指标

def test_extract_stock_codes_forms():
    assert extract_stock_codes("300181.SZ 300181.SZ") == ["300181.SZ"]
    codes = extract_stock_codes("300181.SZ 与 SH600519 及 002415 . SZ")
    assert set(codes) == {"300181.SZ", "600519.SH", "002415.SZ"}


def test_extract_stock_codes_ignores_plain_numbers():
    assert extract_stock_codes("营业收入 1234567 元") == []


def test_extract_report_period_forms():
    assert extract_report_period("2025Q3 业绩") == "2025Q3"
    assert extract_report_period("2025年三季度") == "2025Q3"
    assert extract_report_period("2025H1") == "2025H1"
    assert extract_report_period("2025年中报") == "2025H1"
    assert extract_report_period("2024年报") == "2024A"
    assert extract_report_period("2024年度") == "2024A"
    assert extract_report_period("2025年一季报") == "2025Q1"
    assert extract_report_period("没有报告期") is None


def test_extract_metric_names_from_headers():
    assert extract_metric_names(["项目", "营业收入", "同比增速", "备注"]) == ["营业收入", "同比增速"]
    assert extract_metric_names([]) == []


def test_header_placeholder_names():
    assert header_placeholder(0) == "项目"
    assert header_placeholder(1) == "列2"
    assert header_placeholder(5) == "列6"


# ---------------------------------------------------------------- Markdown 解析

def test_parse_markdown_table_grid_basic():
    grid = parse_markdown_table_grid(MD_TABLE.split("\n"))
    assert grid is not None
    assert grid.has_header is True and grid.is_key_value is False
    assert grid.headers == ["指标", "2025Q3", "2024Q3"]
    assert grid.rows == [["营业收入", "1,234", "1,000"], ["归母净利润", "321", "300"]]


def test_parse_markdown_table_grid_without_separator():
    grid = parse_markdown_table_grid(["| A | B |", "| 1 | 2 |"])
    assert grid.headers == ["A", "B"]
    assert grid.rows == [["1", "2"]]


def test_parse_markdown_table_grid_escaped_pipe():
    grid = parse_markdown_table_grid(["| 指标 | 说明 |", "| --- | --- |", "| 营收 | 含\\|符号 |"])
    assert grid.rows == [["营收", "含|符号"]]


def test_parse_markdown_table_grid_pads_ragged_rows():
    grid = parse_markdown_table_grid(["| A | B | C |", "| --- | --- | --- |", "| 1 | 2 |"])
    assert grid.headers == ["A", "B", "C"]
    assert grid.rows == [["1", "2", ""]]


def test_parse_markdown_table_grid_rejects_header_only():
    assert parse_markdown_table_grid(["| A | B |", "| --- | --- |"]) is None
    assert parse_markdown_table_grid(["| A | B |"]) is None


# ---------------------------------------------------------------- HTML 解析（表头推断）

def test_parse_html_table_grid_with_th():
    grid = parse_html_table_grid(
        "<table><tr><th>指标</th><th>数值</th></tr><tr><td>营收</td><td>100</td></tr></table>"
    )
    assert grid.has_header is True
    assert grid.headers == ["指标", "数值"]
    assert grid.rows == [["营收", "100"]]


def test_parse_html_table_grid_headerless_uses_first_row_as_header():
    grid = parse_html_table_grid(GRID_HTML)
    assert grid.has_header is True and grid.is_key_value is False
    assert grid.headers == ["会计年度", "2024", "2025E"]
    assert grid.rows == [["营业收入", "3,728", "4,411"]]


def test_parse_html_table_grid_two_column_is_key_value():
    grid = parse_html_table_grid(KEY_VALUE_HTML)
    assert grid.is_key_value is True and grid.has_header is False
    assert grid.headers == ["项目", "列2"]
    # 首行是数据（基本数据 | 2025年04月03日），不能被当表头吃掉
    assert grid.rows == [["基本数据", "2025年04月03日"], ["总市值（百万元）", "11,349.65"]]


def test_parse_html_table_grid_colspan_title_row_becomes_caption():
    html = (
        "<table>"
        "<tr><td colspan='3'>盈利预测与估值</td></tr>"
        "<tr><td></td><td>2024</td><td>2025E</td></tr>"
        "<tr><td>营业收入</td><td>3,728</td><td>4,411</td></tr>"
        "</table>"
    )
    grid = parse_html_table_grid(html)
    assert grid.caption == "盈利预测与估值"
    assert grid.headers == ["项目", "2024", "2025E"]
    assert grid.rows == [["营业收入", "3,728", "4,411"]]


def test_parse_html_table_grid_colspan_and_rowspan():
    html = (
        "<table><tr><th>A</th><th>B</th><th>C</th></tr>"
        "<tr><td colspan='2'>X</td><td>1</td></tr>"
        "<tr><td rowspan='2'>Y</td><td>2</td><td>3</td></tr>"
        "<tr><td>4</td><td>5</td></tr></table>"
    )
    grid = parse_html_table_grid(html)
    assert grid.headers == ["A", "B", "C"]
    assert grid.rows == [["X", "X", "1"], ["Y", "2", "3"], ["Y", "4", "5"]]


def test_parse_html_table_grid_rejects_header_only():
    assert parse_html_table_grid("<table><tr><th>只有表头</th></tr></table>") is None
    assert parse_html_table_grid("") is None


# ---------------------------------------------------------------- Table 构造

def test_build_table_infers_metadata():
    grid = _grid(
        ["指标", "营业收入", "同比增速"], [["贵州茅台", "1234", "5%"]],
        caption="表1：主要财务指标",
    )
    table = build_table(0, "a.md", grid, title="贵州茅台2025Q3主要指标（万元）")
    assert isinstance(table, Table)
    assert table.table_id == "a.md#t1"
    assert table.index == 0
    assert table.caption == "表1：主要财务指标"
    assert table.unit == "万元" and table.unit_factor == 10000
    assert table.report_period == "2025Q3"
    assert table.metric_names == ["营业收入", "同比增速"]
    assert table.row_count == 1 and table.col_count == 3
    assert table.parse_ok is True and table.table_format == "markdown"


def test_build_table_unit_from_first_column_when_single():
    grid = _grid(
        ["项目", "2024", "2025E"],
        [["营业收入(百万元)", "3,728", "4,411"], ["归母净利润(百万元)", "443", "528"]],
    )
    table = build_table(0, "a.md", grid)
    assert table.unit == "百万元" and table.unit_factor == 1000000
    assert table.metric_names == ["营业收入(百万元)", "归母净利润(百万元)"]


def test_build_table_mixed_units_leave_unit_none():
    grid = _grid(
        ["项目", "列2"], [["收盘价（元）", "26.33"], ["总市值（百万元）", "11,349.65"]],
        has_header=False, is_key_value=True,
    )
    table = build_table(0, "a.md", grid)
    assert table.unit is None
    assert table.unit_candidates == ["元", "百万元"]


def test_build_table_stock_code_from_rows_when_head_missing():
    grid = _grid(["证券代码", "证券简称", "市值"], [["300181.SZ", "佐力药业", "130.81"]])
    table = build_table(0, "b.md", grid)
    assert table.stock_codes == ["300181.SZ"]


def test_build_table_fills_empty_header_placeholder():
    grid = _grid(["", "营业收入"], [["1", "2"]])
    table = build_table(0, "c.md", grid)
    assert table.headers == ["项目", "营业收入"]


# ---------------------------------------------------------------- 序列化

def test_table_to_text_markdown_layout():
    grid = _grid(["指标", "数值"], [["营收", "100"]], caption="表1：主要指标")
    table = build_table(0, "a.md", grid)
    assert table.to_text().split("\n") == [
        "表1：主要指标", "| 指标 | 数值 |", "| --- | --- |", "| 营收 | 100 |",
    ]


def test_table_to_text_key_value_lines():
    grid = _grid(
        ["项目", "列2"], [["基本数据", "2025年04月03日"], ["总市值（百万元）", "11,349.65"]],
        caption="表1：基本数据", has_header=False, is_key_value=True,
    )
    lines = build_table(0, "a.md", grid).to_text().split("\n")
    assert lines[0] == "表1：基本数据（单位：百万元）"
    assert "基本数据: 2025年04月03日" in lines
    assert "总市值（百万元）: 11,349.65" in lines
    assert not any(line.startswith("|") for line in lines)


def test_table_to_text_without_header_skips_header_row():
    grid = _grid(["项目", "列2", "列3"], [["1", "2", "3"]], has_header=False)
    table = build_table(0, "a.md", grid)
    assert table.to_text() == "| 1 | 2 | 3 |"


def test_table_to_text_includes_title_and_escapes_pipe():
    grid = _grid(["指标"], [["含|竖线"]])
    table = build_table(0, "a.md", grid, title="贵州茅台2024年报")
    text = table.to_text()
    assert text.split("\n")[0] == "贵州茅台2024年报"
    assert "含\\|竖线" in text


def test_table_caption_text_does_not_duplicate_unit():
    grid = _grid(["指标"], [["营收"]], caption="表1：主要指标（万元）")
    table = build_table(0, "a.md", grid)
    assert table.caption_text() == "表1：主要指标（万元）"
    assert table.to_text().count("万元") == 1


# ---------------------------------------------------------------- 切片

def _data_lines(markdown_text):
    """从 Markdown 表里取数据行（去掉标题、表头、分隔行）。"""
    rows = [
        line for line in markdown_text.split("\n")
        if line.startswith("| ") and not line.startswith("| ---")
    ]
    return rows[1:]


def test_to_slices_repeats_header_and_keeps_all_rows():
    grid = _grid(["序号"], [[str(i)] for i in range(95)])
    table = build_table(0, "a.md", grid)
    slices = table.to_slices(40)
    assert len(slices) == 3
    assert all("| 序号 |" in s for s in slices)
    assert [len(_data_lines(s)) for s in slices] == [40, 40, 15]
    collected = [line for s in slices for line in _data_lines(s)]
    assert len(collected) == 95
    assert collected[0] == "| 0 |" and collected[-1] == "| 94 |"


def test_to_slices_disabled_and_empty():
    grid = _grid(["序号"], [[str(i)] for i in range(95)])
    assert len(build_table(0, "a.md", grid).to_slices(0)) == 1
    assert Table(table_id="x#t1", headers=["A"]).to_slices() == []


# ---------------------------------------------------------------- payload

def test_to_payload_truncates_long_table():
    grid = _grid(["字段", "说明"], [["字段", "值" * 200] for _ in range(50)])
    payload = build_table(0, "a.md", grid).to_payload(max_chars=500)
    assert payload["table_truncated"] is True
    assert payload["table_row_count"] == 50
    assert payload["table_id"] == "a.md#t1"
    assert json.loads(payload["table_json"])["headers"] == ["字段", "说明"]


def test_to_payload_keeps_short_table_intact():
    grid = _grid(["A"], [["1"]])
    payload = build_table(0, "a.md", grid).to_payload()
    assert payload["table_truncated"] is False
    assert payload["table_has_header"] is True
    assert json.loads(payload["table_json"]) == {"headers": ["A"], "rows": [["1"]]}


# ---------------------------------------------------------------- scan_tables

def test_scan_tables_extracts_caption_and_source_note():
    content = (
        "正文第一段。\n\n"
        "表1：主要指标（万元）\n"
        "| 指标 | 数值 |\n| --- | --- |\n| 营收 | 100 |\n"
        "资料来源：公司公告\n\n"
        "正文第二段。"
    )
    prose, tables = scan_tables(content, source_file="a.md")
    assert len(tables) == 1
    assert "主要指标" in tables[0].caption and "资料来源" in tables[0].caption
    assert tables[0].rows == [["营收", "100"]]
    assert "| 指标 |" not in prose
    assert "表1：主要指标" not in prose
    assert "资料来源" not in prose
    assert "正文第一段。" in prose and "正文第二段。" in prose


def test_scan_tables_without_table_keeps_prose():
    prose, tables = scan_tables("一段正文。\n\n另一段。", source_file="a.md")
    assert tables == []
    assert "一段正文。" in prose and "另一段。" in prose


def test_scan_tables_keeps_unparsable_html():
    prose, tables = scan_tables("正文<table><tr><th>只有表头</th></tr></table>", source_file="a.md")
    assert tables == []
    assert "<table>" in prose


def test_scan_tables_strips_legacy_placeholder():
    prose, _ = scan_tables("正文\n[TABLE_PLACEHOLDER_3]\n\n另一段", source_file="a.md")
    assert "TABLE_PLACEHOLDER" not in prose
    assert "正文" in prose and "另一段" in prose


def test_scan_tables_parses_html_tables():
    prose, tables = scan_tables(GRID_HTML + KEY_VALUE_HTML, source_file="a.md", title="某公司2024年报")
    assert len(tables) == 2
    assert tables[0].table_format == "html"
    assert tables[0].title == "某公司2024年报"
    assert tables[0].headers == ["会计年度", "2024", "2025E"]
    assert tables[1].is_key_value is True
    assert "<table>" not in prose


# ---------------------------------------------------------------- split_documents 集成

def test_split_documents_structured_mode_emits_table_chunks():
    docs = [{
        "content": MD_TABLE + "\n\n正文段落。" + "内容" * 200,
        "metadata": {"filename": "a.md", "stockCode": "600519.SH", "stockName": "贵州茅台"},
    }]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=0, min_chars=200, table_mode="structured")
    tables = [c for c in chunks if c["metadata"]["chunk_type"] == "table"]
    prose = [c for c in chunks if c["metadata"]["chunk_type"] == "prose"]
    assert len(tables) == 1
    meta = tables[0]["metadata"]
    assert meta["is_table_row"] is True
    assert meta["parent_id"] == "a.md#t1"
    assert meta["stock_code"] == "600519.SH"
    assert meta["stock_name"] == "贵州茅台"
    assert meta["report_period"] == "2025Q3"
    assert meta["table_row_count"] == 2
    assert "| 营业收入 | 1,234 | 1,000 |" in tables[0]["content"]
    assert prose and all("| 指标 |" not in c["content"] for c in prose)
    assert "TABLE_PLACEHOLDER" not in "".join(c["content"] for c in chunks)
    assert meta["chunk_total"] == len(chunks)


def test_split_documents_structured_slices_long_table():
    rows = "\n".join("| 营业收入{} | {} |".format(i, i) for i in range(95))
    content = "| 指标 | 数值 |\n| --- | --- |\n" + rows
    docs = [{"content": content, "metadata": {"filename": "a.md"}}]
    chunks = split_documents(
        docs, chunk_size=512, chunk_overlap=0, table_mode="structured", table_slice_rows=40
    )
    tables = [c for c in chunks if c["metadata"]["chunk_type"] == "table"]
    assert len(tables) == 3
    assert [c["metadata"]["table_slice_index"] for c in tables] == [0, 1, 2]
    assert [c["metadata"]["row_start"] for c in tables] == [0, 40, 80]
    assert all("| 指标 | 数值 |" in c["content"] for c in tables)
    assert sum(c["metadata"]["row_end"] - c["metadata"]["row_start"] for c in tables) == 95


def test_split_documents_structured_without_table_emits_prose_only():
    docs = [{"content": "纯正文" + "内容" * 300, "metadata": {"filename": "a.md"}}]
    chunks = split_documents(docs, chunk_size=256, chunk_overlap=0, table_mode="structured")
    assert chunks
    assert all(c["metadata"]["chunk_type"] == "prose" for c in chunks)
    assert all(c["metadata"]["is_table_row"] is False for c in chunks)


def test_split_documents_legacy_mode_is_unchanged():
    docs = [{"content": MD_TABLE + "\n\n正文段落。" + "内容" * 200, "metadata": {"filename": "a.md"}}]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=0, min_chars=200)
    rows = [c for c in chunks if c["metadata"]["is_table_row"]]
    assert len(rows) == 2
    assert all(c["metadata"]["parent_id"] == "a.md_tables" for c in rows)
    assert all("chunk_type" not in c["metadata"] for c in chunks)


def test_table_constants_defaults():
    assert TABLE_SLICE_ROWS == 40
    assert TABLE_JSON_MAX_CHARS == 6000

# ---------------------------------------------------------------- 分节行 / 次级表头 / 合并单元格

#: 表中间出现「跨满列的单格标题行」（语料 NO=13 的「利润表 / 资产负债表 / 现金流量表」形）
SECTION_HTML = (
    "<table>"
    "<tr><td>会计年度</td><td>2024</td><td>2025E</td></tr>"
    '<tr><td colspan="3">利润表</td></tr>'
    "<tr><td>营业收入</td><td>3,728</td><td>4,411</td></tr>"
    "<tr><td>净利润</td><td>528</td><td>644</td></tr>"
    "</table>"
)

#: 表格**最后一行**是「备注:…」全宽说明行（语料 40 处；旧实现被「后面必须有数据行」漏掉）
TAIL_NOTE_HTML = (
    "<table>"
    "<tr><td>证券评级</td><td>评级</td><td>说明</td></tr>"
    "<tr><td>买入(Buy)</td><td>推荐</td><td>相对强于市场 20% 以上</td></tr>"
    '<tr><td colspan="3">备注:评级标准为以报告日后的6~12个月</td></tr>'
    "</table>"
)

#: 「近满宽」重复文本行：colspan=3 但表宽 4，余下一格为空（语料 NO=益丰药房 形）
NEAR_FULL_HTML = (
    "<table>"
    "<tr><td>会计年度</td><td>a</td><td>b</td><td>c</td></tr>"
    '<tr><td colspan="3">益丰药房</td><td></td></tr>'
    "<tr><td>门店数</td><td>1</td><td>2</td><td>3</td></tr>"
    "</table>"
)

#: 多级表头：NCCN 分节行之后又出现一行次级表头（语料 NO=8 形）
SUB_HEADER_HTML = (
    "<table>"
    "<tr><td></td><td>I级推荐</td><td>II级推荐</td></tr>"
    "<tr><td>一线治疗方案</td><td>奥希替尼</td><td>伏美替尼</td></tr>"
    '<tr><td colspan="3">NCCN指南 (Version 3.2025)</td></tr>'
    "<tr><td></td><td>首选</td><td>其他推荐</td></tr>"
    "<tr><td>一线治疗方案</td><td>奥希替尼</td><td>埃万妥单抗</td></tr>"
    "</table>"
)

#: 跨子表重复表头：分节行之后的整行与表头逐格相同（语料 NO=13 形）
REPEAT_HEADER_HTML = (
    "<table>"
    "<tr><td>会计年度</td><td>2024</td><td>2025E</td></tr>"
    '<tr><td colspan="3">利润表</td></tr>'
    "<tr><td>会计年度</td><td>2024</td><td>2025E</td></tr>"
    "<tr><td>营业收入</td><td>1</td><td>2</td></tr>"
    "</table>"
)

#: rowspan 合并（语料 NO=16 形）
MERGED_HTML = (
    "<table>"
    "<tr><td>评级标准</td><td>评级</td><td>说明</td></tr>"
    '<tr><td rowspan="2">行业</td><td>推荐</td><td>A</td></tr>'
    "<tr><td>中性</td><td>B</td></tr>"
    "</table>"
)

#: 表头行自带的 colspan——坐标为 -1，属于表头行，**不输出**
HEADER_SPAN_HTML = (
    "<table>"
    '<tr><td colspan="2">项目</td><td>2024</td></tr>'
    "<tr><td>a</td><td>b</td><td>c</td></tr>"
    "</table>"
)


def _html_table(html):
    grid = parse_html_table_grid(html)
    assert grid is not None
    return build_table(0, "a.md", grid, table_format="html")


def test_section_rows_detected_by_colspan_fact():
    table = _html_table(SECTION_HTML)
    assert table.rows[0] == ["利润表", "利润表", "利润表"]
    assert table.section_rows == [0]
    assert table.sub_header_rows == []


def test_section_rows_include_trailing_note_row():
    """旧实现要求「后面还有数据行」，会漏掉表格末尾的备注行。"""
    table = _html_table(TAIL_NOTE_HTML)
    assert table.section_rows == [1]
    assert table.rows[1][0].startswith("备注:评级标准")


def test_section_rows_near_full_width_repeated_text():
    table = _html_table(NEAR_FULL_HTML)
    assert table.section_rows == [0]


def test_sub_header_rows_multi_level():
    table = _html_table(SUB_HEADER_HTML)
    assert table.section_rows == [1]
    assert table.sub_header_rows == [2]


def test_sub_header_rows_repeated_header_row():
    table = _html_table(REPEAT_HEADER_HTML)
    assert table.section_rows == [0]
    assert table.sub_header_rows == [1]


def test_merged_cells_mapped_to_rows_coordinates():
    table = _html_table(MERGED_HTML)
    assert table.merged_cells == [{"row": 0, "col": 0, "rowspan": 2, "colspan": 1}]
    assert table.section_rows == [] and table.sub_header_rows == []


def test_merged_cells_exclude_header_row():
    table = _html_table(HEADER_SPAN_HTML)
    assert table.merged_cells == []  # 表头行的 span 折算为 -1，不属于 rows


def test_two_column_table_is_key_value_and_uses_span_fact():
    """该行只有 2 格 → 性质③（要求列数 ≥3）覆盖不到，只能靠 colspan 事实判定。"""
    html = (
        "<table>"
        "<tr><td>项目</td><td>2024</td></tr>"
        '<tr><td colspan="2">利润表</td></tr>'
        "<tr><td>营业收入</td><td>1</td></tr>"
        "</table>"
    )
    table = _html_table(html)
    assert table.is_key_value is True  # 2 列表走「键值卡片」分支，不消费表头行
    assert table.section_rows == [1]


def test_section_rows_ignore_pure_numeric_label_in_markdown():
    grid = parse_markdown_table_grid(
        [
            "| 项目 | 2024 | 2025E |",
            "| --- | --- | --- |",
            "| 营业收入 | 1 | 2 |",
            "| 2023.07 |  |  |",
        ]
    )
    assert grid is not None
    table = build_table(0, "a.md", grid)  # markdown：无 colspan 事实
    assert table.section_rows == []


def test_section_rows_markdown_label_row():
    grid = parse_markdown_table_grid(
        [
            "| 项目 | 2024 | 2025E |",
            "| --- | --- | --- |",
            "| 利润表 |  |  |",
            "| 营业收入 | 1 | 2 |",
        ]
    )
    assert grid is not None
    table = build_table(0, "a.md", grid)
    assert table.section_rows == [0]


def test_key_value_table_has_no_structure_annotations():
    table = _html_table(KEY_VALUE_HTML)
    assert table.section_rows == [] and table.sub_header_rows == [] and table.merged_cells == []


def test_to_payload_exposes_structure_annotations():
    table = _html_table(MERGED_HTML)
    payload = table.to_payload()
    assert payload["table_section_rows"] == []
    assert payload["table_sub_header_rows"] == []
    assert payload["table_merged_cells"] == [{"row": 0, "col": 0, "rowspan": 2, "colspan": 1}]


def test_scan_tables_annotates_section_and_merge():
    prose, tables = scan_tables(SECTION_HTML, source_file="a.md")
    assert prose.strip() == ""
    assert len(tables) == 1
    assert tables[0].section_rows == [0]


def test_doc_level_fields_reads_stock_and_period():
    """§3.2：每 chunk 都带主体 / 报告期（文档级字段）。"""
    from data.splitter import doc_level_fields

    fields = doc_level_fields({
        "title": "凯莱英2025年三季度报点评：新兴业务维持高增长",
        "stockCode": "002821",
        "stockName": "凯莱英",
    })
    assert fields["stock_code"] == "002821"
    assert fields["stock_name"] == "凯莱英"
    assert fields["report_period"] == "2025Q3"
    assert fields["doc_type"] == "stock"


def test_doc_level_fields_does_not_infer_period_from_publish_year():
    """标题不含报告类型词时**不猜**报告期（`2024年…行业概览` 的年份是发布年，不是报告期）。"""
    from data.splitter import doc_level_fields

    fields = doc_level_fields({"title": "2024年中国备案中药材保健食品行业概览", "stockCode": "", "stockName": ""})
    assert fields["report_period"] is None
    assert fields["doc_type"] == "industry"


def test_structured_prose_chunks_carry_doc_level_metadata():
    docs = [{
        "content": "# 投资要点\n\n" + "公司业绩稳健增长，营业收入同比提升。" * 40,
        "metadata": {
            "filename": "a.md",
            "title": "凯莱英2025年三季度报点评：新兴业务维持高增长",
            "stockCode": "002821",
            "stockName": "凯莱英",
        },
    }]
    chunks = split_documents(docs, chunk_size=256, chunk_overlap=0, table_mode="structured")
    assert chunks
    for chunk in chunks:
        assert chunk["metadata"]["stock_code"] == "002821"
        assert chunk["metadata"]["stock_name"] == "凯莱英"
        assert chunk["metadata"]["report_period"] == "2025Q3"
        assert chunk["metadata"]["chunk_type"] == "prose"


def test_legacy_mode_has_no_doc_level_fields():
    """legacy 口径逐位复现基线：不得新增 `stock_code` / `doc_type` 等键。"""
    docs = [{
        "content": "# 投资要点\n\n" + "公司业绩稳健增长。" * 40,
        "metadata": {"filename": "a.md", "title": "凯莱英2025年三季度报点评"},
    }]
    chunks = split_documents(docs, chunk_size=256, chunk_overlap=0, table_mode="legacy")
    assert chunks
    assert all("stock_code" not in c["metadata"] and "chunk_type" not in c["metadata"] for c in chunks)


def test_to_slices_respects_char_budget():
    """**硬约束**：切片渲染长度必须 ≤ 字符上限（嵌入 API 单条 ≤2048 token，整表会超限）。"""
    rows = [["货币资金", "1,234,567,890.12", "9,876,543,210.98", "同比 +12.34%"] for _ in range(40)]
    table = build_table(0, "a.md", _grid(["项目", "2024A", "2025E", "同比"], rows))
    slices = table.to_slices(40, 600)
    assert len(slices) > 1
    assert all(len(s) <= 600 for s in slices)
    # 每片都重复表头，且总行数守恒（不丢行）
    assert all("| 项目 | 2024A | 2025E | 同比 |" in s for s in slices)
    assert sum(s.count("货币资金") for s in slices) == 40


def test_to_slices_keeps_rows_when_single_row_exceeds_budget():
    """单行自身超限时**仍单独成片**（保证至少 1 行/片，不丢行）。"""
    rows = [["超长行", "X" * 900]]
    table = build_table(0, "a.md", _grid(["项目", "内容"], rows))
    slices = table.to_slices(40, 300)
    assert len(slices) == 1
    assert "X" * 900 in slices[0]


def test_to_slices_returns_single_slice_when_small():
    table = build_table(0, "a.md", _grid(["项目", "值"], [["营业收入", "100"]]))
    assert len(table.to_slices(40, 1200)) == 1


def test_split_documents_structured_table_chunks_stay_within_char_budget():
    rows = "\n".join("| 营业收入{} | {} |".format(i, "9" * 60) for i in range(60))
    content = "| 指标 | 数值 |\n| --- | --- |\n" + rows
    docs = [{"content": content, "metadata": {"filename": "a.md"}}]
    chunks = split_documents(
        docs, chunk_size=512, chunk_overlap=0, table_mode="structured",
        table_slice_rows=40, table_slice_max_chars=600,
    )
    tables = [c for c in chunks if c["metadata"]["chunk_type"] == "table"]
    assert len(tables) > 3
    assert all(len(c["content"]) <= 600 for c in tables)
    assert sum(c["metadata"]["row_end"] - c["metadata"]["row_start"] for c in tables) == 60
