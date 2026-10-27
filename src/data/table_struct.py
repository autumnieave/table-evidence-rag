# -*- coding: utf-8 -*-
"""研报表格结构化（数据管道重构 · 第一步）。

问题
----
现状 ``splitter.py`` 把表格按「一行一个 chunk」入库：全库 57,178 个 chunk 中
**49,913 个（87.5%）是表格行**，平均仅 83.7 字，且只来自 **449 张表**。这些碎片行
（如 ``300181. SZ  佐力药业  0.72  2.73  3.71``）不含可回答内容，却因「公司名 + 一串数字」
与「公司 + 指标」类查询词面重合而劫持精排 top-10（B-59 实测 36/40）。

本模块把表格解析为**结构化对象**（:class:`Table`），一张表一个 chunk：

- **解析**：HTML ``<table>``（含 colspan / rowspan）与 Markdown 管道表 → 表头 + 二维行；
- **序列化**：有表头 → Markdown 表；**2 列键值卡片 → ``键: 值`` 行**（见下）；超长表按
  「表头重复 + 每 K 行」切片（:data:`TABLE_SLICE_ROWS`）；
- **元数据**：``table_id`` / ``row_index`` / ``stock_code`` / ``report_period`` /
  ``metric_name`` / ``unit``，供 L5-B 实体过滤、B-59 A 表行闸门、L1 数值核验共用；
- **兜底**：解析失败或异常结构保留 ``raw`` 并置 ``parse_ok=False``，**不丢内容**。

语料实测（2026-10-16，473 篇 / 3,343 张表，``训练结果数据/table_struct_smoke/diag.json``）
------------------------------------------------------------------------------------------
- **3,343 张表里 0 张带 ``<th>``**（表格由 PDF 解析器生成，表头也是 ``<td>``）；
- 因此表头只能按结构推断，实测规则为：
  1. 有 ``<th>`` → 首行即表头；
  2. 无 ``<th>`` 但首行是**跨满列的单格标题行**（如 ``<td colspan="6">盈利预测与估值</td>``）
     → 该行移入 ``caption``，**下一行才是表头**；
  3. 否则 → **首行即表头**（如 ``会计年度 | 2024 | 2025E | …``）；
  4. **特例：2 列表 = 键值卡片**（``总市值（百万元） | 11,349.65``）——没有表头行，
     首行就是数据，序列化为 ``键: 值``（若按规则 3 处理会**吃掉首行数据**）。
- 列数分布（前 120 篇抽样）：2 列 131 张、5/6/7 列 682 张、其余少量。

边界与取舍
----------
- ``unit`` 登记为**原始单位标签**（``万元`` / ``亿元`` …）并附 ``unit_factor``（折算为「元」的倍数），
  **不改写单元格文本**——静默乘数会在 L1 数值核验里制造「对不上的数」。
- 一张表**混用多种单位**时（键值卡片常见）``unit`` 置 ``None``，不猜。
- 表格序号（``index``）按「本文档内出现顺序」编列，跨文档不唯一；``table_id`` 才唯一。
- 纯逻辑、零外部依赖（``bs4`` 可选；缺失时退化为正则解析，不支持 colspan / rowspan）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - 与 B-59 A 共用同一份指标词表，避免两处漂移
    from filters.row_gate import METRIC_TERMS
except ImportError:  # pragma: no cover - filters 包不可用时退化为空词表
    METRIC_TERMS = ()

#: 单位 → 折算为「元」的倍数
UNIT_TO_YUAN: Dict[str, int] = {
    "元": 1,
    "千元": 10 ** 3,
    "万元": 10 ** 4,
    "十万元": 10 ** 5,
    "百万元": 10 ** 6,
    "千万元": 10 ** 7,
    "亿元": 10 ** 8,
    "十亿元": 10 ** 9,
}

#: 常见简写 → 规范单位名
UNIT_ALIASES: Dict[str, str] = {"万": "万元", "亿": "亿元", "百万": "百万元", "千": "千元"}

#: 表格类型（阶段 1 契约；用于分层覆盖率与下游用例）
TABLE_TYPES = (
    "financial_statement", "valuation", "forecast", "industry_summary",
    "ranking", "pipeline", "other",
)

#: 表格主体范围（阶段 1 契约）
TABLE_SCOPES = ("stock", "industry", "mixed", "unknown")

_PIPELINE_TERMS = ("管线", "靶点", "适应症", "临床", "获批", "药品名称", "药物", "疫苗", "器械", "上市申请")
_RANKING_TERMS = ("排名", "排序", "涨幅", "跌幅", "涨跌幅", "TOP", "前5", "后5", "涨幅前", "涨幅后")
_VALUATION_TERMS = ("估值", "PE", "P/E", "PB", "P/B", "市值", "目标价", "EPS", "股息率")
_FORECAST_TERMS = ("预测", "盈利预测", "2024E", "2025E", "2026E", "2027E", "2028E")
_STATEMENT_TITLE_TERMS = ("资产负债表", "利润表", "现金流量表", "财务摘要", "主要财务指标")
_FIN_METRIC_TERMS = ("营业收入", "净利润", "归母净利润", "总资产", "净资产", "负债合计",
                     "经营活动现金流", "每股收益", "EPS")
_FIN_STATEMENT_TERMS = _STATEMENT_TITLE_TERMS + _FIN_METRIC_TERMS
_INDUSTRY_SCOPE_TERMS = ("行业", "板块", "子板块", "行业汇总", "行业对比", "市场", "政策", "产业链", "景气度")

#: 单表切片行数上限：超过则「表头重复 + 每 K 行」切片，保留行列语义
TABLE_SLICE_ROWS: int = 40

#: 单表切片**字符**上限（硬约束）：嵌入 API（DashScope text-embedding-v2）单条输入 ≤ 2048 token，
#: 整表 Markdown 会超限（实测最长 13,239 字 / 702 块 > 2048 字）⇒ 切片同时受行数与字符数约束
TABLE_SLICE_MAX_CHARS: int = 1200

#: 结构化数据内联进 payload 的字符上限（超长时逐步截行；chunk 正文不受影响）
TABLE_JSON_MAX_CHARS: int = 6000

#: 空表头列的占位名（非首列）
_COL_PLACEHOLDER = "列{}"
#: 首列占位名（财务表里首列惯例是「项目 / 指标」这类标签列）
_ROW_LABEL_PLACEHOLDER = "项目"
#: 键值卡片的行内分隔符（2 列表）
_KV_SEPARATOR = ": "

#: ``300181.SZ`` / ``300181 . SZ``
_CODE_DOT_RE = re.compile(r"(?<!\d)(\d{6})\s*[.·。]?\s*(SZ|SH|BJ)(?![A-Za-z])", re.IGNORECASE)
#: ``SZ300181`` / ``SH. 600519``
_CODE_PRE_RE = re.compile(r"(SZ|SH|BJ)\s*[.·。]?\s*(\d{6})(?!\d)", re.IGNORECASE)

#: 年份前缀：4 位（20xx）或 2 位（``24`` / ``19`` → 2024 / 2019）；``(?<!\d)`` 防从长数字里截取
_YEAR_PAT = r"(?<!\d)((?:20\d{2}|[0-3]\d))"
#: 强模式（必须带年期词，**可 2 位年**）—— 报告期的「强证据」
_QUARTER_RE = re.compile(_YEAR_PAT + r"\s*年?\s*[Qq]([1-4])(?![0-9])")
_CN_QUARTER_RE = re.compile(_YEAR_PAT + r"\s*年?\s*(?:第)?([一二三四1234])\s*季度")
_HALF_RE = re.compile(_YEAR_PAT + r"\s*年?\s*[Hh]([12])(?![0-9])")
_Q3_CN_RE = re.compile(_YEAR_PAT + r"\s*年?\s*(?:三季报|前三季度)")
_HALF_CN_RE = re.compile(_YEAR_PAT + r"\s*年?\s*(?:中报|半年报|半年度|上半年)")
_Q1_CN_RE = re.compile(_YEAR_PAT + r"\s*年?\s*一季报")
_ANNUAL_RE = re.compile(_YEAR_PAT + r"\s*年?\s*(?:年度|年报|全年)")
#: 弱模式（裸年份，**仅 4 位**）：单独出现不足以判定报告期（可能是发布年）
_YEAR_CN_RE = re.compile(r"(20\d{2})\s*年(?!报)")

_CJK_DIGIT: Dict[str, str] = {"一": "1", "二": "2", "三": "3", "四": "4"}

#: 带括号的单位（``（万元）`` / ``(单位:亿元)``）；``（元/股）`` 这类复合单位不命中
_UNIT_PAREN_RE = re.compile(
    r"[（(【\[]\s*(?:单位\s*[:：]?)?\s*(?:人民币|港币|美元|RMB|CNY|元)?\s*"
    r"(十亿元|千万吨|千万元|百万元|十万元|千元|万元|亿元|万|亿|元)\s*[)）】\]]"
)
#: ``单位：万元`` / ``单位：人民币百万元`` 这类显式标注
_UNIT_LABEL_RE = re.compile(
    r"单位\s*[:：]\s*(?:人民币|港币|美元|RMB|CNY)?\s*(十亿元|千万元|百万元|十万元|千元|万元|亿元|万|亿|元)"
)

#: 表标题行（如「表1：2025Q3 主要财务指标」「图表 3 可比公司估值」）
_CAPTION_RE = re.compile(r"^\s*(?:表|图表|图)\s*[0-9一二三四五六七八九十]+\s*[:：.、]?\s*\S")
#: 表格来源 / 注释行（并入 caption，从正文剥离）
_NOTE_RE = re.compile(r"^\s*(?:资料来源|数据来源|注|来源)\s*[:：]")
#: 表下注释行（unit 回退链的 footnote 来源；比 caption 注释更宽，含「备注」）
_FOOTNOTE_RE = re.compile(r"^\s*(?:资料来源|数据来源|注|备注|来源)\s*[:：]")
#: 文本字符（汉字 / 拉丁字母）——用于把纯数字行排除在「分节标题行」之外
_TEXT_CHAR_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")


def _clean_cell(text: str) -> str:
    """单元格文本归一：去零宽字符、折叠空白、去首尾。"""
    s = (text or "").replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", s).strip()


def header_placeholder(index: int) -> str:
    """空表头列的占位名：首列用「项目」（标签列惯例），其余用「列N」。"""
    return _ROW_LABEL_PLACEHOLDER if index == 0 else _COL_PLACEHOLDER.format(index + 1)


def _has_text_char(text: str) -> bool:
    """是否含汉字或拉丁字母（区分「利润表」类标题行与「2023.07」类纯数字行）。"""
    return bool(_TEXT_CHAR_RE.search(text or ""))


def normalize_unit(label: Optional[str]) -> Optional[str]:
    """单位写法归一为规范标签（``万`` → ``万元``，``亿`` → ``亿元``）；无法识别返回 ``None``。"""
    if not label:
        return None
    key = _clean_cell(label).replace(" ", "")
    if key in UNIT_TO_YUAN:
        return key
    return UNIT_ALIASES.get(key)


def unit_factor(label: Optional[str]) -> int:
    """单位折算为「元」的倍数；无法识别时返回 ``1``。"""
    normalized = normalize_unit(label)
    return UNIT_TO_YUAN.get(normalized, 1) if normalized else 1


def detect_unit(text: str) -> Tuple[Optional[str], int]:
    """从标题 / 表头里识别单位，返回 ``(规范单位, 折算为元的倍数)``；识别不到返回 ``(None, 1)``。"""
    if not text:
        return None, 1
    match = _UNIT_PAREN_RE.search(text) or _UNIT_LABEL_RE.search(text)
    if not match:
        return None, 1
    normalized = normalize_unit(match.group(1))
    if normalized is None:
        return None, 1
    return normalized, UNIT_TO_YUAN.get(normalized, 1)


def detect_units(text: str, limit: int = 5) -> List[str]:
    """列出文本里出现的全部单位（去重、保持顺序），用于判断「一张表是否混用多单位」。"""
    if not text:
        return []
    out: List[str] = []
    for pattern in (_UNIT_PAREN_RE, _UNIT_LABEL_RE):
        for match in pattern.finditer(text):
            label = normalize_unit(match.group(1))
            if label and label not in out:
                out.append(label)
                if len(out) >= limit:
                    return out
    return out


def extract_stock_codes(text: str, limit: int = 20) -> List[str]:
    """抽取证券代码并统一为 ``300181.SZ``；去重、保持出现顺序。"""
    if not text:
        return []
    found: List[str] = []
    for match in _CODE_DOT_RE.finditer(text):
        code = "{}.{}".format(match.group(1), match.group(2).upper())
        if code not in found:
            found.append(code)
        if len(found) >= limit:
            return found
    for match in _CODE_PRE_RE.finditer(text):
        code = "{}.{}".format(match.group(2), match.group(1).upper())
        if code not in found:
            found.append(code)
        if len(found) >= limit:
            break
    return found


def normalize_year(year: str) -> str:
    """年份归一：2 位补世纪（``24`` → ``2024``，``98`` → ``1998``）。"""
    if len(year) == 4:
        return year
    value = int(year)
    return str(2000 + value if value < 50 else 1900 + value)


def extract_report_period_strict(text: str) -> Optional[str]:
    """**强口径**报告期：必须带年期词（``Q3`` / ``三季报`` / ``半年度`` / ``年报``…），**接受 2 位年**。

    与 :func:`extract_report_period` 的区别：**不接受光秃秃的年份**（``2024年`` 可能是发布年），
    因此可安全用于「正文里找报告期」这种容易误判的位置。返回 ``2025Q3`` / ``2025H1`` / ``2024A``。
    """
    if not text:
        return None
    match = _QUARTER_RE.search(text)
    if match:
        return "{}Q{}".format(normalize_year(match.group(1)), match.group(2))
    match = _CN_QUARTER_RE.search(text)
    if match:
        digit = _CJK_DIGIT.get(match.group(2), match.group(2))
        return "{}Q{}".format(normalize_year(match.group(1)), digit)
    match = _HALF_RE.search(text)
    if match:
        return "{}H{}".format(normalize_year(match.group(1)), match.group(2))
    match = _Q3_CN_RE.search(text)
    if match:
        return "{}Q3".format(normalize_year(match.group(1)))
    match = _HALF_CN_RE.search(text)
    if match:
        return "{}H1".format(normalize_year(match.group(1)))
    match = _Q1_CN_RE.search(text)
    if match:
        return "{}Q1".format(normalize_year(match.group(1)))
    match = _ANNUAL_RE.search(text)
    if match:
        return "{}A".format(normalize_year(match.group(1)))
    return None


def extract_report_period(text: str) -> Optional[str]:
    """识别报告期（**宽松口径**）：强口径 + 裸年份兜底（``2024年`` → ``2024A``）。

    裸年份兜底只应用在**有报告类型词佐证**的位置（标题）；正文里请用
    :func:`extract_report_period_strict`，否则会把发布年 / 叙述年份当成报告期。
    """
    period = extract_report_period_strict(text)
    if period:
        return period
    match = _YEAR_CN_RE.search(text or "")
    if match:
        return "{}A".format(match.group(1))
    return None


def extract_metric_names(headers: Sequence[str], limit: int = 20) -> List[str]:
    """从表头 / 首列里挑出指标名（含 :data:`METRIC_TERMS` 任一即算），保持顺序、去重。"""
    out: List[str] = []
    for header in headers or []:
        text = _clean_cell(header)
        if text and any(term in text for term in METRIC_TERMS) and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out

@dataclass
class TableGrid:
    """解析出的表格网格 + 表头判定（:func:`parse_html_table_grid` / :func:`parse_markdown_table_grid` 的返回）。"""

    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""
    has_header: bool = True
    """``headers`` 是否来自真实表头行；``False`` 表示用的是「项目 / 列N」占位名。"""
    is_key_value: bool = False
    """2 列键值卡片（如 ``总市值（百万元） | 11,349.65``）：**没有表头行**，按 ``键: 值`` 序列化。"""
    raw_spans: List[Dict[str, int]] = field(default_factory=list)
    """源网格里的合并单元格（``{"row": 源网格行号, "col": 列号, "rowspan": n, "colspan": m}``）。"""
    consumed_rows: int = 0
    """从源网格**剥掉的前置行数**（跨列标题行 + 表头行）——用于把 ``raw_spans`` 折算到 ``rows`` 坐标。"""

    @property
    def col_count(self) -> int:
        """列数。"""
        return len(self.headers)

    def to_dict(self) -> Dict[str, Any]:
        """调试 / 落盘用的字典视图。"""
        return {
            "caption": self.caption,
            "has_header": self.has_header,
            "is_key_value": self.is_key_value,
            "headers": list(self.headers),
            "rows": [list(row) for row in self.rows],
        }


def _fit_cells(cells: List[str], width: int) -> List[str]:
    """把一行单元格补齐 / 截断到 ``width`` 列。"""
    row = list(cells[:width])
    if len(row) < width:
        row.extend([""] * (width - len(row)))
    return row


def _is_md_row(line: str) -> bool:
    """判断一行是否为 Markdown 管道表行（``| a | b |``）。"""
    stripped = (line or "").strip()
    return len(stripped) >= 2 and stripped.startswith("|") and stripped.endswith("|")


def _split_md_row(line: str) -> List[str]:
    """拆 Markdown 表行；支持 ``\\|`` 转义。"""
    s = (line or "").strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    parts = re.split(r"(?<!\\)\|", s)
    return [_clean_cell(p.replace("\\|", "|")) for p in parts]


def _is_md_separator(line: str) -> bool:
    """判断是否为 Markdown 表头分隔行（``| --- | :--: |``）。"""
    s = (line or "").strip()
    if not s:
        return False
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = [c.strip() for c in s.split("|")]
    if not cells:
        return False
    meaningful = [c for c in cells if c != ""]
    if not meaningful:
        return False
    return all(re.fullmatch(r":?-{2,}:?", c) for c in meaningful) and len(meaningful) == len(cells)


def _pop_span_title_row(grid: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """剥掉「跨满列的单格标题行」（``<td colspan=6>盈利预测与估值</td>``）。

    返回 ``(标题, 剩余行)``；标题行可连续多行（如「表 3-A」+「盈利预测」）。
    仅在列数 ≥3 时判定，避免把 2 列卡片里内容雷同的数据行误当标题。
    """
    titles: List[str] = []
    rows = grid
    while len(rows) >= 2:
        first = rows[0]
        nonempty = [cell for cell in first if cell]
        if len(first) >= 3 and len(nonempty) >= 2 and len(set(nonempty)) == 1:
            titles.append(nonempty[0])
            rows = rows[1:]
            continue
        break
    return " ".join(titles), rows


def _finalize_grid(
    grid: List[List[str]],
    caption: str = "",
    has_th: bool = False,
    spans: Optional[List[Dict[str, int]]] = None,
) -> Optional[TableGrid]:
    """把二维网格收敛为 :class:`TableGrid`（统一补齐列宽 + 推断表头）。

    判定顺序见模块 docstring「语料实测」一节。
    """
    if not grid:
        return None
    width = max(len(row) for row in grid)
    if width <= 0:
        return None
    grid = [_fit_cells(row, width) for row in grid if any(row)]
    if not grid:
        return None

    if has_th:
        if len(grid) < 2:
            return None
        return TableGrid(
            headers=[_clean_cell(c) or header_placeholder(i) for i, c in enumerate(grid[0])],
            rows=grid[1:],
            caption=caption,
            has_header=True,
            raw_spans=list(spans or []),
            consumed_rows=1,
        )

    rows_before_pop = len(grid)
    title, grid = _pop_span_title_row(grid)
    title_rows = rows_before_pop - len(grid)
    caption = " ".join(part for part in (caption, title) if part).strip()

    if width <= 2:  # 键值卡片：首行就是数据，不能当表头
        if not grid:
            return None
        return TableGrid(
            headers=[header_placeholder(i) for i in range(width)],
            rows=grid,
            caption=caption,
            has_header=False,
            is_key_value=width == 2,
            raw_spans=list(spans or []),
            consumed_rows=title_rows,
        )

    if len(grid) >= 2:
        header_row = grid[0]
        nonempty = [cell for cell in header_row if cell]
        if len(nonempty) >= 2:  # 首行是表头（「会计年度 | 2024 | 2025E | …」）
            return TableGrid(
                headers=[_clean_cell(c) or header_placeholder(i) for i, c in enumerate(header_row)],
                rows=grid[1:],
                caption=caption,
                has_header=True,
                raw_spans=list(spans or []),
                consumed_rows=title_rows + 1,
            )

    return TableGrid(
        headers=[header_placeholder(i) for i in range(width)],
        rows=grid,
        caption=caption,
        has_header=False,
        raw_spans=list(spans or []),
        consumed_rows=title_rows,
    )


def parse_markdown_table_grid(lines: Sequence[str]) -> Optional[TableGrid]:
    """解析 Markdown 管道表为 :class:`TableGrid`；不可解析返回 ``None``。

    - 第 2 行是分隔行 → 第 1 行作表头、第 3 行起为数据；
    - 无分隔行 → 第 1 行作表头、第 2 行起为数据；
    - 单元格数不齐的行按表头列数补齐 / 截断；
    - 无数据行时返回 ``None``（调用方应保留原文，避免丢内容）。
    """
    lines = [ln for ln in (lines or []) if _is_md_row(ln)]
    if len(lines) < 2:
        return None
    headers = _split_md_row(lines[0])
    if not headers or not any(headers):
        return None
    data_lines = lines[2:] if _is_md_separator(lines[1]) else lines[1:]
    rows: List[List[str]] = []
    for line in data_lines:
        cells = _split_md_row(line)
        if not any(cells):
            continue
        rows.append(_fit_cells(cells, len(headers)))
    if not rows:
        return None
    return TableGrid(headers=list(headers), rows=rows, caption="", has_header=True, consumed_rows=1)


def _parse_html_table_grid_regex(html_table: str) -> Optional[TableGrid]:
    """无 bs4 时的兜底 HTML 解析（不支持 colspan / rowspan）。"""
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
    tag_pattern = re.compile(r"<[^>]+>")
    trs = row_pattern.findall(html_table or "")
    grid: List[List[str]] = []
    has_th = False
    for tr in trs:
        if re.search(r"<th[^>]*>", tr, re.IGNORECASE):
            has_th = True
        cells = [_clean_cell(tag_pattern.sub(" ", cell)) for cell in cell_pattern.findall(tr)]
        if any(cells):
            grid.append(cells)
    return _finalize_grid(grid, has_th=has_th)


def parse_html_table_grid(html_table: str) -> Optional[TableGrid]:
    """解析 HTML 表格为 :class:`TableGrid`；不可解析返回 ``None``。

    - ``colspan``：按列数复制该单元格文本占位；``rowspan``：向后续行填充；
    - 表头判定（语料实测，见模块 docstring）：``<th>`` → 首行即表头；
      **跨满列的单格标题行** → 移入 ``caption``，下一行作表头；
      **2 列 → 键值卡片（无表头）**；其余 → 首行即表头；
    - 解析不出数据行时返回 ``None``（调用方应保留原文）。
    """
    if not (html_table or "").strip():
        return None
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _parse_html_table_grid_regex(html_table)

    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    if table is None:
        return _parse_html_table_grid_regex(html_table)

    grid: List[List[str]] = []
    spans: List[Dict[str, int]] = []
    pending: Dict[int, List[Any]] = {}  # 列号 -> [文本, 还需向下填充的行数]
    has_th = False

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        if any(cell.name == "th" for cell in cells):
            has_th = True
        row: List[str] = []
        row_index = len(grid)
        row_spans: List[Dict[str, int]] = []
        col = 0
        for cell in cells:
            while col in pending:  # 先补齐上方 rowspan 遗留
                text, left = pending[col][0], pending[col][1]
                row.append(text)
                if left <= 1:
                    pending.pop(col, None)
                else:
                    pending[col][1] = left - 1
                col += 1
            text = _clean_cell(cell.get_text(" ", strip=True))
            try:
                colspan = max(1, int(cell.get("colspan") or 1))
            except (TypeError, ValueError):
                colspan = 1
            try:
                rowspan = max(1, int(cell.get("rowspan") or 1))
            except (TypeError, ValueError):
                rowspan = 1
            if colspan > 1 or rowspan > 1:
                row_spans.append(
                    {"row": row_index, "col": col, "rowspan": rowspan, "colspan": colspan}
                )
            for offset in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col + offset] = [text, rowspan - 1]
                col += 1
        while col in pending:  # 行尾再补一次
            text, left = pending[col][0], pending[col][1]
            row.append(text)
            if left <= 1:
                pending.pop(col, None)
            else:
                pending[col][1] = left - 1
            col += 1
        if any(row):
            grid.append(row)
            spans.extend(row_spans)

    return _finalize_grid(grid, has_th=has_th, spans=spans)

@dataclass
class Table:
    """一张结构化表格（结构化集合里「一张表 = 一个 / 若干 chunk」的来源）。"""

    table_id: str
    source_file: str = ""
    index: int = 0
    title: str = ""
    """所属研报标题（作为 chunk 正文的上下文前缀；不是表标题）。"""
    caption: str = ""
    """表标题（来自 colspan 标题行 / 「表N：…」行 / 资料来源注释行）。"""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    has_header: bool = True
    """``headers`` 是否来自真实表头行（``False`` = 「项目 / 列N」占位名）。"""
    is_key_value: bool = False
    """2 列键值卡片：无表头行，按 ``键: 值`` 序列化。"""
    section_rows: List[int] = field(default_factory=list)
    """**全宽标签行**在 ``rows`` 中的下标：整行只承载 1 段文本，**不是数据行**。

    含分节标题（``利润表`` / ``成长能力`` / ``NCCN指南 (Version 3.2025)``）、说明与脚注
    （``备注:评级标准…`` / ``资料来源：Wind``）。下游逐行取数时应跳过，否则会把标题当指标。

    判定见 :func:`_detect_section_rows`；对全库 473 篇 / 2,990 张 HTML 表做过源 HTML 客观校验
    （precision 98.5% / recall 99.8%，口径与残差见 ``docs/评估报告/数据管道重构方案_20261016.md``）。
    """
    sub_header_rows: List[int] = field(default_factory=list)
    """**次级表头行**在 ``rows`` 中的下标：① 整行逐格等于表头行（跨子表重复表头）；
    ② 紧跟分节标题行且「空列位置」与表头行一致（多级表头）。同样**不是数据行**。"""
    merged_cells: List[Dict[str, int]] = field(default_factory=list)
    """合并单元格：``{"row": ``rows`` 下标, "col": 列号, "rowspan": n, "colspan": m}``。

    仅含落在 ``rows`` 内的 span（表头行自带的合并折算为 -1，不输出）；列表页的跨列合并即分节行。
    """
    unit: Optional[str] = None
    unit_factor: int = 1
    unit_source: str = "none"
    """unit 来源：``cell`` / ``caption`` / ``footnote`` / ``none`` / ``multi``。"""
    unit_candidates: List[str] = field(default_factory=list)
    """表内出现过的全部单位；**恰好 1 个**时才写进 ``unit``，多单位置 ``None``。"""
    table_type: str = "other"
    """表格类型：financial_statement / valuation / forecast / industry_summary / ranking / pipeline / other。"""
    table_scope: str = "unknown"
    """表格主体范围：stock / industry / mixed / unknown。"""
    stock_codes: List[str] = field(default_factory=list)
    report_period: Optional[str] = None
    metric_names: List[str] = field(default_factory=list)
    table_format: str = "markdown"
    parse_ok: bool = True
    raw: str = ""

    @property
    def row_count(self) -> int:
        """数据行数（不含表头）。"""
        return len(self.rows)

    @property
    def col_count(self) -> int:
        """列数。"""
        return len(self.headers)

    def caption_text(self) -> str:
        """chunk 正文的标题行：表标题 + 研报标题（上下文）+ 单位标注。"""
        parts: List[str] = []
        for text in (_clean_cell(self.caption), _clean_cell(self.title)):
            if text and text not in parts:
                parts.append(text)
        head = " ".join(parts)
        if self.unit and self.unit not in head:
            head = "{}\uff08\u5355\u4f4d\uff1a{}\uff09".format(head, self.unit)
        return head

    @staticmethod
    def _escape(text: str) -> str:
        """Markdown 单元格转义（竖线会破坏表结构）。"""
        return _clean_cell(text).replace("|", "\\|")

    def to_text(self, row_start: int = 0, row_end: Optional[int] = None) -> str:
        """序列化为 chunk 正文。

        - 键值卡片（``is_key_value``）→ 每行 ``键: 值``；
        - 有表头 → Markdown 表（表头 + 分隔行 + 数据行）；
        - 无表头且非卡片 → Markdown 数据行（不造假的表头行）。
        """
        lines: List[str] = []
        head = self.caption_text()
        if head:
            lines.append(head)
        rows = self.rows[row_start:row_end]
        if self.is_key_value:
            for row in rows:
                cells = _fit_cells(list(row), 2)
                value = cells[1].strip()
                lines.append("{}: {}".format(cells[0], value) if value else cells[0])
            return "\n".join(lines)
        if self.has_header and self.headers:
            lines.append("| " + " | ".join(self._escape(h) for h in self.headers) + " |")
            lines.append("| " + " | ".join("---" for _ in self.headers) + " |")
        for row in rows:
            cells = _fit_cells([_clean_cell(c) for c in row], max(1, self.col_count))
            lines.append("| " + " | ".join(self._escape(c) for c in cells) + " |")
        return "\n".join(lines)

    def slice_ranges(
        self, max_rows: int = TABLE_SLICE_ROWS, max_chars: int = TABLE_SLICE_MAX_CHARS
    ) -> List[Tuple[int, int]]:
        """切片的**行区间** ``[(row_start, row_end), ...]``（``rows`` 下标，前闭后开）。

        与 :meth:`to_slices` 同源——**元数据里的 ``row_start`` / ``row_end`` 必须用这里的真实区间**，
        不能按行数步长估算（字符上限会让每片行数不等，估算会错位）。
        """
        total = len(self.rows)
        if total == 0:
            return []
        row_cap = max_rows if max_rows and max_rows > 0 else total
        if row_cap >= total and (max_chars <= 0 or len(self.to_text()) <= max_chars):
            return [(0, total)]
        out: List[Tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(start + row_cap, total)
            if max_chars > 0:
                low, high = start + 1, end
                while low < high:  # 二分：找「渲染长度 ≤ max_chars」的最大行数
                    mid = (low + high + 1) // 2
                    if len(self.to_text(start, mid)) <= max_chars:
                        low = mid
                    else:
                        high = mid - 1
                end = low
            out.append((start, end))
            start = end
        return out

    def to_slices(
        self, max_rows: int = TABLE_SLICE_ROWS, max_chars: int = TABLE_SLICE_MAX_CHARS
    ) -> List[str]:
        """按「表头重复 + 行数 / 字符数」双上限切片；都不过上限时返回单块。

        Args:
            max_rows: 单切片行数上限（0 = 不限行数）
            max_chars: 单切片**渲染后字符数**上限（0 = 不限；默认 :data:`TABLE_SLICE_MAX_CHARS`）

        Note:
            **字符上限是硬约束**——嵌入 API 单条输入 ≤ 2048 token，整表 Markdown（实测最长 13,239 字）
            会超限报 400。行数上限保留原设计（切片可读性）。单行自身超限时仍单独成片
            （**保证每片至少 1 行，不丢行**）。行区间见 :meth:`slice_ranges`。
        """
        return [self.to_text(start, end) for start, end in self.slice_ranges(max_rows, max_chars)]

    def to_payload(self, max_chars: int = TABLE_JSON_MAX_CHARS) -> Dict[str, Any]:
        """结构化数据的 payload 片段。

        ``table_json`` 是 ``{"headers": [...], "rows": [[...]]}`` 的 JSON 字符串；
        超过 ``max_chars`` 时逐步截行并置 ``table_truncated=True``（**chunk 正文不受影响**）。
        """
        rows = [list(row) for row in self.rows]
        truncated = False
        blob = ""
        while True:
            blob = json.dumps({"headers": self.headers, "rows": rows}, ensure_ascii=False)
            if len(blob) <= max_chars:
                break
            if len(rows) <= 1:  # 无法再截，保留最后一行并打标
                truncated = True
                break
            keep = int(len(rows) * max_chars / len(blob)) - 1
            keep = min(max(1, keep), len(rows) - 1)
            rows = rows[:keep]
            truncated = True
        return {
            "table_id": self.table_id,
            "table_json": blob,
            "table_truncated": truncated,
            "table_row_count": self.row_count,
            "table_col_count": self.col_count,
            "table_has_header": self.has_header,
            "table_is_key_value": self.is_key_value,
            "table_type": self.table_type,
            "table_scope": self.table_scope,
            "table_unit_source": self.unit_source,
            "table_unit_candidates": list(self.unit_candidates),
            "table_section_rows": list(self.section_rows),
            "table_sub_header_rows": list(self.sub_header_rows),
            "table_merged_cells": [dict(cell) for cell in self.merged_cells],
        }


def _detect_section_rows(
    rows: List[List[str]],
    headers: List[str],
    has_header: bool,
    spans: Optional[List[Dict[str, int]]] = None,
    width: int = 0,
    span_facts: bool = False,
) -> List[int]:
    """检测**分节标题行**在 ``rows`` 中的下标：整行只承载 **1 个语义单元**。

    信号按可靠性排序：

    ① **结构信号（精确）**：源 ``<td colspan == 列宽>`` 独占整行——如
       ``<td colspan="7">利润表</td>``、``<td colspan="3">备注:评级标准…</td>``；
    ② **启发式（兜底）**：整行只剩 1 个非空格（``资本结构 |  |  |``；不限列位置，
       因为源可能把它写在中间的 ``<td>`` 里），且该文本含汉字 / 拉丁字母
       （排除 ``2023.07`` 这类纯数字行）；
    ③ **启发式（兜底）**：列数 ≥3、**≥2 个非空格且文本全同**（跨列合并展开后的重复文本，
       含「近满宽」形如 ``<td colspan="7">益丰药房</td><td></td>`` 的 8 列表）。
       要求文本含汉字 / 拉丁字母——``是 | 是 | 是``、``- | - | -`` 这类短符号数据行因此被排除。

    不带「后面必须有数据行」约束——表格末尾的 ``备注`` / ``资料来源`` 行同样是全宽标签行。
    """
    if not rows:
        return []
    col_count = width or len(headers)
    structural = {
        span["row"]
        for span in (spans or [])
        if col_count
        and span.get("col", 0) == 0
        and span.get("colspan", 1) >= col_count
        and 0 <= span["row"] < len(rows)
    }
    found: List[int] = []
    for index, row in enumerate(rows):
        if index in structural:
            continue  # 已由结构信号判定
        cells = [_clean_cell(c) for c in row]
        nonempty = [c for c in cells if c]
        if not nonempty:
            continue
        is_label = len(nonempty) == 1 and _has_text_char(nonempty[0])
        is_span = (
            len(cells) >= 3
            and len(nonempty) >= 2
            and len(set(nonempty)) == 1
            and _has_text_char(nonempty[0])
        )
        if is_label or is_span:
            found.append(index)
    return sorted(structural | set(found))


def _detect_sub_header_rows(
    rows: List[List[str]], headers: List[str], section_rows: List[int], has_header: bool
) -> List[int]:
    """检测**次级表头行**在 ``rows`` 中的下标。两种形态：

    ① **跨子表重复表头**：整行逐格等于表头行——把一张表按「利润表 / 资产负债表 / 现金流量表」切成
       三段时，每段都重抄一遍 ``会计年度 | 2022 | … | 2027E``；
    ② **多级表头**：紧跟分节标题行，且「空列位置」与表头行一致（``NCCN指南`` 行之后的 ``|  | 首选 | 其他推荐``）。

    两类行都**不是数据行**：下游逐行取数时必须跳过，否则会把 ``2027E`` 之类当成年份数值。
    """
    if not has_header or not headers:
        return []
    width = len(headers)
    head = [_clean_cell(h) for h in headers]
    head_norm = ["" if t == header_placeholder(i) else t for i, t in enumerate(head)]
    pattern = [bool(t) for t in head_norm]
    use_pattern = not (all(pattern) or not any(pattern))  # 表头行无空列特征时不做形态②判定
    found: List[int] = []
    for index, row in enumerate(rows):
        cells = _fit_cells([_clean_cell(c) for c in row], width)
        if any(cells) and cells == head_norm:  # ① 逐格等于表头行
            found.append(index)
            continue
        if use_pattern and index - 1 in section_rows and [bool(c) for c in cells] == pattern:  # ②
            found.append(index)
    return found


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    """大小写不敏感的任一词命中。"""
    upper = (text or "").upper()
    return any(term.upper() in upper for term in terms)


def _is_footnote_row(row: Sequence[str]) -> bool:
    """是否表下注释行（unit 的 footnote 来源）。"""
    for cell in list(row)[:2]:
        if _FOOTNOTE_RE.match(_clean_cell(cell)):
            return True
    return False


def _resolve_unit_sources(
    cell_text: str, caption_text: str, footnote_text: str
) -> Tuple[List[str], Optional[str], str]:
    """按 cell -> caption -> footnote 解析单位，返回 ``(candidates, unit, source)``。

    任一层出现多个不同货币单位 → 整体 ``multi``，不猜、不取众数。
    """
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
    if not candidates:
        return [], None, "none"
    if len(candidates) > 1:
        return candidates, None, "multi"
    unit = candidates[0]
    for name, units in sources:
        if unit in units:
            return candidates, unit, name
    return candidates, unit, "none"


def classify_table_type(
    caption: str,
    title: str,
    headers: Sequence[str],
    first_column: str,
    metric_names: Sequence[str],
    doc_type: str = "",
) -> str:
    """表类型启发式（阶段 1 契约；先规则、后人工抽检校准）。"""
    text = " ".join(part for part in (
        caption or "", " ".join(headers or []), first_column or "",
        " ".join(metric_names or []),
    ) if part)
    if _contains_any(text, _STATEMENT_TITLE_TERMS):
        return "financial_statement"
    if _contains_any(text, _PIPELINE_TERMS):
        return "pipeline"
    if _contains_any(text, _RANKING_TERMS):
        return "ranking"
    # 预测标记（2025E/盈利预测）优先于 EPS/PE：预测表常同时带估值列。
    if _contains_any(text, _FORECAST_TERMS):
        return "forecast"
    # 明确写「估值」的标题优先于财务指标：可比公司估值表常带营业收入列。
    if "估值" in (caption or "") and not _contains_any(text, _FORECAST_TERMS):
        return "valuation"
    if _contains_any(text, _FIN_METRIC_TERMS):
        return "financial_statement"
    if _contains_any(text, _VALUATION_TERMS):
        return "valuation"
    if "评级" in text and "说明" in text and not _contains_any(text, _VALUATION_TERMS + _FIN_STATEMENT_TERMS):
        return "other"
    if doc_type == "industry" or _contains_any(text, _INDUSTRY_SCOPE_TERMS):
        return "industry_summary"
    return "other"


def classify_table_scope(doc_type: str, stock_codes: Sequence[str], text: str) -> str:
    """表主体范围启发式：表级证据优先，``doc_type`` 只作先验。"""
    n_codes = len(stock_codes or [])
    industry_hit = _contains_any(text, _INDUSTRY_SCOPE_TERMS)
    if doc_type == "stock":
        if n_codes > 1 or (industry_hit and n_codes == 0):
            return "mixed"
        return "stock"
    if doc_type == "industry":
        if n_codes == 1 and not industry_hit:
            return "stock"
        return "industry"
    if n_codes == 1 and not industry_hit:
        return "stock"
    if industry_hit or n_codes >= 2:
        return "industry"
    return "unknown"


def build_table(
    index: int,
    source_file: str,
    grid: TableGrid,
    title: str = "",
    caption: str = "",
    table_format: str = "markdown",
    parse_ok: bool = True,
    raw: str = "",
    doc_type: str = "",
    unit_mode: str = "fallback",
) -> Table:
    """由 :class:`TableGrid` 构造 :class:`Table`，并推断单位 / 主体 / 报告期 / 指标列。

    元数据抽取**分两级**：先看「表标题 / 研报标题 / 表头」，取不到再退回「首列 / 前 20 行」——
    语料里指标名通常写在**首列**（``营业收入(百万元)``），而不是列头（列头是年份）。
    """
    headers = [
        _clean_cell(header) or header_placeholder(i) for i, header in enumerate(grid.headers or [])
    ]
    width = max(1, len(headers))
    rows = [[_clean_cell(cell) for cell in _fit_cells(list(row), width)] for row in (grid.rows or [])]
    title_text = _clean_cell(title)
    caption_text = " ".join(
        part for part in (_clean_cell(caption), _clean_cell(grid.caption)) if part
    )
    head_text = " ".join(
        part
        for part in (caption_text, title_text, " ".join(headers) if grid.has_header else "")
        if part
    )
    first_column = " ".join(row[0] for row in rows[:40] if row and row[0])
    core_rows = [row for row in rows[:40] if not _is_footnote_row(row)]
    first_column_core = " ".join(row[0] for row in core_rows if row and row[0])
    body_text = " ".join(" ".join(row) for row in rows[:20])

    # 全表文本（含正文串入的表）；作为元数据扫描的最后一级来源

    if unit_mode == "legacy":
        # 仅供已归档审计脚本复现旧 65.0% 口径；生产默认走 fallback。
        candidates = detect_units(head_text) or detect_units(first_column) or detect_units(body_text)
        unit = candidates[0] if len(candidates) == 1 else None
        unit_source = "cell" if unit else ("multi" if len(candidates) > 1 else "none")
    else:
        # 回退链：cell -> caption/title -> footnote；多来源冲突一律 multi，不猜。
        cell_rows = [row for row in rows[:40] if not _is_footnote_row(row)]
        footnote_rows = [row for row in rows if _is_footnote_row(row)]
        cell_text = " ".join(headers + [" ".join(row) for row in cell_rows])
        caption_unit_text = " ".join(part for part in (caption_text, title_text) if part)
        footnote_text = " ".join(" ".join(row) for row in footnote_rows)
        candidates, unit, unit_source = _resolve_unit_sources(
            cell_text, caption_unit_text, footnote_text
        )
    factor = UNIT_TO_YUAN.get(unit, 1) if unit else 1

    period = (
        extract_report_period(head_text)
        or extract_report_period(first_column)
        or extract_report_period(body_text)
    )
    codes = (
        extract_stock_codes(head_text)
        or extract_stock_codes(first_column)
        or extract_stock_codes(body_text)
    )
    metric_names = extract_metric_names(headers)
    for name in extract_metric_names([row[0] for row in rows if row]):
        if name not in metric_names:
            metric_names.append(name)
    if not metric_names:
        # 兜底：表内任意**单元格**里出现的指标词（首列不是指标名的那类表）。
        # 这里只收「命中的词」而不是整格文本，避免把数值一起塞进指标名（如「营业收入 100」）。
        for row in rows:
            for cell in row:
                text = _clean_cell(cell)
                if not text:
                    continue
                for term in METRIC_TERMS:
                    if term in text and term not in metric_names:
                        metric_names.append(term)

    # 合并单元格：把源网格坐标折算到 ``rows`` 坐标（``raw_row - consumed_rows``；表头行为 -1）。
    merged_cells = [
        {
            "row": span["row"] - grid.consumed_rows,
            "col": span["col"],
            "rowspan": span["rowspan"],
            "colspan": span["colspan"],
        }
        for span in (grid.raw_spans or [])
        if span["row"] - grid.consumed_rows >= 0
    ]
    section_rows = _detect_section_rows(
        rows,
        headers,
        grid.has_header,
        spans=merged_cells,
        width=width,
        span_facts=table_format == "html",
    )
    sub_header_rows = _detect_sub_header_rows(rows, headers, section_rows, grid.has_header)
    table_type = classify_table_type(
        caption=caption_text, title=title_text, headers=headers,
        first_column=first_column_core, metric_names=metric_names, doc_type=doc_type,
    )
    scope_text = " ".join(part for part in (caption_text, " ".join(headers), first_column_core) if part)
    table_scope = classify_table_scope(doc_type, codes, scope_text)

    return Table(
        table_id="{}#t{}".format(source_file or "unknown", index + 1),
        source_file=source_file,
        index=index,
        title=title_text,
        caption=caption_text,
        headers=headers,
        rows=rows,
        has_header=grid.has_header,
        is_key_value=grid.is_key_value,
        section_rows=section_rows,
        sub_header_rows=sub_header_rows,
        merged_cells=merged_cells,
        unit=unit,
        unit_factor=factor,
        unit_source=unit_source,
        unit_candidates=candidates,
        table_type=table_type,
        table_scope=table_scope,
        stock_codes=codes,
        report_period=period,
        metric_names=metric_names,
        table_format=table_format,
        parse_ok=parse_ok,
        raw=raw,
    )


_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"^[ \t]*\[TABLE_PLACEHOLDER_\d+\][ \t]*$", re.MULTILINE)


def scan_tables(content: str, source_file: str = "", title: str = "", doc_type: str = "",
                unit_mode: str = "fallback") -> Tuple[str, List[Table]]:
    """扫描正文，抽出全部表格，返回 ``(剥离表格后的正文, Table 列表)``。

    规则：

    - HTML 表格解析成功 → 从正文删除；**解析失败 → 原样保留**（宁可留噪声，不丢内容）；
    - Markdown 管道表 → 抽为 :class:`Table`；紧邻的「表N：…」标题行与
      「资料来源：…」注释行（最多 3 行）并入 ``caption``，并从正文删除；
    - ``[TABLE_PLACEHOLDER_*]`` 一律剥离（结构化后表格自成 chunk，不再需要占位符）。
    """
    tables: List[Table] = []
    text = content or ""

    def _html_sub(match: "re.Match[str]") -> str:
        grid = parse_html_table_grid(match.group(0))
        if grid is None:
            return match.group(0)
        tables.append(
            build_table(
                len(tables), source_file, grid, title=title, table_format="html",
                raw=match.group(0), doc_type=doc_type, unit_mode=unit_mode,
            )
        )
        return "\n"

    text = _HTML_TABLE_RE.sub(_html_sub, text)

    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        if not _is_md_row(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and _is_md_row(lines[j]):
            j += 1
        block = lines[i:j]
        grid = parse_markdown_table_grid(block)
        if grid is None:
            out.extend(block)
            i = j
            continue
        caption = ""
        if out and _CAPTION_RE.match(out[-1].strip()):
            caption = out.pop().strip()
        notes: List[str] = []
        k = j
        while k < len(lines) and len(notes) < 3 and _NOTE_RE.match(lines[k].strip()):
            notes.append(lines[k].strip())
            k += 1
        if notes:
            caption = " ".join(part for part in (caption, " ".join(notes)) if part)
            j = k
        tables.append(
            build_table(
                len(tables), source_file, grid, title=title, caption=caption,
                table_format="markdown", raw="\n".join(block), doc_type=doc_type, unit_mode=unit_mode,
            )
        )
        out.append("")
        i = j

    prose = _PLACEHOLDER_RE.sub("", "\n".join(out))
    prose = re.sub(r"[ \t]*\n{3,}[ \t]*", "\n\n", prose)
    return prose, tables