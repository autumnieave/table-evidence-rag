# -*- coding: utf-8 -*-
"""比对原文表格与 `data.table_struct` 的解析结果，输出差异报告。

背景：20 张表人工核对工作单只展示 ``rows`` 前 8 行 + 原文截断 700 字，
人工无法独立核对「行数守恒」，故用机器做全量比对。

四项比对（每张表一条记录）：

1. **行数差异**：原文行数（colspan / rowspan 展开、含文本）vs 解析 ``rows`` 数。
   被「消费」的前置行**逐行举证**——每行必须是表头行或「跨满列的单格标题行」，
   举证不出即为丢行 / 多吃行；
2. **列数差异**：原文最大列数 vs 解析 ``col_count``，并检查解析每行宽度是否自洽；
3. **表头差异**：``headers`` 文本能否在原文里找到（反转义 + 折叠空白），
   且表头行是否落在原文的正确位置；
4. **关键单元格抽样**：抽解析结果前 3 行 + 后 3 行，与原文同一行逐格比对。

原文网格的 colspan / rowspan 展开语义与解析器**同定义**（「原文网格」本身即该语义），
但文本直接从原始 HTML 取，故可与解析输出做内容级比对。四项互补：
「行数」管整体守恒，「单元格」管丢行后错位，两者都不通过才是硬伤。

输出：``训练结果数据/table_struct_review_20261016/compare_report.json``。

用法::

    .venv\\Scripts\\python.exe src/tools/data_scripts/compare_table_parse.py
    .venv\\Scripts\\python.exe src/tools/data_scripts/compare_table_parse.py --all
    .venv\\Scripts\\python.exe src/tools/data_scripts/compare_table_parse.py --show 20
"""

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data.table_struct import Table, scan_tables  # noqa: E402

_REPO_ROOT = _SRC_ROOT.parent

DEFAULT_CORPUS = _REPO_ROOT / "data_corpus" / "全部数据" / "正式数据" / "附件5：研报数据"
DEFAULT_WORKLIST = _REPO_ROOT / "训练结果数据" / "table_struct_review_20261016" / "worklist.json"
DEFAULT_OUT = _REPO_ROOT / "训练结果数据" / "table_struct_review_20261016" / "compare_report.json"

#: 解析器的空表头占位名（``项目`` / ``列N``）
_PLACEHOLDER_RE = re.compile(r"^(?:项目|列\d+)$")
_MD_SEPARATOR_RE = re.compile(r"^:?-{2,}:?$")
_WS_RE = re.compile(r"\s+")

_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ATTR_RE = re.compile(r"""([A-Za-z_:][-\w:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")


def clean_cell_text(text: str) -> str:
    """单元格文本归一：去零宽字符、折叠空白、去首尾（与 `table_struct._clean_cell` 同口径）。"""
    s = (text or "").replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return _WS_RE.sub(" ", s).strip()


def normalize_raw_text(text: str) -> str:
    """原文归一：反转义 HTML 实体 + 折叠空白（用于「表头文本是否在原文中」）。"""
    return _WS_RE.sub(" ", html.unescape(text or "")).strip()


def is_placeholder(text: str) -> bool:
    """是否为解析器的空表头占位名（``项目`` / ``列N``）。

    注意：``项目`` 也是语料里常见的**真实表头**，故占位判定必须结合原文网格
    （原文该格为空才是占位），不能只看文本——见 :func:`_row_matches_headers`。
    """
    return bool(_PLACEHOLDER_RE.match(text or ""))


def _attrs(raw_attrs: str) -> Dict[str, str]:
    """解析标签属性串为字典（小写键）。"""
    out: Dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw_attrs or ""):
        value = match.group(2)
        if value is None:
            value = match.group(3)
        if value is None:
            value = match.group(4)
        out[match.group(1).lower()] = value or ""
    return out


def _span(attrs: Dict[str, str], key: str) -> int:
    """读取 colspan / rowspan，非法值按 1 处理。"""
    try:
        return max(1, int(attrs.get(key) or 1))
    except (TypeError, ValueError):
        return 1


def _take_pending(row: List[str], pending: Dict[int, List[Any]], col: int) -> int:
    """把当前列上方遗留的 rowspan 单元格补进本行，返回推进后的列号。"""
    while col in pending:
        text, left = pending[col][0], pending[col][1]
        row.append(text)
        if left <= 1:
            pending.pop(col, None)
        else:
            pending[col][1] = left - 1
        col += 1
    return col


def _expand_cells(
    cells: Sequence[Tuple[str, int, int]], pending: Dict[int, List[Any]]
) -> List[str]:
    """按 colspan（复制文本占位）与 rowspan（向下填充）把一行单元格展开为一维列表。"""
    row: List[str] = []
    col = 0
    for text, colspan, rowspan in cells:
        col = _take_pending(row, pending, col)
        for offset in range(colspan):
            row.append(text)
            if rowspan > 1:
                pending[col + offset] = [text, rowspan - 1]
            col += 1
    _take_pending(row, pending, col)
    return row


def _pad_grid(grid: List[List[str]]) -> List[List[str]]:
    """把二维网格补齐到统一列宽（与解析器的 `_fit_cells` 同口径）。"""
    width = max((len(row) for row in grid), default=0)
    return [list(row) + [""] * (width - len(row)) for row in grid]


def _html_cells_regex(row_html: str) -> List[Tuple[str, int, int]]:
    """正则兜底提取一行的 ``(文本, colspan, rowspan)``。"""
    out: List[Tuple[str, int, int]] = []
    for _tag, raw_attrs, inner in _CELL_RE.findall(row_html):
        attrs = _attrs(raw_attrs)
        out.append(
            (
                clean_cell_text(_TAG_RE.sub(" ", inner)),
                _span(attrs, "colspan"),
                _span(attrs, "rowspan"),
            )
        )
    return out


def _build_html_source_grid_regex(raw: str) -> List[List[str]]:
    """无 bs4 时的兜底 HTML 展开（不支持嵌套表格，colspan / rowspan 仍展开）。"""
    grid: List[List[str]] = []
    pending: Dict[int, List[Any]] = {}
    for row_html in _TR_RE.findall(raw or ""):
        cells = _html_cells_regex(row_html)
        if not cells:
            continue
        row = _expand_cells(cells, pending)
        if any(row):
            grid.append(row)
    return _pad_grid(grid)


def build_html_source_grid(raw: str) -> List[List[str]]:
    """把原文 HTML 展开为二维网格（文本直接取自原始 HTML，不依赖解析结果）。"""
    if not (raw or "").strip():
        return []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _build_html_source_grid_regex(raw)
    soup = BeautifulSoup(raw, "html.parser")
    table = soup.find("table")
    if table is None:
        return _build_html_source_grid_regex(raw)
    grid: List[List[str]] = []
    pending: Dict[int, List[Any]] = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        expanded = [
            (
                clean_cell_text(cell.get_text(" ", strip=True)),
                _span({"colspan": str(cell.get("colspan") or 1)}, "colspan"),
                _span({"rowspan": str(cell.get("rowspan") or 1)}, "rowspan"),
            )
            for cell in cells
        ]
        row = _expand_cells(expanded, pending)
        if any(row):
            grid.append(row)
    return _pad_grid(grid)


def build_markdown_source_grid(raw: str) -> List[List[str]]:
    """Markdown 管道表原文 → 二维网格（丢弃表头分隔行）。"""
    rows: List[List[str]] = []
    for line in (raw or "").split("\n"):
        text = line.strip()
        if not text.startswith("|"):
            continue
        cells = [clean_cell_text(cell) for cell in text.strip("|").split("|")]
        meaningful = [cell for cell in cells if cell]
        is_separator = (
            bool(meaningful)
            and len(meaningful) == len(cells)
            and all(_MD_SEPARATOR_RE.match(cell) for cell in meaningful)
        )
        if is_separator:
            continue
        if any(cells):
            rows.append(cells)
    return _pad_grid(rows)


def build_source_grid(raw: str, table_format: str) -> List[List[str]]:
    """把原文展开成二维网格。

    **按原文内容嗅探**（``<tr`` / ``<table`` → HTML，管道符 → Markdown），
    ``table_format`` 仅作兜底——标签与实际原文不一致时不至于产出空网格。
    """
    text = raw or ""
    lowered = text.lower()
    if "<tr" in lowered or "<table" in lowered:
        return build_html_source_grid(text)
    if table_format == "markdown" or "|" in text:
        return build_markdown_source_grid(text)
    return build_html_source_grid(text)


def _row_matches_headers(row: Sequence[str], headers: Sequence[str]) -> bool:
    """某源行是否就是解析所用的表头行。

    匹配规则：逐格「文本相等」，或「源格为空 且 解析用占位名」。
    两种都接受是必要的——``项目`` 既是占位名也是真实表头文本。
    """
    if len(row) != len(headers):
        return False
    for cell, header in zip(row, headers):
        left, right = clean_cell_text(cell), clean_cell_text(header)
        if left == right:
            continue
        if left == "" and is_placeholder(right):
            continue
        return False
    return True


def _is_span_title_row(row: Sequence[str]) -> bool:
    """是否为「跨满列的单格标题行」（``<td colspan=6>盈利预测与估值</td>`` 展开后各列同文）。"""
    nonempty = [cell for cell in row if cell]
    return len(row) >= 3 and len(nonempty) >= 2 and len(set(nonempty)) == 1


def align_table(grid: List[List[str]], table: Table) -> Dict[str, Any]:
    """比对项 1：行数差异——对齐原文网格与解析结果，并给被消费的前置行逐行举证。"""
    src_rows = len(grid)
    parsed_rows = len(table.rows)
    consumed = src_rows - parsed_rows

    header_index: Optional[int] = None
    if table.has_header:
        for index in range(max(0, consumed)):
            if _row_matches_headers(grid[index], table.headers):
                header_index = index
                break
        if header_index is None:
            for index, row in enumerate(grid):
                if _row_matches_headers(row, table.headers):
                    header_index = index
                    break

    issues: List[str] = []
    if consumed < 0:
        issues.append("解析数据行 %d 多于原文行 %d（多吃行）" % (parsed_rows, src_rows))

    unexplained: List[Dict[str, Any]] = []
    for index in range(max(0, consumed)):
        if index == header_index:
            continue
        if _is_span_title_row(grid[index]):
            continue
        unexplained.append(
            {"源行": index, "文本": " | ".join(cell for cell in grid[index] if cell)[:60]}
        )
    if unexplained:
        issues.append(
            "%d 个被消费的前置行既不是表头行也不是跨列标题行（疑丢行）" % len(unexplained)
        )
    if table.is_key_value and consumed != 0:
        issues.append("键值卡片消费了 %d 行（应为 0，疑吃掉首行数据）" % consumed)
    if table.has_header and consumed < 1:
        issues.append("有表头却未消费任何前置行（疑表头重复计入数据行）")
    if table.has_header and header_index is None:
        issues.append("解析表头行未能在原文中定位")

    return {
        "源行数": src_rows,
        "解析数据行": parsed_rows,
        "消费行数": consumed,
        "表头行": header_index,
        "数据起点": max(0, consumed),
        "未解释消费行": unexplained[:10],
        "通过": not issues,
        "说明": "；".join(issues),
    }


def compare_columns(grid: List[List[str]], table: Table) -> Dict[str, Any]:
    """比对项 2：列数差异（原文最大列数 vs 解析列数 + 解析每行宽度自洽性）。"""
    src_cols = max((len(row) for row in grid), default=0)
    parsed_cols = table.col_count
    ragged = [i for i, row in enumerate(table.rows) if len(row) != parsed_cols]
    issues: List[str] = []
    if parsed_cols < src_cols:
        issues.append("解析列数 %d < 原文最大列数 %d（横向截断）" % (parsed_cols, src_cols))
    elif parsed_cols > src_cols:
        issues.append("解析列数 %d > 原文最大列数 %d（多造列）" % (parsed_cols, src_cols))
    if ragged:
        issues.append("解析有 %d 行宽度 != 列数 %d（行内列数不齐）" % (len(ragged), parsed_cols))
    return {
        "源最大列数": src_cols,
        "解析列数": parsed_cols,
        "宽度异常行": ragged[:10],
        "通过": not issues,
        "说明": "；".join(issues),
    }


def compare_headers(grid: List[List[str]], table: Table, align: Dict[str, Any]) -> Dict[str, Any]:
    """比对项 3：表头差异（文本是否在原文中 + 表头行位置是否正确）。"""
    headers = list(table.headers)
    header_index = align["表头行"]
    source_row = grid[header_index] if isinstance(header_index, int) else None
    placeholders: List[str] = []
    for index, header in enumerate(headers):
        if source_row is not None and index < len(source_row) and not source_row[index]:
            placeholders.append(header)
        elif source_row is None and is_placeholder(header):
            placeholders.append(header)
    verified = set(placeholders)

    raw_norm = normalize_raw_text(table.raw)
    missing = [
        header
        for header in headers
        if header and header not in verified and clean_cell_text(header) not in raw_norm
    ]
    issues: List[str] = []
    if table.has_header:
        if header_index is None:
            issues.append("解析表头行未能在原文中定位（原文 %d 行）" % len(grid))
        elif header_index != align["数据起点"] - 1:
            issues.append(
                "表头行位于原文第 %d 行，与数据起点 %d 不衔接"
                % (header_index, align["数据起点"])
            )
    if missing:
        issues.append("表头文本不在原文中：%s" % " / ".join(missing[:3]))
    return {
        "has_header": table.has_header,
        "占位名": placeholders,
        "不在原文": missing,
        "原文表头行": header_index,
        "通过": not issues,
        "说明": "；".join(issues),
    }


def _sample_indices(count: int) -> List[int]:
    """抽检行：前 3 行 + 后 3 行（去重、保持顺序）。"""
    if count <= 0:
        return []
    indices = list(range(min(3, count)))
    for index in range(max(0, count - 3), count):
        if index not in indices:
            indices.append(index)
    return indices


def compare_key_cells(grid: List[List[str]], table: Table, data_start: int) -> Dict[str, Any]:
    """比对项 4：关键单元格抽样（前 3 行 + 后 3 行逐格比对）。"""
    indices = _sample_indices(len(table.rows))
    diffs: List[Dict[str, Any]] = []
    for index in indices:
        src_index = data_start + index
        if src_index >= len(grid):
            diffs.append(
                {"解析行": index, "源行": src_index, "列": -1, "解析": "（无对应源行）", "源": ""}
            )
            continue
        src_row, parsed_row = grid[src_index], table.rows[index]
        for col in range(max(len(src_row), len(parsed_row))):
            left = clean_cell_text(parsed_row[col]) if col < len(parsed_row) else ""
            right = clean_cell_text(src_row[col]) if col < len(src_row) else ""
            if left != right:
                diffs.append(
                    {"解析行": index, "源行": src_index, "列": col, "解析": left, "源": right}
                )
    return {
        "抽检行": indices,
        "不一致数": len(diffs),
        "不一致": diffs[:20],
        "通过": not diffs,
    }


def compare_table(raw: str, table: Table, bucket: str = "") -> Dict[str, Any]:
    """比对一张表：原文 ``raw`` vs 解析结果 ``table``，返回差异记录。"""
    grid = build_source_grid(raw, table.table_format)
    align = align_table(grid, table)
    cols_cmp = compare_columns(grid, table)
    head_cmp = compare_headers(grid, table, align)
    cell_cmp = compare_key_cells(grid, table, align["数据起点"])

    types: List[str] = []
    detail: List[str] = []
    if not align["通过"]:
        types.append("行数差异")
        detail.append(align["说明"])
    if not cols_cmp["通过"]:
        types.append("列数差异")
        detail.append(cols_cmp["说明"])
    if not head_cmp["通过"]:
        types.append("表头差异")
        detail.append(head_cmp["说明"])
    if not cell_cmp["通过"]:
        types.append("单元格差异")
        detail.append(
            "抽检 %d 行有 %d 处单元格不一致" % (len(cell_cmp["抽检行"]), cell_cmp["不一致数"])
        )
    return {
        "table_id": table.table_id,
        "file": table.source_file,
        "bucket": bucket,
        "format": table.table_format,
        "has_header": table.has_header,
        "is_key_value": table.is_key_value,
        "caption": table.caption,
        "解析数据行": table.row_count,
        "解析列数": table.col_count,
        "差异类型": types,
        "差异详情": detail,
        "比对明细": {
            "行数": align,
            "列数": cols_cmp,
            "表头": head_cmp,
            "关键单元格": cell_cmp,
        },
        "是否通过": not types,
    }


def load_corpus_tables(corpus: Path) -> Dict[str, Table]:
    """扫描语料目录，返回 ``table_id -> Table``（解析器输出）。"""
    tables: Dict[str, Table] = {}
    for path in sorted(corpus.rglob("*.md")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        _, found = scan_tables(content, source_file=path.name, title=path.stem)
        for table in found:
            tables[table.table_id] = table
    return tables


def load_worklist(path: Path) -> List[Dict[str, str]]:
    """读工作单抽样，返回 ``[{table_id, file, bucket}, ...]``。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, str]] = []
    for item in payload.get("picked", []):
        out.append(
            {
                "table_id": str(item.get("table_id", "")),
                "file": str(item.get("file", "")),
                "bucket": str(item.get("bucket", "")),
            }
        )
    return out


def build_report(records: List[Dict[str, Any]], scope: str, corpus: Path) -> Dict[str, Any]:
    """汇总报告（含统计 + 逐表记录）。"""
    type_counter: Dict[str, int] = {}
    for record in records:
        for name in record["差异类型"]:
            type_counter[name] = type_counter.get(name, 0) + 1
    passed = sum(1 for record in records if record["是否通过"])
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": scope,
        "corpus": str(corpus),
        "统计": {
            "比对表数": len(records),
            "通过": passed,
            "不通过": len(records) - passed,
            "通过率": round(passed / len(records), 4) if records else 0.0,
            "差异类型计数": type_counter,
        },
        "records": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：比对工作单 20 张表（或 ``--all`` 全库）并落盘差异报告。"""
    parser = argparse.ArgumentParser(description="比对原文表格与 table_struct 解析结果")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="语料根目录")
    parser.add_argument(
        "--worklist", type=Path, default=DEFAULT_WORKLIST, help="人工核对工作单 json"
    )
    parser.add_argument(
        "--all", action="store_true", help="比对全库所有表（默认只比对工作单抽样）"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="差异报告输出路径")
    parser.add_argument("--limit", type=int, default=0, help="只比对前 N 张（0 = 不限制）")
    parser.add_argument("--show", type=int, default=10, help="控制台打印前 N 条不通过记录")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - 控制台不支持时忽略
            pass

    tables = load_corpus_tables(args.corpus)
    if args.all:
        targets = [
            {"table_id": tid, "file": table.source_file, "bucket": ""}
            for tid, table in tables.items()
        ]
        scope = "all"
    else:
        targets = load_worklist(args.worklist)
        scope = "worklist"
    if args.limit > 0:
        targets = targets[: args.limit]

    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for target in targets:
        table = tables.get(target["table_id"])
        if table is None:
            missing.append(target["table_id"])
            continue
        records.append(compare_table(table.raw, table, bucket=target["bucket"]))

    report = build_report(records, scope, args.corpus)
    report["未命中"] = missing
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = report["统计"]
    print(
        "比对表数: %d  通过: %d  不通过: %d  通过率: %.4f"
        % (stats["比对表数"], stats["通过"], stats["不通过"], stats["通过率"])
    )
    print("差异类型计数: %s" % json.dumps(stats["差异类型计数"], ensure_ascii=False))
    if missing:
        print("工作单未命中语料: %d 张 %s" % (len(missing), missing[:3]))
    failures = [record for record in records if not record["是否通过"]]
    for record in failures[: max(0, args.show)]:
        print(
            "- %s | %s | %s"
            % (record["table_id"], "/".join(record["差异类型"]), "；".join(record["差异详情"]))
        )
    print("报告: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())