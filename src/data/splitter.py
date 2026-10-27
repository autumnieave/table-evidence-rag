import logging
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .boilerplate import is_boilerplate_block, strip_trailing_boilerplate
from .row_facts import build_row_facts
from .table_struct import (
    TABLE_SLICE_MAX_CHARS,
    TABLE_SLICE_ROWS,
    extract_report_period,
    extract_report_period_strict,
    scan_tables,
    unit_factor,
)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

logger = logging.getLogger(__name__)

#: 标题里必须出现这些词，才从**标题**推报告期（防「2024年中国XX行业概览」被推成 `2024A`）
REPORT_PERIOD_TITLE_KEYWORDS: Tuple[str, ...] = (
    "年报", "季报", "半年报", "中报", "一季报", "三季报", "业绩", "财报", "点评",
)


def resolve_report_period(title: str, body_head: str = "", doc_type: str = "",
                          pub_year: int = 0) -> Tuple[Optional[str], str]:
    """报告期解析（三级来源），返回 ``(报告期, 来源标签)``。

    1. ``title``：标题里的**强期词**（``2025Q3`` / ``25H1`` / ``2025年三季报`` / ``2024年报``…）——强证据，直接采信；
    2. ``title_weak``：标题含**报告类型词**（年报/季报/点评/业绩…）时的**裸年份**（``2024年`` → ``2024A``）
       ——有类型词佐证才认，避免把「2024年…行业概览」的**发布年**当报告期；
    3. ``body``：**正文首段**里的强期词——只对**个股研报**（``doc_type == "stock"``）启用。

    第 3 级为什么不用于行业研报：行业研报是市场/策略研究，正文里的季度多半是**顺带提及**
    （如周报回顾里的 ``2025Q1``），当报告期就是假阳性（实测会给出 ``2024Q4`` / ``2025Q2`` 这类错值）。
    """
    if title:
        period = extract_report_period_strict(title)
        if period and (not pub_year or _year_of(period) <= pub_year):
            return period, "title"
        if any(k in title for k in REPORT_PERIOD_TITLE_KEYWORDS):
            period = extract_report_period(title)
            if period and (not pub_year or _year_of(period) <= pub_year):
                return period, "title_weak"
    if body_head and doc_type in BODY_PERIOD_DOC_TYPES:
        period = extract_report_period_strict(body_head[:BODY_HEAD_CHARS])
        if period and (not pub_year or _year_of(period) <= pub_year):
            return period, "body"
    return None, "none"


def doc_level_fields(metadata: Dict[str, Any], body_head: str = "") -> Dict[str, Any]:
    """文档级主体 / 报告期元数据（§3.2「每 chunk」口径；正文块与表块共用）。

    - ``stock_code`` / ``stock_name``：取自 loader 的 Excel 元数据（``stockCode`` / ``stockName``），行业研报为空；
    - ``report_period`` / ``report_period_source``：见 :func:`resolve_report_period`
      （来源 ∈ ``title`` / ``title_weak`` / ``body`` / ``none``，便于审计覆盖率来自哪一级）；
    - ``publish_year``：**文档发布年**（``publishDate`` → ``filename`` → ``title``），取不到为 ``None``。
      这是**补充字段、不替代 ``report_period``**：行业研报是市场/策略/周报，**没有报告期语义**但**有发布年**，
      L5-C 期间过滤对行业研报用它、对个股研报用 ``report_period``（见 ``filters.period_gate``）；
    - ``doc_type``：``stock`` / ``industry``（loader 已给 ``doc_type`` 的行业语料优先）。
    """
    title = metadata.get("title") or ""
    code = metadata.get("stockCode") or None
    name = metadata.get("stockName") or None
    doc_type = metadata.get("doc_type") or ("stock" if code else "industry")
    pub_year = publish_year(metadata)
    period, source = resolve_report_period(title, body_head, doc_type, pub_year)
    return {"stock_code": code, "stock_name": name, "report_period": period,
            "report_period_source": source, "publish_year": pub_year or None,
            "doc_type": doc_type}

#: 取「发布年」的元数据键顺序（loader 的 ``publishDate`` 优先）
_PUBLISH_YEAR_KEYS: Tuple[str, ...] = ("publishDate", "filename", "title")
_FOUR_DIGIT_YEAR_RE = re.compile(r"(20\d{2})")


def _year_of(text: str) -> int:
    """取文本里第一个 4 位年份；取不到返回 0。"""
    match = _FOUR_DIGIT_YEAR_RE.search(text or "")
    return int(match.group(1)) if match else 0


def publish_year(metadata: Dict[str, Any]) -> int:
    """文档发布年（``publishDate`` → ``filename`` → ``title``）；都取不到返回 0。"""
    for key in _PUBLISH_YEAR_KEYS:
        year = _year_of(str(metadata.get(key) or ""))
        if year:
            return year
    return 0


def period_is_plausible(period: Optional[str], metadata: Dict[str, Any]) -> bool:
    """报告期合理性：**报告期年份不得晚于发布年**（报告不可能覆盖发布之后的报告期）。

    发布年取不到时**不做判断**（返回 True），宁可留空也不误杀。
    这条闸门剔除的是「策略报告把**预测年度**当报告期」这类假阳性
    （如 2025 年发布的《2026 年医药行业投资策略》推出来的 ``2026A``）。
    """
    if not period:
        return False
    pub = publish_year(metadata)
    if not pub:
        return True
    return _year_of(period) <= pub


#: 「正文首段」用于推断报告期的字符窗口（仅个股研报启用）
BODY_HEAD_CHARS: int = 1500
#: 允许用「正文首段」推断报告期的 doc_type（行业研报**不启用**：会把顺带提及的季度当报告期）
BODY_PERIOD_DOC_TYPES: Tuple[str, ...] = ("stock",)
#: 分块最小长度（字符）：过短的相邻文本块会被合并，避免单字/碎片块入库。
#: 与 config.rag_config 的 CHUNK_MIN_CHARS 默认值保持一致；改这里必须同步改那里（同 overlap 约定）。
CHUNK_MIN_CHARS: int = 200
#: 合并相邻文本块时补的分隔符（切分时原分隔符已被消费，这里用换行做通用再拼接）
CHUNK_JOIN_SEPARATOR: str = "\n"


def extract_tables_and_text(content: str) -> List:
    html_table_pattern = re.compile(r'<table>.*?</table>', re.DOTALL)
    html_tables = html_table_pattern.findall(content)
    placeholder_content = html_table_pattern.sub('__HTML_TABLE_PLACEHOLDER__', content)
    lines = placeholder_content.split('\n')
    processed_lines = []
    table_rows = []
    in_md_table = False
    md_table_buffer = []
    for line in lines:
        stripped = line.strip()
        if '__HTML_TABLE_PLACEHOLDER__' in line:
            for html_table in html_tables:
                rows = parse_html_table(html_table)
                table_rows.extend(rows)
            processed_lines.append(f"[TABLE_PLACEHOLDER_{len(table_rows)}]")
            continue
        if stripped.startswith('|') and stripped.endswith('|'):
            if not in_md_table:
                in_md_table = True
                md_table_buffer = [stripped]
            else:
                md_table_buffer.append(stripped)
        else:
            if in_md_table:
                if len(md_table_buffer) >= 2:
                    rows = parse_markdown_table(md_table_buffer)
                    table_rows.extend(rows)
                processed_lines.append(f"[TABLE_PLACEHOLDER_{len(table_rows)}]")
                md_table_buffer = []
                in_md_table = False
            processed_lines.append(line)
    if in_md_table and len(md_table_buffer) >= 2:
        rows = parse_markdown_table(md_table_buffer)
        table_rows.extend(rows)
        processed_lines.append(f"[TABLE_PLACEHOLDER_{len(table_rows)}]")
    return '\n'.join(processed_lines), table_rows


def parse_html_table(html_table: str) -> List[str]:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_table, 'html.parser')
        table = soup.find('table')
        if not table:
            return []
        rows = table.find_all('tr')
        if len(rows) < 1:
            return []
        header_row = rows[0]
        headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]
        result = []
        for row in rows[1:]:
            cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            if len(cells) == len(headers):
                pairs = [f"{headers[i]}: {cells[i]}" for i in range(len(headers))]
                row_text = '; '.join(pairs)
            else:
                row_text = row.get_text(strip=True)
            if row_text:
                result.append(row_text)
        return result
    except ImportError:
        return _parse_html_table_regex(html_table)


def _parse_html_table_regex(html_table: str) -> List[str]:
    row_pattern = re.compile(r'<tr>(.*?)</tr>', re.DOTALL)
    cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL)
    rows = row_pattern.findall(html_table)
    if len(rows) < 1:
        return []
    header_cells = cell_pattern.findall(rows[0])
    headers = [re.sub(r'<.*?>', '', cell).strip() for cell in header_cells]
    result = []
    for row_html in rows[1:]:
        cells = cell_pattern.findall(row_html)
        cell_texts = [re.sub(r'<.*?>', '', cell).strip() for cell in cells]
        if len(cell_texts) == len(headers):
            pairs = [f"{headers[i]}: {cell_texts[i]}" for i in range(len(headers))]
            row_text = '; '.join(pairs)
        else:
            row_text = re.sub(r'<.*?>', ' ', row_html).strip()
        if row_text:
            result.append(row_text)
    return result


def parse_markdown_table(table_lines: List[str]) -> List[str]:
    if len(table_lines) < 2:
        return []
    header_line = table_lines[0]
    headers = [cell.strip() for cell in header_line.strip('|').split('|')]
    data_lines = table_lines[2:] if len(table_lines) > 2 else []
    rows = []
    for line in data_lines:
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) != len(headers):
            row_text = line
        else:
            pairs = [f"{headers[i]}: {cells[i]}" for i in range(len(headers))]
            row_text = '; '.join(pairs)
        rows.append(row_text)
    return rows


def recursive_split_text(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    chunks = []
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    for separator in separators:
        if separator == "":
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks
        parts = text.split(separator)
        if len(parts) > 1:
            current_chunk = ""
            for part in parts:
                test_chunk = current_chunk + separator + part if current_chunk else part
                if len(test_chunk) <= chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk)
                    if len(part) > chunk_size:
                        sub_chunks = recursive_split_text(part, chunk_size, chunk_overlap, separators[separators.index(separator)+1:])
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part
            if current_chunk.strip():
                chunks.append(current_chunk)
            return chunks
    return chunks


def merge_short_chunks(chunks: List[str], min_chars: int = CHUNK_MIN_CHARS) -> List[str]:
    """把过短的相邻文本块合并，避免入库出现「状」这类单字/碎片块（L5-A）。

    规则与边界：

    - ``min_chars <= 0`` 时只做去空、不合并（等价旧行为，便于对照实验）；
    - 顺序不变、内容不丢：合并只在相邻块之间补一个 ``CHUNK_JOIN_SEPARATOR``；
    - 逐块累积到 ``>= min_chars`` 才落一块；**末块不足 min_chars 时并入前一块**，
      避免尾部留一个碎片；
    - 仅用于**文本块**：表格行（``is_table_row=True``）不参与合并——父表回拉
      （``parent_id`` / ``TABLE_AGG_TOPK``）依赖「一行一块」的结构。

    Args:
        chunks: 切分后的文本块列表（按原文顺序）
        min_chars: 合并阈值（字符数）

    Returns:
        合并后的文本块列表
    """
    cleaned = [c for c in (chunks or []) if c and c.strip()]
    if min_chars <= 0 or not cleaned:
        return cleaned

    merged: List[str] = []
    buffer = ""
    for chunk in cleaned:
        buffer = buffer + CHUNK_JOIN_SEPARATOR + chunk if buffer else chunk
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = merged[-1] + CHUNK_JOIN_SEPARATOR + buffer
        else:
            merged.append(buffer)
    return merged


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    min_chars: int = CHUNK_MIN_CHARS,
    table_mode: str = "legacy",
    table_slice_rows: int = TABLE_SLICE_ROWS,
    table_slice_max_chars: int = TABLE_SLICE_MAX_CHARS,
    strip_boilerplate: bool = False,
    row_fact_mode: str = "off",
) -> List[Dict[str, Any]]:
    """把文档切分为「文本块 + 表格块」两类 chunk。

    Args:
        documents: 文档列表（每项含 content / metadata，metadata 需带 filename）
        chunk_size: 文本块大小（字符）
        chunk_overlap: 文本块重叠（字符）
        min_chars: 文本块最小长度，过短的相邻块会被合并（``0`` = 关闭，保持旧行为）；
            表格行不受影响
        table_mode: 表格入库口径。``"legacy"`` = 一行一个 chunk（现状，线上
            ``research_reports_v3_full`` 口径）；``"structured"`` = 一张表一个 / 若干 chunk（数据管道重构，**需重建索引**）。
        table_slice_rows: 结构化模式下单表切片行数上限（超长表按「表头重复 + 每 K 行」切片）
        table_slice_max_chars: 结构化模式下单表切片**字符**上限（嵌入 API 单条 ≤ 2048 token，
            整表 Markdown 会超限；0 = 不限）
        strip_boilerplate: 锅炉板闸门（§3.2）：剥离文档尾部声明/签名区，残余锅炉板块标
            ``chunk_type = "boilerplate"``（改动内容口径，**需重建索引**；默认 False）
        row_fact_mode: 行级事实口径。``"off"`` = 只写 ``anchor_cells``（默认）；
            ``"chunks"`` = 额外生成 ``chunk_type = "row_fact"`` 的子块（阶段 2/3，
            必须配合 ``table_mode="structured"`` 使用，**需重建索引**）

    Returns:
        chunk 列表；``structured`` 模式下每个 chunk 额外带
        ``chunk_type``（``prose`` / ``table`` / ``boilerplate``）与表级元数据
    """
    if RecursiveCharacterTextSplitter is None:
        raise ImportError("langchain_text_splitters is required for split_documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "……", " ", ""],
        length_function=len,
    )

    all_chunks = []
    seen_hashes = set()
    structured = table_mode == "structured"
    if row_fact_mode == "chunks" and not structured:
        raise ValueError("row_fact_mode='chunks' 仅支持 table_mode='structured'")
    bp_docs = 0
    bp_chars = 0
    total_in_chars = sum(len(d.get("content") or "") for d in documents)

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        if strip_boilerplate:
            content, removed = strip_trailing_boilerplate(content)
            if removed:
                bp_docs += 1
                bp_chars += removed
        filename = metadata.get("filename", "")
        table_chunks = []
        row_fact_chunks = []

        if structured:
            # 结构化口径：一张表一个 / 若干 chunk；正文剥离表格与占位符（数据管道重构）
            doc_type_hint = metadata.get("doc_type") or (
                "stock" if metadata.get("stockCode") else "industry"
            )
            processed_content, tables = scan_tables(
                content, source_file=filename, title=metadata.get("title", ""),
                doc_type=doc_type_hint,
            )
            text_chunks = (
                text_splitter.split_text(processed_content) if processed_content.strip() else []
            )
            if text_chunks:
                text_chunks = merge_short_chunks(text_chunks, min_chars)
            doc_fields = doc_level_fields(metadata, body_head=text_chunks[0] if text_chunks else "")
            for table in tables:
                ranges = table.slice_ranges(table_slice_rows, table_slice_max_chars)
                payload = table.to_payload()
                default_code = table.stock_codes[0] if table.stock_codes else None
                for slice_index, (row_start, row_end) in enumerate(ranges):
                    slice_text = table.to_text(row_start, row_end)
                    period = table.report_period
                    period_source = "table"
                    if period and not period_is_plausible(period, metadata):
                        period, period_source = None, "none"
                    if not period:
                        period = doc_fields["report_period"]
                        period_source = doc_fields["report_period_source"]
                    extra = dict(payload)
                    extra.update({
                        "is_table_row": True,
                        "parent_id": table.table_id,
                        "row_index": row_start,
                        "row_start": row_start,
                        "row_end": row_end,
                        "table_index": table.index,
                        "table_slice_index": slice_index,
                        "table_slice_total": len(ranges),
                        "table_format": table.table_format,
                        "table_parse_ok": table.parse_ok,
                        "stock_code": metadata.get("stockCode") or default_code,
                        "stock_name": metadata.get("stockName") or None,
                        "report_period": period,
                        "report_period_source": period_source,
                        "publish_year": doc_fields["publish_year"],
                        "doc_type": doc_fields["doc_type"],
                        "metric_name": ", ".join(table.metric_names),
                        "metric_names": list(table.metric_names),
                        "unit": table.unit,
                        "unit_factor": table.unit_factor,
                        "unit_source": table.unit_source,
                        "unit_candidates": list(table.unit_candidates),
                        "table_type": table.table_type,
                        "table_scope": table.table_scope,
                        "chunk_id": "{}#s{}".format(table.table_id, slice_index),
                    })
                    facts = build_row_facts(table, metadata, row_start, row_end)
                    extra["anchor_cells"] = [fact["anchor"] for fact in facts]
                    extra["anchor_count"] = len(facts)
                    extra["row_fact_count"] = len(facts)
                    if row_fact_mode == "chunks":
                        for fact in facts:
                            fact_extra = dict(extra)
                            fact_extra.update({
                                "chunk_type": "row_fact",
                                "is_table_row": False,
                                "is_row_fact": True,
                                "parent_id": table.table_id,
                                "table_id": table.table_id,
                                "table_index": table.index,
                                "row_index": fact["row_index"],
                                "row_start": fact["row_index"],
                                "row_end": fact["row_index"] + 1,
                                "col_index": fact["col_index"],
                                "chunk_id": fact["chunk_id"],
                                "anchor_cells": [fact["anchor"]],
                                "anchor_count": 1,
                                "unit": fact["unit"],
                                "unit_factor": unit_factor(fact["unit"]),
                                "unit_source": fact["unit_source"],
                                "currency_raw": fact["currency_raw"],
                                "unit_kind": fact["unit_kind"],
                                "period": fact["period"],
                                "period_type": fact["period_type"],
                                "entity": fact["entity"],
                                "metric": fact["metric"],
                                "metric_name": fact["metric"],
                                "metric_names": [fact["metric"]],
                                "table_caption": fact["table_caption"],
                            })
                            row_fact_chunks.append((fact["text"], fact_extra))
                    table_chunks.append((slice_text, extra))
        else:
            processed_content, table_rows = extract_tables_and_text(content)
            if processed_content.strip():
                text_chunks = text_splitter.split_text(processed_content)
                # L5-A：先合并碎片块再入库（表格行在下方独立处理，不参与合并）
                text_chunks = merge_short_chunks(text_chunks, min_chars)
            else:
                text_chunks = []
            if table_rows:
                parent_id = f"{filename}_tables"
                for idx, row_text in enumerate(table_rows):
                    table_chunks.append((row_text, {
                        "is_table_row": True,
                        "parent_id": parent_id,
                        "row_index": idx,
                    }))

        chunk_total = len(text_chunks) + len(table_chunks) + len(row_fact_chunks)

        for idx, chunk_text in enumerate(text_chunks):
            if not chunk_text:
                continue
            content_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": idx,
                "chunk_total": chunk_total,
                "content_hash": content_hash,
                "is_table_row": False,
                "parent_id": None,
                "chunk_id": "{}#c{}".format(filename, idx),
            })
            if structured:
                chunk_metadata.update(doc_fields)
                chunk_metadata["chunk_type"] = (
                    "boilerplate" if strip_boilerplate and is_boilerplate_block(chunk_text) else "prose"
                )
            all_chunks.append({"content": chunk_text, "metadata": chunk_metadata})
        for idx, (chunk_text, extra) in enumerate(table_chunks):
            if not chunk_text:
                continue
            content_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(text_chunks) + idx,
                "chunk_total": chunk_total,
                "content_hash": content_hash,
            })
            chunk_metadata.update(extra)
            if structured:
                chunk_metadata["chunk_type"] = "table"
            all_chunks.append({"content": chunk_text, "metadata": chunk_metadata})
        for idx, (chunk_text, extra) in enumerate(row_fact_chunks):
            if not chunk_text:
                continue
            content_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(text_chunks) + len(table_chunks) + idx,
                "chunk_total": chunk_total,
                "content_hash": content_hash,
            })
            chunk_metadata.update(extra)
            all_chunks.append({"content": chunk_text, "metadata": chunk_metadata})

    if strip_boilerplate:
        logger.info(
            "锅炉板剥离：%d/%d 篇，%d 字符（占输入 %.2f%%）",
            bp_docs,
            len(documents),
            bp_chars,
            100.0 * bp_chars / max(1, total_in_chars),
        )
    # 统一顶层来源字段：anchor 内已有 source_file，chunk 顶层补一份便于按文件过滤
    for chunk in all_chunks:
        chunk["metadata"].setdefault("source_file", chunk["metadata"].get("filename", ""))
    return all_chunks
