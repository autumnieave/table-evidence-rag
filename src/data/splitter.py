import re
import hashlib
from typing import Any, Dict, List

from .table_struct import TABLE_SLICE_ROWS, scan_tables

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

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

    Returns:
        chunk 列表；``structured`` 模式下每个 chunk 额外带
        ``chunk_type``（``prose`` / ``table``）与表级元数据
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

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        filename = metadata.get("filename", "")
        table_chunks = []

        if structured:
            # 结构化口径：一张表一个 / 若干 chunk；正文剥离表格与占位符（数据管道重构）
            processed_content, tables = scan_tables(
                content, source_file=filename, title=metadata.get("title", "")
            )
            text_chunks = (
                text_splitter.split_text(processed_content) if processed_content.strip() else []
            )
            if text_chunks:
                text_chunks = merge_short_chunks(text_chunks, min_chars)
            for table in tables:
                slices = table.to_slices(table_slice_rows)
                payload = table.to_payload()
                step = table_slice_rows if table_slice_rows > 0 else (table.row_count or 1)
                default_code = table.stock_codes[0] if table.stock_codes else None
                for slice_index, slice_text in enumerate(slices):
                    row_start = slice_index * step
                    row_end = min(row_start + step, table.row_count)
                    extra = dict(payload)
                    extra.update({
                        "is_table_row": True,
                        "parent_id": table.table_id,
                        "row_index": row_start,
                        "row_start": row_start,
                        "row_end": row_end,
                        "table_index": table.index,
                        "table_slice_index": slice_index,
                        "table_slice_total": len(slices),
                        "table_format": table.table_format,
                        "table_parse_ok": table.parse_ok,
                        "stock_code": metadata.get("stockCode") or default_code,
                        "stock_name": metadata.get("stockName") or None,
                        "report_period": table.report_period,
                        "metric_name": ", ".join(table.metric_names),
                        "metric_names": list(table.metric_names),
                        "unit": table.unit,
                        "unit_factor": table.unit_factor,
                    })
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

        chunk_total = len(text_chunks) + len(table_chunks)

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
            })
            if structured:
                chunk_metadata["chunk_type"] = "prose"
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

    return all_chunks
