import re
import hashlib
from typing import Any, Dict, List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None


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


def split_documents(documents: List[Dict[str, Any]], chunk_size: int = 1024, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
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

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        processed_content, table_rows = extract_tables_and_text(content)
        if processed_content.strip():
            text_chunks = text_splitter.split_text(processed_content)
        else:
            text_chunks = []
        parent_id = None
        if table_rows:
            parent_id = f"{metadata['filename']}_tables"
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
                "chunk_total": len(text_chunks) + len(table_rows),
                "content_hash": content_hash,
                "is_table_row": False,
                "parent_id": None,
            })
            all_chunks.append({"content": chunk_text, "metadata": chunk_metadata})
        for idx, row_text in enumerate(table_rows):
            if not row_text:
                continue
            content_hash = hashlib.md5(row_text.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(text_chunks) + idx,
                "chunk_total": len(text_chunks) + len(table_rows),
                "content_hash": content_hash,
                "is_table_row": True,
                "parent_id": parent_id,
                "row_index": idx,
            })
            all_chunks.append({"content": row_text, "metadata": chunk_metadata})

    return all_chunks
