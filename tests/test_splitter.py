"""data.splitter 分块与表格抽取单元测试（纯逻辑）。"""

from data.splitter import (
    CHUNK_MIN_CHARS,
    extract_tables_and_text,
    merge_short_chunks,
    parse_html_table,
    parse_markdown_table,
    recursive_split_text,
    split_documents,
)


def test_recursive_split_text_basic():
    # 与 split_documents 一致的分隔符链（含 "" 兜底字符级切分）
    text = "段1。" + "内容" * 60 + "\n段2。" + "内容" * 60
    chunks = recursive_split_text(text, chunk_size=64, chunk_overlap=8, separators=["\n\n", "\n", "。", " ", ""])
    assert isinstance(chunks, list) and len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)
    # 每块不超过 chunk_size；重叠区会重复部分内容（分块设计如此）
    assert all(len(c) <= 64 for c in chunks)
    assert "段1" in chunks[0]


def test_recursive_split_text_short_text():
    assert recursive_split_text("短文本", 64, 8, ["\n"]) == ["短文本"]
    assert recursive_split_text("", 64, 8, ["\n"]) == []
    assert recursive_split_text("   ", 64, 8, ["\n"]) == []


def test_extract_tables_and_text_markdown():
    md = "| 指标 | 数值 |\n| --- | --- |\n| 营收 | 100 |\n\n正文段落"
    content, rows = extract_tables_and_text(md)
    assert "[TABLE_PLACEHOLDER_" in content
    assert rows, "markdown 表格应被抽取为行"
    assert any("营收" in "".join(r) for r in rows)


def test_parse_markdown_table():
    lines = ["| 指标 | 数值 |", "| --- | --- |", "| 营收 | 100 |"]
    rows = parse_markdown_table(lines)
    assert any("营收" in r and "100" in r for r in rows)


def test_extract_tables_and_text_html():
    html = "<p>正文</p><table><tr><th>指标</th><th>数值</th></tr><tr><td>营收</td><td>100</td></tr></table>"
    content, rows = extract_tables_and_text(html)
    assert "[TABLE_PLACEHOLDER_" in content
    assert any("营收" in "".join(r) for r in rows)


def test_parse_html_table():
    rows = parse_html_table("<table><tr><th>指标</th><th>数值</th></tr><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>")
    assert len(rows) >= 2
    assert any("A" in r and "1" in r for r in rows)
    assert any("B" in r and "2" in r for r in rows)


def test_split_documents_keeps_metadata():
    docs = [{"content": "研报正文" + "内容" * 300, "metadata": {"stockName": "贵州茅台"}}]
    chunks = split_documents(docs, chunk_size=128, chunk_overlap=16)
    assert chunks
    assert all(c.get("metadata", {}).get("stockName") == "贵州茅台" for c in chunks)

# ------------------------------------------------- L5-A 碎片合并（最小块 ≥200 字）


def test_merge_short_chunks_merges_fragments():
    frags = ["状", "行业: A股价(2025/3/18); 医药生物: 58.26", "第三块内容" * 50]
    merged = merge_short_chunks(frags, 200)
    assert len(merged) == 1
    assert len(merged[0]) >= 200
    # 内容不丢、顺序不变
    pos = -1
    for frag in frags:
        idx = merged[0].find(frag)
        assert idx > pos
        pos = idx


def test_merge_short_chunks_tail_absorbed_by_previous():
    merged = merge_short_chunks(["长块" * 120, "尾碎片"], 200)
    assert len(merged) == 1
    assert merged[0].endswith("尾碎片")


def test_merge_short_chunks_disabled_when_min_zero():
    frags = ["状", "   ", "短句"]
    assert merge_short_chunks(frags, 0) == ["状", "短句"]


def test_merge_short_chunks_skips_blank_and_handles_empty():
    assert merge_short_chunks([], 200) == []
    assert merge_short_chunks(["   ", ""], 200) == []


def test_merge_short_chunks_single_short_chunk_kept():
    # 全篇不足 min_chars 时不能凭空造内容，原样保留
    merged = merge_short_chunks(["只有很短的一句"], 200)
    assert merged == ["只有很短的一句"]


def test_split_documents_merges_short_text_chunks():
    # chunk_size 调小，确保 langchain 先切出多个 <200 的小块，再验证合并
    docs = [{"content": "\n\n".join("短段落%d" % i for i in range(60)), "metadata": {"filename": "a.md"}}]
    chunks = split_documents(docs, chunk_size=60, chunk_overlap=0, min_chars=200)
    text_chunks = [c for c in chunks if not c["metadata"]["is_table_row"]]
    assert text_chunks
    assert all(len(c["content"]) >= 200 for c in text_chunks)
    assert sum(len(c["content"]) for c in text_chunks) >= 300


def test_split_documents_min_chars_zero_keeps_old_behavior():
    docs = [{"content": "\n\n".join("短段落%d" % i for i in range(60)), "metadata": {"filename": "a.md"}}]
    off = split_documents(docs, chunk_size=60, chunk_overlap=0, min_chars=0)
    on = split_documents(docs, chunk_size=60, chunk_overlap=0, min_chars=200)
    assert len(off) > len(on)
    assert any(len(c["content"]) < 200 for c in off)
    assert all(len(c["content"]) >= 200 for c in on)


def test_split_documents_does_not_merge_table_rows():
    md = "| 指标 | 数值 |\n| --- | --- |\n| 营收 | 100 |\n| 净利 | 20 |\n\n正文" + "内容" * 200
    docs = [{"content": md, "metadata": {"filename": "b.md"}}]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=0, min_chars=200)
    rows = [c for c in chunks if c["metadata"]["is_table_row"]]
    texts = [c for c in chunks if not c["metadata"]["is_table_row"]]
    assert len(rows) >= 2, "表格行应保持一行一块，不参与合并"
    assert all(c["metadata"]["parent_id"] == "b.md_tables" for c in rows)
    assert texts and all(len(c["content"]) >= 200 for c in texts)


def test_chunk_min_chars_default_constant():
    assert CHUNK_MIN_CHARS == 200
