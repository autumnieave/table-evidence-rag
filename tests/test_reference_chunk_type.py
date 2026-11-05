# -*- coding: utf-8 -*-
"""引用构造的 chunk_type / 表格 Markdown / 图文分离单测（零外部依赖）。

背景 badcase：图表标题被挂到纯文本引用下、表格数据以纯文本展示、不同口径数字被缝合。
本文件锁定：①旧索引无 chunk_type 时按行级锚点回退为 table；②表格引用输出结构化
Markdown（只做结构还原，不新增数值）；③paper_image 只来自真实含图块，表格块不挂图。
"""

from types import SimpleNamespace

from pipelines.rag_pipeline import (
    RAGPipeline,
    _resolve_chunk_type,
    _table_text_to_markdown,
)


class _LLMStub:
    """若走 LLM 图片标题提取则直接失败，确保测试只验证确定性路径。"""

    class client:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    raise AssertionError("纯文本块不应调用 LLM 提取图片标题")


def _pipeline():
    p = RAGPipeline.__new__(RAGPipeline)
    p.config = SimpleNamespace()
    p.llm_generator = _LLMStub()
    p._img_title_cache = {}
    p._img_title_lock = __import__("threading").Lock()
    return p


def test_resolve_chunk_type_prefers_explicit_value():
    assert _resolve_chunk_type({"chunk_type": "table"}) == "table"
    assert _resolve_chunk_type({"chunk_type": "row_fact"}) == "table"
    assert _resolve_chunk_type({"chunk_type": "prose"}) == "text"
    assert _resolve_chunk_type({"chunk_type": "boilerplate"}) == "text"


def test_resolve_chunk_type_falls_back_to_legacy_row_anchor():
    # 旧索引没有 chunk_type，只能靠 is_table_row / parent_id 判定
    assert _resolve_chunk_type({"is_table_row": True, "parent_id": 7}) == "table"
    assert _resolve_chunk_type({"parent_id": 7}) == "table"
    assert _resolve_chunk_type({"is_table_row": False}) == "text"
    assert _resolve_chunk_type({}) == "text"


def test_table_text_to_markdown_row_fact_keeps_values_verbatim():
    md, truncated = _table_text_to_markdown(
        "报告期: 收盘价（元）; 2025年04月03日: 26.33"
    )
    assert truncated is False
    assert md.splitlines()[0] == "| 字段 | 值 |"
    assert "| 报告期 | 收盘价（元） |" in md
    assert "| 2025年04月03日 | 26.33 |" in md


def test_table_text_to_markdown_pipe_rows():
    md, _ = _table_text_to_markdown("指标 | 2023 | 2024\n毛利率 | 78.4 | 64.7")
    assert md.splitlines()[0] == "| 指标 | 2023 | 2024 |"
    assert "| 毛利率 | 78.4 | 64.7 |" in md


def test_table_text_to_markdown_truncates_long_table():
    rows = "\n".join(f"字段{i}: 值{i}" for i in range(60))
    md, truncated = _table_text_to_markdown(rows, max_rows=5)
    assert truncated is True
    assert "此处展示前 5 行" in md


def test_table_text_to_markdown_empty():
    assert _table_text_to_markdown("") == ("", False)


def test_reference_marks_legacy_row_as_table_and_emits_markdown():
    p = _pipeline()
    search_results = [{
        "payload": {
            "is_table_row": True,
            "parent_id": 1,
            "content": "报告期: 毛利率; 2024年: 64.7",
            "source": "a.md",
            # 表格块即使含“图表 1：”也不应挂图表
            "summary": "图表 1：毛利率走势",
        }
    }]
    ref = p._build_reference_for_doc(0, search_results, [search_results[0]["payload"]["content"]])
    assert ref["chunk_type"] == "table"
    assert ref["markdown"].startswith("| 字段 | 值 |")
    assert "64.7" in ref["markdown"]
    assert ref["paper_image"] == ""


def test_text_chunk_without_image_marker_has_no_paper_image():
    p = _pipeline()
    search_results = [{
        "payload": {
            "chunk_type": "prose",
            "content": "图表 1：医药工业毛利率由 78.4% 降至 64.7%",
            "source": "b.md",
        }
    }]
    ref = p._build_reference_for_doc(0, search_results, [search_results[0]["payload"]["content"]])
    assert ref["chunk_type"] == "text"
    assert ref["markdown"] == ""
    assert ref["paper_image"] == ""


def test_text_chunk_with_real_image_keeps_caption():
    p = _pipeline()
    text = "![](images/c1.png)\n图表 1：医药工业毛利率走势"
    search_results = [{"payload": {"chunk_type": "prose", "content": text, "source": "c.md"}}]
    ref = p._build_reference_for_doc(0, search_results, [text])
    assert ref["chunk_type"] == "text"
    assert ref["paper_image"] == "图表 1：医药工业毛利率走势"


def test_table_text_to_markdown_falls_back_to_single_column():
    """扁平表格行无法还原列关系时，退回单列，绝不臆造键值配对。"""
    md, _ = _table_text_to_markdown("600436.SH 片仔癀 5.52 24.72 27.97")
    assert md.splitlines()[0] == "| 表格行内容 |"
    assert "| 600436.SH 片仔癀 5.52 24.72 27.97 |" in md


def test_table_text_to_markdown_distrusts_numeric_keys():
    md, _ = _table_text_to_markdown("收盘价(元):: 第一大股东/持股(元):; 208.98: 188.75/283.19")
    assert md.splitlines()[0] == "| 表格行内容 |"
