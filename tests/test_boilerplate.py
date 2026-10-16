# -*- coding: utf-8 -*-
"""锅炉板识别（§3.2）单元测试：纯逻辑、零外部依赖。"""

from data.boilerplate import (
    find_trailing_boilerplate_start,
    heading_label,
    is_boilerplate_block,
    is_boilerplate_heading,
    strip_trailing_boilerplate,
)
from data.splitter import split_documents

BODY = "# 投资要点\n\n" + "公司业绩稳健增长，营业收入同比提升。" * 40 + "\n\n"
TAIL = (
    "# 免责声明\n\n" + "本报告仅供客户参考，不构成任何投资建议。" * 8 + "\n\n"
    "# 东吴证券投资评级标准\n\n" + "买入：相对同期基准指数涨幅 20% 以上。" * 6
)
DOC = {"content": BODY + TAIL, "metadata": {"filename": "demo.md", "title": "demo"}}


def test_heading_label_forms():
    assert heading_label("# 免责声明") == "免责声明"
    assert heading_label("**法律声明**") == "法律声明"
    assert heading_label("免责声明") == "免责声明"
    assert heading_label("免责声明", strong_only=True) is None
    assert heading_label("这是一段正文段落，含句读符号，不该被当成标题。") is None


def test_is_boilerplate_heading():
    assert is_boilerplate_heading("免责声明")
    assert is_boilerplate_heading("东吴证券投资评级标准")  # 子串命中（带公司前缀）
    assert is_boilerplate_heading("法律主体声明")  # 「…声明」兜底
    assert is_boilerplate_heading("国信证券经济研究所")
    assert is_boilerplate_heading("IMPORTANT DISCLAIMER")  # 英文（大小写不敏感）
    assert not is_boilerplate_heading("投资建议")
    assert not is_boilerplate_heading("结论及建议")
    assert not is_boilerplate_heading(None)


def test_find_trailing_boilerplate_start_points_at_first_disclaimer_heading():
    text = BODY + TAIL
    start = find_trailing_boilerplate_start(text)
    assert start is not None
    assert text[start:].startswith("# 免责声明")


def test_ratio_guard_refuses_when_disclaimer_is_near_head():
    """护栏：声明区超过全文 40% ⇒ **不切**（宁漏切不误伤）。"""
    text = "# 免责声明\n\n" + "模板文本。" * 60 + "\n\n# 正文\n\n" + "真正的内容。" * 40
    assert find_trailing_boilerplate_start(text) is None


def test_trailing_signature_block_does_not_block_detection():
    """声明之后的签名区（`# 联系` / 城市名）不能把判定掐掉。"""
    text = BODY + "# 免责声明\n\n" + "模板。" * 60 + "\n\n# 联系\n\n上海 北京 深圳\n"
    start = find_trailing_boilerplate_start(text)
    assert start is not None
    assert text[start:].startswith("# 免责声明")


def test_mid_document_marker_does_not_pull_boundary_into_body():
    """正文中段的 `# 相关研究报告` 不能被当成起点（否则会切掉正文）。"""
    text = (
        BODY
        + "# 相关研究报告\n\n" + "过往报告清单。" * 30 + "\n\n"
        + "# 风险提示\n\n" + "政策风险、研发风险。" * 30 + "\n\n"
        + "# 免责声明\n\n" + "本报告仅供客户参考。" * 20
    )
    start = find_trailing_boilerplate_start(text)
    assert start is not None
    assert text[start:].startswith("# 免责声明")


def test_strip_trailing_boilerplate_keeps_content():
    new_text, removed = strip_trailing_boilerplate(BODY + TAIL)
    assert removed > 0
    assert "免责声明" not in new_text
    assert "投资要点" in new_text and "营业收入同比提升" in new_text


def test_strip_is_noop_when_no_boilerplate():
    text = "# 投资要点\n\n" + "正常内容。" * 80
    assert strip_trailing_boilerplate(text) == (text, 0)


def test_is_boilerplate_block():
    assert is_boilerplate_block("# 免责声明\n\n模板……")
    assert is_boilerplate_block("\n\n**法律声明**\n\n模板……")
    assert not is_boilerplate_block("# 投资要点\n\n内容……")
    assert not is_boilerplate_block("")


def test_split_documents_strips_boilerplate_when_enabled():
    chunks = split_documents([DOC], table_mode="structured", strip_boilerplate=True)
    joined = "".join(c["content"] for c in chunks)
    assert "免责声明" not in joined
    assert "营业收入同比提升" in joined
    assert all(c["metadata"]["chunk_type"] in {"prose", "table", "boilerplate"} for c in chunks)


def test_split_documents_keeps_boilerplate_by_default():
    chunks = split_documents([DOC], table_mode="structured")
    joined = "".join(c["content"] for c in chunks)
    assert "免责声明" in joined
    assert all(c["metadata"]["chunk_type"] == "prose" for c in chunks)
