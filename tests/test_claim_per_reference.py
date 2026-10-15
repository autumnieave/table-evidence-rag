"""产品层 claim 级引用定位（locate_claims_per_reference）单测。

口径：与评测层同一套定位逻辑（数字 claim 走规则），主序改为「以引用为主」——
返回列表与 references 同长度同顺序，供前端「这条引用支撑了答案里的哪几句话」展示。
零外部依赖：不触网、不读盘、不依赖 LLM/embedding 服务。
"""

from tools.claim_evidence_locator import locate_claims_per_reference

ANSWER = "2025 年公司营收 12.3 亿元，同比增长 8.5%。净利率 10.2%。"
REF_A = {
    "paper_path": "研报A.md",
    "chunk_index": 3,
    "text": "片段",
    "full_text": "公司营收 12.3 亿元，同比增长 8.5%。渠道扩张带动提升。",
}
REF_B = {
    "paper_path": "研报B.md",
    "chunk_index": 0,
    "text": "片段",
    "full_text": "净利率 10.2%，较上年同期提升 1.1 个百分点。",
}
REF_C = {"paper_path": "研报C.md", "chunk_index": 1, "text": "无关内容", "full_text": "行业景气度回暖，需求端改善。"}


def test_ref_major_order_and_locating():
    blocks = locate_claims_per_reference(ANSWER, [REF_A, REF_B, REF_C])
    assert len(blocks) == 3  # 与 references 同长度同顺序

    a = blocks[0]
    assert a, "引用 A 应支撑榜首的数字断言"
    first = a[0]
    assert "12.3" in first["numbers"]
    assert "12.3" in first["matched_numbers"]
    assert first["missing_numbers"] == []
    assert first["evidence"], "命中数字必须带证据片段"
    span = first["evidence"][0]
    assert REF_A["full_text"][span["start"]:span["end"]] == span["text"]
    assert span["tier"] == "高"
    assert "8.5" in first["matched_numbers"]

    b = blocks[1]
    assert b and "10.2" in b[0]["matched_numbers"]
    assert all("10.2" not in blk["matched_numbers"] for blk in a)  # 不串台

    assert blocks[2] == []  # 无关引用返回空列表


def test_missing_numbers_reported():
    answer = "公司营收 12.3 亿元，同比增长 9.9%。"
    blocks = locate_claims_per_reference(answer, [REF_A])
    claim = blocks[0][0]
    assert "12.3" in claim["matched_numbers"]
    assert claim["missing_numbers"] == ["9.9"]  # 材料里没有 9.9 → 显式标未命中


def test_no_number_answer_returns_empty_lists():
    blocks = locate_claims_per_reference("公司经营稳健，行业前景向好。", [REF_A, REF_B])
    assert blocks == [[], []]


def test_top_per_ref_limits_evidence():
    full = "营收 12.3 亿元。" * 20
    ref = {"paper_path": "研报D.md", "chunk_index": 0, "text": "", "full_text": full}
    blocks = locate_claims_per_reference("营收 12.3 亿元。", [ref], top_per_ref=1)
    assert len(blocks[0][0]["evidence"]) == 1


def test_falls_back_to_text_when_full_text_missing():
    ref = {"paper_path": "研报E.md", "chunk_index": 0, "text": "公司营收 12.3 亿元，同比增长 8.5%。"}
    blocks = locate_claims_per_reference(ANSWER, [ref])
    assert blocks[0], "缺 full_text 时应回退 text 定位"


def test_empty_references_and_empty_answer_are_safe():
    assert locate_claims_per_reference(ANSWER, []) == []
    assert locate_claims_per_reference("", [REF_A]) == [[]]
