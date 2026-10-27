"""claim 级证据定位单测（零外部依赖：不触网、不读盘、不依赖 LLM/embedding 服务）。

覆盖：切句/切单元的字符偏移自洽、数字 claim 的规则抽取与定位（复用 L1 口径）、
观点 claim 的注入式抽取与向量定位、三档口径边界、payload 开关三态、
数字证据收紧（只保留放大方向 + 命中句须含单位）、SQL 结果材料接入、观点证据线索标记。
"""

import os

import pytest

from tools.claim_evidence_locator import (
    CLUE_NOTE,
    SQL_MATERIAL_LABEL,
    DEFAULT_SENTENCE_WINDOW,
    FULL_TEXT_PAYLOAD_ENV,
    Claim,
    attach_full_text,
    build_unit_index,
    build_opinion_prompt,
    cosine,
    detect_material_unit,
    extract_number_claims,
    extract_opinion_claims,
    full_text_in_payload,
    is_plausible_variant,
    is_year_like,
    locate_answer,
    locate_number_claim,
    locate_opinion_claim,
    mask_quarter_tokens,
    merge_windows,
    parse_opinion_output,
    ref_text,
    split_sentences,
    split_units,
    tier_of,
    has_unit_context,
    sql_rows_to_material,
    sql_value_matches,
    stock_code_numbers,
)

ANSWER = "2025 年公司营收 12.3 亿元，同比增长 8.5%。公司营收同比提升主要来自渠道扩张。"
FULL = "公司营收 12.3 亿元，同比增长 8.5%。\n\n渠道扩张带动营收提升。"


def _ref(text=FULL, path="研报A.md", ci=0):
    return {"paper_path": path, "chunk_index": ci, "text": text[:50], "full_text": text, "text_total": len(text)}


def _fake_embed(texts):
    """关键词计数向量（确定性、无外部依赖）。"""
    vocab = ("营收", "毛利", "下滑", "提升")
    return [[float(t.count(w)) for w in vocab] for t in texts]


def _fake_llm(reply):
    return lambda _prompt: reply


# ----------------------------------------------------------------- 档位 / 相似度


def test_tier_boundaries():
    assert tier_of(1.0) == "高"
    assert tier_of(0.81) == "高"
    assert tier_of(0.8) == "中"
    assert tier_of(0.61) == "中"
    assert tier_of(0.6) == "低"
    assert tier_of(0.0) == "低"


def test_cosine_edges():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([], []) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0], [1.0, 2.0]) == 0.0


# ----------------------------------------------------------------- 切分（偏移自洽）


def test_split_sentences_offsets_are_exact():
    spans = split_sentences(ANSWER)
    assert [s.text for s in spans] == ["2025 年公司营收 12.3 亿元，同比增长 8.5%。", "公司营收同比提升主要来自渠道扩张。"]
    for s in spans:
        assert ANSWER[s.start : s.end] == s.text
    assert spans[0].start == 0 and spans[-1].end == len(ANSWER)


def test_split_sentences_skips_blank_and_keeps_tail():
    spans = split_sentences("甲。\n\n乙")
    assert [s.text for s in spans] == ["甲。", "乙"]
    assert spans[1].start == 4 and spans[1].end == 5


def test_split_units_splits_long_block_and_keeps_offsets():
    text = "\n".join("行%d 内容" % i for i in range(60))
    units = split_units(text, max_chars=50)
    assert len(units) > 1
    assert all(len(u.text) <= 50 for u in units)
    for u in units:
        assert text[u.start : u.end] == u.text


def test_split_units_keeps_short_paragraphs_intact():
    units = split_units("第一段。\n\n第二段。", max_chars=800)
    assert [u.text for u in units] == ["第一段。", "第二段。"]


def test_ref_text_prefers_full_text():
    assert ref_text({"text": "短", "full_text": "长全文"}) == "长全文"
    assert ref_text({"text": "摘要"}) == "摘要"
    assert ref_text({}) == ""


# ----------------------------------------------------------------- 数字 claim


def test_is_year_like():
    assert is_year_like("2025") is True
    assert is_year_like("1900") is True
    assert is_year_like("2101") is False
    assert is_year_like("12.3") is False
    assert is_year_like("12345") is False


def test_extract_number_claims_drops_year_noise():
    claims = extract_number_claims(ANSWER, drop_year_like=True)
    assert len(claims) == 1
    c = claims[0]
    assert c.kind == "number"
    assert c.numbers == ["12.3", "8.5"]
    assert ANSWER[c.start : c.end] == c.text


def test_extract_number_claims_keeps_year_when_disabled():
    claims = extract_number_claims(ANSWER, drop_year_like=False)
    assert claims[0].numbers == ["2025", "12.3", "8.5"]


def test_locate_number_claim_exact_hit_window_and_offsets():
    claim = extract_number_claims(ANSWER)[0]
    evidence = locate_number_claim(claim, [_ref()], window=DEFAULT_SENTENCE_WINDOW)
    assert len(evidence) == 1
    e = evidence[0]
    assert e.hit_type == "number"
    assert e.score == pytest.approx(1.0)
    assert e.tier == "高"
    assert e.paper_path == "研报A.md" and e.chunk_index == 0
    assert sorted(e.matched_numbers) == ["12.3", "8.5"]
    assert FULL[e.start : e.end] == e.text
    assert "渠道扩张" in e.text  # ±2 句窗口把下一句并进来


def test_locate_number_claim_unit_variant_scores_lower():
    claim = Claim(text="营业成本 58.86 亿元", kind="number", start=0, end=10, numbers=["58.86"])
    evidence = locate_number_claim(claim, [_ref("营业成本 5886 万元，同比持平。")])
    assert len(evidence) == 1
    assert evidence[0].score == pytest.approx(0.9)
    assert evidence[0].tier == "高"


def test_locate_number_claim_miss_returns_empty():
    claim = Claim(text="净利润 99.9 亿元", kind="number", start=0, end=5, numbers=["99.9"])
    assert locate_number_claim(claim, [_ref()]) == []


def test_is_plausible_variant_rejects_degenerate_values():
    assert is_plausible_variant("5886") is True
    assert is_plausible_variant("5.89") is True
    assert is_plausible_variant("0") is False
    assert is_plausible_variant("1") is False
    assert is_plausible_variant("0.01") is False
    assert is_plausible_variant("") is False
    assert is_plausible_variant("abc") is False


def test_locate_number_claim_ignores_degenerate_variant_hits():
    """12.3 的 ÷10000 变体是 "0"，若不做收紧会命中任何带 0 的句子。"""
    claim = Claim(text="营收 12.3 亿元", kind="number", start=0, end=5, numbers=["12.3"])
    assert locate_number_claim(claim, [_ref("第0 行与本题无关。\n第1 行也无关。")]) == []


def test_locate_number_claim_rejects_variant_substring_false_positive():
    """99.9 的万元变体之一是 "1"，纯子串匹配会命中 "12.3" 内部 —— 必须被 token 边界挡住。"""
    claim = Claim(text="净利润 99.9 亿元", kind="number", start=0, end=5, numbers=["99.9"])
    assert locate_number_claim(claim, [_ref("公司营收 12.3 亿元。")]) == []
    # 关掉变体匹配后，精确路径仍与 L1 一致（12.3 命中 12.35 内部）
    claim2 = Claim(text="营收 12.3", kind="number", start=0, end=4, numbers=["12.3"])
    assert locate_number_claim(claim2, [_ref("营收 12.35 亿元。")]) != []


def test_locate_number_claim_falls_back_to_text_field():
    claim = Claim(text="营收 12.3", kind="number", start=0, end=5, numbers=["12.3"])
    evidence = locate_number_claim(claim, [{"paper_path": "B.md", "text": "公司营收 12.3 亿元。"}])
    assert len(evidence) == 1


def test_merge_windows_merges_overlaps_and_clips():
    assert merge_windows([2], 2, 10) == [[0, 1, 2, 3, 4]]
    assert merge_windows([0, 9], 1, 10) == [[0, 1], [8, 9]]
    assert merge_windows([3, 4], 1, 6) == [[2, 3, 4, 5]]
    assert merge_windows([], 2, 5) == []


# ----------------------------------------------------------------- 观点 claim


def test_parse_opinion_output_variants():
    assert parse_opinion_output('["甲", "乙"]') == ["甲", "乙"]
    assert parse_opinion_output('{"claims": ["甲"]}') == ["甲"]
    assert parse_opinion_output('好的，结果如下：["甲", "乙"] 完毕') == ["甲", "乙"]
    assert parse_opinion_output("- 甲\n- 乙") == ["甲", "乙"]
    assert parse_opinion_output("") == []


def test_extract_opinion_claims_locates_verbatim_and_flags_paraphrase():
    reply = '["公司营收同比提升主要来自渠道扩张。", "渠道扩张是唯一原因"]'
    claims = extract_opinion_claims(ANSWER, llm=_fake_llm(reply))
    assert len(claims) == 2
    assert ANSWER[claims[0].start : claims[0].end] == claims[0].text
    assert claims[0].note == ""
    assert claims[1].start == -1 and claims[1].note


def test_extract_opinion_claims_without_llm_is_empty():
    assert extract_opinion_claims(ANSWER, llm=None) == []


def test_extract_opinion_claims_llm_failure_degrades():
    def boom(_prompt):
        raise RuntimeError("quota")

    claims = extract_opinion_claims(ANSWER, llm=boom)
    assert len(claims) == 1 and "quota" in claims[0].note


def test_build_opinion_prompt_contains_answer_and_limit():
    prompt = build_opinion_prompt("测试答案", max_claims=3)
    assert "测试答案" in prompt and "3" in prompt


def test_locate_opinion_claim_ranks_by_similarity():
    refs = [_ref("毛利下滑明显，成本高企。\n\n营收同比提升来自渠道扩张。")]
    claim = Claim(text="营收提升源于渠道扩张", kind="opinion", start=0, end=9)
    evidence = locate_opinion_claim(claim, refs, embed=_fake_embed, top_k=2)
    assert evidence[0].hit_type == "semantic"
    assert "渠道扩张" in evidence[0].text
    assert evidence[0].score > evidence[1].score
    assert evidence[0].tier in ("高", "中", "低")
    for e in evidence:
        assert refs[0]["full_text"][e.start : e.end] == e.text


def test_locate_opinion_claim_without_embed_is_empty():
    claim = Claim(text="营收提升", kind="opinion", start=0, end=4)
    assert locate_opinion_claim(claim, [_ref()], embed=None) == []


# ----------------------------------------------------------------- payload 开关


def test_full_text_payload_switch_off_by_default(monkeypatch):
    monkeypatch.delenv(FULL_TEXT_PAYLOAD_ENV, raising=False)
    ref = {"paper_path": "a.md"}
    attach_full_text(ref, "全文")
    assert "full_text" not in ref
    assert full_text_in_payload() is False


def test_full_text_payload_switch_on(monkeypatch):
    monkeypatch.setenv(FULL_TEXT_PAYLOAD_ENV, "1")
    ref = {"paper_path": "a.md"}
    attach_full_text(ref, "全文")
    assert ref["full_text"] == "全文"


@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_full_text_payload_switch_falsey(monkeypatch, value):
    monkeypatch.setenv(FULL_TEXT_PAYLOAD_ENV, value)
    assert full_text_in_payload() is False


# ----------------------------------------------------------------- 总装


def test_locate_number_claim_respects_quota():
    parts = []
    for i in range(6):
        parts.append("第%d 行营收 12.3 亿元。" % i)
        parts.append("第%d 行净利为零。" % i)  # 非命中句，隔开窗口避免合并
    text = "\n".join(parts)
    claim = Claim(text="营收 12.3", kind="number", start=0, end=4, numbers=["12.3"])
    evidence = locate_number_claim(claim, [_ref(text)], window=0, max_per_ref=2, max_total=5)
    assert len(evidence) == 2  # 单引用上限先生效（共 6 处非相邻命中）
    for e in evidence:
        assert text[e.start : e.end] == e.text
    assert len(locate_number_claim(claim, [_ref(text)], window=0, max_per_ref=0, max_total=3)) == 3
    assert len(locate_number_claim(claim, [_ref(text)], window=0, max_per_ref=0, max_total=0)) == 6


def test_locate_answer_reports_missing_numbers_and_pre_truncation_total():
    answer = "公司营收 12.3 亿元，净利润 99.9 亿元。"
    out = locate_answer(answer, [_ref("公司营收 12.3 亿元。")])
    ce = out["claims"][0]
    assert ce["matched_numbers"] == ["12.3"]
    assert ce["missing_numbers"] == ["99.9"]
    assert ce["n_evidence_total"] >= 1
    assert len(ce["evidence"]) <= 5
    assert out["meta"]["n_numbers_missing"] == 1
    assert out["meta"]["n_claims_with_missing_numbers"] == 1
    assert out["meta"]["n_numbers_matched"] == 1


def test_locate_answer_end_to_end(monkeypatch):
    monkeypatch.delenv(FULL_TEXT_PAYLOAD_ENV, raising=False)
    reply = '["公司营收同比提升主要来自渠道扩张。"]'
    out = locate_answer(ANSWER, [_ref()], llm=_fake_llm(reply), embed=_fake_embed)
    meta = out["meta"]
    assert meta["llm_injected"] is True and meta["embed_injected"] is True
    assert meta["n_number_claims"] == 1 and meta["n_opinion_claims"] == 1
    assert meta["n_claims"] == 2
    starts = [c["claim"]["start"] for c in out["claims"]]
    assert starts == sorted(starts)
    assert meta["n_claims_with_evidence"] == 2
    assert meta["n_evidence"] == len(out["claims"][0]["evidence"]) + len(out["claims"][1]["evidence"])
    assert meta["window"] == DEFAULT_SENTENCE_WINDOW


def test_locate_answer_embeds_units_once_for_all_claims():
    """成本回归：单元向量只嵌一次，之后每条观点 claim 只嵌 claim 自身。"""
    calls = []

    def counting_embed(texts):
        texts = list(texts)
        calls.append(len(texts))
        return _fake_embed(texts)

    answer = "公司营收 12.3 亿元。营收提升来自渠道扩张。毛利下滑明显。"
    reply = '["营收提升来自渠道扩张", "毛利下滑明显"]'
    out = locate_answer(answer, [_ref()], llm=_fake_llm(reply), embed=counting_embed)
    assert out["meta"]["n_opinion_claims"] == 2
    assert calls[0] == len(split_units(FULL))
    assert calls[1:] == [1, 1]


def test_build_unit_index_edges():
    assert build_unit_index([_ref()], embed=None) is None
    assert build_unit_index([], embed=_fake_embed) is None
    assert build_unit_index([_ref()], embed=lambda texts: []) is None
    index = build_unit_index([_ref()], embed=_fake_embed)
    assert index is not None and len(index.units) == len(index.vectors) == len(split_units(FULL))


def test_locate_answer_rules_only_without_injection():
    out = locate_answer(ANSWER, [_ref()])
    meta = out["meta"]
    assert meta["llm_injected"] is False and meta["embed_injected"] is False
    assert meta["n_opinion_claims"] == 0
    assert all(c["claim"]["kind"] == "number" for c in out["claims"])
    assert meta["n_evidence_low"] == 0


def test_locate_answer_marks_missing_evidence():
    answer = "公司净利润 77.7 亿元，同比增长 12.0%。"
    out = locate_answer(answer, [_ref("材料与答案无关。")])
    assert out["meta"]["n_claims_without_evidence"] == 1

# ----------------------------------------------------------------- 收紧 (a)(b)：方向 + 单位语境


def test_is_plausible_variant_rejects_shrink_direction():
    """S5 实测假阳性：2120.99 万元 ÷100 = "21.21" 命中研报「PE(TTM): 21.21」。

    收紧 (c) 之 (a)：传入 base 时只保留放大方向（|variant| >= |base|），缩小方向一律拒绝。
    """
    assert is_plausible_variant("21.21", base="2120.99") is False
    assert is_plausible_variant("5.29", base="52.92") is False
    assert is_plausible_variant("5886", base="58.86") is True  # 万元 -> 元：放大方向保留
    assert is_plausible_variant("21.21") is True  # 不传 base：只做退化值收紧（向后兼容）


@pytest.mark.parametrize(
    "text,expected",
    [
        ("营收 12.3 亿元", True),
        ("同比增长 8.5%", True),
        ("约为去年同期的 3 倍", True),
        ("提升 5 个百分点", True),
        ("序号: 10", False),
        ("PE(TTM): 21.21", False),
        ("2025年第三季度", False),
        ("", False),
    ],
)
def test_has_unit_context(text, expected):
    """收紧 (c) 之 (b)：命中句必须含单位，否则不算数字证据。"""
    assert has_unit_context(text) is expected


def test_locate_number_claim_requires_unit_context():
    """无单位命中默认被挡；关掉收紧后回到旧行为（便于与历史产物逐条对照）。"""
    claim = Claim(text="净利润 21.21", kind="number", start=0, end=6, numbers=["21.21"])
    text = "序号: 21.21 与本题无关。"
    assert locate_number_claim(claim, [_ref(text)]) == []
    assert len(locate_number_claim(claim, [_ref(text)], require_unit_context=False)) == 1


def test_locate_number_claim_blocks_shrink_variant_false_positive():
    """2120.99 命中「21.21 倍」不得成为证据（方向 + 单位双收紧共同生效）。"""
    claim = Claim(text="净利润 2120.99 万元", kind="number", start=0, end=13, numbers=["2120.99"])
    assert locate_number_claim(claim, [_ref("公司当期市盈率 21.21 倍。")]) == []


def test_locate_number_claim_q3_digit_noise_blocked():
    """裸 "3"（Q3 的 3）在整库表格里到处命中 —— 无单位语境必须被挡。"""
    claim = Claim(text="第三季度 3", kind="number", start=0, end=5, numbers=["3"])
    assert locate_number_claim(claim, [_ref("Q3 报告期内公司经营正常。")]) == []


# ----------------------------------------------------------------- SQL 材料（S 层 / mixed）


def test_sql_rows_to_material_structure():
    rows = [{"利润总额(万元)": 248500}, {"利润总额(万元)": 201000}]
    mat = sql_rows_to_material(rows, sql_text="SELECT 利润总额 FROM t", label="SQL 结果", max_rows=1)
    assert mat["paper_path"] == SQL_MATERIAL_LABEL
    assert mat["chunk_index"] is None
    assert mat["is_sql_material"] is True
    assert mat["full_text"].startswith("SQL: SELECT 利润总额 FROM t")
    assert "利润总额(万元): 248500" in mat["full_text"]
    assert "201000" not in mat["full_text"]  # max_rows=1 生效


def test_sql_rows_to_material_tolerates_scalar_rows_and_blank_sql():
    assert sql_rows_to_material(["24.85", 20.1], sql_text="   ")["full_text"] == "24.85\n20.1"
    assert sql_rows_to_material([])["full_text"] == ""
    assert sql_rows_to_material([{"a": 1}], max_rows=0)["full_text"] == ""


def test_locate_answer_sql_material_covers_sql_only_layer():
    """B2005 口径：sql_only 题答案数字来自 SQL 结果，接入 extra_materials 后不再"无证据"。"""
    answer = "片仔癀的利润总额为 24.85 亿元。"
    mat = sql_rows_to_material([{"利润总额(万元)": 248500}], sql_text="SELECT 利润总额 FROM financial")
    out = locate_answer(answer, [], extra_materials=[mat])
    ce = out["claims"][0]
    assert ce["matched_numbers"] == ["24.85"]
    assert ce["missing_numbers"] == []
    ev = ce["evidence"]
    assert len(ev) == 1
    assert ev[0]["paper_path"] == SQL_MATERIAL_LABEL
    assert ev[0]["score"] == pytest.approx(0.9)  # 仅靠单位换算变体命中
    assert out["meta"]["n_materials"] == 1
    assert out["meta"]["n_extra_materials"] == 1


def test_locate_answer_meta_counts_materials_and_clues():
    """meta 新增字段：材料数 / SQL 材料数 / 线索数 / 收紧开关。"""
    reply = '["营收提升来自渠道扩张"]'
    mat = sql_rows_to_material([{"利润总额(万元)": 248500}])
    out = locate_answer(ANSWER, [_ref()], llm=_fake_llm(reply), embed=_fake_embed, extra_materials=[mat])
    meta = out["meta"]
    assert meta["n_materials"] == 2
    assert meta["n_extra_materials"] == 1
    assert meta["require_unit_context"] is True
    clue = [e for c in out["claims"] for e in c["evidence"] if e["is_clue"]]
    assert clue and meta["n_evidence_clue"] == len(clue)
    assert all(e["note"] == CLUE_NOTE for e in clue)


def test_locate_opinion_claim_evidence_flagged_as_clue():
    """语义定位只作"线索"：必须带 is_clue + note，不得当证据定论。"""
    claim = Claim(text="营收提升来自渠道扩张", kind="opinion", start=0, end=10)
    ev = locate_opinion_claim(claim, [_ref()], embed=_fake_embed)
    assert ev
    assert all(e.hit_type == "semantic" and e.is_clue and e.note == CLUE_NOTE for e in ev)
    assert all(e.tier in ("高", "中", "低") for e in ev)


def test_number_evidence_is_not_clue():
    """数字证据走规则命中，可当证据，不带"线索"标记。"""
    claim = extract_number_claims(ANSWER)[0]
    ev = locate_number_claim(claim, [_ref()])
    assert ev
    assert all(not e.is_clue and e.note == "" for e in ev)

# ----------------------------------------------------------------- SQL 材料容差（答案精度内相等）


def test_sql_value_matches_rounding():
    """SQL 原值 -> 答案展示位舍入：248506.00 万元 应能对上答案 24.85（亿元）。"""
    assert sql_value_matches("24.85", "total_profit: 248506.00") == "248506.00 /10000"
    assert sql_value_matches("21.02", "total_operating_revenue: 210157.59") == "210157.59 /10000"
    assert sql_value_matches("5681.44", "net_profit: 5681.44") == "5681.44 x1"


def test_sql_value_matches_guardrails():
    """收紧：整数答案不做容差；差得远的不命中；空文本安全。"""
    assert sql_value_matches("10", "x: 10.4") == ""  # 整数不容差
    assert sql_value_matches("99.99", "total_profit: 248506.00") == ""
    assert sql_value_matches("24.85", "") == ""
    assert sql_value_matches("abc", "x: 1.0") == ""


def test_locate_number_claim_sql_tolerance_requires_sql_material_flag():
    """容差只对 SQL 材料生效；同一段文本当研报引用时仍走严格口径（不命中）。"""
    claim = Claim(text="利润总额 24.85", kind="number", start=0, end=9, numbers=["24.85"])
    text = "total_profit: 248506.00"
    ref = {"paper_path": "研报A.md", "chunk_index": 0, "full_text": text}
    assert locate_number_claim(claim, [ref]) == []  # 研报口径：严格，不命中
    mat = {"paper_path": SQL_MATERIAL_LABEL, "chunk_index": None, "is_sql_material": True, "full_text": text}
    evidence = locate_number_claim(claim, [mat])
    assert len(evidence) == 1
    e = evidence[0]
    assert e.hit_type == "sql_tolerance"
    assert e.score == pytest.approx(0.8)
    assert e.tier == "中"  # <= 高分线：标记为需人工确认，不冒充精确命中
    assert "~=" in e.note and "248506.00" in e.note
    assert locate_number_claim(claim, [mat], sql_number_tolerance=False) == []


def test_locate_answer_b2005_rounding_gap_closed():
    """B2005 实测：SQL 原值 248506.00 万元 vs 答案 24.85 亿元，容差接入后不再整层"无证据"。"""
    answer = "片仔癀的利润总额为 24.85 亿元。"
    mat = sql_rows_to_material(
        [{"stock_abbr": "片仔癀", "total_profit": "248506.00"}],
        sql_text="SELECT t1.stock_abbr, t1.total_profit FROM income_sheet t1",
    )
    out = locate_answer(answer, [], extra_materials=[mat])
    ce = out["claims"][0]
    assert ce["matched_numbers"] == ["24.85"]
    assert ce["missing_numbers"] == []
    assert ce["evidence"][0]["hit_type"] == "sql_tolerance"
    assert out["meta"]["n_extra_materials"] == 1
    assert out["meta"]["sql_number_tolerance"] is True


# ----------------------------------------------------------------- 评测器修复回归（单位/代码/季度/配额/引表）


def test_detect_material_unit_prefers_long_currency_unit():
    assert detect_material_unit("营业收入（百万元）\n12386.02") == "百万元"
    assert detect_material_unit("金额（亿元）") == "亿元"
    assert detect_material_unit("PE(TTM) 21.21 倍") == "倍"
    # 表头排版空格：最新市 值(亿 元) 必须识别为亿元
    assert detect_material_unit("最新市 值(亿 元)") == "亿元"
    assert detect_material_unit("无单位文本") is None


def test_locate_number_claim_rounding_against_spaced_header_unit():
    """B2080 实测：答案 561 亿元 vs 表头“最新市 值(亿 元)” 的 560.93。"""
    claim = Claim(text="同仁堂（市值约 561 亿元）", kind="number", start=0, end=14, numbers=["561"])
    text = "| 项目 | 最新市 值(亿 元) |\n| 同仁堂 | 560.93 |"
    evidence = locate_number_claim(claim, [_ref(text)])
    assert evidence
    assert evidence[0].matched_numbers == ["561"]


def test_locate_number_claim_percent_row_inherits_claim_unit():
    """B2002 实测：排行表行内只有 2.17，单位 % 只出现在答案 claim 里。"""
    claim = Claim(text="中恒集团（涨幅 2.17%）", kind="number", start=0, end=13, numbers=["2.17"])
    text = "| 涨幅前5 | 2 | 中恒集团 | 2.17 |"
    evidence = locate_number_claim(claim, [_ref(text)])
    assert evidence
    assert evidence[0].matched_numbers == ["2.17"]


def test_extract_number_claims_masks_quarter_and_stock_code():
    answer = "2025Q3 与 2025H1 公司营收 12.3 亿元，代码：423；PZH2108 处于 II 期临床，来源 t2。"
    claims = extract_number_claims(answer)
    assert [n for c in claims for n in c.numbers] == ["12.3"]
    assert mask_quarter_tokens("2025Q3") == "      "
    assert mask_quarter_tokens("2025H1") == "      "
    assert stock_code_numbers("代码：423；代码：000538") == {"423", "000538"}


def test_locate_number_claim_converts_currency_units_with_header_inheritance():
    """123.86 亿元 == 12386.02 百万元，行内无单位时继承表头单位。"""
    claim = Claim(text="营收 123.86 亿元", kind="number", start=0, end=11, numbers=["123.86"])
    evidence = locate_number_claim(claim, [_ref("营业收入（百万元）\n12386.02")])
    assert len(evidence) == 1
    assert evidence[0].matched_numbers == ["123.86"]


def test_locate_number_claim_converts_currency_sign_agnostic():
    """资产减值损失 4319 万元 vs 源表 -43.19 百万元：金额单位换算按绝对值。"""
    claim = Claim(text="资产减值损失 4319 万元", kind="number", start=0, end=14, numbers=["4319"])
    evidence = locate_number_claim(claim, [_ref("项目（百万元）\n资产减值损失 -43.19")])
    assert len(evidence) == 1
    assert evidence[0].matched_numbers == ["4319"]


def test_locate_number_claim_converts_plain_currency_pairs():
    for answer_value, source_value in (("2200", "22"), ("6600", "66")):
        claim = Claim(
            text="金额 %s 万元" % answer_value,
            kind="number",
            start=0,
            end=10,
            numbers=[answer_value],
        )
        evidence = locate_number_claim(claim, [_ref("项目（百万元）\n金额 %s" % source_value)])
        assert len(evidence) == 1, (answer_value, source_value)


def test_locate_number_claim_does_not_treat_multiple_as_currency():
    """2120.99 万元 不应命中 PE(TTM): 21.21 倍。"""
    claim = Claim(text="营业成本 2120.99 万元", kind="number", start=0, end=15, numbers=["2120.99"])
    assert locate_number_claim(claim, [_ref("PE(TTM): 21.21 倍")]) == []


def test_number_evidence_quota_covers_multiple_numbers():
    answer = "营收 12.3 亿元，同比 8.5%。"
    claim = extract_number_claims(answer)[0]
    text = "营收 12.3 亿元。\n说明一。\n营收 12.3 亿元。\n说明二。\n营收 12.3 亿元。\n说明三。\n同比 8.5%。"
    evidence = locate_number_claim(claim, [_ref(text)], window=0, max_per_ref=0, max_total=2)
    covered = {n for e in evidence for n in e.matched_numbers}
    assert len(evidence) == 2
    assert covered == {"12.3", "8.5"}


def test_number_evidence_prefers_claim_cited_table():
    claim = Claim(
        text="营收 12.3 亿元【表：研报B.md#t4】",
        kind="number",
        start=0,
        end=20,
        numbers=["12.3"],
    )
    refs = [
        {"paper_path": "研报A.md", "chunk_index": 0, "full_text": "营收 12.3 亿元。"},
        {"paper_path": "研报B.md#t4", "chunk_index": 0, "full_text": "营收 12.3 亿元。"},
    ]
    evidence = locate_number_claim(claim, refs, max_per_ref=0, max_total=1)
    assert len(evidence) == 1
    assert evidence[0].paper_path == "研报B.md#t4"
