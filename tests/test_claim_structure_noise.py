"""markdown 结构行噪声过滤（is_structure_noise）单测。

背景：答案里的 `### 1. 利润总额历史趋势分析` 会被抽成数字 claim（"命中"数字 = 标题序号），
产品层点开引用会显示成荒谬断言。规则收紧为双重条件：结构行 + 数字全为小整数序号。
两层（评测层 locate_answer / 产品层 locate_claims_per_reference）共享同一开关。
零外部依赖：不触网、不读盘、不依赖 LLM/embedding 服务。
"""

import pytest

from tools.claim_evidence_locator import (
    STRUCTURE_NOISE_MAX_ORDINAL,
    extract_number_claims,
    is_number_sequence_noise,
    is_structure_noise,
    locate_answer,
    locate_claims_per_reference,
)

ANSWER = """### 1. 利润总额历史趋势分析
公司营收 12.3 亿元，同比增长 8.5%。
### 2. 影响利润总额变化的关键因素
- 2 家公司净利润为负。
3. 2024 年毛利率降至 42.7%。
"""

REF = {
    "paper_path": "研报A.md",
    "chunk_index": 0,
    "text": "片段",
    "full_text": "公司营收 12.3 亿元，同比增长 8.5%。2024 年毛利率降至 42.7%。2 家公司净利润为负。",
}


@pytest.mark.parametrize(
    "text,numbers,expected",
    [
        ("### 1. 利润总额历史趋势分析", ["1"], True),
        ("### 2. 影响利润总额变化的关键因素", ["2"], True),
        ("3. 结论与展望", ["3"], True),
        ("[4] 风险提示", ["4"], True),
        ("（5）后续计划", ["5"], True),
        # 含真实数值 → 不判噪声
        ("### 2024 年营收 12.3 亿元", ["12.3"], False),
        ("1. 公司营收 12.3 亿元", ["1", "12.3"], False),
        # 列表符号不是"序号"结构行 → 保留
        ("- 2 家公司净利润为负", ["2"], False),
        ("*   **毛利率下降**：46.8% → 42.7%", ["46.8", "42.7"], False),
        # 序号超阈值 → 不判噪声（防误伤真实小数值）
        (f"{STRUCTURE_NOISE_MAX_ORDINAL + 1}. 某指标为 13", ["13"], False),
        # 空值安全
        ("", ["1"], False),
        ("### 1. 无数字标题", [], False),
    ],
)
def test_is_structure_noise(text, numbers, expected):
    assert is_structure_noise(text, numbers) is expected


@pytest.mark.parametrize(
    "text,numbers,expected",
    [
        # ASCII 柱状图坐标轴：无指标名、无单位、连续 3+ 数字 -> 噪声
        ("             0        1000       2000       3000       4000\n",
         ["0", "1000", "2000", "3000", "4000"], True),
        # 回归：L1 的空格折叠会把“0  1000”折成“01000”，numbers 只剩 2 个；
        # 噪声判定必须按原始 token 再数一次，否则轴行会漏过。
        ("             0        1000       2000       3000\n",
         ["01000", "3000"], True),
        ("0, 1000, 2000", ["0", "1000", "2000"], True),
        ("| 0 | 1000 | 2000 |", ["0", "1000", "2000"], True),
        # 少于 3 个数字 -> 保留
        ("0 1000", ["0", "1000"], False),
        # 含指标名 -> 保留
        ("营收 0 1000 2000", ["0", "1000", "2000"], False),
        # 含单位 -> 保留
        ("0% 10% 20%", ["0", "10", "20"], False),
        # 空值安全
        ("", ["0", "1000", "2000"], False),
    ],
)
def test_is_number_sequence_noise(text, numbers, expected):
    assert is_number_sequence_noise(text, numbers) is expected


def test_extract_number_claims_drops_ascii_axis_row():
    answer = "```\n+------------------------>\n0   1000   2000   3000   4000\n```\n"
    claims = extract_number_claims(answer)
    assert not any("1000" in c.numbers and "4000" in c.numbers for c in claims)


def test_extract_number_claims_drops_structure_noise_by_default():
    claims = extract_number_claims(ANSWER)
    texts = [c.text for c in claims]
    assert not any(t.startswith("### 1.") for t in texts)
    assert not any(t.startswith("### 2.") for t in texts)
    # 真断言保留：含 12.3 / 8.5 的那句
    assert any("12.3" in c.numbers for c in claims)
    assert any("42.7" in c.numbers for c in claims)


def test_extract_number_claims_can_keep_structure_noise():
    kept = extract_number_claims(ANSWER, drop_structure_noise=False)
    assert any(c.text.startswith("### 1.") for c in kept)
    assert len(kept) > len(extract_number_claims(ANSWER))


def test_locate_answer_reports_dropped_structure_claims():
    out = locate_answer(ANSWER, [REF])
    meta = out["meta"]
    assert meta["drop_structure_noise"] is True
    assert meta["n_claims_dropped_structure"] >= 2  # 两个 `### N.` 标题

    out2 = locate_answer(ANSWER, [REF], drop_structure_noise=False)
    assert out2["meta"]["n_claims_dropped_structure"] == 0


def test_product_layer_shares_the_rule():
    blocks = locate_claims_per_reference(ANSWER, [REF])
    claims = [c["claim"] for c in blocks[0]]
    assert not any(c.startswith("### 1.") for c in claims)
    assert any("12.3" in c["matched_numbers"] for c in blocks[0])
