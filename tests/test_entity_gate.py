# -*- coding: utf-8 -*-
"""L5-B：实体硬闸门单测（纯逻辑、零外部依赖）

覆盖：简称/代码匹配、闭词表归一（resolve_names）、命中本体判定（entity_hit）、
无实体标注候选的取舍、安全闸门（min_keep，默认 0）的开与关、顺序保持。
"""
from __future__ import annotations

from filters.entity_gate import (
    MIN_CONTAIN_NAME_LEN,
    apply_entity_gate,
    code_matches,
    entity_hit,
    entity_matches,
    name_matches,
    resolve_names,
)

VOCAB = ["桂林三金", "康缘药业", "云南白药", "华兰生物"]


def _cand(stock_name="", stock_code="", tag=None):
    """构造候选（payload 结构最小化）"""
    return {"payload": {"stockName": stock_name, "stockCode": stock_code}, "tag": tag}


# --------------------------------------------------------------------------
# 简称匹配
# --------------------------------------------------------------------------
def test_name_exact_match() -> None:
    assert name_matches("康缘药业", "康缘药业") is True
    assert name_matches(" 康缘药业 ", "康缘药业") is True


def test_name_prefix_match_absorbs_short_name() -> None:
    """简称/全称差异：「康缘」↔「康缘药业」（较短一侧是前缀）"""
    assert name_matches("康缘药业", "康缘") is True
    assert name_matches("康缘", "康缘药业") is True
    assert name_matches("康缘药业", "康缘药业股份有限公司") is True


def test_name_generic_suffix_word_not_matched() -> None:
    """通用后缀词不是前缀 → 不匹配（防「药业」「生物」误伤）"""
    assert MIN_CONTAIN_NAME_LEN == 2
    assert name_matches("康缘药业", "药业") is False
    assert name_matches("华兰生物", "生物") is False
    assert name_matches("康缘药业", "药") is False


def test_name_empty_is_no_match() -> None:
    assert name_matches("", "康缘药业") is False
    assert name_matches("康缘药业", None) is False


# --------------------------------------------------------------------------
# 代码匹配
# --------------------------------------------------------------------------
def test_code_matches_int_and_str() -> None:
    """payload 侧 stockCode 是 int，查询侧是 str"""
    assert code_matches(600993, "600993") is True
    assert code_matches("600993", " 600993 ") is True


def test_code_mismatch_and_empty() -> None:
    assert code_matches(600993, "000538") is False
    assert code_matches("", "600993") is False
    assert code_matches(600993, None) is False


# --------------------------------------------------------------------------
# 闭词表归一
# --------------------------------------------------------------------------
def test_resolve_names_unique_substring_hit() -> None:
    """「三金」在闭词表里唯一命中「桂林三金」→ 归一"""
    assert resolve_names("三金", VOCAB) == ["桂林三金"]


def test_resolve_names_exact_hit_returns_as_is() -> None:
    assert resolve_names("桂林三金", VOCAB) == ["桂林三金"]


def test_resolve_names_ambiguous_returns_raw() -> None:
    """闭词表内多命中（歧义）→ 不归一，退回原字符串"""
    ambiguous = ["康缘药业", "康缘生物"]
    assert resolve_names("康缘", ambiguous) == ["康缘"]


def test_resolve_names_no_vocab_or_empty_query() -> None:
    """无词表时不归一（退化为原字符串匹配）；空查询返回空列表"""
    assert resolve_names("三金", None) == ["三金"]
    assert resolve_names("三金", []) == ["三金"]
    assert resolve_names("", VOCAB) == []
    assert resolve_names(None, VOCAB) == []


def test_resolve_names_absent_from_vocab_returns_raw() -> None:
    """不在语料内（如「香雪制药」）→ 原样返回，由调用方按池内零命中处理"""
    assert resolve_names("香雪制药", VOCAB) == ["香雪制药"]


# --------------------------------------------------------------------------
# 单候选判定
# --------------------------------------------------------------------------
def test_no_query_entity_keeps_everything() -> None:
    assert entity_matches({}, None, None) is True
    assert entity_matches({"stockName": ""}, None, "") is True


def test_entity_hit_by_name_or_code() -> None:
    assert entity_matches({"stockName": "康缘药业", "stockCode": 600557}, "康缘药业", None) is True
    assert entity_matches({"stockName": "康缘药业", "stockCode": 600557}, None, "600557") is True
    assert entity_matches({"stockName": "云南白药", "stockCode": 538}, "康缘药业", "600557") is False


def test_entity_hit_uses_vocabulary_normalization() -> None:
    """「三金」靠闭词表才能命中「桂林三金」；无词表则命中失败（B-58 名称匹配缺口）"""
    payload = {"stockName": "桂林三金", "stockCode": 302}
    assert entity_hit(payload, "三金", None, VOCAB) is True
    assert entity_hit(payload, "三金", None) is False
    assert entity_matches(payload, "三金", None, known_names=VOCAB) is True


def test_entity_hit_ignores_unlabeled_candidates() -> None:
    """entity_hit 只看实体本体；无标注候选一律不算命中（与 keep_unknown 无关）"""
    assert entity_hit({"stockName": "", "stockCode": ""}, "康缘药业", None) is False
    assert entity_hit({}, "康缘药业", None) is False


def test_entity_hit_false_without_query_entity() -> None:
    assert entity_hit({"stockName": "康缘药业"}, None, None) is False


def test_candidate_without_entity_label_kept_by_default() -> None:
    """行业研报（stockName/stockCode 皆空）默认**保留**（keep_unknown_entity 默认 True）"""
    assert entity_matches({"stockName": "", "stockCode": ""}, "康缘药业", None) is True
    assert (
        entity_matches({"stockName": "", "stockCode": ""}, "康缘药业", None, keep_unknown_entity=False)
        is False
    )


# --------------------------------------------------------------------------
# 闸门（安全阈值）
# --------------------------------------------------------------------------
def _pool(n_match: int, n_other: int = 40, name: str = "康缘药业", code: int = 600557):
    return [_cand(name, code, "m%d" % i) for i in range(n_match)] + [
        _cand("云南白药", 538, "o%d" % i) for i in range(n_other)
    ]


def test_gate_applied_when_matched_reaches_min_keep() -> None:
    cands = _pool(25, 25)
    kept, meta = apply_entity_gate(cands, stock_name="康缘药业", min_keep=20)
    assert meta["applied"] is True
    assert meta["n_in"] == 50
    assert meta["n_matched"] == 25
    assert meta["n_kept"] == 25
    assert all(c["payload"]["stockName"] == "康缘药业" for c in kept)


def test_gate_keeps_unlabeled_by_default() -> None:
    """默认 keep_unknown_entity=True：无标注候选（行业研报）保留 ⇒ 保留数 > 实体命中数"""
    cands = _pool(25, 20) + [_cand("", "")] * 5
    kept, meta = apply_entity_gate(cands, stock_name="康缘药业", min_keep=3)
    assert meta["applied"] is True
    assert meta["n_matched"] == 25
    assert meta["n_kept"] == 30
    assert meta["n_kept"] > meta["n_matched"]


def test_gate_drops_unlabeled_when_asked() -> None:
    cands = _pool(25, 20) + [_cand("", "")] * 5
    kept, meta = apply_entity_gate(
        cands, stock_name="康缘药业", min_keep=3, keep_unknown_entity=False
    )
    assert meta["applied"] is True
    assert meta["n_kept"] == 25


def test_gate_skipped_when_matched_below_min_keep() -> None:
    """显式开计数闸门时的保护：实体命中不足 → 原样返回，不做任何过滤"""
    cands = _pool(8, 42)
    kept, meta = apply_entity_gate(cands, stock_name="康缘药业", min_keep=20)
    assert meta["applied"] is False
    assert meta["n_matched"] == 8
    assert "min_keep" in meta["reason"]
    assert kept == cands


def test_gate_default_min_keep_is_zero() -> None:
    """默认 min_keep=0（不启用计数闸门）：命中 5 条的小池也会过滤"""
    cands = _pool(5, 45)
    kept, meta = apply_entity_gate(cands, stock_name="康缘药业")
    assert meta["min_keep"] == 0
    assert meta["applied"] is True
    assert meta["n_kept"] == 5


def test_gate_skipped_without_entity() -> None:
    cands = _pool(25, 25)
    kept, meta = apply_entity_gate(cands, stock_name=None, stock_code=None, min_keep=20)
    assert meta["applied"] is False
    assert meta["reason"] == "未解析出实体"
    assert kept == cands


def test_gate_empty_candidates_is_safe() -> None:
    kept, meta = apply_entity_gate([], stock_name="康缘药业")
    assert kept == []
    assert meta["applied"] is False
    assert meta["reason"] == "空候选"


def test_gate_preserves_input_order() -> None:
    cands = [_cand("康缘药业", 600557, "m0"), _cand("云南白药", 538, "o0")] * 24
    kept, meta = apply_entity_gate(cands, stock_code="600557", min_keep=20)
    assert meta["applied"] is True
    assert [c["tag"] for c in kept] == [c["tag"] for c in cands if c["tag"].startswith("m")]


def test_gate_min_keep_zero_always_applies() -> None:
    """min_keep=0（默认）时只要解析出实体就过滤，不看命中数"""
    kept, meta = apply_entity_gate(_pool(1, 10), stock_name="康缘药业", min_keep=0)
    assert meta["applied"] is True
    assert len(kept) == 1


def test_gate_zero_hit_pool_filters_other_but_keeps_unlabeled() -> None:
    """池内零命中（公司不在语料内，7 家 B-58 案例）默认仍过滤：他公司候选移除、无标注保留

    这是**刻意的**：公司不在语料时答案不该引别家公司的数字（跨公司污染是幻觉入口），
    而行业研报（无标注）保留保证上下文不会被清空。
    """
    cands = [_cand("云南白药", 538, "o%d" % i) for i in range(30)] + [_cand("", "")] * 5
    kept, meta = apply_entity_gate(cands, stock_name="香雪制药", known_names=VOCAB)
    assert meta["applied"] is True
    assert meta["n_matched"] == 0
    assert len(kept) == 5
    assert all(c["payload"]["stockName"] == "" for c in kept)


def test_gate_zero_hit_pool_skipped_when_min_keep_explicitly_set() -> None:
    """显式开计数闸门（min_keep=3）时，池内零命中 → 不过滤（保留旧行为，供 A/B）"""
    cands = [_cand("云南白药", 538, "o%d" % i) for i in range(30)]
    kept, meta = apply_entity_gate(cands, stock_name="香雪制药", min_keep=3, known_names=VOCAB)
    assert meta["applied"] is False
    assert meta["n_matched"] == 0
    assert len(kept) == 30