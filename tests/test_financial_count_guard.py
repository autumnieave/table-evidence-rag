# -*- coding: utf-8 -*-
"""B-37 计数一致性守卫单测（零外部依赖）。

覆盖：
- _count_consistency_error：合计类计数 vs 查询结果公司行数的自洽性核验；
- 假阳性防御：聚合行不计入、分组计数之和等于总数、TopN 不与总行数比、差距过大只记录；
- 小样本与无计数表述不触发。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.native_financial import _company_rows, _count_consistency_error  # noqa: E402


def _rows(n: int) -> List[Dict[str, Any]]:
    """构造 n 家公司行（含公司标识）。"""
    return [
        {
            "stock_abbr": f"公司{i}",
            "stock_code": f"60000{i}",
            "asset_liability_ratio": "30.00",
        }
        for i in range(1, n + 1)
    ]


def test_company_rows_ignores_aggregate_rows() -> None:
    """聚合行（无公司标识，如 avg_* ）不计入公司行数（B2042 假阳性根因）。"""
    rows = _rows(37) + [{"avg_rnd_ratio": "0.048", "count": 37}]
    assert _company_rows(rows) == 37


def test_company_rows_dedups_multi_period() -> None:
    """同一公司多期数据只算一家。"""
    rows = [
        {"stock_abbr": "华润三九", "report_year": 2023},
        {"stock_abbr": "华润三九", "report_year": 2024},
        {"stock_abbr": "华润三九", "report_year": 2025},
    ]
    assert _company_rows(rows) == 1


def test_count_ok_when_total_matches() -> None:
    """合计计数与行数一致 → 不报错（B2053 修复后形态）。"""
    assert _count_consistency_error(_rows(14), "2025 年第三季度共有 14 家公司满足条件。") is None


def test_count_flags_off_by_one() -> None:
    """B-37 原始缺陷形态：14 行写“13 家” → 返回纠错说明，且说明含真实行数。"""
    err = _count_consistency_error(_rows(14), "本次共筛选出 13 家公司，盈利质量较高。")
    assert err is not None
    assert "14" in err


def test_count_ok_when_group_sum_matches() -> None:
    """分组计数之和等于总行数 → 自洽（B2051 假阳性防御）。"""
    text = "72 家医药上市公司分组：高盈利组共 18 家，中盈利组共 29 家，低盈利组共 25 家。"
    assert _count_consistency_error(_rows(72), text) is None


def test_count_ok_when_total_present_alongside_groups() -> None:
    """总计数正确、同时出现分组计数 → 自洽。"""
    text = "共 72 家公司，其中高盈利组共 18 家，中盈利组共 29 家，低盈利组共 25 家。"
    assert _count_consistency_error(_rows(72), text) is None


def test_count_skips_far_off_counts() -> None:
    """与行数差距过大 → 无法判定为同一样本笔误，只记录不返回错误。"""
    assert _count_consistency_error(_rows(72), "其中共 12 家披露了研发费用明细。") is None


def test_count_skips_when_aggregate_row_present() -> None:
    """结果含聚合行时，按公司行数核验（37 家正确，不应被 38 行误判）。"""
    rows = _rows(37) + [{"avg_rnd_ratio": "0.048"}]
    assert _count_consistency_error(rows, "2025 年第三季度共有 37 家净利润同比增长的医药公司。") is None
    err = _count_consistency_error(rows, "2025 年第三季度共有 38 家净利润同比增长的医药公司。")
    assert err is not None


def test_count_skips_small_samples() -> None:
    """1~2 行小样本不核验（噪声大）。"""
    assert _count_consistency_error(_rows(2), "共有 1 家公司。") is None


def test_count_skips_without_total_claim() -> None:
    """只有 TopN/排名表述、无合计类计数 → 不核验（B2046 假阳性防御）。"""
    text = "研发费用占比排名前 10 的公司中，成都先导与百普赛斯实现高增长。"
    assert _count_consistency_error(_rows(20), text) is None


def test_count_handles_item_units() -> None:
    """“合计 N 项/条”等非“家”单位同样纳入核验。"""
    err = _count_consistency_error(_rows(10), "本次查询合计 9 项指标需要关注。")
    assert err is not None
    assert "10" in err


def test_count_skips_empty_analysis() -> None:
    assert _count_consistency_error(_rows(10), "") is None


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """记录每次调用的 messages，按序返回预设文案。"""

    def __init__(self, contents: List[str]) -> None:
        self.contents = list(contents)
        self.calls: List[List[Dict[str, Any]]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs.get("messages") or [])
        idx = min(len(self.calls) - 1, len(self.contents) - 1)
        return _FakeResponse(self.contents[idx])


class _FakeClient:
    def __init__(self, contents: List[str]) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(contents)})()


class _FakeRag:
    """_generate_analysis 所需的最小 rag 替身（零外部依赖）。"""

    def __init__(self, contents: List[str]) -> None:
        self.llm_generator = type("_Gen", (), {"client": _FakeClient(contents)})()
        self.config = type("_Cfg", (), {"LLM_MODEL": "fake-model", "AGENT_ENABLE_THINKING": False})()


def _user_content_of(call_messages: List[Dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in call_messages)


def test_generate_analysis_injects_row_count_and_retries_on_mismatch() -> None:
    """B-37 接线：用户消息注入【结果行数】，计数不一致时带纠错说明重试一次。"""
    from tools.native_financial import _generate_analysis

    rag = _FakeRag(["本次共筛选出 13 家公司，盈利质量较高。", "本次共筛选出 14 家公司，盈利质量较高。"])
    out = _generate_analysis(rag, "找出满足条件的公司数量", _rows(14))

    calls = rag.llm_generator.client.chat.completions.calls
    assert len(calls) == 2, "计数不一致应触发一次重试"
    assert "结果行数：14 家公司行" in _user_content_of(calls[0])
    retry_text = _user_content_of(calls[1])
    assert "计数不一致" in retry_text and "14" in retry_text
    assert "14" in out


def test_generate_analysis_accepts_consistent_count_without_retry() -> None:
    """计数一致时不重试（只调用一次）。"""
    from tools.native_financial import _generate_analysis

    rag = _FakeRag(["本次共筛选出 14 家公司，盈利质量较高。"])
    out = _generate_analysis(rag, "找出满足条件的公司数量", _rows(14))

    assert len(rag.llm_generator.client.chat.completions.calls) == 1
    assert "14" in out
