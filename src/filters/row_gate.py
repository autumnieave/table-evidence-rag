# -*- coding: utf-8 -*-
"""低信息「裸表行」闸门（B-59 A）——把**提到公司但不讲公司**的表格碎片从候选里排除。

为什么需要
----------
公司类问题的最终 top-10 会被一种碎片系统性劫持（实测 2026-10-16）：

- 全库 57,178 个 chunk 中，**86%（49,025）不足 200 字**、60%（34,045）不足 100 字；表格行按「一行一 chunk」入库。
- 含「语料公司名 + 表行数字」的 chunk 全库只有 **864 条（1.5%）**，却占了 4 道典型
  「公司 + 财务指标」题 top-10 的 **36/40**；
- 这些劫持 chunk 实际只有 **54 字 / 135 字**（如 ``300181. SZ  佐力药业  0.72  2.73  3.71  5.08  2.53  130.81``），
  **不含任何可回答内容**，但因为「公司名 + 一串数字」与「公司 + 指标」类查询词面高度重合，精排给它们 0.65~0.70 的高分。

因此本闸门做的是**候选层的结构性排除**（不是排序调整）：长 ≤ ``max_chars``、以数字为主、
提到某家公司、且**不含任何指标词**的 chunk = 「裸表行」，直接丢弃。

为什么不会误杀「业绩对比表」
----------------------------
判据里有一条硬性豁免：**只要含指标词（营业收入 / 归母净利润 / 毛利率 / 研发费用 / 现金流 / ROE …）就保留**。
真实业绩对比表 / 财务预测表 / 三表都有表头指标词 ⇒ 不受影响（单测覆盖）。

边界
----
- 只做**排除**，不做降权（降权需要改精排打分链路，留待数据结构化重构后一并考虑）。
- 依赖 ``known_names``（语料公司简称闭词表）识别「提到哪家公司」；词表为空时退化为「证券代码样式」识别。
- 长 > ``max_chars`` 的正文 chunk 一律保留（正文里出现的表格行属于正常上下文）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: 「裸表行」允许的最大长度；超过该长度视为正文，一律保留
MAX_ROW_CHARS: int = 200

#: 指标词表（**硬性豁免**）：出现任一即视为「业绩/财务表或分析正文」，不排除。
#: 刻意不收「利润」「资金」这类单字/通用词，避免把「待分配利润 / 货币资金」数据行豁免掉。
METRIC_TERMS: Tuple[str, ...] = (
    "营业收入", "营收", "营业成本", "销售收入", "营业收入同比",
    "归母净利润", "净利润", "扣非", "毛利", "毛利率", "净利率", "利润率",
    "研发费用", "研发投入", "研发支出", "销售费用", "管理费用", "财务费用", "费用率",
    "经营活动现金流", "现金流", "资产负债率", "净资产收益率", "ROE", "每股收益", "EPS",
    "存货", "应收账款", "应付账款", "商誉", "合同负债", "分红", "股息", "派息", "股利",
    "市值", "市盈率", "PE", "PB", "估值", "目标价", "市占率", "产能", "产量", "销量",
    "同比", "环比", "增速", "增长率", "复合增速",
)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
#: 同一行内 ≥3 个空白分隔的数字（如 `0.72  2.73  3.71  5.08`）
_ROW_RE = re.compile(r"(?:\d+(?:\.\d+)?%?\s+){3,}")
#: `键: 数字` 形态（如 `待分配利润 (2023年报): 7`）
_KV_RE = re.compile(r"[:：]\s*\d")
#: 证券代码样式（`300181.SZ` / `300181 . SZ` / `SH603998`）
_CODE_RE = re.compile(r"(?:\d{6}\s*\.?\s*(?:SZ|SH|sz|sh)|(?:SZ|SH)\s*\d{6})")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def count_numbers(text: str) -> int:
    """数字 token 数（含百分比 / 千分位）"""
    return len(_NUM_RE.findall(text or ""))


def cjk_ratio(text: str) -> float:
    """中文字符占比（分母为去空白后的长度）"""
    compact = re.sub(r"\s", "", text or "")
    if not compact:
        return 0.0
    return len(_CJK_RE.findall(compact)) / len(compact)


def has_stock_reference(text: str, known_names: Optional[Iterable[str]] = None) -> bool:
    """chunk 是否提到某家公司（语料简称闭词表命中，或出现证券代码样式）"""
    if _CODE_RE.search(text or ""):
        return True
    return any(n and n in text for n in (known_names or []))


def is_rowlike(text: str, min_numbers: int = 3) -> bool:
    """是否呈现「数据行」形态：同行 ≥3 个数字 / ≥3 个 `键: 数字` / 数字多且中文极少"""
    if _ROW_RE.search(text or ""):
        return True
    if len(_KV_RE.findall(text or "")) >= min_numbers:
        return True
    return count_numbers(text) >= 4 and cjk_ratio(text) < 0.45


def is_lowinfo_row(
    content: str,
    known_names: Optional[Iterable[str]] = None,
    max_chars: int = MAX_ROW_CHARS,
    metric_terms: Optional[Iterable[str]] = None,
) -> bool:
    """是否为应排除的「裸表行」碎片。

    Args:
        content: chunk 正文
        known_names: 语料公司简称闭词表
        max_chars: 长度上限（超过即视为正文，默认 :data:`MAX_ROW_CHARS`）
        metric_terms: 指标词表（默认 :data:`METRIC_TERMS`，出现任一即豁免）

    Returns:
        True = 应排除
    """
    text = (content or "").strip()
    if not text or len(text) > max_chars:
        return False
    terms = tuple(metric_terms) if metric_terms is not None else METRIC_TERMS
    if any(t and t in text for t in terms):
        return False
    if not has_stock_reference(text, known_names):
        return False
    if count_numbers(text) < 3:
        return False
    return is_rowlike(text)


def apply_row_gate(
    candidates: List[Dict[str, Any]],
    known_names: Optional[Iterable[str]] = None,
    max_chars: int = MAX_ROW_CHARS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """排除候选里的「裸表行」碎片，保持输入顺序。

    Args:
        candidates: 检索结果列表，每项需含 ``payload`` 字段（读 ``content``）
        known_names: 语料公司简称闭词表
        max_chars: 长度上限

    Returns:
        ``(kept, meta)``；``meta`` = ``{"applied", "dropped", "n_in", "n_kept", "max_chars"}``
    """
    items = list(candidates or [])
    meta: Dict[str, Any] = {
        "applied": False,
        "dropped": 0,
        "n_in": len(items),
        "n_kept": len(items),
        "max_chars": int(max_chars),
    }
    if not items:
        return items, meta
    kept = [
        c for c in items
        if not is_lowinfo_row(
            str((c.get("payload") or {}).get("content") or ""), known_names, max_chars
        )
    ]
    meta["dropped"] = len(items) - len(kept)
    meta["n_kept"] = len(kept)
    meta["applied"] = meta["dropped"] > 0
    return kept, meta