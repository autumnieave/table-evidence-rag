"""L5-C「期间过滤」纯函数（**默认不启用、不接检索主链路**，供 L5-C 实验与后续线上过滤共用）。

**口径（2026-10-17 决策）**：不同 ``doc_type`` 用**不同的期间键**——

+--------------------------+-------------------------+--------------------------------------------------------+
| ``doc_type``             | 期间键                  | 为什么                                                 |
+==========================+=========================+========================================================+
| ``stock``（个股研报）    | ``report_period``       | 个股研报是对**某个报告期**的点评/跟踪，报告期有语义     |
+--------------------------+-------------------------+--------------------------------------------------------+
| ``industry``（行业研报） | ``publish_year``        | 行业研报是市场/策略/周报，**没有报告期语义**，只有发布年 |
+--------------------------+-------------------------+--------------------------------------------------------+

**不猜原则**：键取不到时一律「**不过滤**」（返回 ``None`` / ``True``），宁可放过、不误杀。

**边界**：``publish_year`` 是**补充字段**，**不替代** ``report_period``；本模块只做「键选择 + 匹配判定」，
不含阈值、不含召回策略，也不改检索排序。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

#: ``doc_type`` → 期间键字段名（未登记的 ``doc_type`` 回退到 :data:`_FALLBACK_KEY`）
PERIOD_KEY_BY_DOC_TYPE: Dict[str, str] = {
    "stock": "report_period",
    "industry": "publish_year",
}

#: 未登记 ``doc_type`` 的回退键（本期只有 ``report_period`` 这一个语义可用的键）
_FALLBACK_KEY: str = "report_period"

_YEAR_RE = re.compile(r"(20\d{2})")
_QUARTER_RE = re.compile(r"[QH](\d)")


def _is_empty(value: Any) -> bool:
    """空值统一判定（``None`` / ``""`` / ``[]`` / ``{}`` 都算「取不到」）。"""
    return value in (None, "", [], {})


def period_key_name(payload: Dict[str, Any]) -> str:
    """该 chunk 用于「期间过滤」的**字段名**（按 ``doc_type`` 选键）。"""
    return PERIOD_KEY_BY_DOC_TYPE.get(str(payload.get("doc_type") or ""), _FALLBACK_KEY)


def period_key(payload: Dict[str, Any]) -> Optional[str]:
    """该 chunk 的**期间键值**（个股形如 ``2025Q3``、行业形如 ``2025``）；取不到返回 ``None``。"""
    value = payload.get(period_key_name(payload))
    return None if _is_empty(value) else str(value)


def period_year(payload: Dict[str, Any]) -> Optional[int]:
    """从期间键里取 4 位年份；取不到返回 ``None``（= 不参与过滤）。"""
    key = period_key(payload)
    if not key:
        return None
    match = _YEAR_RE.search(key)
    return int(match.group(1)) if match else None


def period_quarter(payload: Dict[str, Any]) -> Optional[int]:
    """从期间键里取季度（``25Q3`` / ``25H1`` → ``3`` / ``1``）；没有季度信息返回 ``None``。"""
    key = period_key(payload)
    if not key:
        return None
    match = _QUARTER_RE.search(key.upper())
    return int(match.group(1)) if match else None


def matches_period(payload: Dict[str, Any], year: int, quarter: Optional[int] = None) -> Optional[bool]:
    """该 chunk 是否落在目标期间。

    - ``None``：**无法判断**（该 chunk 取不到期间键）⇒ 调用方应**放行**（不过滤），不要当不匹配；
    - ``True`` / ``False``：年份匹配（且季度可比时季度也匹配）/ 年份不匹配。

    季度规则：调用方给了 ``quarter`` 而 chunk 只有年份（如行业研报的 ``2025``）时判 **True**（放行，不误杀）。
    """
    chunk_year = period_year(payload)
    if chunk_year is None:
        return None
    if chunk_year != year:
        return False
    if quarter is None:
        return True
    chunk_quarter = period_quarter(payload)
    if chunk_quarter is None:
        return True
    return chunk_quarter == quarter
