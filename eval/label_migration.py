# -*- coding: utf-8 -*-
"""评测标注的坐标迁移（阶段 2 · 契约实现）。

原则
----
- **只按坐标映射，不按文本相似度**：``source_file + table_id + row_index + col_index``；
- ``table_id`` 变化时，允许用 ``source_file + table_caption + row_index + col_index``
  做二次匹配；
- 旧标注只有 ``row_start / row_end`` 时，按行区间落到新 anchor；命中多条即
  ``ambiguous``，不猜；
- 三态：``mapped`` / ``ambiguous`` / ``unmapped``；后两者必须进人工复核，不静默丢弃。

边界
----
当前 L5 工作单的 10 条人工判定是「题 × rank 片段」结构，不含
``table_id / row_index / col_index``，**不属于本迁移契约的输入**；
等阶段 3 检索冻结后重新标注时，再按本契约生成坐标化标注。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _as_int(value: Any) -> Optional[int]:
    """宽容转 int；失败返回 ``None``。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field(anchor: Mapping[str, Any], *names: str) -> Any:
    """按候选字段名取第一个非空值。"""
    for name in names:
        value = anchor.get(name)
        if value not in (None, ""):
            return value
    return None


def _source_file(value: Mapping[str, Any]) -> str:
    """取源文件名（兼容旧字段名）。"""
    return str(_field(value, "source_file", "file_path", "file", "filename") or "")


def _table_id(value: Mapping[str, Any]) -> str:
    """取表 ID（兼容核心六项里的 ``source``）。"""
    return str(_field(value, "table_id", "source") or "")


def _caption(value: Mapping[str, Any]) -> str:
    """取表标题（兼容 ``caption`` / ``table_caption``）。"""
    return str(_field(value, "table_caption", "caption") or "")


def _match_exact(old: Mapping[str, Any], new_anchors: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """精确匹配 ``source_file + table_id + row_index + col_index``。"""
    source_file = _source_file(old)
    table_id = _table_id(old)
    row = _as_int(old.get("row_index"))
    col = _as_int(old.get("col_index"))
    if not source_file or not table_id or row is None or col is None:
        return []
    return [
        a for a in new_anchors
        if _source_file(a) == source_file
        and _table_id(a) == table_id
        and _as_int(a.get("row_index")) == row
        and _as_int(a.get("col_index")) == col
    ]


def _match_caption(old: Mapping[str, Any], new_anchors: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """表 ID 变化时的二次匹配：``source_file + table_caption + row + col``。"""
    source_file = _source_file(old)
    caption = _caption(old)
    row = _as_int(old.get("row_index"))
    col = _as_int(old.get("col_index"))
    if not source_file or not caption or row is None or col is None:
        return []
    return [
        a for a in new_anchors
        if _source_file(a) == source_file
        and _caption(a) == caption
        and _as_int(a.get("row_index")) == row
        and _as_int(a.get("col_index")) == col
    ]


def _match_range(old: Mapping[str, Any], new_anchors: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """旧标注只给 ``row_start / row_end`` 时，按区间匹配。"""
    source_file = _source_file(old)
    table_id = _table_id(old)
    row_start = _as_int(old.get("row_start"))
    row_end = _as_int(old.get("row_end"))
    col = _as_int(old.get("col_index"))
    if not source_file or row_start is None or row_end is None or col is None:
        return []
    return [
        a for a in new_anchors
        if _source_file(a) == source_file
        and (not table_id or _table_id(a) == table_id)
        and row_start <= (_as_int(a.get("row_index")) if _as_int(a.get("row_index")) is not None else -1) < row_end
        and _as_int(a.get("col_index")) == col
    ]


def migrate_anchor(old: Mapping[str, Any], new_anchors: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """迁移单个旧标注，返回三态结果。

    Args:
        old: 旧标注（至少含 ``source_file`` / ``row_index`` / ``col_index``；
            或含 ``row_start`` / ``row_end``）
        new_anchors: 新集合的 anchor 列表（:func:`data.row_facts.build_row_facts` 输出）

    Returns:
        ``{"status": "mapped|ambiguous|unmapped", "rule": ..., "old": ..., "new": ...}``
    """
    matches = _match_exact(old, new_anchors)
    rule = "exact"
    if not matches:
        matches = _match_caption(old, new_anchors)
        rule = "caption"
    if not matches:
        matches = _match_range(old, new_anchors)
        rule = "range"
    if len(matches) == 1:
        return {"status": "mapped", "rule": rule, "old": dict(old), "new": dict(matches[0])}
    if len(matches) > 1:
        return {"status": "ambiguous", "rule": rule, "old": dict(old), "matches": [dict(m) for m in matches]}
    return {
        "status": "unmapped",
        "rule": "none",
        "old": dict(old),
        "reason": "缺少 source_file/row/col 或新 anchor 中无同坐标项",
    }


def migrate_labels(
    labels: Sequence[Mapping[str, Any]],
    new_anchors: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """批量迁移，返回逐条结果与三态汇总。"""
    items = [migrate_anchor(label, new_anchors) for label in labels]
    summary = {"total": len(items), "mapped": 0, "ambiguous": 0, "unmapped": 0}
    for item in items:
        summary[item["status"]] += 1
    return {"summary": summary, "items": items}
