# -*- coding: utf-8 -*-
"""答案实体覆盖检查：问题点名的主体是否被回答覆盖、是否被替换成其他主体。

纯规则模块，不调用 LLM、不访问 Qdrant。公司词表与代码/简称别名由调用方注入，
模块内只保留确定的一对一别名（999 → 华润三九）。

判定口径
--------
- 问题没有点名公司主体（行业级 / 市场级）→ 跳过，不限制答案提到哪些公司；
- 问题点名了主体：
  - ``missing_query_entities``：答案没提到的点名主体；
  - ``unexpected_entities``：答案提到、但不在点名集合里的主体；
  - ``drift``：答案出现了其他主体，且**没有覆盖任何**点名主体（主体被整体替换）；
  - ``needs_repair``：``drift`` 或答案完全没提点名主体。

这样既抓住 B2007|2 的“999/白云山 → 云南白药/同仁堂”，又不会误伤
“华润三九与同行相比”这类点名主体已覆盖、额外公司只是对照的正常回答。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from filters.query_understanding import extract_query_entities

#: 语料内确定的一对一简称；与 query_understanding 的口径保持一致。
DEFAULT_ENTITY_ALIASES: Tuple[Tuple[str, str], ...] = (("999", "华润三九"),)


@dataclass(frozen=True)
class EntityCoverage:
    """一次答案实体覆盖检查的结果。"""

    query_entities: Tuple[str, ...] = ()
    answer_entities: Tuple[str, ...] = ()
    missing_query_entities: Tuple[str, ...] = ()
    unexpected_entities: Tuple[str, ...] = ()
    drift: bool = False
    needs_repair: bool = False
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """转为可写入 JSON / trace 的字典。"""
        return asdict(self)


def _apply_aliases(text: str, aliases: Mapping[str, str]) -> str:
    """把简称/代码替换成标准公司名；纯数字别名要求前后不紧邻数字。"""
    out = str(text or "")
    for alias, canonical in sorted(aliases.items(), key=lambda item: -len(item[0])):
        alias = str(alias or "")
        canonical = str(canonical or "")
        if not alias or not canonical:
            continue
        if alias.isdigit():
            out = re.sub(r"(?<!\d)%s(?!\d)" % re.escape(alias), canonical, out)
        else:
            out = out.replace(alias, canonical)
    return out


def _dedup(values: Sequence[str]) -> Tuple[str, ...]:
    """保持顺序去重。"""
    out = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in out:
            out.append(value)
    return tuple(out)


def _canonical(entity: str, aliases: Mapping[str, str]) -> str:
    """把实体代码/简称归一到标准名；没有别名时原样返回。"""
    return str(aliases.get(entity) or entity)


def audit_entity_coverage(
    query: str,
    answer: str,
    *,
    known_entities: Optional[Any] = None,
    context_queries: Optional[Sequence[str]] = None,
    entity_aliases: Optional[Mapping[str, str]] = None,
) -> EntityCoverage:
    """检查答案是否覆盖问题点名的主体。

    Args:
        query: 当前问题
        answer: 生成的答案
        known_entities: 公司名词表（dict 或 sequence）
        context_queries: 同一会话更早的子问题；只用于补全点名主体
        entity_aliases: 额外的代码/简称 → 标准名映射（例如 000538 → 云南白药）

    Returns:
        ``EntityCoverage``；``needs_repair=True`` 时应带约束重生成或提示澄清。
    """
    aliases: Dict[str, str] = dict(DEFAULT_ENTITY_ALIASES)
    for key, value in (entity_aliases or {}).items():
        if str(key or "").strip() and str(value or "").strip():
            aliases[str(key)] = str(value)

    query_text = " ".join(
        [str(query or ""), *[str(item or "") for item in (context_queries or [])]]
    ).strip()
    query_text = _apply_aliases(query_text, aliases)
    answer_text = _apply_aliases(str(answer or ""), aliases)

    query_entities = _dedup(
        _canonical(entity, aliases)
        for entity in extract_query_entities(query_text, known_entities)
    )
    answer_entities = _dedup(
        _canonical(entity, aliases)
        for entity in extract_query_entities(answer_text, known_entities)
    )

    if not query_entities:
        return EntityCoverage(
            answer_entities=answer_entities,
            reason="问题未点名公司主体，跳过实体覆盖检查",
        )

    missing = tuple(entity for entity in query_entities if entity not in answer_entities)
    unexpected = tuple(entity for entity in answer_entities if entity not in query_entities)
    covered = tuple(entity for entity in query_entities if entity in answer_entities)
    drift = bool(unexpected) and not covered
    needs_repair = drift or (bool(missing) and not answer_entities)

    if drift:
        reason = "答案主体不在问题点名实体中：问题=%s；答案=%s" % (
            "、".join(query_entities),
            "、".join(unexpected),
        )
    elif missing and not answer_entities:
        reason = "答案没有提到问题点名的实体：%s" % "、".join(missing)
    elif missing:
        reason = "答案覆盖了点名主体，但未提及：%s" % "、".join(missing)
    else:
        reason = "实体覆盖通过"

    return EntityCoverage(
        query_entities=query_entities,
        answer_entities=answer_entities,
        missing_query_entities=missing,
        unexpected_entities=unexpected,
        drift=drift,
        needs_repair=needs_repair,
        reason=reason,
    )


def build_entity_constraint(coverage: EntityCoverage) -> str:
    """把实体覆盖结果转成一次重生成用的硬约束；无需修复时返回空串。"""
    if not coverage.needs_repair or not coverage.query_entities:
        return ""
    return (
        "【实体约束】只允许讨论问题点名的以下主体：%s。"
        "参考资料没有覆盖的主体，请直接说明“资料未覆盖”，不得替换成其他公司。"
        % "、".join(coverage.query_entities)
    )
