# -*- coding: utf-8 -*-
"""tools/typo_normalizer.py —— B-28 错别字/口语归一化（2026-09-10 落地）

背景：B-22 挑战集 C2011 人工口径不通过——「云白要2025年三季度的资产负债绿是多少？」
双错别字（云白要→云南白药、资产负债绿→资产负债率）未还原，SQL 按错字查询返回空后直接拒答。

本模块两类归一化（在原生财务链路 SQL 生成前调用）：
1. 精简错字映射（CURATED_TYPO_MAP）：确定性替换已识别的高频错字（公司+指标），低误伤；
2. 库内公司简称编辑距离纠错：对规范简称（≥3 字、且问句未含该规范名）在问句连续窗口内
   做增/删/改 ≤1 个字符的模糊匹配，命中则还原为规范名（覆盖未见过的错字变体）。

纯逻辑模块：不依赖 LLM/MySQL；公司简称列表由调用侧注入（native 链路从 MySQL 加载并缓存），
可离线单测。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: 精简错字映射（确定性优先；仅收录确认过的高频错字，宁少勿滥）
CURATED_TYPO_MAP: Dict[str, str] = {
    "云白要": "云南白药",
    "云白药": "云南白药",
    "片仔簧": "片仔癀",
    "净利闰": "净利润",
    "资产负债绿": "资产负债率",
}

#: 库内公司简称缓存（模块级，进程内一次加载）
_COMPANY_ABBR_CACHE: Optional[List[str]] = None


def edit_distance(a: str, b: str, max_d: int = 1) -> int:
    """字符串编辑距离（Levenshtein），超过 max_d 提前返回 max_d+1（短串快速剪枝）。

    Args:
        a: 字符串 a
        b: 字符串 b
        max_d: 最大关注距离（用于剪枝）

    Returns:
        编辑距离；若超过 max_d 返回 max_d+1
    """
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


def _replace_curated(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """按精简映射做确定性替换（长词优先，避免子串冲突）。"""
    out = text
    replaced: List[Tuple[str, str]] = []
    for wrong in sorted(CURATED_TYPO_MAP, key=len, reverse=True):
        if wrong in out:
            out = out.replace(wrong, CURATED_TYPO_MAP[wrong])
            replaced.append((wrong, CURATED_TYPO_MAP[wrong]))
    return out, replaced


def _fuzzy_company_fix(text: str, canonical_names: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    """对未含规范名的公司简称做 ≤1 编辑距离纠错（单窗口替换，取最长命中优先）。

    Args:
        text: 待纠错问句
        canonical_names: 库内规范公司简称（≥3 字）

    Returns:
        (纠错后文本, [(原文片段, 规范名), ...])
    """
    out = text
    replaced: List[Tuple[str, str]] = []
    chosen: List[Tuple[int, int, str]] = []
    if not out:
        return out, replaced
    candidates: List[Tuple[int, int, str]] = []
    for name in canonical_names:
        if not name or len(name) < 3 or name in out:
            continue  # 已含规范名/过短 → 不参与模糊匹配
        for ln in range(max(2, len(name) - 1), len(name) + 2):
            if ln > len(out):
                continue
            for i in range(0, len(out) - ln + 1):
                seg = out[i:i + ln]
                if seg == name:
                    continue
                if edit_distance(seg, name) <= 1:
                    candidates.append((i, i + ln, name))
                    break  # 该 name 只取最先命中窗口
    # 取最长规范名优先；同长取更靠前窗口；窗口重叠时跳过（保守，避免误替换）
    for start, end, name in sorted(candidates, key=lambda c: (-len(c[2]), c[0])):
        if any(start < c_end and end > c_start for c_start, c_end, _ in chosen):
            continue
        chosen.append((start, end, name))
        replaced.append((out[start:end], name))
        if len(replaced) >= 2:
            break
    # 按下标从右到左执行替换，避免左侧替换改变右侧窗口下标
    for start, end, name in sorted(chosen, key=lambda c: -c[0]):
        out = out[:start] + name + out[end:]
    return out, replaced


def normalize_question_typos(question: str, canonical_names: Optional[List[str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """错别字归一化主入口：先精简映射，再做公司简称模糊纠错。

    Args:
        question: 用户原始问句
        canonical_names: 库内规范公司简称列表（None=仅做精简映射）

    Returns:
        (归一化后问句, [(原文片段, 替换后), ...])
    """
    text = question or ""
    changed: List[Tuple[str, str]] = []
    text, rep = _replace_curated(text)
    changed.extend(rep)
    if canonical_names:
        text, rep = _fuzzy_company_fix(text, [n for n in canonical_names])
        changed.extend(rep)
    return text, changed


def load_company_abbrs_from_config(config: Any) -> List[str]:
    """从 RAGConfig 建一次性 MySQL 连接加载公司简称（模块级缓存；失败回退 []）。

    供 Agent 入口（拆解前）做错字归一化使用；连接只建一次，后续命中缓存。

    Args:
        config: RAGConfig（含 MYSQL_HOST/USER/PASSWORD/DATABASE）

    Returns:
        规范公司简称列表（≥3 字）；不可用时为空列表（不影响主流程）
    """
    global _COMPANY_ABBR_CACHE
    if _COMPANY_ABBR_CACHE is not None:
        return _COMPANY_ABBR_CACHE
    try:
        import pymysql  # noqa: PLC0415

        conn = pymysql.connect(
            host=config.MYSQL_HOST,
            port=int(getattr(config, "MYSQL_PORT", 3306) or 3306),
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            charset="utf8mb4",
            connect_timeout=3,
        )
        try:
            return load_company_abbrs(conn)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("B-28 公司简称字典加载失败（Agent 入口仅做精简映射）: %s", exc)
        _COMPANY_ABBR_CACHE = []
    return _COMPANY_ABBR_CACHE


def load_company_abbrs(conn: Any) -> List[str]:
    """从 MySQL 加载规范公司简称列表（进程内缓存，失败返回空列表不影响主流程）。

    Args:
        conn: 可执行 SQL 的连接对象（cursor 接口）

    Returns:
        规范公司简称列表（≥3 字）
    """
    global _COMPANY_ABBR_CACHE
    if _COMPANY_ABBR_CACHE is None:
        names: List[str] = []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT stock_abbr FROM core_performance_indicators_sheet "
                    "WHERE stock_abbr IS NOT NULL AND CHAR_LENGTH(stock_abbr) >= 3"
                )
                rows = cur.fetchall()
            names = sorted({str(r[0]).strip() for r in rows if str(r[0]).strip()})
            logger.info("B-28 公司简称字典加载：%d 家", len(names))
        except Exception as exc:  # noqa: BLE001
            logger.warning("B-28 公司简称加载失败（跳过公司模糊纠错）: %s", exc)
        _COMPANY_ABBR_CACHE = names
    return _COMPANY_ABBR_CACHE
