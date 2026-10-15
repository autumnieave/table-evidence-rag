# -*- coding: utf-8 -*-
"""claim 级证据定位（纯函数模块：零外部依赖、可离线单测；LLM 抽取与 embedding 均注入）。

职责：输入「答案 + 引用材料」，输出「claim -> 证据片段」的定位结果，供两层共享：

- **评测层**：生成 claim-level 盲判工作单（人工逐 claim 核对）。
- **产品层**：引用卡片从"随机摘要"改为"claim 对应定位片段"，并可展开全文。

两层共用同一份定位逻辑，避免"人看到的材料"与"产品展示的材料"再次漂移。

设计约束
--------
1. 只依赖标准库；**不触网、不读盘**。LLM 抽取（``llm``）与向量化（``embed``）均为可注入
   依赖，默认 ``None`` 即只用规则路径，因此无网络环境也能全量跑、可单测。
2. 数字 claim 的抽取与匹配**复用 L1**（``pipelines.citation_validator``），不另造口径
   （comma 归一化 / LaTeX 空格折叠 / 单位换算变体与 L1 完全一致）。
3. **定位不使用 LLM**：数字 claim 走规则（命中句前后各 N 句），观点 claim 走向量相似度。
   这样"是否命中"可复算、可全量跑；LLM 只负责切 claim 边界。
4. 观点 claim 的语义相似度**永远会返回一个最相近段落**，因此必须报分数 + 档位；低档必须
   能被上游识别为「证据不足」，不得静默当作命中。
5. **SQL 材料**（``sql_rows_to_material``）走独立口径：SQL 返回原值、答案按展示位四舍五入
   （实测 ``248506.00`` 万元 -> 答案 ``24.85 亿元``），等值换算必然漏判，故对 SQL 材料额外
   开「答案精度内相等」容差（见 ``sql_value_matches``）；且 SQL 行内通常不带单位，
   单位语境收紧（``has_unit_context``）对 SQL 材料不适用。研报引用的严格口径不变。

档位口径
--------
``高`` = 分数 > 0.8；``中`` = 0.6 < 分数 <= 0.8；``低`` = 分数 <= 0.6（= 证据不足，交人工）。

输出结构（``locate_answer``）
----------------------------
::

    {
      "claims": [
        {"claim": {"text", "kind", "start", "end", "numbers", "note"},
         "evidence": [{"paper_path", "chunk_index", "start", "end", "text",
                       "hit_type", "score", "tier", "matched_numbers",
                       "is_clue", "note"}]}
      ],
      "meta": {...}
    }

``start`` / ``end`` 均为**字符偏移**：claim 的偏移相对答案正文，evidence 的偏移相对该条引用的
``full_text``，可直接用于原文回溯与高亮。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pipelines.citation_validator import CitationValidator

# --------------------------------------------------------------- 常量（口径，勿散落）

TIER_HIGH_MIN: float = 0.8
TIER_MID_MIN: float = 0.6
SCORE_EXACT_MATCH: float = 1.0
SCORE_UNIT_VARIANT: float = 0.9
SCORE_SQL_TOLERANCE: float = 0.8  # SQL 材料容差命中：<= 高分线，档位落「中」，提示需人工确认
DEFAULT_SENTENCE_WINDOW: int = 2
DEFAULT_TOP_K: int = 3
DEFAULT_UNIT_CHARS: int = 800
DEFAULT_MAX_OPINION_CLAIMS: int = 8
DEFAULT_MAX_EVIDENCE_PER_CLAIM: int = 5
DEFAULT_MAX_EVIDENCE_PER_REF: int = 2
YEAR_MIN: int = 1900
YEAR_MAX: int = 2100
STRUCTURE_NOISE_MAX_ORDINAL: int = 12  # 结构行噪声的最大序号（1~12 视为标题/列表序号，非真实数值）
STRUCTURE_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s|[(（]\d{1,2}[)）]\s?|\d{1,2}\s*[.、)）]\s|\[\d{1,2}\]\s?)"
)  # 结构行：markdown 标题 / 数字序号 / [数字]；**不含** -/* 列表符号（列表项常含真实断言）
SENTENCE_DELIMITERS: str = "。！？!?；;\n"
FULL_TEXT_PAYLOAD_ENV: str = "CITATION_FULL_TEXT_IN_PAYLOAD"
UNIT_TOKENS: Tuple[str, ...] = ("元", "万", "亿", "%", "％", "倍", "个百分点")
SQL_MATERIAL_LABEL: str = "SQL 结果"
CLUE_NOTE: str = "语义线索：相似度近似，不足以单独支撑观点判据"
DEFAULT_SQL_MATERIAL_ROWS: int = 30
DEFAULT_PRODUCT_MAX_CLAIMS: int = 60  # 产品层单次定位的 claim 上限（超出截断，防超长答案拖慢在线链路）
DEFAULT_PRODUCT_TOP_PER_REF: int = 3  # 产品层单条 claim 在单条引用里保留的证据片段上限
# SQL 材料容差匹配的换算量级（行值 x 量级 ~= 答案值）
TOLERANCE_MULTIPLIERS: Tuple[float, ...] = (1.0, 10.0, 100.0, 10000.0, 0.1, 0.01, 0.0001)
_NUMBER_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?")
TRUTHY: Tuple[str, ...] = ("1", "true", "yes", "on")
UNABLE_LOCATE_NOTE: str = "未能回文定位（LLM 非逐字摘录），claim 偏移记为 -1"
NO_EMBED_NOTE: str = "未注入 embed，观点证据未定位"

LlmCallable = Callable[[str], str]
EmbedCallable = Callable[[Sequence[str]], List[List[float]]]


# --------------------------------------------------------------- 数据结构


@dataclass
class Span:
    """带字符偏移的文本片段。"""

    start: int
    end: int
    text: str


@dataclass
class Claim:
    """答案里的一条待核验断言。

    Attributes:
        text: 断言原文
        kind: ``number``（数字 claim）/ ``opinion``（观点 claim）
        start: 在答案正文中的起始字符偏移（观点 claim 未能回文定位时为 -1）
        end: 在答案正文中的结束字符偏移（不含）
        numbers: 数字 claim 的关键数字（已去重、已按口径过滤年份噪声）
        note: 异常说明（空串表示正常）
    """

    text: str
    kind: str
    start: int
    end: int
    numbers: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Evidence:
    """一条 claim 在某条引用材料里的定位结果。

    Attributes:
        paper_path: 引用文件路径（表格聚合块为多源合并）
        chunk_index: 引用块序号（表格聚合块为 None）
        start: 在 ``full_text`` 中的起始字符偏移
        end: 在 ``full_text`` 中的结束字符偏移（不含）
        text: 证据片段原文
        hit_type: ``number`` / ``semantic``
        score: 0~1 分数（规则命中为 1.0 / 0.9；语义为余弦相似度裁剪到 0~1）
        tier: 档位 ``高`` / ``中`` / ``低``
        matched_numbers: 本片段内命中的数字（供判据①逐数字核对）
        is_clue: 是否为"线索"（语义定位结果只作线索，不可当证据定论）
        note: 附加说明
    """

    paper_path: str
    chunk_index: Optional[int]
    start: int
    end: int
    text: str
    hit_type: str
    score: float
    tier: str
    matched_numbers: List[str] = field(default_factory=list)
    is_clue: bool = False
    note: str = ""


@dataclass
class ClaimEvidence:
    """一条 claim 及其证据片段（受配额截断）。

    Attributes:
        claim: 断言
        evidence: 证据片段（按分数 / 片段长度择优，已按配额截断）
        n_evidence_total: 截断前的命中总数（不静默丢弃命中的诚实标记）
        matched_numbers: 该 claim 的数字中，在材料里至少命中过一次的
        missing_numbers: 该 claim 的数字中，材料里完全没命中的（判据① 的直接失败信号）
    """

    claim: Claim
    evidence: List[Evidence] = field(default_factory=list)
    n_evidence_total: int = 0
    matched_numbers: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)


@dataclass
class UnitIndex:
    """观点证据的检索索引（单元 + 预计算向量）。

    存在的意义是成本：一条答案通常有若干条观点 claim，若每条 claim 都重新嵌入全部单元，
    成本是 O(claims x units)。事先建一次索引后，每条 claim 只需嵌入 claim 本身。

    Attributes:
        units: (引用条目, 全文, 片段) 三元组
        vectors: 与 units 等长的单元向量
    """

    units: List[Tuple[Dict[str, Any], str, Span]] = field(default_factory=list)
    vectors: List[List[float]] = field(default_factory=list)


# --------------------------------------------------------------- 基础工具（纯函数）


def tier_of(score: float) -> str:
    """把证据分数映射为档位。

    Args:
        score: 0~1 分数

    Returns:
        ``高``（>0.8）/ ``中``（0.6~0.8）/ ``低``（<=0.6，即证据不足）
    """
    if score > TIER_HIGH_MIN:
        return "高"
    if score > TIER_MID_MIN:
        return "中"
    return "低"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（纯 Python，无量纲依赖）。

    Args:
        a: 向量 A
        b: 向量 B

    Returns:
        相似度；任一向量为空 / 零向量 / 维度不一致时返回 0.0
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def split_sentences(text: str) -> List[Span]:
    """按中英文句末标点与换行切句。

    Args:
        text: 待切分文本

    Returns:
        片段列表（偏移相对入参 text，标点归属前一句）
    """
    spans: List[Span] = []
    src = text or ""
    start = 0
    for i, ch in enumerate(src):
        if ch in SENTENCE_DELIMITERS:
            seg = src[start : i + 1]
            if seg.strip():
                spans.append(Span(start, i + 1, seg))
            start = i + 1
    if start < len(src):
        seg = src[start:]
        if seg.strip():
            spans.append(Span(start, len(src), seg))
    return spans


def split_units(text: str, max_chars: int = DEFAULT_UNIT_CHARS) -> List[Span]:
    """把长文切成"段落级"检索单元（供观点 claim 的向量定位）。

    先按空行切段；仍超长的段再按行累加切；单行超长则硬切。

    Args:
        text: 待切分文本
        max_chars: 单个单元字符上限

    Returns:
        片段列表（偏移相对入参 text）
    """
    out: List[Span] = []
    src = text or ""
    if not src:
        return out
    limit = max_chars if max_chars and max_chars > 0 else DEFAULT_UNIT_CHARS
    for b_start, b_end, block in _split_blocks(src):
        if len(block) <= limit:
            out.append(Span(b_start, b_end, block))
        else:
            out.extend(_split_block_by_lines(src, b_start, b_end, limit))
    return out


def _split_blocks(text: str) -> List[Tuple[int, int, str]]:
    """按空行切块（返回 (start, end, block)）。

    Args:
        text: 待切分文本

    Returns:
        块列表，仅含非空白块
    """
    blocks: List[Tuple[int, int, str]] = []
    last = 0
    for match in re.finditer(r"\n[ \t]*\n+", text):
        if last < match.start():
            seg = text[last : match.start()]
            if seg.strip():
                blocks.append((last, match.start(), seg))
        last = match.end()
    if last < len(text):
        seg = text[last:]
        if seg.strip():
            blocks.append((last, len(text), seg))
    return blocks


def _split_block_by_lines(text: str, start: int, end: int, limit: int) -> List[Span]:
    """把单个超长块按行累加切成 <= limit 的片段（单行超长则硬切）。

    Args:
        text: 完整文本（用于切片取原文）
        start: 块起始偏移
        end: 块结束偏移
        limit: 单片段字符上限

    Returns:
        片段列表
    """
    spans: List[Span] = []
    cur_start: Optional[int] = None
    cur_len = 0
    pos = start
    for line in text[start:end].splitlines(keepends=True):
        ln = len(line)
        if ln > limit:
            if cur_start is not None:
                spans.append(Span(cur_start, pos, text[cur_start:pos]))
                cur_start, cur_len = None, 0
            off = pos
            while off < pos + ln:
                nxt = min(off + limit, pos + ln)
                spans.append(Span(off, nxt, text[off:nxt]))
                off = nxt
            pos += ln
            continue
        if cur_start is None:
            cur_start, cur_len = pos, ln
        elif cur_len + ln > limit:
            spans.append(Span(cur_start, pos, text[cur_start:pos]))
            cur_start, cur_len = pos, ln
        else:
            cur_len += ln
        pos += ln
    if cur_start is not None:
        spans.append(Span(cur_start, pos, text[cur_start:pos]))
    return [s for s in spans if s.text.strip()]


def ref_text(ref: Dict[str, Any]) -> str:
    """取引用的可用材料文本（优先全文 full_text，回退摘要片段 text）。

    Args:
        ref: 引用条目

    Returns:
        材料文本
    """
    return str(ref.get("full_text") or ref.get("text") or "")


def default_validator() -> CitationValidator:
    """进程内共享的 L1 校验器（纯内存，不读盘；数字口径与 L1 完全一致）。

    Returns:
        ``CitationValidator`` 单例
    """
    global _VALIDATOR_SINGLETON
    if _VALIDATOR_SINGLETON is None:
        _VALIDATOR_SINGLETON = CitationValidator(match_mode="comma")
    return _VALIDATOR_SINGLETON


_VALIDATOR_SINGLETON: Optional[CitationValidator] = None


# --------------------------------------------------------------- claim 抽取


def is_year_like(token: str) -> bool:
    """判断数字 token 是否为"年份噪声"（1900~2100 的四位整数）。

    Args:
        token: 数字字符串

    Returns:
        是否视为年份
    """
    if not re.fullmatch(r"\d{4}", token or ""):
        return False
    return YEAR_MIN <= int(token) <= YEAR_MAX


def is_structure_noise(text: str, numbers: Sequence[str]) -> bool:
    """判断一条 claim 是否只是 markdown 结构行（标题 / 序号）造成的数字噪声。

    典型噪声：``### 1. 利润总额历史趋势分析``、``2. 影响…的关键因素`` 会被抽成数字 claim，
    而"命中"的数字其实是标题序号，产品层点开引用会显示成荒谬断言。判定收紧为**双重条件**：

    1. 该行是结构行（``#{1,6}`` 标题 / ``1.`` 序号 / ``[1]``），**不含** ``-`` / ``*`` 列表符号；
    2. 该行数字**全部**是小整数序号（1 ~ ``STRUCTURE_NOISE_MAX_ORDINAL``）。

    含任一真实数值就不判噪声：``1. 公司营收 12.3 亿元``（12.3 非小整数）保留、
    ``### 2024 年营收 12.3 亿元`` 保留；``- 2 家公司净利润为负`` 因不是结构行也保留。

    Args:
        text: 断言原文（通常是一句）
        numbers: 该断言命中的数字（已按年份口径过滤）

    Returns:
        是否为结构行噪声（True 表示应丢弃）
    """
    if not numbers or not text:
        return False
    if not STRUCTURE_LINE_RE.match(text):
        return False
    for num in numbers:
        if not re.fullmatch(r"\d{1,2}", num or ""):
            return False
        if not (1 <= int(num) <= STRUCTURE_NOISE_MAX_ORDINAL):
            return False
    return True


def extract_number_claims(
    answer: str,
    drop_year_like: bool = True,
    validator: Optional[CitationValidator] = None,
    drop_structure_noise: bool = True,
) -> List[Claim]:
    """用规则抽取数字 claim（复用 L1 的数字口径）。

    Args:
        answer: 答案正文
        drop_year_like: 是否剔除纯年份 token（避免"2025 年"制造噪声 claim）
        validator: L1 校验器（默认取共享单例）
        drop_structure_noise: 是否剔除 markdown 标题 / 序号造成的数字噪声
            （见 ``is_structure_noise``；评测层与产品层共享同一开关）

    Returns:
        claim 列表（按出现顺序）
    """
    v = validator or default_validator()
    claims: List[Claim] = []
    for span in split_sentences(answer or ""):
        nums = _dedup(v.extract_numbers(span.text))
        if drop_year_like:
            nums = [x for x in nums if not is_year_like(x)]
        if nums and drop_structure_noise and is_structure_noise(span.text, nums):
            continue
        if nums:
            claims.append(
                Claim(text=span.text, kind="number", start=span.start, end=span.end, numbers=nums)
            )
    return claims


def build_opinion_prompt(answer: str, max_claims: int = DEFAULT_MAX_OPINION_CLAIMS) -> str:
    """组装观点 claim 抽取提示词（供注入的 LLM 调用）。

    Args:
        answer: 答案正文
        max_claims: 最多抽取条数

    Returns:
        完整提示词
    """
    return OPINION_PROMPT.format(max_claims=max_claims, answer=(answer or "").strip())


def extract_opinion_claims(
    answer: str,
    llm: Optional[LlmCallable] = None,
    max_claims: int = DEFAULT_MAX_OPINION_CLAIMS,
) -> List[Claim]:
    """抽取观点 claim（LLM 抽取 + 回文定位）。

    Args:
        answer: 答案正文
        llm: 注入的模型调用（入参提示词，返回模型原文）；为 None 时不做观点抽取
        max_claims: 最多抽取条数

    Returns:
        claim 列表；未能逐字回文定位的项 ``start`` / ``end`` 记为 -1 并带 note
    """
    src = answer or ""
    if llm is None or not src.strip():
        return []
    try:
        raw = llm(build_opinion_prompt(src, max_claims))
    except Exception as exc:  # noqa: BLE001 注入实现可能抛任意异常，失败即降级为不抽取
        return [Claim(text="", kind="opinion", start=-1, end=-1, note=f"观点抽取失败：{exc}")]
    claims: List[Claim] = []
    for item in parse_opinion_output(raw)[:max_claims]:
        text = (item or "").strip()
        if not text:
            continue
        idx = src.find(text)
        if idx < 0:
            idx = _find_loose(src, text)
        if idx < 0:
            claims.append(Claim(text=text, kind="opinion", start=-1, end=-1, note=UNABLE_LOCATE_NOTE))
        else:
            claims.append(Claim(text=src[idx : idx + len(text)], kind="opinion", start=idx, end=idx + len(text)))
    return claims


def parse_opinion_output(raw: str) -> List[str]:
    """解析 LLM 输出的 claim 列表（容忍 JSON 数组 / 对象 / 逐行文本）。

    Args:
        raw: 模型原始输出

    Returns:
        claim 文本列表
    """
    text = (raw or "").strip()
    if not text:
        return []
    for candidate in (text, _first_json_block(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except Exception:  # noqa: BLE001 模型输出不可控，解析失败走文本兜底
            continue
        if isinstance(data, dict):
            data = data.get("claims") or data.get("items") or []
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    lines: List[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.、)])\s*", "", line).strip().strip('",')
        if cleaned and not cleaned.startswith("{") and not cleaned.startswith("["):
            lines.append(cleaned)
    return lines


def _first_json_block(text: str) -> str:
    """截取首个成对括号片段（模型常在 JSON 前后加解释）。

    Args:
        text: 原始文本

    Returns:
        片段字符串；找不到返回空串
    """
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            return text[i : j + 1]
    return ""


def _find_loose(haystack: str, needle: str) -> int:
    """忽略空白差异地定位子串（返回原文偏移）。

    Args:
        haystack: 原文本
        needle: 待定位文本

    Returns:
        原文起始偏移；找不到返回 -1
    """
    target = re.sub(r"\s+", "", needle)
    if not target:
        return -1
    stripped_chars: List[str] = []
    index_map: List[int] = []
    for i, ch in enumerate(haystack):
        if ch.isspace():
            continue
        stripped_chars.append(ch)
        index_map.append(i)
    joined = "".join(stripped_chars)
    pos = joined.find(target)
    if pos < 0:
        return -1
    end_pos = min(pos + len(target) - 1, len(index_map) - 1)
    start_off = index_map[pos]
    end_off = index_map[end_pos] + 1
    return start_off if end_off - start_off >= len(target) else -1


# --------------------------------------------------------------- 证据定位


def locate_number_claim(
    claim: Claim,
    references: Sequence[Dict[str, Any]],
    window: int = DEFAULT_SENTENCE_WINDOW,
    validator: Optional[CitationValidator] = None,
    accept_unit_variants: bool = True,
    max_per_ref: int = DEFAULT_MAX_EVIDENCE_PER_REF,
    max_total: int = DEFAULT_MAX_EVIDENCE_PER_CLAIM,
    require_unit_context: bool = True,
    sql_number_tolerance: bool = True,
) -> List[Evidence]:
    """用规则定位数字 claim 的证据（命中句前后各 window 句）。

    Args:
        claim: 数字 claim
        references: 引用列表（需含 full_text；缺失时回退 text）
        window: 命中句前后扩展的句数
        validator: L1 校验器（默认取共享单例）
        accept_unit_variants: 是否接受单位换算变体（判据①要求"单位口径一致"，故默认开）
        max_per_ref: 单条引用最多保留的证据片段数
        max_total: 单条 claim 最多保留的证据片段总数
        require_unit_context: 命中句必须含单位（元 / 万 / 亿 / % / 倍）才记为证据，
            用于挡掉"序号: 10"、"PE(TTM): 21.21" 这类无单位数字命中（收紧 (b)）；
            对 SQL 材料（``is_sql_material``）不适用 —— SQL 行是结构化数值，单位在表结构里；
    sql_number_tolerance: SQL 材料是否启用「答案精度内相等」容差（见 ``sql_value_matches``）

    Returns:
        证据列表；全部未命中时返回空列表（= 材料中找不到该数字）
    """
    return _number_evidence(claim, references, window, validator, accept_unit_variants,
                            max_per_ref, max_total, require_unit_context, sql_number_tolerance)[0]


def _number_evidence(
    claim: Claim,
    references: Sequence[Dict[str, Any]],
    window: int = DEFAULT_SENTENCE_WINDOW,
    validator: Optional[CitationValidator] = None,
    accept_unit_variants: bool = True,
    max_per_ref: int = DEFAULT_MAX_EVIDENCE_PER_REF,
    max_total: int = DEFAULT_MAX_EVIDENCE_PER_CLAIM,
    require_unit_context: bool = True,
    sql_number_tolerance: bool = True,
) -> Tuple[List[Evidence], int]:
    """``locate_number_claim`` 的内部实现，额外返回**截断前**的命中总数。

    Args:
        （同 locate_number_claim）

    Returns:
        (按配额截断后的证据列表, 截断前命中总数)
    """
    v = validator or default_validator()
    numbers = list(claim.numbers or [])
    if not numbers:
        return [], 0
    out: List[Evidence] = []
    for ref in references or []:
        full = ref_text(ref)
        if not full:
            continue
        is_sql = bool(ref.get("is_sql_material"))
        need_unit = require_unit_context and not is_sql
        sentences = split_sentences(full)
        if not sentences:
            continue
        hits: Dict[int, Tuple[float, List[str], List[str]]] = {}
        for i, span in enumerate(sentences):
            normalized = v.normalize_for_match(span.text)
            matched: List[str] = []
            exact = False
            for num in numbers:
                ok, variant_only = _match_number(v, num, normalized, accept_unit_variants)
                if ok:
                    matched.append(num)
                    if not variant_only:
                        exact = True
            if matched and (not need_unit or has_unit_context(span.text)):
                hits[i] = (SCORE_EXACT_MATCH if exact else SCORE_UNIT_VARIANT, matched, [])
                continue
            if is_sql and sql_number_tolerance:
                tol_numbers: List[str] = []
                tol_notes: List[str] = []
                for num in numbers:
                    found = sql_value_matches(num, span.text)
                    if found:
                        tol_numbers.append(num)
                        tol_notes.append("SQL 行值 %s ~= 答案 %s" % (found, num))
                if tol_numbers:
                    hits[i] = (SCORE_SQL_TOLERANCE, tol_numbers, tol_notes)
        ref_items: List[Evidence] = []
        for group in merge_windows(sorted(hits), window, len(sentences)):
            lo, hi = group[0], group[-1]
            s, e = sentences[lo].start, sentences[hi].end
            hit_items = [hits[i] for i in group if i in hits]  # group 含窗口补位下标，需过滤
            score = max(item[0] for item in hit_items)
            matched = _dedup([n for item in hit_items for n in item[1]])
            notes = _dedup([n for item in hit_items for n in item[2]])
            tolerance_only = all(item[0] == SCORE_SQL_TOLERANCE for item in hit_items)
            ref_items.append(
                Evidence(
                    paper_path=str(ref.get("paper_path") or ""),
                    chunk_index=ref.get("chunk_index"),
                    start=s,
                    end=e,
                    text=full[s:e],
                    hit_type="sql_tolerance" if tolerance_only else "number",
                    score=score,
                    tier=tier_of(score),
                    matched_numbers=matched,
                    note="；".join(notes),
                )
            )
        out.extend(_select_evidence(ref_items, max_per_ref))
    return _select_evidence(out, max_total), len(out)


def build_unit_index(
    references: Sequence[Dict[str, Any]],
    embed: Optional[EmbedCallable] = None,
    max_unit_chars: int = DEFAULT_UNIT_CHARS,
) -> Optional[UnitIndex]:
    """建观点证据检索索引（把全部单元嵌入一次，供多条 claim 复用）。

    Args:
        references: 引用列表（需含 full_text）
        embed: 注入的向量化函数；为 None 时返回 None
        max_unit_chars: 检索单元字符上限

    Returns:
        ``UnitIndex``；未注入 embed、无材料或向量数不匹配时返回 None
    """
    if embed is None:
        return None
    units: List[Tuple[Dict[str, Any], str, Span]] = []
    for ref in references or []:
        full = ref_text(ref)
        if not full:
            continue
        for span in split_units(full, max_unit_chars):
            units.append((ref, full, span))
    if not units:
        return None
    vectors = embed([u[2].text for u in units])
    if not vectors or len(vectors) != len(units):
        return None
    return UnitIndex(units=units, vectors=vectors)


def locate_opinion_claim(
    claim: Claim,
    references: Optional[Sequence[Dict[str, Any]]] = None,
    embed: Optional[EmbedCallable] = None,
    top_k: int = DEFAULT_TOP_K,
    max_unit_chars: int = DEFAULT_UNIT_CHARS,
    index: Optional[UnitIndex] = None,
) -> List[Evidence]:
    """用向量相似度定位观点 claim 的证据。

    Args:
        claim: 观点 claim
        references: 引用列表（需含 full_text）；已传 index 时可省略
        embed: 注入的向量化函数（入参文本列表，返回等长向量列表）；为 None 时不定位
        top_k: 最多返回的证据条数
        max_unit_chars: 检索单元字符上限（未传 index 时用于建索引）
        index: 已建好的单元索引（多条 claim 复用以避免重复嵌入单元）

    Returns:
        证据列表（按分数降序）；未注入 embed 或无材料时返回空列表
    """
    if embed is None or not (claim.text or "").strip():
        return []
    idx = index if index is not None else build_unit_index(references or [], embed, max_unit_chars)
    if idx is None or not idx.units:
        return []
    query_vectors = embed([claim.text])
    if not query_vectors or len(query_vectors) != 1:
        return []
    query = query_vectors[0]
    scored: List[Tuple[float, Dict[str, Any], str, Span]] = []
    for (ref, full, span), vec in zip(idx.units, idx.vectors):
        raw = cosine(query, vec)
        scored.append((max(0.0, min(1.0, raw)), ref, full, span))
    scored.sort(key=lambda item: -item[0])
    out: List[Evidence] = []
    for score, ref, full, span in scored[: max(1, top_k)]:
        out.append(
            Evidence(
                paper_path=str(ref.get("paper_path") or ""),
                chunk_index=ref.get("chunk_index"),
                start=span.start,
                end=span.end,
                text=span.text,
                hit_type="semantic",
                score=score,
                tier=tier_of(score),
                is_clue=True,
                note=CLUE_NOTE,
            )
        )
    return out


def _select_evidence(items: List[Evidence], limit: int) -> List[Evidence]:
    """择优保留证据（分数高者优先；同分取片段更短者 —— 更精准；再按位置稳定排序）。

    Args:
        items: 候选证据
        limit: 保留上限（<=0 表示不限）

    Returns:
        截断后的证据列表（输出按引用路径 + 偏移排序，便于人工按原文顺序核对）
    """
    if not items:
        return []
    picked = items if limit is None or limit <= 0 else sorted(
        items, key=lambda e: (-e.score, len(e.text), e.paper_path, e.start)
    )[:limit]
    return sorted(picked, key=lambda e: (e.paper_path, e.start))


def _has_number_token(haystack: str, needle: str) -> bool:
    """判断 needle 是否以"独立数字 token"出现（前后都不接数字或小数点）。

    用途：收紧**单位换算变体**的匹配。L1 的 ``number_in_text`` 是纯子串匹配，而变体可能
    是很短的整数（如 "99.9" 的万元变体 "1"），会命中 "12.3" 这类数字的内部子串，造成假阳性。
    **仅收紧变体路径**，精确匹配路径保持与 L1 完全一致。

    Args:
        haystack: 已归一化文本
        needle: 待查找的数字串

    Returns:
        是否存在独立 token 形式的出现
    """
    if not needle:
        return False
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        before_ok = start == 0 or not (haystack[start - 1].isdigit() or haystack[start - 1] == ".")
        after_ok = end >= len(haystack) or not (haystack[end].isdigit() or haystack[end] == ".")
        if before_ok and after_ok:
            return True
        start = haystack.find(needle, start + 1)
    return False


def is_plausible_variant(variant: str, base: Optional[str] = None) -> bool:
    """判断单位换算变体是否"有意义"。

    两层收紧：
      1. **退化值**：``_unit_variants`` 会产出 "0" / "1" 这类值（99.9 ÷ 10000 舍入到 "0.01"、
         12.3 ÷ 10000 舍入到 "0"），它们在任意文本里几乎必中，故要求"长度 >= 2 且 |值| >= 1"。
      2. **方向**（传入 base 时）：只接受**放大方向**（|variant| >= |base|）。``_unit_variants``
         同时给出放大与缩小两个方向，而缩小方向在实务里几乎不成立，实测会命中无关数字——
         例：2120.99 万元 ÷ 100 = "21.21" 命中了研报里的「PE(TTM): 21.21」，被记成高分证据。

    Args:
        variant: 变体数字串
        base: 原始数字串；给定时启用方向收紧

    Returns:
        是否可用作证据匹配
    """
    if len(variant or "") < 2:
        return False
    try:
        value = abs(float(variant))
    except ValueError:
        return False
    if value < 1.0:
        return False
    if base is not None:
        try:
            if value < abs(float(base)):
                return False
        except ValueError:
            return False
    return True


def has_unit_context(text: str) -> bool:
    """判断片段里是否出现金额 / 比例类单位。

    用途（收紧 (b)）：纯数字命中在长文档里几乎必然出现（"序号: 10"、"PE(TTM): 21.21"、Q3 的 "3"），
    要求命中句内出现单位（元 / 万 / 亿 / % / 倍 / 个百分点）才能作为数字证据，
    把"无单位数字"这类噪声挡在判据①之外。

    Args:
        text: 命中句文本

    Returns:
        是否含单位
    """
    return any(token in (text or "") for token in UNIT_TOKENS)


def sql_rows_to_material(
    rows: Sequence[Any],
    sql_text: str = "",
    label: str = SQL_MATERIAL_LABEL,
    max_rows: int = DEFAULT_SQL_MATERIAL_ROWS,
) -> Dict[str, Any]:
    """把 SQL 结果行转成"材料"结构，供定位器与研报引用同口径使用。

    S 层（sql_only）与 mixed 题的答案数字来自 SQL 结果，而非研报；不接这一类材料，
    claim 级工作单在 S 层会整层显示"无证据"（实测：B2005「利润总额 24.85 亿元」）。

    Args:
        rows: SQL 结果行（dict 列表；也容忍标量行）
        sql_text: SQL 原文（一并放进材料首行，便于人工核对）
        label: 材料展示名
        max_rows: 最多纳入的行数

    Returns:
        材料结构（与引用条目同形：``paper_path`` / ``chunk_index`` / ``full_text``）
    """
    lines: List[str] = []
    if (sql_text or "").strip():
        lines.append("SQL: " + " ".join(sql_text.split()))
    for row in list(rows or [])[: max(0, max_rows)]:
        if isinstance(row, dict):
            lines.append("; ".join(f"{k}: {v}" for k, v in row.items()))
        else:
            lines.append(str(row))
    return {
        "paper_path": label,
        "chunk_index": None,
        "is_sql_material": True,
        "full_text": "\n".join(lines),
    }


def _decimals_of(number: str) -> int:
    """数字串的小数位数（无小数点返回 0）。"""
    text = number or ""
    return len(text.split(".")[1]) if "." in text else 0


def sql_value_matches(number: str, text: str) -> str:
    """SQL 材料专用：判断 ``number`` 是否与文本内某个数值「在答案精度内相等」。

    背景：SQL 返回的是**原值**（``total_profit = 248506.00`` 万元），答案按展示位四舍五入
    （写成 ``24.85 亿元``）。等值换算（``unit_variants``）要求严格相等，在舍入场景必然
    漏判 —— 实测 B2005（24.85 vs 248506.00）与 B2075（21.02 vs 210157.59）两条题都因此漏判。

    规则（宁紧勿松）：

    - 仅当答案数字有 **1~2 位小数**时启用，容差 = 答案末位的半个单位（``24.85`` -> 0.005）；
    - 整数答案不做容差（避免 "10" 命中 "10.4" 这类放大）；
    - 行内数值按 ``TOLERANCE_MULTIPLIERS`` 换算后落进容差即命中。

    Args:
        number: 答案里的数字串
        text: SQL 材料文本（``列: 值`` 逐行）

    Returns:
        命中的行内原值与换算方向描述（如 ``"248506.00 /10000"``）；未命中返回空串
    """
    decimals = _decimals_of(number)
    if decimals not in (1, 2):
        return ""
    try:
        target = float(number)
    except ValueError:
        return ""
    tolerance = 0.5 * (10 ** -decimals)
    for token in _NUMBER_TOKEN_RE.findall(text or ""):
        try:
            value = float(token)
        except ValueError:
            continue
        for multiplier in TOLERANCE_MULTIPLIERS:
            if abs(value * multiplier - target) <= tolerance:
                direction = ("x%g" % multiplier) if multiplier >= 1 else ("/%g" % (1.0 / multiplier))
                return "%s %s" % (token, direction)
    return ""


def _match_number(
    validator: CitationValidator,
    number: str,
    normalized: str,
    accept_unit_variants: bool,
) -> Tuple[bool, bool]:
    """在已归一化文本里匹配一个数字（精确路径同 L1；变体路径加 token 边界收紧）。

    Args:
        validator: L1 校验器
        number: 待匹配数字
        normalized: 已归一化的文本
        accept_unit_variants: 是否接受单位换算变体

    Returns:
        (是否命中, 是否仅靠单位换算变体命中)
    """
    if not accept_unit_variants:
        return validator.number_in_text(number, normalized, accept_unit_variants=False)
    needle = validator.normalize_for_match(number)
    if needle and needle in normalized:
        return True, False
    for variant in validator.unit_variants(number):
        if is_plausible_variant(variant, base=number) and _has_number_token(normalized, variant):
            return True, True
    return False, False


def merge_windows(indices: Sequence[int], window: int, total: int) -> List[List[int]]:
    """把命中下标扩展为 ±window 的窗口并按重叠合并。

    Args:
        indices: 命中句下标（升序）
        window: 前后扩展句数
        total: 句总数（用于裁剪）

    Returns:
        合并后的连续下标组列表
    """
    if not indices:
        return []
    w = max(0, int(window))
    groups: List[List[int]] = []
    cur_lo = max(0, indices[0] - w)
    cur_hi = min(total - 1, indices[0] + w)
    for idx in indices[1:]:
        lo, hi = max(0, idx - w), min(total - 1, idx + w)
        if lo <= cur_hi + 1:
            cur_hi = max(cur_hi, hi)
        else:
            groups.append(list(range(cur_lo, cur_hi + 1)))
            cur_lo, cur_hi = lo, hi
    groups.append(list(range(cur_lo, cur_hi + 1)))
    return groups


# --------------------------------------------------------------- 总装


def locate_answer(
    answer: str,
    references: Sequence[Dict[str, Any]],
    llm: Optional[LlmCallable] = None,
    embed: Optional[EmbedCallable] = None,
    window: int = DEFAULT_SENTENCE_WINDOW,
    top_k: int = DEFAULT_TOP_K,
    max_claims: int = DEFAULT_MAX_OPINION_CLAIMS,
    drop_year_like: bool = True,
    drop_structure_noise: bool = True,
    validator: Optional[CitationValidator] = None,
    max_evidence_per_claim: int = DEFAULT_MAX_EVIDENCE_PER_CLAIM,
    max_evidence_per_ref: int = DEFAULT_MAX_EVIDENCE_PER_REF,
    extra_materials: Optional[Sequence[Dict[str, Any]]] = None,
    require_unit_context: bool = True,
    sql_number_tolerance: bool = True,
) -> Dict[str, Any]:
    """总装：抽取 claim（数字规则 + 观点 LLM）并逐条定位证据。

    Args:
        answer: 答案正文
        references: 引用列表（需含 full_text）
        llm: 注入的 LLM 抽取函数；None 时只抽数字 claim
        embed: 注入的向量化函数；None 时观点 claim 不定位证据
        window: 数字 claim 的命中句前后扩展句数
        top_k: 观点 claim 的证据条数上限
        max_claims: 观点 claim 条数上限
        drop_year_like: 是否剔除纯年份 token
        validator: L1 校验器（默认取共享单例）
        max_evidence_per_claim: 单条 claim 保留的证据片段上限
        max_evidence_per_ref: 单条引用保留的证据片段上限
        extra_materials: 第二类材料（如 ``sql_rows_to_material`` 造出的 SQL 结果行）；
            与研报引用同结构、同等参与定位。S 层 / mixed 题的答案数字来自 SQL 结果，
            不传这一类材料时 S 层会整层显示"无证据"
        require_unit_context: 数字证据是否要求命中句含单位（见 ``has_unit_context``；
            SQL 材料不受此限）
        sql_number_tolerance: SQL 材料是否启用「答案精度内相等」容差

    Returns:
        ``{"claims": [...], "meta": {...}}``（结构见模块 docstring）
    """
    v = validator or default_validator()
    materials: List[Dict[str, Any]] = list(references or []) + list(extra_materials or [])
    claims: List[Claim] = extract_number_claims(
        (answer or ""), drop_year_like, v, drop_structure_noise
    )
    opinion = extract_opinion_claims((answer or ""), llm, max_claims)
    claims.extend(opinion)
    claims.sort(key=lambda c: (c.start if c.start >= 0 else 10 ** 9))

    has_opinion = any(c.kind == "opinion" for c in claims)
    unit_index = build_unit_index(materials, embed) if (has_opinion and embed is not None) else None

    results: List[ClaimEvidence] = []
    for claim in claims:
        if claim.kind == "number":
            evidence, found_total = _number_evidence(
                claim, materials, window, v, True,
                max_evidence_per_ref, max_evidence_per_claim, require_unit_context,
                sql_number_tolerance,
            )
            found = {n for e in evidence for n in e.matched_numbers}
            results.append(
                ClaimEvidence(
                    claim=claim,
                    evidence=evidence,
                    n_evidence_total=found_total,
                    matched_numbers=[n for n in claim.numbers if n in found],
                    missing_numbers=[n for n in claim.numbers if n not in found],
                )
            )
            continue
        evidence = locate_opinion_claim(claim, materials, embed, top_k, index=unit_index)
        if embed is None and not claim.note:
            claim.note = NO_EMBED_NOTE
        results.append(ClaimEvidence(claim=claim, evidence=evidence, n_evidence_total=len(evidence)))

    meta = {
        "n_claims": len(results),
        "n_claims_dropped_structure": (
            len(extract_number_claims((answer or ""), drop_year_like, v, False)) - len(claims)
            if drop_structure_noise
            else 0
        ),
        "n_number_claims": sum(1 for r in results if r.claim.kind == "number"),
        "n_opinion_claims": sum(1 for r in results if r.claim.kind == "opinion"),
        "n_claims_with_evidence": sum(1 for r in results if r.evidence),
        "n_claims_without_evidence": sum(1 for r in results if not r.evidence),
        "n_evidence": sum(len(r.evidence) for r in results),
        "n_numbers_matched": sum(len(r.matched_numbers) for r in results),
        "n_numbers_missing": sum(len(r.missing_numbers) for r in results),
        "n_claims_with_missing_numbers": sum(1 for r in results if r.missing_numbers),
        "n_evidence_high": sum(1 for r in results for e in r.evidence if e.tier == "高"),
        "n_evidence_mid": sum(1 for r in results for e in r.evidence if e.tier == "中"),
        "n_evidence_low": sum(1 for r in results for e in r.evidence if e.tier == "低"),
        "n_evidence_clue": sum(1 for r in results for e in r.evidence if e.is_clue),
        "n_materials": len(materials),
        "n_extra_materials": len(extra_materials or []),
        "require_unit_context": require_unit_context,
        "sql_number_tolerance": sql_number_tolerance,
        "llm_injected": llm is not None,
        "embed_injected": embed is not None,
        "window": window,
        "top_k": top_k,
        "drop_year_like": drop_year_like,
        "drop_structure_noise": drop_structure_noise,
        "max_evidence_per_claim": max_evidence_per_claim,
        "max_evidence_per_ref": max_evidence_per_ref,
    }
    return {"claims": [asdict(r) for r in results], "meta": meta}


# --------------------------------------------------------------- 引用材料 payload 开关


def full_text_in_payload() -> bool:
    """是否把引用全文放进 API payload（默认关；评测 / 人工复核时置 1 开启）。

    与已废弃的 ``CITATION_FULL_TEXT_DUMP`` 不同：那个是"落盘到 JSONL 文件"，本开关只决定
    **响应体里带不带** ``full_text``，不写任何文件。

    Returns:
        是否随 payload 返回 full_text
    """
    return (os.environ.get(FULL_TEXT_PAYLOAD_ENV) or "").strip().lower() in TRUTHY


def attach_full_text(ref: Dict[str, Any], full_text: str) -> None:
    """按 ``full_text_in_payload()`` 决定是否给引用条目补 ``full_text`` 字段。

    Args:
        ref: 引用条目（开启时就地写入 full_text）
        full_text: 该块原文全文（= text_total 对应的字符串）

    Returns:
        None
    """
    if full_text_in_payload():
        ref["full_text"] = full_text


def locate_claims_per_reference(
    answer: str,
    references: Sequence[Dict[str, Any]],
    *,
    window: int = DEFAULT_SENTENCE_WINDOW,
    max_claims: int = DEFAULT_PRODUCT_MAX_CLAIMS,
    top_per_ref: int = DEFAULT_PRODUCT_TOP_PER_REF,
    drop_structure_noise: bool = True,
    validator: Optional[CitationValidator] = None,
) -> List[List[Dict[str, Any]]]:
    """产品层：把「答案 + 引用材料」定位成**以引用为主序**的 claim 列表（纯规则，不用 LLM）。

    与 :func:`locate_answer` 的区别是主序：评测层按 claim 聚合（逐条 claim 看材料），
    产品层按**引用**聚合（点开某条引用，看它支撑了答案里的哪几条断言），因此这里返回
    与 ``references`` **同长度同顺序** 的列表，第 i 个元素是第 i 条引用的 claim 块。

    定位逻辑与评测层完全共享：claim 抽取复用 :func:`extract_number_claims`（内部走 L1 数字口径），
    证据定位复用 :func:`_number_evidence`（同窗口 / 同单位变体 / 同 SQL 容差口径），
    不另造第二套匹配规则。观点 claim 需要 LLM 抽取，按口径只在评测层开启，本函数不抽。

    Args:
        answer: 答案正文
        references: 引用列表；每项需含 ``full_text``（缺失时回退 ``text``）
        window: 数字 claim 命中句前后扩展的句数
        max_claims: claim 上限（超出按出现顺序截断）
        top_per_ref: 单条 claim 在单条引用里保留的证据片段上限
        drop_structure_noise: 是否剔除 markdown 标题 / 序号噪声（与评测层同一开关）
        validator: L1 校验器（默认取共享单例）

    Returns:
        ``List[List[dict]]``，与外层 ``references`` 一一对应。每条 claim 块结构::

            {"claim": 断言原文, "claim_start": 偏移, "claim_end": 偏移,
             "numbers": [该断言的数字], "matched_numbers": [...], "missing_numbers": [...],
             "evidence": [{"start", "end", "text", "tier", "score", "hit_type",
                           "matched_numbers", "note"}]}

        材料里完全定位不到证据时该引用对应空列表。
    """
    refs = list(references or [])
    claims = extract_number_claims(
        answer or "", validator=validator, drop_structure_noise=drop_structure_noise
    )[:max_claims]
    if not claims:
        return [[] for _ in refs]

    out: List[List[Dict[str, Any]]] = []
    for ref in refs:
        blocks: List[Dict[str, Any]] = []
        if not ref_text(ref):
            out.append(blocks)
            continue
        for claim in claims:
            evidence, _total = _number_evidence(
                claim,
                [ref],
                window=window,
                validator=validator,
                max_per_ref=top_per_ref,
                max_total=top_per_ref,
            )
            if not evidence:
                continue
            matched = _dedup([n for item in evidence for n in item.matched_numbers])
            blocks.append(
                {
                    "claim": claim.text,
                    "claim_start": claim.start,
                    "claim_end": claim.end,
                    "numbers": list(claim.numbers),
                    "matched_numbers": matched,
                    "missing_numbers": [n for n in claim.numbers if n not in matched],
                    "evidence": [
                        {
                            "start": item.start,
                            "end": item.end,
                            "text": item.text,
                            "tier": item.tier,
                            "score": item.score,
                            "hit_type": item.hit_type,
                            "matched_numbers": list(item.matched_numbers),
                            "note": item.note,
                        }
                        for item in evidence
                    ],
                }
            )
        # 命中数字多的断言排前（该引用对它支撑最强），同分按答案出现顺序
        blocks.sort(key=lambda b: (-len(b["matched_numbers"]), b["claim_start"]))
        out.append(blocks)
    return out


def _dedup(items: Sequence[str]) -> List[str]:
    """保序去重。

    Args:
        items: 字符串序列

    Returns:
        去重后的列表
    """
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


OPINION_PROMPT = """你是金融问答的 claim 抽取器。请从下面的回答里挑出**需要证据支撑的断言**（观点、结论、判断、因果、对比、排名）。

规则：
- 只摘"断言性 / 结论性"句子；纯过渡句、免责声明、格式标题不要。
- 尽量**逐字原文**摘录（可以是句子的连续片段），便于回文定位；不要改写、不要合并。
- 最多 {max_claims} 条，按重要性降序。
- 只输出 JSON 数组，形如 ["断言句 1", "断言句 2"]，不要任何多余文字。

【回答】
{answer}
"""
