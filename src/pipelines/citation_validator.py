"""
pipelines/citation_validator.py
引用核验器（L1） - 引用可溯源性与数字可核验性

核验两层：
1. 文件可溯源：引用 paper_path 是否能在语料库中定位到真实文件
   - exact：路径字符串与磁盘文件名完全一致
   - fuzzy：归一化（引号 / 双反斜杠 / 空白 / 目录层级差异）后在全库索引中唯一定位
   - missing：两层均失败，引用文件在语料库中不存在
2. 数字可溯源：引文 text 中的数字是否能在目标文件全文中找到
   - 匹配口径 raw / comma / loose，默认 comma（千分位逗号归一化）
   - comma 口径额外折叠数字内部的排版空白（LaTeX 空格数字，如 "8 7 . 5" -> "87.5"）
   - 可选单位换算变体匹配：accept_unit_variants=True 时接受
     百万元 / 万元 / 千万元 <-> 亿元 的换算值（用于答案级端到端核验，默认关闭）
"""

import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_ASCII_QUOTE = '"'
# 数字/空白/小数点连续片段（用于折叠 LaTeX 排版空格，如 "8 7 . 5"）
_DIGIT_RUN_RE = re.compile(r"\d[\d\s\u00a0.]*")
# 常见单位换算系数（对应：百万元 -> 亿元 /100，万元 -> 亿元 /10000，千万元 -> 亿元 /10）
_UNIT_FACTORS: Tuple[Tuple[int, str], ...] = (
    (100, "百万元<->亿元"),
    (10000, "万元<->亿元"),
    (10, "千万元<->亿元"),
)


def _collapse_number_spaces(text: str) -> str:
    """折叠数字内部的排版空白（LaTeX 空格数字：'8 7 . 5' -> '87.5'，'1 2 3 4' -> '1234'）。

    规则：
    - 仅折叠后**至多一个小数点**的片段，避免合并表格行中空格分隔的多个数字
      （如 '0.72 2.73 3.71' 保持原样，'8 7 . 5' 折叠为 '87.5'）；
    - 保护“4 位年份 + 空格 + 4 位年份”并列（如 '2023 2024'），避免误合并成 8 位数字。

    Args:
        text: 原始文本

    Returns:
        折叠数字内部空白后的文本
    """
    protected = re.sub(r"(?<=\d{4})\s+(?=\d{4})", "\x00", text or "")

    def _fix(match: "re.Match[str]") -> str:
        run = match.group(0)
        collapsed = run.replace("\x00", " ").replace(" ", "").replace("\u00a0", "")
        if collapsed.count(".") <= 1:
            return collapsed
        return run.replace("\x00", " ")

    collapsed = _DIGIT_RUN_RE.sub(_fix, protected)
    return collapsed.replace("\x00", " ")


def _unit_variants(number: str) -> Dict[str, str]:
    """生成数字的单位换算变体（保留 2 位小数并去尾零）。

    Args:
        number: 数字字符串（如 "58.86"）

    Returns:
        变体 -> 换算方向 字典（如 {"5886": "百万元<->亿元"}）
    """
    try:
        value = float(number)
    except ValueError:
        return {}
    variants: Dict[str, str] = {}
    for factor, label in _UNIT_FACTORS:
        for sign in (1, -1):
            converted = value * (factor ** sign)
            variant = f"{converted:.2f}".rstrip("0").rstrip(".")
            if variant and variant != number:
                variants.setdefault(variant, label)
    return variants


def _norm_key(filename: str) -> str:
    """文件名归一化，用于模糊定位比对。

    归一化规则：中文弯引号转 ASCII 引号、双反斜杠折叠为路径分隔符、
    去除全部空白、Unicode NFC 规范化。

    Args:
        filename: 原始文件名或路径片段

    Returns:
        归一化后的文件名
    """
    s = filename.replace("\u201c", _ASCII_QUOTE).replace("\u201d", _ASCII_QUOTE).replace("\uff02", _ASCII_QUOTE)
    s = s.replace("\\\\", "\\").replace("\\", os.sep).replace("/", os.sep)
    s = re.sub(r"\s+", "", s)
    return unicodedata.normalize("NFC", s)


class CitationValidator:
    """引用核验器（L1）

    对引用列表执行两层核验：文件可溯源（exact / fuzzy / missing）与数字可溯源。

    Attributes:
        corpus_root: 语料库根目录（默认取 RAGConfig.CITATION_CORPUS_ROOT）
        match_mode: 数字匹配口径，raw=原样 / comma=逗号归一化+折叠数字内空白 /
                    loose=comma+去全部空白
    """

    # 命中数字上下文提取参数（供前端"查看原文"展示数字在文件全文中的位置）
    MAX_CONTEXT_CHARS: int = 300
    MAX_HIT_CONTEXTS: int = 3

    def __init__(self, corpus_root: Optional[str] = None, match_mode: str = "comma") -> None:
        self.corpus_root = corpus_root
        self.match_mode = match_mode
        self._index: Optional[Dict[str, str]] = None

    def build_index(self) -> Dict[str, str]:
        """扫描语料库，构建 归一化文件名 -> 文件绝对路径 索引。

        Returns:
            全库文件索引字典；语料根目录不可访问时返回空字典
        """
        if not self.corpus_root or not os.path.isdir(self.corpus_root):
            logger.warning("语料根目录不存在或不可访问: %s，引用将全部判定为 missing", self.corpus_root)
            self._index = {}
            return self._index
        index: Dict[str, str] = {}
        for dirpath, _dirnames, filenames in os.walk(self.corpus_root):
            for filename in filenames:
                if not filename.lower().endswith(".md"):
                    continue
                key = _norm_key(filename)
                # 同名文件取第一个，与全量 L1 核验口径保持一致
                index.setdefault(key, os.path.join(dirpath, filename))
        self._index = index
        logger.info("语料库索引构建完成：%s 篇研报（根目录 %s）", len(index), self.corpus_root)
        return index

    @property
    def index(self) -> Dict[str, str]:
        """语料库文件索引（惰性构建，首次访问时全库扫描）。"""
        if self._index is None:
            self.build_index()
        return self._index

    def locate(self, paper_path: str) -> Tuple[str, Optional[str]]:
        """定位引用文件。

        Args:
            paper_path: 引用中的文件路径

        Returns:
            (状态, 解析后路径)：状态为 exact / fuzzy / missing
        """
        if not paper_path:
            return "missing", None
        normalized = paper_path.replace("\\\\", "\\").replace("\\", os.sep)
        if os.path.exists(normalized):
            return "exact", os.path.abspath(normalized)
        key = _norm_key(os.path.basename(normalized))
        resolved = self.index.get(key)
        if resolved is not None:
            return "fuzzy", resolved
        return "missing", None

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """从文本中抽取数字（整数 / 小数），先折叠数字内部的排版空白。

        Args:
            text: 待抽取文本

        Returns:
            数字字符串列表，按文本中出现顺序排列
        """
        return _NUMBER_RE.findall(_collapse_number_spaces(text or ""))

    def _normalize_for_match(self, text: str) -> str:
        """按匹配口径归一化文本。

        Args:
            text: 原始文本

        Returns:
            归一化后的文本
        """
        if self.match_mode == "raw":
            return text
        s = text.replace(",", "").replace("\uff0c", "")
        if self.match_mode == "loose":
            s = re.sub(r"\s+", "", s)
        else:
            # comma 口径：折叠数字内部的排版空白（LaTeX 空格数字）
            s = _collapse_number_spaces(s)
        return s

    def _extract_context(
        self, number: str, raw_content: str, near_text: Optional[str] = None
    ) -> Optional[str]:
        """在原始文件文本中定位数字，返回命中位置附近的上下文片段。

        优先定位在引用片段（near_text）内出现的命中位置，使上下文与引文语义相关；
        片段定位失败时回退到全文第一次出现位置。

        Args:
            number: 已命中的数字字符串（如 "29.77"）
            raw_content: 目标文件的原始全文（未归一化）
            near_text: 引用片段文本（用于优先定位片段内命中位置）

        Returns:
            压缩空白后的上下文片段；原始文本中定位不到该数字时返回 None
            （例如仅靠排版空格折叠或单位换算变体才命中的数字）
        """
        if not number or not raw_content:
            return None
        matches = [m.start() for m in re.finditer(re.escape(number), raw_content)]
        if not matches:
            return None
        pos = matches[0]
        if near_text:
            off = near_text.find(number)
            if off >= 0:
                probe = near_text[max(0, off - 40) : off + 40]
                p = raw_content.find(probe)
                if p >= 0:
                    pos = p + (off - max(0, off - 40))
        start = max(0, pos - 130)
        end = min(len(raw_content), pos + len(number) + 170)
        ctx = raw_content[start:end]
        return re.sub(r"\s+", " ", ctx).strip()

    def number_in_text(
        self, number: str, haystack: str, accept_unit_variants: bool = False
    ) -> Tuple[bool, bool]:
        """判断数字是否命中已归一化文本。

        Args:
            number: 待匹配数字字符串
            haystack: 已用 _normalize_for_match 归一化的文本
            accept_unit_variants: 是否接受单位换算变体（百万元/万元/千万元 <-> 亿元）

        Returns:
            (是否命中, 是否仅靠单位换算变体命中)
        """
        needle = self._normalize_for_match(number)
        if needle and needle in haystack:
            return True, False
        if accept_unit_variants and needle:
            for variant in _unit_variants(needle):
                if variant and variant in haystack:
                    return True, True
        return False, False

    def check_reference(
        self, paper_path: str, text: str, accept_unit_variants: bool = False
    ) -> Dict[str, Any]:
        """核验单条引用（文件可溯源 + 数字可溯源）。

        Args:
            paper_path: 引用文件路径
            text: 引用文本（用于数字抽检）
            accept_unit_variants: 是否接受单位换算变体命中（默认关闭，保持 L1 口径）

        Returns:
            核验记录字典：
            {
                paper_path: 原始引用路径,
                text: 引用文本,
                status: exact / fuzzy / missing,
                located: 解析后文件路径或 None,
                nums: 数字总数,
                num_hit: 命中数字数,
                num_ratio: 命中率（无数字时为 None）,
                unhit: 未命中数字列表,
                unit_hit: 仅靠单位换算变体命中的数字列表
            }
        """
        status, located = self.locate(paper_path)
        numbers = self.extract_numbers(text)
        num_hit = 0
        unhit: List[str] = []
        unit_hit: List[str] = []
        hits_context: List[Dict[str, Any]] = []
        if located and numbers:
            content = ""
            try:
                with open(located, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError as exc:
                logger.warning("读取目标文件失败: %s（%s）", located, exc)
            haystack = self._normalize_for_match(content)
            for number in numbers:
                hit, unit_only = self.number_in_text(number, haystack, accept_unit_variants)
                if hit:
                    num_hit += 1
                    if unit_only:
                        unit_hit.append(number)
                    elif len(hits_context) < self.MAX_HIT_CONTEXTS:
                        ctx = self._extract_context(number, content, near_text=text)
                        if ctx is not None and not any(h["context"] == ctx for h in hits_context):
                            hits_context.append({"num": number, "context": ctx})
                else:
                    unhit.append(number)
        return {
            "paper_path": paper_path,
            "text": text,
            "status": status,
            "located": located,
            "nums": len(numbers),
            "num_hit": num_hit,
            "num_ratio": round(num_hit / len(numbers), 4) if numbers else None,
            "unhit": unhit,
            "unit_hit": unit_hit,
            "hits_context": hits_context,
        }

    def check_references(
        self, references: List[Dict[str, Any]], accept_unit_variants: bool = False
    ) -> List[Dict[str, Any]]:
        """批量核验引用列表。

        Args:
            references: 引用列表，元素须含 paper_path / text 字段
                        （如 {paper_path, text, paper_image}）
            accept_unit_variants: 是否接受单位换算变体命中（默认关闭，保持 L1 口径）

        Returns:
            逐条核验记录列表
        """
        return [
            self.check_reference(
                ref.get("paper_path", ""), ref.get("text", ""), accept_unit_variants
            )
            for ref in references
        ]

    def summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总核验结果，输出 L1 指标。

        Args:
            records: check_references 返回的逐条核验记录

        Returns:
            汇总字典：引用总数、文件可溯源率、数字命中率、全命中 / 零命中引用数、
            缺失引用明细、零命中引用明细
        """
        total = len(records)
        exact = sum(1 for r in records if r["status"] == "exact")
        fuzzy = sum(1 for r in records if r["status"] == "fuzzy")
        missing = sum(1 for r in records if r["status"] == "missing")
        traceable = exact + fuzzy
        nums_total = sum(r["nums"] for r in records)
        nums_hit = sum(r["num_hit"] for r in records)
        with_numbers = [r for r in records if r["nums"] > 0]
        all_hit = sum(1 for r in with_numbers if r["num_hit"] == r["nums"])
        zero_hit = sum(1 for r in with_numbers if r["num_hit"] == 0)
        return {
            "total": total,
            "exact": exact,
            "fuzzy": fuzzy,
            "missing": missing,
            "traceable": traceable,
            "traceable_rate": round(traceable / total, 4) if total else None,
            "num_total": nums_total,
            "num_hit": nums_hit,
            "num_rate": round(nums_hit / nums_total, 4) if nums_total else None,
            "all_hit_refs": all_hit,
            "zero_hit_refs": zero_hit,
            "missing_refs": [
                {"paper_path": r["paper_path"], "text": r["text"][:80]}
                for r in records if r["status"] == "missing"
            ],
            "zero_hit_refs_detail": [
                {"paper_path": r["paper_path"], "unhit": r["unhit"]}
                for r in records if r["status"] != "missing" and r["nums"] > 0 and r["num_hit"] == 0
            ],
        }
