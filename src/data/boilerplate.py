# -*- coding: utf-8 -*-
"""锅炉板（免责声明 / 分析师声明 / 评级标准 / 研究所签名 …）识别。

研报的**尾部**通常是一串固定声明段（`# 免责声明` → `# 东吴证券投资评级标准` → 研究所签名），
它们与「问数」无关，却在旧的「一行一 chunk」口径下贡献大量短块（模板文本稀释正文）。

本模块只做**纯文本判定**（零外部依赖、可离线单测），供两个调用方共用：

- `data.splitter`：入库时**剥离尾部锅炉板段**，并把残余锅炉板块标 `chunk_type = "boilerplate"`；
- `src/tools/data_scripts/measure_boilerplate.py`：语料测量（改判定先跑测量，再决定是否重建）。

**刻意排除「风险提示」**：它不是模板，而是**可被提问的内容**（「该行业有哪些风险」），
切入会伤召回；实测 2 篇行业周报会因此多切 ~6,000 字。

判定的三条硬约束（防误伤正文，口径为「**宁可漏切，不可误伤**」）：

1. **只认强标题**：Markdown 标题行（`# 免责声明`）或整行加粗；**裸短行不算**——
   实测裸短行会把声明正文里的「重庆」「联系」当成标题，反而把链条走断；
2. **只切尾部**：取「其后所有标题都是声明类 / 签名类」的那一段起点，链必须一直延伸到文末；
3. **比例护栏**：可切长度 ≤ ``max_tail_ratio`` × 全文（默认 40%）且 ≥ ``min_tail_chars``（默认 200），
   否则放弃切割（登记，不硬切）。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

#: 锅炉板标题词表（**子串**命中；覆盖 `东吴证券投资评级标准` 这类带公司前缀的写法）
BOILERPLATE_MARKERS: Tuple[str, ...] = (
    "免责声明",
    "免责条款",
    "免责申明",
    "免责及风险",
    "分析师声明",
    "分析师承诺",
    "重要声明",
    "特别声明",
    "法律声明",
    "信息披露声明",
    "版权声明",
    "公司声明",
    "投资评级说明",
    "评级说明",
    "评级标准",
    "相关研究报告",
    "方法论",
)

#: 尾部签名 / 评级体系类标题（研报末尾与声明同属「模板区」）
SIGNATURE_MARKERS: Tuple[str, ...] = (
    "研究所",
    "评级体系",
    "投资评级",
    "公司评级",
    "行业评级",
    "评级定义",
    "联系方式",
    "联系地址",
    "分析师团队",
    "证券投资咨询业务",
    "分发和地区通知",
    "IMPORTANTDISCLAIM",
    "DISTRIBUTIONAND",
)

#: 默认比例护栏：可切尾部不得超过全文的 40%
DEFAULT_MAX_TAIL_RATIO: float = 0.40
#: 默认最小可切长度（字符）
DEFAULT_MIN_TAIL_CHARS: int = 200
#: 标题行长度上限（超过视为正文段落）
_MAX_HEADING_CHARS: int = 40
#: 裸标题行（无 `#`、无 `**`）长度上限
_MAX_BARE_HEADING_CHARS: int = 20
#: 词表（大写化，用于英文 / 大小写不敏感的**子串**命中）
_MARKERS_UPPER: Tuple[str, ...] = tuple(m.upper() for m in BOILERPLATE_MARKERS + SIGNATURE_MARKERS)

#: 声明类标题文本长度上限（防「长段落里提到免责声明」被误判成标题）
_MAX_MARKER_LABEL_CHARS: int = 24

_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}(?P<hashes>#{1,6})[ \t]*(?P<label>.+?)[ \t]*$")
_BARE_STOP_CHARS = set("。，；：！？,.;:!?()（）「」【】")


def heading_label(line: str, strong_only: bool = False) -> Optional[str]:
    """把一行识别成「标题」并返回标题文本；不是标题返回 ``None``。

    认三种形态：① Markdown 标题（``# 免责声明``）；② 整行加粗（``**法律声明**``）；
    ③ 裸短行（≤20 字、不含句读符号）。``strong_only = True`` 时**只认 ①②**。
    """
    stripped = (line or "").strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return None
    match = _MD_HEADING_RE.match(stripped)
    if match:
        return match.group("label").strip().strip("*").strip() or None
    if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
        return stripped.strip("*").strip() or None
    if not strong_only and len(stripped) <= _MAX_BARE_HEADING_CHARS and not (set(stripped) & _BARE_STOP_CHARS):
        return stripped
    return None


def is_boilerplate_heading(label: Optional[str]) -> bool:
    """标题文本是否属于锅炉板 / 尾部签名（**子串**命中，含 ``东吴证券投资评级标准`` 这类前缀写法）。"""
    if not label:
        return False
    compact = re.sub(r"[\s:：\-—_/（）()]+", "", label)
    if len(compact) > _MAX_MARKER_LABEL_CHARS:
        return False
    upper = compact.upper()
    if any(marker in upper for marker in _MARKERS_UPPER):
        return True
    # 兜底：「…声明」类标题（`法律主体声明` / `分析师声明` 的变体写法）
    return compact.endswith("声明") and len(compact) <= 12


def _iter_headings(text: str, strong_only: bool = True) -> List[Tuple[int, str]]:
    """返回 ``[(字符偏移, 标题文本)]``。"""
    out: List[Tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        label = heading_label(line, strong_only)
        if label:
            out.append((offset, label))
        offset += len(line) + 1
    return out


def find_trailing_boilerplate_start(
    text: str,
    max_tail_ratio: float = DEFAULT_MAX_TAIL_RATIO,
    min_tail_chars: int = DEFAULT_MIN_TAIL_CHARS,
) -> Optional[int]:
    """返回**尾部锅炉板段**的起始字符偏移；判定不成立返回 ``None``。

    规则：从后往前扫**强标题**链，取「其后所有标题都是声明 / 签名类」的那一段起点，
    再套比例与长度护栏。护栏不过即返回 ``None``（宁漏切不误伤）。
    """
    if not text:
        return None
    headings = _iter_headings(text, strong_only=True)
    if not headings:
        return None
    # 1) 找**最后一个**声明 / 签名类标题（其后可能有「联系」「上海/北京」等签名短行，
    #    它们同样是模板区，不能因为它们而放弃整段判定）
    marker_positions = [i for i, (_, label) in enumerate(headings) if is_boilerplate_heading(label)]
    if not marker_positions:
        return None
    # 2) 从该标题往前，把**连续**的声明 / 签名类标题一并纳入（`免责声明` → `分析师声明` → `评级标准`）
    index = marker_positions[-1]
    while index > 0 and is_boilerplate_heading(headings[index - 1][1]):
        index -= 1
    # 3) 连续链里取**仍满足护栏的最早**一个：链条可能一路回溯到中段（如正文里出现过
    #    `# 相关研究报告`），直接取最早会把正文一起切掉 ⇒ 逐候选挑，护栏不过就往后再取
    limit = max_tail_ratio * len(text)
    for candidate in range(index, len(headings)):
        start = headings[candidate][0]
        tail = len(text) - start
        if min_tail_chars <= tail <= limit:
            return start
    return None


def strip_trailing_boilerplate(
    text: str,
    max_tail_ratio: float = DEFAULT_MAX_TAIL_RATIO,
    min_tail_chars: int = DEFAULT_MIN_TAIL_CHARS,
) -> Tuple[str, int]:
    """剥离尾部锅炉板段，返回 ``(新文本, 被剥离字符数)``；不成立时原样返回 ``(text, 0)``。"""
    start = find_trailing_boilerplate_start(text, max_tail_ratio, min_tail_chars)
    if start is None:
        return text, 0
    return text[:start].rstrip() + "\n", len(text) - start


def is_boilerplate_block(text: str) -> bool:
    """整块文本是否**以声明 / 签名类标题开头**（用于给残余块打 ``chunk_type = boilerplate``）。"""
    for line in (text or "").split("\n"):
        if not line.strip():
            continue
        return is_boilerplate_heading(heading_label(line))
    return False
