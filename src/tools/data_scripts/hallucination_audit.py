"""幻觉回查底稿自动生成（B-23A，2026-09-10）

目的：为「人工幻觉回查 SOP（B-23B）」自动准备底稿，把人工需要核对的两件事机械化：
1. 答案中出现的全部数值（金额 / 百分比 / 年份 / 比率，含表格单元格）抽成「待核对清单」，
   并给出所在句子与命中的引用序号，人工只需逐行判定口径是否成立；
2. 每条引用（paper_path + 片段）在本地语料中是否存在、片段能否在原文中定位，
   把「引用源文件是否真实存在」这一类可自动判定的事实先算出来。

两种模式：
  --generate   分层抽样（问题类型 × 有/无 SQL）抽 N 题，真实生成（agent_query 口径，
               QUERY_CACHE_ENABLED 必须为 false），产物落 训练结果数据/halluc_audit_<日期>/
  --audit      读取一次生成的产物 → 输出「待核对清单.csv」「引用存在性比对.csv」「audit_summary.json」

用法：
  python tools/data_scripts/hallucination_audit.py --generate --n 10
  python tools/data_scripts/hallucination_audit.py --audit

口径声明：本脚本产出的是**底稿**，不是幻觉判定结论；数值是否「口径一致」必须由人工（B-23B）判定。
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]  # 仓库根（数据路径）
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

GOLDEN = _REPO_ROOT / "database" / "golden" / "v1_2026-08-22.json"
SQL_REGRESSION = _REPO_ROOT / "训练结果数据" / "sql_full_regression_native.json"
DEFAULT_OUT_DIR = _REPO_ROOT / "训练结果数据" / "halluc_audit_20260910"
AUDIT_DATE = "2026-09-10"

# 数字正则（与 CitationValidator 同源：先折叠数字内部排版空白）
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^:?-{2,}:?$")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*")
_AGGREGATED_SOURCES = {"聚合表格/多源", "未知来源", ""}

MAX_SENT_CHARS = 200
MAX_SNIPPET_CHARS = 4000


# ---------------------------------------------------------------- 纯函数（可离线单测）


def collapse_number_spaces(text: str) -> str:
    """折叠数字内部的排版空白（如 "1 234.5" -> "1234.5"）。"""
    return re.sub(r"(?<=\d)\s+(?=[\d.])", "", text or "")


def classify_number(number: str, context: str) -> str:
    """按上下文判定数值类型：年份 / 百分比 / 金额或数值。"""
    if re.fullmatch(r"\d{4}", number) and 1900 <= int(number) <= 2100:
        return "年份"
    trailing = context.split(number, 1)[-1][:4] if number in context else ""
    if trailing.lstrip().startswith(("%", "％")):
        return "百分比"
    return "金额或数值"


def extract_unit_context(sentence: str, number: str, window: int = 8) -> str:
    """截取数值后紧邻的单位/修饰片段（如 "亿元" "%"），供人工判断口径。"""
    idx = sentence.find(number)
    if idx < 0:
        return ""
    tail = sentence[idx + len(number) : idx + len(number) + window].strip()
    return tail.replace("\n", " ")


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点切句（保留标点）。"""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text or "")]
    return [p for p in parts if p]


def iter_number_items(text: str) -> List[Dict[str, str]]:
    """逐行抽取数值条目，区分正文/表格来源。

    Returns:
        条目列表，每条含 数值 / 数值类型 / 单位上下文 / 所在句子 / 来源（正文|表格）
    """
    items: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _TABLE_LINE_RE.match(line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and all(_TABLE_SEP_RE.match(c) for c in cells if c):
                continue  # markdown 表格分隔行
            for cell in cells:
                if not cell:
                    continue
                for number in _NUMBER_RE.findall(collapse_number_spaces(cell)):
                    key = (number, cell)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(
                        {
                            "数值": number,
                            "数值类型": classify_number(number, cell),
                            "单位上下文": extract_unit_context(cell, number),
                            "所在句子": cell[:MAX_SENT_CHARS],
                            "来源": "表格",
                        }
                    )
            continue
        for sentence in split_sentences(line):
            for number in _NUMBER_RE.findall(collapse_number_spaces(sentence)):
                key = (number, sentence)
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "数值": number,
                        "数值类型": classify_number(number, sentence),
                        "单位上下文": extract_unit_context(sentence, number),
                        "所在句子": sentence[:MAX_SENT_CHARS],
                        "来源": "正文",
                    }
                )
    return items


def iter_number_items_with_lines(text: str) -> List[Dict[str, str]]:
    """逐行抽取数值条目，并附行号（CSV 定位用；口径同 iter_number_items）。"""
    items: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for lineno, raw_line in enumerate((text or "").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        is_table = bool(_TABLE_LINE_RE.match(line))
        if is_table:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and all(_TABLE_SEP_RE.match(c) for c in cells if c):
                continue
            units: Sequence[Tuple[str, str]] = [(c, c) for c in cells if c]
        else:
            units = [(s, s) for s in split_sentences(line)]
        for raw_unit, sentence in units:
            for number in _NUMBER_RE.findall(collapse_number_spaces(raw_unit)):
                key = (number, sentence)
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "数值": number,
                        "数值类型": classify_number(number, sentence),
                        "单位上下文": extract_unit_context(sentence, number),
                        "所在句子": sentence[:MAX_SENT_CHARS],
                        "来源": "表格" if is_table else "正文",
                        "行号": str(lineno),
                    }
                )
    return items


def snippet_sha256(snippet: str) -> str:
    """片段哈希（归一化后计算，避免空白差异导致哈希漂移）。"""
    normalized = re.sub(r"\s+", "", snippet or "")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def strip_html_tags(text: str) -> str:
    """剥离 HTML 标签（MinerU 解析稿中表格以 <table><td> 形式存储）。

    Args:
        text: 原始文本

    Returns:
        标签替换为空格后的文本
    """
    return re.sub(r"<[^>]+>", " ", text or "")


_PUNCT_STRIP_RE = re.compile(r"[：:；;，,、。·“”\"\'（）()\[\]【】]")


def normalize_token(text: str) -> str:
    """归一化 token：去掉中英文标点（保留小数点与负号，避免破坏数值）。

    引用片段中表格单元格会被重组成「列名: 值;」形式，标点附着在词尾会导致
    纯字符串比对假阴性，因此比对前两边都要去标点。
    """
    return _PUNCT_STRIP_RE.sub("", text or "")


def snippet_hit_ratio(snippet: str, content: str, max_tokens: int = 40) -> Tuple[float, int, int]:
    """计算片段 token 在原文中的命中率（用于表格行跨单元格拼接的场景）。

    Args:
        snippet: 引用片段
        content: 已剥离 HTML 标签的原文
        max_tokens: 最多参与统计的 token 数

    Returns:
        (命中率, 命中 token 数, 参与统计 token 数)；无有效 token 时返回 (0.0, 0, 0)
    """
    tokens = [normalize_token(t) for t in re.split(r"\s+", snippet or "")]
    tokens = [t for t in tokens if len(t) >= 2][:max_tokens]
    if not tokens:
        return 0.0, 0, 0
    normalized = _PUNCT_STRIP_RE.sub(" ", content or "")
    hits = sum(1 for t in tokens if t in normalized)
    return hits / len(tokens), hits, len(tokens)


def match_hit_refs(number: str, references: Sequence[Dict[str, Any]]) -> List[int]:
    """返回引用文本中命中该数值的引用序号（1-based，按逗号分隔字符串返回）。"""
    hits: List[int] = []
    for i, ref in enumerate(references or [], 1):
        text = str(ref.get("text") or "")
        normalized = collapse_number_spaces(text).replace(",", "").replace("，", "")
        if number in normalized:
            hits.append(i)
    return hits


def audit_one_reference(
    ref: Dict[str, Any], validator: Any, fallback_corpus_root: Optional[Path] = None
) -> Dict[str, Any]:
    """核验单条引用：文件是否存在/可定位 + 片段能否在原文中定位。

    Args:
        ref: 引用条目（paper_path / text）
        validator: CitationValidator 实例
        fallback_corpus_root: 定位失败时用于兜底搜索的语料根目录

    Returns:
        核验明细字典
    """
    paper_path = str(ref.get("paper_path") or "")
    snippet = str(ref.get("text") or "")
    status, located = validator.locate(paper_path)
    if status == "missing" and paper_path in _AGGREGATED_SOURCES:
        status = "aggregated"  # 表格聚合/多源引用，天然无单一源文件
    if status == "missing" and fallback_corpus_root is not None and paper_path:
        # 兜底：按文件名在语料根目录下直接找（不依赖索引的 .md 后缀过滤）
        name = Path(paper_path.replace("\\", "/")).name
        if name:
            for cand in fallback_corpus_root.rglob(name):
                status, located = "fallback", str(cand)
                break

    file_exists = bool(located and Path(located).exists())
    snippet_locatable = ""
    hit_ratio, hit_tokens, total_tokens = 0.0, 0, 0
    if file_exists and snippet:
        try:
            content = Path(located).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            content = ""
        normalized_content = re.sub(r"\s+", "", content)
        normalized_snippet = re.sub(r"\s+", "", snippet)
        # 去 HTML 标签口径：MinerU 表格以 <td> 存储，表格行引用需剥标签后才连续
        dehtml_content = re.sub(r"\s+", "", strip_html_tags(content))
        hit_ratio, hit_tokens, total_tokens = snippet_hit_ratio(snippet, strip_html_tags(content))
        if normalized_snippet and normalized_snippet in normalized_content:
            snippet_locatable = "全文命中"
        elif normalized_snippet and normalized_snippet in dehtml_content:
            snippet_locatable = "去HTML后命中"
        elif snippet[:80] and re.sub(r"\s+", "", snippet[:80]) in dehtml_content:
            snippet_locatable = "前80字命中"
        elif hit_ratio >= 0.8:
            snippet_locatable = "部分命中（token≥80%）"
        elif not dehtml_content:
            snippet_locatable = "文件为空"
        else:
            snippet_locatable = "低命中（token<80%）"
    elif not file_exists:
        snippet_locatable = "未比对（文件缺失）"

    return {
        "paper_path": paper_path,
        "定位状态": status,
        "解析后路径": str(located or ""),
        "文件存在": "是" if file_exists else "否",
        "片段字符数": len(snippet),
        "片段sha256": snippet_sha256(snippet) if snippet else "",
        "片段定位": snippet_locatable,
        "片段命中率": f"{hit_ratio:.2f}",
        "命中token": f"{hit_tokens}/{total_tokens}",
        "片段首40字": snippet[:40].replace("\n", " "),
    }


def stratified_sample(
    items: Sequence[Dict[str, Any]],
    sql_flags: Dict[str, bool],
    n: int = 10,
    sub_index: int = 0,
) -> List[Dict[str, Any]]:
    """按「问题类型 × 有/无 SQL」分层抽样（轮转取数，确定性顺序）。

    Args:
        items: golden items（含 编号 / 问题类型 / 子问题）
        sql_flags: 编号 -> 是否有 SQL（来自最近一次原生链路回归）
        n: 抽样题数
        sub_index: 每题取第几个子问题（默认第 1 个）

    Returns:
        样本列表（含 编号 / 问题类型 / 分层键 / 子问题 / 问题）
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        subs = item.get("子问题") or []
        if len(subs) <= sub_index:
            continue
        has_sql = bool(sql_flags.get(item["编号"], False))
        key = f"{item['问题类型']}|{'有SQL' if has_sql else '无SQL'}"
        buckets[key].append(
            {
                "编号": item["编号"],
                "问题类型": item["问题类型"],
                "分层键": key,
                "子问题序号": sub_index + 1,
                "子问题": subs[sub_index],
                "问题": subs[sub_index],
                "有SQL": has_sql,
            }
        )

    # 轮转取数：按分层桶 key 排序后依次取一条，保证覆盖所有分层
    for key in buckets:
        buckets[key].sort(key=lambda r: r["编号"])
    ordered_keys = sorted(buckets.keys(), key=lambda k: (not buckets[k], k))
    samples: List[Dict[str, Any]] = []
    cursor = 0
    while len(samples) < n and ordered_keys:
        key = ordered_keys[cursor % len(ordered_keys)]
        if buckets[key]:
            samples.append(buckets[key].pop(0))
        else:
            ordered_keys.remove(key)
            if not ordered_keys:
                break
            continue
        cursor += 1
    return samples[:n]


# ---------------------------------------------------------------- 生成模式


def load_golden_items() -> List[Dict[str, Any]]:
    """读取 golden v1 的 items。"""
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return data["items"]


def load_sql_flags() -> Dict[str, bool]:
    """从最近一次原生链路回归产物读取每题「是否有 SQL」（缺失时全 False）。"""
    if not SQL_REGRESSION.exists():
        return {}
    records = json.loads(SQL_REGRESSION.read_text(encoding="utf-8"))
    return {r["编号"]: bool(r.get("有SQL")) for r in records if isinstance(r, dict) and r.get("编号")}


def run_generation(samples: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    """真实生成（agent_query 口径），产物写入 out_dir/raw_generation.json。"""
    from config.rag_config import RAGConfig
    from pipelines.rag_pipeline import RAGPipeline

    config = RAGConfig()
    if config.QUERY_CACHE_ENABLED:
        raise RuntimeError(
            "QUERY_CACHE_ENABLED 必须为 false（避免命中旧结果）；请先设置环境变量 QUERY_CACHE_ENABLED=false"
        )

    pipeline = RAGPipeline(config)
    records: List[Dict[str, Any]] = []
    t_all = time.time()
    for i, s in enumerate(samples, 1):
        user_id = f"halluc-audit-{s['编号']}"
        pipeline.reset_conversation(user_id=user_id)
        time.sleep(0.3)
        t0 = time.time()
        answer: Any = None
        error = ""
        try:
            answer = pipeline.agent_query(s["问题"], user_id=user_id, verbose=False)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        if isinstance(answer, dict):
            content = str(answer.get("content") or "")
            raw_refs = answer.get("references") or []
        else:
            content = str(answer or "")
            raw_refs = []
        references = [
            {
                "paper_path": str(r.get("paper_path") or ""),
                "text": str(r.get("text") or "")[:MAX_SNIPPET_CHARS],
                "paper_image": str(r.get("paper_image") or ""),
            }
            for r in raw_refs
            if isinstance(r, dict)
        ]
        try:
            sql = (pipeline.get_accumulated_sql(user_id) or "").strip()
        except Exception:  # noqa: BLE001
            sql = ""
        records.append({**s, "答案": content, "引用": references, "SQL": sql,
                        "耗时": round(time.time() - t0, 1), "错误": error})
        print(f"[{i}/{len(samples)}] {s['编号']} {s['分层键']} 耗时 {records[-1]['耗时']}s "
              f"答案 {len(content)} 字 / 引用 {len(references)} 条 / SQL {len(sql)} 字",
              file=sys.stderr, flush=True)
        time.sleep(0.5)

    payload = {
        "日期": AUDIT_DATE,
        "口径": "agent_query（supervisor-workers 主链路），QUERY_CACHE_ENABLED=false 真实生成",
        "样本数": len(records),
        "总耗时秒": round(time.time() - t_all, 1),
        "样本": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_generation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------- 审计模式


CHECKLIST_COLUMNS = [
    "编号", "子问题", "数值", "所在句子", "引用序号",
    "数值类型", "单位上下文", "来源", "答案行号", "命中引用数", "核对依据", "人工核对结论", "备注",
]
CITATION_COLUMNS = [
    "编号", "引用序号", "paper_path", "定位状态", "文件存在", "片段定位",
    "片段命中率", "命中token", "片段字符数", "片段sha256", "片段首40字",
    "人工核对结论", "备注",
]


def build_checklist(
    records: Sequence[Dict[str, Any]], validator: Any
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """从生成产物构建「待核对清单」与「引用存在性比对」两张表。"""
    checklist: List[Dict[str, str]] = []
    citation_rows: List[Dict[str, str]] = []
    for rec in records:
        question = str(rec.get("子问题") or rec.get("问题") or "")
        has_sql = bool(str(rec.get("SQL") or "").strip())
        for item in iter_number_items_with_lines(str(rec.get("答案") or "")):
            hits = match_hit_refs(item["数值"], rec.get("引用") or [])
            if hits:
                basis = "引用片段"
            elif has_sql:
                basis = "SQL 结果"
            else:
                basis = "无引用且无SQL（重点核查）"
            checklist.append(
                {
                    "编号": str(rec.get("编号") or ""),
                    "子问题": question,
                    "数值": item["数值"],
                    "所在句子": item["所在句子"],
                    "引用序号": ",".join(str(h) for h in hits) if hits else "（不在引用片段内）",
                    "数值类型": item["数值类型"],
                    "单位上下文": item["单位上下文"],
                    "来源": item["来源"],
                    "答案行号": item["行号"],
                    "命中引用数": str(len(hits)),
                    "核对依据": basis,
                    "人工核对结论": "",
                    "备注": "",
                }
            )
        for idx, ref in enumerate(rec.get("引用") or [], 1):
            detail = audit_one_reference(ref, validator)
            citation_rows.append(
                {
                    "编号": str(rec.get("编号") or ""),
                    "引用序号": str(idx),
                    "paper_path": detail["paper_path"],
                    "定位状态": detail["定位状态"],
                    "文件存在": detail["文件存在"],
                    "片段定位": detail["片段定位"],
                    "片段命中率": detail["片段命中率"],
                    "命中token": detail["命中token"],
                    "片段字符数": str(detail["片段字符数"]),
                    "片段sha256": detail["片段sha256"],
                    "片段首40字": detail["片段首40字"],
                    "人工核对结论": "",
                    "备注": "",
                }
            )
    return checklist, citation_rows


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    """写 CSV（utf-8-sig，Excel/WPS 直接打开不乱码）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(
    records: Sequence[Dict[str, Any]],
    checklist: Sequence[Dict[str, str]],
    citations: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    """汇总底稿统计（不含幻觉判定）。"""
    by_type = Counter(c["定位状态"] for c in citations)
    by_snippet = Counter(c["片段定位"] for c in citations)
    number_kind = Counter(c["数值类型"] for c in checklist)
    no_ref_hit = sum(1 for c in checklist if c["命中引用数"] == "0")
    basis_counter = Counter(c["核对依据"] for c in checklist)
    return {
        "日期": AUDIT_DATE,
        "样本题数": len(records),
        "答案为空题数": sum(1 for r in records if not str(r.get("答案") or "").strip()),
        "生成报错题数": sum(1 for r in records if r.get("错误")),
        "有 SQL 题数": sum(1 for r in records if str(r.get("SQL") or "").strip()),
        "待核对数值条目": len(checklist),
        "数值类型分布": dict(number_kind),
        "数值未落在引用片段内": no_ref_hit,
        "核对依据分布": dict(basis_counter),
        "引用条目数": len(citations),
        "引用定位状态分布": dict(by_type),
        "片段定位分布": dict(by_snippet),
        "口径声明": "本产物为回查底稿；数值口径一致性与幻觉判定须由人工（B-23B）完成，非自动结论。",
    }


def run_audit(out_dir: Path) -> Dict[str, Any]:
    """执行审计：读取 raw_generation.json → 落 CSV + summary。"""
    from config.rag_config import RAGConfig
    from pipelines.citation_validator import CitationValidator

    raw_path = out_dir / "raw_generation.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"未找到生成产物：{raw_path}，请先运行 --generate")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    records = payload.get("样本") or []

    config = RAGConfig()
    validator = CitationValidator(corpus_root=config.CITATION_CORPUS_ROOT, match_mode="comma")
    corpus_root = Path(config.CITATION_CORPUS_ROOT) if config.CITATION_CORPUS_ROOT else None
    if corpus_root is not None and not corpus_root.exists():
        corpus_root = None

    checklist, citations = build_checklist(records, validator)
    write_csv(out_dir / "待核对清单.csv", CHECKLIST_COLUMNS, checklist)
    write_csv(out_dir / "引用存在性比对.csv", CITATION_COLUMNS, citations)
    summary = summarize(records, checklist, citations)
    summary["语料根目录"] = str(corpus_root or config.CITATION_CORPUS_ROOT)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="幻觉回查底稿自动生成（B-23A）")
    parser.add_argument("--generate", action="store_true", help="分层抽样并真实生成")
    parser.add_argument("--audit", action="store_true", help="从生成产物构建回查底稿")
    parser.add_argument("--n", type=int, default=10, help="抽样题数（默认 10）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="产物目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not args.generate and not args.audit:
        args.audit = True

    if args.generate:
        items = load_golden_items()
        sql_flags = load_sql_flags()
        samples = stratified_sample(items, sql_flags, n=args.n)
        print(f"分层抽样 {len(samples)} 题：{json.dumps(Counter(s['分层键'] for s in samples), ensure_ascii=False)}",
              file=sys.stderr, flush=True)
        payload = run_generation(samples, out_dir)
        print(f"生成完成：{payload['样本数']} 题 / {payload['总耗时秒']}s → {out_dir / 'raw_generation.json'}",
              file=sys.stderr, flush=True)

    if args.audit:
        summary = run_audit(out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())