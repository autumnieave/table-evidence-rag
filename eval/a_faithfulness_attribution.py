# -*- coding: utf-8 -*-
"""A 类 12 题 faithfulness 归因：整表命中 / 检索遗漏 / 生成外推 / 口径噪声。

输入：
- ``faithfulness_a_live.json``：答案 vs 当前多阶段整表全文的 claim 明细；
- ``faithfulness_refs.json`` 对应的答案与引用卡片（重新跑一遍以得到全部 claim，
  不能只用 compact 后的 unsupported_examples）；
- 正式研报 Markdown 语料（个股 + 行业）。

输出 ``训练结果数据/stage5_measure/a_faithfulness_attribution.json``。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.claim_faithfulness import evaluate_claim_faithfulness
from eval.faithfulness_eval import _load_answers, _refs_to_hits
from eval.question_ab_classify import A_SET

OUT_DIR = ROOT / "训练结果数据" / "stage5_measure"
LIVE = OUT_DIR / "faithfulness_a_live.json"
OUT = OUT_DIR / "a_faithfulness_attribution.json"
STOCK_NAMES = ROOT / "训练结果数据" / "l5b_measure" / "stockname_dist.json"
CORPUS_ROOTS = (
    ROOT / "测试数据" / "附件5：研报数据" / "个股研报-解析结果-完整版-2.0",
    ROOT / "测试数据" / "附件5：研报数据" / "行业研报-解析结果-完整版-2.0",
    ROOT / "data_corpus" / "全部数据" / "正式数据" / "附件5：研报数据" / "个股研报-解析结果-2.0",
    ROOT / "data_corpus" / "全部数据" / "正式数据" / "附件5：研报数据" / "行业研报-解析结果-2.0",
)

WEAK_MAX = 10.0
CONTEXT_ANCHORS = (
    "中药材", "种植面积", "云药追溯", "研发支出", "研发投入", "综合200", "康美",
    "灵芝", "西洋参", "中药饮片", "配方颗粒", "医保", "创新药", "儿科", "医药工业",
    "疫苗", "CXO", "化学制剂", "医疗设备", "中药板块", "持仓", "流通市值", "上证指数",
    "收盘价",
)


def _load_stock_names() -> List[str]:
    try:
        data = json.loads(STOCK_NAMES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    counts = data.get("counts") or {}
    return sorted((str(k) for k in counts), key=len, reverse=True)


STOCK_NAME_LIST = _load_stock_names()


def _norm(text: str) -> str:
    return (
        str(text or "")
        .replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("\t", "")
        .replace("\r", "")
        .replace("\n", "")
    )


def _num_variants(num: str) -> List[str]:
    raw = str(num or "").replace(",", "").strip()
    if not raw:
        return []
    vals = {raw}
    if "." not in raw:
        return [v for v in vals if v]
    decimals = len(raw.split(".", 1)[1])
    try:
        value = float(raw)
        # 小数只允许保留到小数点后 1 位，绝不回退成整数：
        # ``1.2`` 不能变成 ``1``，``30.04`` 不能变成 ``30``。
        for digits in range(decimals, 0, -1):
            text = f"{value:.{digits}f}"
            if digits == decimals:
                text = text.rstrip("0").rstrip(".")
            vals.add(text)
        if value < 0:
            vals.add(str(abs(value)))
    except ValueError:
        pass
    return [v for v in vals if v]


def _contains_number(text: str, num: str) -> bool:
    norm = _norm(text)
    for v in _num_variants(num):
        # 不要用裸子串：``271`` 会命中图片哈希/``1271``，``7`` 会命中年份。
        if re.search(r"(?<![\dA-Za-z.])" + re.escape(v) + r"(?![\dA-Za-z.])", norm):
            return True
    return False


def _load_corpus() -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for root in CORPUS_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            key = path.name + "|" + str(path.stat().st_size)
            if key in seen:
                continue
            seen.add(key)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            docs.append((str(path), text))
    return docs


def _number_occurrences(claim_text: str, num: str) -> List[Tuple[int, int]]:
    raw = str(claim_text or "")
    variants = {str(num or "").replace(",", ""), str(num or "")}
    out: List[Tuple[int, int]] = []
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", raw):
        token = match.group(0).replace(",", "")
        if token in variants:
            out.append((match.start(), match.end()))
    return out


def _is_noise_occurrence(claim_text: str, start: int, end: int) -> bool:
    pre = claim_text[max(0, start - 8) : start]
    post = claim_text[end : end + 8]
    token = claim_text[start:end].replace(",", "")
    if token.count(".") >= 2:
        return True
    if re.search(r"(图|表|第|Top|前|§|参考[:：]?\s*)$", pre, re.IGNORECASE):
        return True
    if re.match(r"\s*(年|月|日|版|类|节|章|页|条|款|项)", post):
        return True
    if re.fullmatch(r"(19|20)\d{2}", token) and re.search(r"(年|版|报告|期|A|E|H)", pre + post):
        return True
    return False


def _noise_numbers(claim_text: str, numbers: Sequence[str]) -> List[str]:
    noise: List[str] = []
    for num in numbers or []:
        occ = _number_occurrences(claim_text, num)
        if occ and all(_is_noise_occurrence(claim_text, s, e) for s, e in occ):
            noise.append(str(num))
    return noise


def _claim_anchors(claim_text: str) -> List[str]:
    """提取 claim 的公司/产品/指标锚点，避免把无关文档里的同数字当成证据。"""
    anchors: List[str] = []
    for name in STOCK_NAME_LIST:
        if name and name in claim_text:
            anchors.append(name)
    for code in re.findall(r"\b\d{6}\.(?:SH|SZ|BJ)\b", claim_text, re.IGNORECASE):
        anchors.append(code)
    for anchor in CONTEXT_ANCHORS:
        if anchor in claim_text:
            anchors.append(anchor)
    out: List[str] = []
    for anchor in anchors:
        if anchor not in out:
            out.append(anchor)
    return out


def _find_corpus_hit(
    docs: Sequence[Tuple[str, str]], num: str, anchors: Sequence[str]
) -> Dict[str, Any]:
    weak = ""
    for path, text in docs:
        if not _contains_number(text, num):
            continue
        if not weak:
            weak = path
        for anchor in anchors:
            if anchor and anchor in text:
                return {"path": path, "anchor": anchor, "relevant": True}
    return {"path": weak, "anchor": "", "relevant": False}


def _row_search(tables: Sequence[Dict[str, Any]], numbers: Sequence[str]) -> Dict[str, Any]:
    rows = 0
    any_rows = 0
    row_evidence: List[Dict[str, Any]] = []
    for table in tables or []:
        content = str(table.get("content") or "")
        for line in content.splitlines():
            if not line.strip():
                continue
            rows += 1
            hit = [n for n in numbers if _contains_number(line, n)]
            if hit:
                any_rows += 1
            if numbers and len(hit) == len(numbers):
                row_evidence.append(
                    {
                        "table_id": table.get("table_id"),
                        "file_path": table.get("file_path"),
                        "row": line.strip()[:400],
                    }
                )
    return {
        "n_rows": rows,
        "n_rows_any_number": any_rows,
        "row_all": bool(row_evidence),
        "row_any": any_rows > 0,
        "row_evidence": row_evidence[:3],
    }


def _classify(
    *,
    live_supported: bool,
    refs_supported: bool,
    meaningful_missing: Sequence[str],
    row: Dict[str, Any],
    corpus_hits: Dict[str, Dict[str, Any]],
) -> str:
    if live_supported:
        return "supported_both" if refs_supported else "refs_material_gap"
    if refs_supported:
        return "live_regression"
    if not meaningful_missing:
        return "numeric_noise"
    if row.get("row_all"):
        return "returned_row_evidence"
    if row.get("row_any"):
        return "returned_table_partial_evidence"
    found_relevant = [n for n in meaningful_missing if (corpus_hits.get(n) or {}).get("relevant")]
    found_any = [n for n in meaningful_missing if (corpus_hits.get(n) or {}).get("path")]
    if len(found_relevant) == len(meaningful_missing):
        return "retrieval_miss"
    if found_relevant:
        return "partial_retrieval_miss"
    if found_any:
        return "corpus_collision_unverified"
    meaningful_values = []
    for num in meaningful_missing:
        try:
            meaningful_values.append(abs(float(str(num).replace(",", ""))))
        except ValueError:
            meaningful_values.append(999.0)
    if meaningful_values and max(meaningful_values) < WEAK_MAX:
        return "low_specificity_unverified"
    return "generation_or_unverified"


def _claim_key(claim: Dict[str, Any]) -> Tuple[Any, Any, str]:
    return (claim.get("start"), claim.get("end"), str(claim.get("text") or ""))


def _index_claims(claims: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    return {(c.get("start"), c.get("end")): c for c in claims}


def main() -> int:
    live_data = json.loads(LIVE.read_text(encoding="utf-8"))
    answers = dict(_load_answers())
    docs = _load_corpus()
    print(f"[attrib] corpus_docs={len(docs)}", flush=True)

    rows: List[Dict[str, Any]] = []
    total_counts: Counter = Counter()
    refs_unsupported_counts: Counter = Counter()
    live_unsupported_counts: Counter = Counter()

    for item in live_data.get("per_question") or []:
        qid = str(item.get("qid") or "")
        if qid not in A_SET:
            continue
        answer_item = answers.get(qid) or {}
        refs_result = evaluate_claim_faithfulness(
            str(answer_item.get("答案") or answer_item.get("answer") or ""),
            _refs_to_hits(answer_item),
        )
        refs_index = _index_claims(refs_result.get("claims") or [])
        claims = []
        for claim in item.get("claims") or []:
            refs_claim = refs_index.get((claim.get("start"), claim.get("end"))) or {}
            refs_supported = bool(refs_claim.get("supported"))
            live_supported = bool(claim.get("supported"))
            numbers = [str(n) for n in (claim.get("numbers") or [])]
            missing = [str(n) for n in (claim.get("missing_numbers") or [])]
            noise = _noise_numbers(str(claim.get("text") or ""), numbers)
            all_noise = bool(numbers) and len(noise) == len(numbers)
            meaningful = [n for n in missing if n not in noise]
            row = _row_search(item.get("tables") or [], meaningful)
            anchors = _claim_anchors(str(claim.get("text") or ""))
            corpus_hits = {n: _find_corpus_hit(docs, n, anchors) for n in meaningful}
            if all_noise:
                category = "noise_supported" if (live_supported or refs_supported) else "numeric_noise"
            else:
                category = _classify(
                    live_supported=live_supported,
                    refs_supported=refs_supported,
                    meaningful_missing=meaningful,
                    row=row,
                    corpus_hits=corpus_hits,
                )
            total_counts[category] += 1
            if not refs_supported:
                refs_unsupported_counts[category] += 1
            if not live_supported:
                live_unsupported_counts[category] += 1
            claims.append(
                {
                    "text": claim.get("text"),
                    "start": claim.get("start"),
                    "end": claim.get("end"),
                    "numbers": numbers,
                    "missing_numbers": missing,
                    "noise_numbers": noise,
                    "meaningful_missing": meaningful,
                    "refs_supported": refs_supported,
                    "live_supported": live_supported,
                    "row": row,
                    "corpus_hits": {
                        k: str(Path(v.get("path") or "").name) if v.get("path") else ""
                        for k, v in corpus_hits.items()
                    },
                    "corpus_hit_meta": corpus_hits,
                    "anchors": anchors,
                    "category": category,
                }
            )
        rows.append(
            {
                "qid": qid,
                "question": item.get("question"),
                "refs_summary": item.get("refs_summary") or {},
                "live_summary": item.get("summary") or {},
                "claims": claims,
                "counts": dict(Counter(c["category"] for c in claims)),
            }
        )

    def pct(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    refs_total = sum(len(r["claims"]) for r in rows)
    refs_supported = sum(1 for r in rows for c in r["claims"] if c["refs_supported"])
    refs_unsupported = refs_total - refs_supported
    live_supported = sum(1 for r in rows for c in r["claims"] if c["live_supported"])
    live_unsupported = refs_total - live_supported
    result = {
        "summary": {
            "n_questions": len(rows),
            "n_claims": refs_total,
            "refs_supported": refs_supported,
            "refs_unsupported": refs_unsupported,
            "live_supported": live_supported,
            "live_unsupported": live_unsupported,
            "refs_mean_support_rate": round(
                sum(float(r["refs_summary"].get("support_rate") or 0.0) for r in rows) / len(rows), 6
            ),
            "live_mean_support_rate": round(
                sum(float(r["live_summary"].get("support_rate") or 0.0) for r in rows) / len(rows), 6
            ),
            "refs_unsupported_categories": dict(refs_unsupported_counts),
            "live_unsupported_categories": dict(live_unsupported_counts),
            "all_categories": dict(total_counts),
            "refs_unsupported_pct": {k: pct(v, refs_unsupported) for k, v in refs_unsupported_counts.items()},
        },
        "per_question": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"[attrib] WROTE {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
