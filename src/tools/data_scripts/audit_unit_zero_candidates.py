"""审计「表里没写单位」结论（2026-10-17）。

回答三个问题（可复现）：
1. **分母核对**：``compare_table_parse`` 的 3,343 张 vs 本次 3,148 张，差在哪；
2. **零候选抽样**：从「``unit_candidates`` 为空」的表里**固定 seed** 抽 10 张，输出原文证据（标题/表头/单元格）；
3. **正则覆盖**：用**宽松探针**（¥/￥/RMB/人民币/单位/裸单位词）反向检查零候选表，量化漏检上界。

用法（仓库根）::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m tools.data_scripts.audit_unit_zero_candidates

产物：``训练结果数据/table_struct_smoke/unit_zero_audit.json``
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

from data.boilerplate import strip_trailing_boilerplate
from data.loader import load_industry_documents, load_markdown_documents
from data.table_struct import Table, scan_tables
from tools.data_scripts.measure_metadata_coverage import build_config

OUT = "训练结果数据/table_struct_smoke/unit_zero_audit.json"
SEED = 20261017
SAMPLE_N = 10

#: 宽松探针：**故意比正式正则宽**，用于估「漏检上界」
LOOSE_PATTERNS: Dict[str, str] = {
    "yen_symbol": r"[¥￥]",
    "RMB": r"(?i)\bRMB\b",
    "CNY": r"(?i)\bCNY\b",
    "USD": r"(?i)\bUSD\b",
    "人民币": r"人民币",
    "港币": r"港币",
    "美元": r"美元",
    "单位字样": r"单位",
    "裸_百万": r"(?<!\d)百万(?!元)",
    "裸_千": r"(?<!\d)千(?!元)",
    "裸_百万元": r"百万元",
    "裸_万元": r"万元",
    "裸_亿元": r"亿元",
    "英文million": r"(?i)\bmillion\b",
    "英文thousand": r"(?i)\bthousand\b",
    "百分比单位": r"[%％]",
}


def load_docs() -> List[Dict[str, Any]]:
    # 必须用 measure_metadata_coverage 的语料口径（473 篇），不能用 RAGConfig() 默认值（164 篇）
    cfg = build_config()
    docs = load_markdown_documents(cfg.MARKDOWN_DIR, excel_metadata_path=cfg.EXCEL_METADATA_PATH)
    docs += load_industry_documents(cfg.INDUSTRY_MARKDOWN_DIR, excel_metadata_path=cfg.INDUSTRY_EXCEL_PATH)
    return docs


def scan(docs: List[Dict[str, Any]], strip: bool) -> Dict[str, List[Table]]:
    """按文件返回表列表（``strip=True`` 先剥尾部锅炉板，与入库口径一致）。"""
    out: Dict[str, List[Table]] = {}
    for doc in docs:
        content = doc.get("content") or ""
        if strip:
            content, _removed = strip_trailing_boilerplate(content)
        meta = doc.get("metadata") or {}
        _text, tables = scan_tables(content, source_file=meta.get("filename", ""),
                                    title=meta.get("title", ""), unit_mode="legacy")
        out[meta.get("filename", "")] = tables
    return out


def table_evidence(table: Table, limit_rows: int = 3, width: int = 120) -> Dict[str, Any]:
    rows = []
    for row in (table.rows or [])[:limit_rows]:
        rows.append([str(cell)[:width] for cell in row[:8]])
    return {
        "table_id": table.table_id,
        "title": (table.title or "")[:200],
        "caption": (table.caption or "")[:200],
        "headers": [str(h)[:60] for h in (table.headers or [])[:8]],
        "rows_head": rows,
        "col_count": table.col_count,
        "row_count": table.row_count,
        "unit_candidates": list(table.unit_candidates),
    }


def loose_hits(text: str) -> List[str]:
    return [name for name, pat in LOOSE_PATTERNS.items() if re.search(pat, text or "")]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    docs = load_docs()
    raw = scan(docs, strip=False)
    stripped = scan(docs, strip=True)
    n_raw = sum(len(v) for v in raw.values())
    n_str = sum(len(v) for v in stripped.values())
    print("=== 1) 分母核对 ===")
    print("  未剥锅炉板: %d 张表" % n_raw)
    print("  剥锅炉板后（入库口径）: %d 张表" % n_str)
    print("  差: %d 张（仅在尾部锅炉板区域内）" % (n_raw - n_str))

    # 差值出处：按文件，raw 有而 stripped 没有的表
    dropped_by_file: Counter = Counter()
    dropped_titles: List[Dict[str, Any]] = []
    for fn, raw_tables in raw.items():
        kept = {(t.title, t.index) for t in stripped.get(fn, [])}
        for tb in raw_tables:
            if (tb.title, tb.index) not in kept:
                dropped_by_file[fn] += 1
                if len(dropped_titles) < 12:
                    dropped_titles.append({"file": fn, "index": tb.index,
                                           "title": (tb.title or "")[:80],
                                           "caption": (tb.caption or "")[:80],
                                           "rows": tb.row_count, "cols": tb.col_count})
    print("  涉及文件数: %d；前 8 个文件: %s" % (
        len(dropped_by_file), dropped_by_file.most_common(8)))
    for item in dropped_titles[:8]:
        print("    dropped:", json.dumps(item, ensure_ascii=False))

    # 2) 零候选
    zero: List[Tuple[str, Table]] = []
    for fn, tables in stripped.items():
        for tb in tables:
            if len(tb.unit_candidates) == 0:
                zero.append((fn, tb))
    print("\n=== 2) 零候选表 ===")
    print("  总数: %d / %d 张（%.1f%%）" % (len(zero), n_str, 100.0 * len(zero) / max(1, n_str)))

    rng = random.Random(SEED)
    sample = rng.sample(zero, min(SAMPLE_N, len(zero)))
    print("  固定 seed=%d 抽 %d 张：" % (SEED, len(sample)))
    sample_out = []
    for i, (fn, tb) in enumerate(sample, 1):
        ev = table_evidence(tb)
        blob = "\n".join([
            ev["title"], ev["caption"], " | ".join(ev["headers"]),
            " ".join(" | ".join(r) for r in ev["rows_head"]),
        ])
        hits = loose_hits(blob)
        sample_out.append({"no": i, "file": fn, "evidence": ev, "loose_hits": hits})
        print("  [%d] %s" % (i, fn))
        print("      title : %s" % ev["title"])
        print("      caption: %s" % ev["caption"])
        print("      headers: %s" % ev["headers"])
        print("      rows  : %s" % ev["rows_head"])
        print("      loose : %s" % (hits or "（无）"))

    # 3) 全量宽松探针（估漏检上界）
    print("\n=== 3) 宽松探针（仅作漏检上界）===")
    counts: Counter = Counter()
    any_hit = 0
    strict_only_ok = 0
    for fn, tb in zero:
        blob = "\n".join([tb.title or "", tb.caption or "", " ".join(tb.headers or []),
                           " ".join(" ".join(str(c) for c in row) for row in (tb.rows or []))])
        hits = loose_hits(blob)
        if hits:
            any_hit += 1
            for h in hits:
                counts[h] += 1
        else:
            strict_only_ok += 1
    print("  零候选 %d 张中，**宽松探针命中任一** = %d（%.1f%%）⇒ 仍需人工判" %
          (len(zero), any_hit, 100.0 * any_hit / max(1, len(zero))))
    print("  完全无任何单位/币种/百分号痕迹 = %d（%.1f%%）" %
          (strict_only_ok, 100.0 * strict_only_ok / max(1, len(zero))))
    print("  各探针命中数: %s" % counts.most_common())

    payload = {
        "denominator": {
            "raw_tables": n_raw,
            "stripped_tables": n_str,
            "diff": n_raw - n_str,
            "files_with_dropped_tables": len(dropped_by_file),
            "dropped_sample": dropped_titles,
        },
        "zero_candidate": {
            "count": len(zero),
            "total": n_str,
            "ratio": round(len(zero) / max(1, n_str), 4),
            "seed": SEED,
            "sample": sample_out,
            "loose_any_hit": any_hit,
            "loose_any_hit_ratio": round(any_hit / max(1, len(zero)), 4),
            "no_trace_at_all": strict_only_ok,
            "no_trace_ratio": round(strict_only_ok / max(1, len(zero)), 4),
            "loose_counts": dict(counts),
        },
    }
    out = pathlib.Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
