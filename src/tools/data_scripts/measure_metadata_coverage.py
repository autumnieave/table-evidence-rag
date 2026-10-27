# -*- coding: utf-8 -*-
"""measure_metadata_coverage.py —— §3.2 元数据覆盖率实测（离线、零 API 成本）

用法（仓库根）::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m tools.data_scripts.measure_metadata_coverage

产出：``训练结果数据/table_struct_smoke/metadata_coverage_report.json``（含逐项覆盖率与自检）。

口径：
- 分块参数与重建一致（structured / strip_boilerplate / 1024 / overlap 100 / min 200 / 表切片 40 行 1200 字）；
- **不调用任何 API**：只跑 loader + splitter；
- 「适用域」= 个股研报 chunk（行业研报是市场/策略研究，本身没有报告期语义）。
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from config.rag_config import RAGConfig
from data import splitter as splitter_mod
from data.boilerplate import strip_trailing_boilerplate
from data.loader import load_industry_documents, load_markdown_documents
from data.splitter import split_documents
from data.table_struct import detect_units, scan_tables

logger = logging.getLogger("measure_metadata_coverage")

CORPUS_ROOT = pathlib.Path("data_corpus/全部数据/正式数据/附件5：研报数据")
OUT = pathlib.Path("训练结果数据/table_struct_smoke/metadata_coverage_report.json")
FIELDS = ["report_period", "publish_year", "unit", "unit_source", "unit_candidates",
          "stock_code", "stock_name", "metric_name", "doc_type", "chunk_type",
          "table_id", "table_type", "table_scope", "row_start", "row_end"]
_YEAR_RE = re.compile(r"(20\d{2})")


def build_config() -> RAGConfig:
    return RAGConfig(
        MARKDOWN_DIR=str(CORPUS_ROOT / "个股研报-解析结果-2.0"),
        EXCEL_METADATA_PATH=str(CORPUS_ROOT / "个股_研报信息.xlsx"),
        INDUSTRY_MARKDOWN_DIR=str(CORPUS_ROOT / "行业研报-解析结果-2.0"),
        INDUSTRY_EXCEL_PATH=str(CORPUS_ROOT / "行业_研报信息.xlsx"),
    )


def load_documents(config: RAGConfig) -> List[Dict[str, Any]]:
    """加载全部研报（个股 + 行业）。"""
    docs = load_markdown_documents(config.MARKDOWN_DIR, excel_metadata_path=config.EXCEL_METADATA_PATH)
    docs += load_industry_documents(config.INDUSTRY_MARKDOWN_DIR, excel_metadata_path=config.INDUSTRY_EXCEL_PATH)
    logger.info("加载研报 %d 篇", len(docs))
    return docs


def load_chunks(config: RAGConfig, docs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    docs = load_documents(config) if docs is None else docs
    return split_documents(
        docs,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        min_chars=config.CHUNK_MIN_CHARS,
        table_mode="structured",
        table_slice_rows=config.TABLE_SLICE_ROWS,
        table_slice_max_chars=config.TABLE_SLICE_MAX_CHARS,
        strip_boilerplate=True,
    )


def coverage(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    n = len(rows)
    hit = sum(1 for r in rows if r.get(field) not in (None, "", [], {}))
    return {"覆盖": hit, "分母": n, "覆盖率": round(hit / n, 4) if n else 0.0}


def distribution(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    """字段值分布（空值统一记为 ``(empty)``）。"""
    return dict(Counter(str(r.get(field) or "(empty)") for r in rows).most_common())


def layered_table_coverage(
    rows: List[Dict[str, Any]], fields: List[str]
) -> Dict[str, Any]:
    """按 ``doc_type × table_type × table_scope`` 分层统计表格块覆盖率。

    分母只含 ``chunk_type == "table"`` 的块；每个分组都返回分子/分母/覆盖率，
    避免用一个全库数字掩盖不同表格类型的差异。
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (row.get("chunk_type") or "") != "table":
            continue
        key = "{}|{}|{}".format(
            row.get("doc_type") or "unknown",
            row.get("table_type") or "other",
            row.get("table_scope") or "unknown",
        )
        groups[key].append(row)
    out: Dict[str, Any] = {}
    for key, sub in sorted(groups.items()):
        item: Dict[str, Any] = {"n": len(sub)}
        item.update({field: coverage(sub, field) for field in fields})
        item["unit_source 分布"] = distribution(sub, "unit_source")
        out[key] = item
    return out


def unit_candidate_distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """按候选个数分桶（0 / 1 / >=2），与表级 ``unit_candidates`` 口径一致。"""
    buckets = Counter()
    for row in rows:
        candidates = row.get("unit_candidates")
        count = len(candidates) if isinstance(candidates, list) else 0
        buckets["0" if count == 0 else "1" if count == 1 else ">=2"] += 1
    return dict(buckets)


def unit_ceiling(rows: List[Dict[str, Any]], contents: List[str]) -> Dict[str, Any]:
    """表格块的 ``unit`` 上限扫描（报告 §六 的分母来源，可复现）。

    - 「已填写」= 表格块 ``unit`` 非空；
    - 「上限」= 表格块**文本里能扫到单位标注**（``detect_units`` 命中 ≥ 1）；
    - 「多单位」= 同一块扫到 **≥ 2 个不同**单位（解析器按「不猜」策略置 ``None``）；
    - 「完全没有单位标注」= 一个也没扫到。

    注：扫的是**渲染后的 chunk 文本**（表标题 + 行），与解析器 ``head or first_column or body`` 的短路口径
    略有差异（解析器短路后可能漏掉后面几处的单位），因此这里的「上限」是**乐观上界**。
    """
    pairs = [(r, c) for r, c in zip(rows, contents) if (r.get("chunk_type") or "") == "table"]
    n = len(pairs)
    filled = sum(1 for r, _ in pairs if r.get("unit") not in (None, "", [], {}))
    any_unit = multi = 0
    for _r, text in pairs:
        units = detect_units(text or "")
        if units:
            any_unit += 1
            if len(set(units)) >= 2:
                multi += 1
    ceiling = max(any_unit, filled)
    return {
        "表格块数": n,
        "已填写unit": filled,
        "上限_表内扫到单位": any_unit,
        "其中多单位_策略置None": multi,
        "完全没有单位标注": n - any_unit,
        "已填_占上限": round(filled / ceiling, 4) if ceiling else 0.0,
    }


def unit_candidates_scan(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """**表级**口径：解析器 ``unit_candidates`` 的分布（分母 = **表数**，不是 chunk 数）。

    与 :func:`unit_ceiling`（chunk 级、独立正则扫描）**口径不同，两个数不可互相除**。

    - ``0`` 个候选 = 表里**完全没写单位**（无可提取）；
    - ``恰好 1`` 个 = 可写 ``unit``（解析器必然写入）；
    - ``>= 2`` 个 = **多单位**，按「不猜」策略置 ``None``。

    同时输出按 ``doc_type × table_type × table_scope`` 的表级分层覆盖率，
    以及 ``unit`` 适用域汇总；避免用一个全库数字掩盖不同表格类型的差异。
    """
    applicable_types = ("financial_statement", "valuation", "forecast", "industry_summary")
    fields = ("unit", "report_period", "publish_year", "metric_name", "stock_code")
    tables = one = many = zero = 0
    source_dist: Counter = Counter()
    type_dist: Counter = Counter()
    scope_dist: Counter = Counter()
    layers: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "unit": 0, "report_period": 0, "publish_year": 0,
                 "metric_name": 0, "stock_code": 0}
    )
    applicable_n = applicable_unit = applicable_zero = 0
    for doc in docs:
        content, _ = strip_trailing_boilerplate(doc.get("content") or "")
        meta = doc.get("metadata") or {}
        doc_type = meta.get("doc_type") or ("stock" if meta.get("stockCode") else "industry")
        year = publish_year(meta)
        _text, parsed = scan_tables(content, source_file=meta.get("filename", ""),
                                    title=meta.get("title", ""), doc_type=doc_type)
        for table in parsed:
            tables += 1
            source_dist[table.unit_source] += 1
            type_dist[table.table_type] += 1
            scope_dist[table.table_scope] += 1
            count = len(table.unit_candidates)
            if count == 0:
                zero += 1
            elif count == 1:
                one += 1
            else:
                many += 1

            key = "{}|{}|{}".format(doc_type, table.table_type, table.table_scope)
            layer = layers[key]
            layer["n"] += 1
            if table.unit:
                layer["unit"] += 1
            if table.report_period:
                layer["report_period"] += 1
            if year:
                layer["publish_year"] += 1
            if table.metric_names:
                layer["metric_name"] += 1
            if table.stock_codes:
                layer["stock_code"] += 1

            if table.table_type in applicable_types:
                applicable_n += 1
                if table.unit:
                    applicable_unit += 1
                if not table.unit_candidates:
                    applicable_zero += 1

    layered: Dict[str, Any] = {}
    for key, layer in sorted(layers.items()):
        n = layer["n"]
        item: Dict[str, Any] = {"n": n}
        item.update({
            field: {"覆盖": layer[field], "分母": n,
                    "覆盖率": round(layer[field] / n, 4) if n else 0.0}
            for field in fields
        })
        layered[key] = item

    return {
        "表数": tables,
        "零候选_表里没写单位": zero,
        "恰好1个_可提取": one,
        "多单位_不猜置None": many,
        "unit_source 分布": dict(source_dist.most_common()),
        "table_type 分布": dict(type_dist.most_common()),
        "table_scope 分布": dict(scope_dist.most_common()),
        "unit 适用域": {
            "定义": "table_type ∈ {financial_statement, valuation, forecast, industry_summary}；排除 ranking / pipeline / other",
            "表数": applicable_n,
            "unit 覆盖": applicable_unit,
            "unit 覆盖率": round(applicable_unit / applicable_n, 4) if applicable_n else 0.0,
            "零候选": applicable_zero,
            "零候选率": round(applicable_zero / applicable_n, 4) if applicable_n else 0.0,
        },
        "表级分层覆盖率": layered,
    }


def period_year(period: str) -> int:
    match = _YEAR_RE.search(period or "")
    return int(match.group(1)) if match else 0


def publish_year(meta: Dict[str, Any]) -> int:
    for key in ("publishDate", "filename", "title"):
        match = _YEAR_RE.search(str(meta.get(key) or ""))
        if match:
            return int(match.group(1))
    return 0


def invalid_period_count(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """报告期年份**晚于发布年** = 不可能，计为可疑（自动正确性自检）。"""
    total = bad = unknown = 0
    bad_docs = set()
    for r in rows:
        period = r.get("report_period")
        if not period:
            continue
        total += 1
        py = publish_year(r)
        per = period_year(period)
        if not py or not per:
            unknown += 1
            continue
        if per > py:
            bad += 1
            bad_docs.add(str(r.get("filename")))
    return {"有报告期的块": total, "报告期晚于发布年（可疑）": bad,
            "无法比对发布年": unknown, "可疑文档数": len(bad_docs),
            "可疑文档样例": sorted(bad_docs)[:10]}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = build_config()
    docs = load_documents(config)
    chunks = load_chunks(config, docs)
    rows = [dict(c["metadata"]) for c in chunks]
    contents = [c.get("content") or "" for c in chunks]
    logger.info("chunk 数 %d", len(rows))

    type_of = lambda r: r.get("chunk_type") or "legacy"
    slices = {
        "全部": rows,
        "正文块 prose": [r for r in rows if type_of(r) == "prose"],
        "表格块 table": [r for r in rows if type_of(r) == "table"],
        "锅炉板块 boilerplate": [r for r in rows if type_of(r) == "boilerplate"],
        "个股研报 chunk（适用域）": [r for r in rows if r.get("doc_type") == "stock"],
        "行业研报 chunk": [r for r in rows if r.get("doc_type") == "industry"],
    }
    report: Dict[str, Any] = {"chunk 总数": len(rows), "字段覆盖": {}, "报告期来源": {}, "自检": {}}
    for name, sub in slices.items():
        report["字段覆盖"][name] = {f: coverage(sub, f) for f in FIELDS}
    report["报告期来源"] = dict(Counter(r.get("report_period_source") or "(无字段)" for r in rows))
    report["报告期来源_by_doc_type"] = {
        dt: dict(Counter(r.get("report_period_source") or "(无字段)"
                         for r in rows if r.get("doc_type") == dt))
        for dt in ("stock", "industry")
    }
    report["自检"] = {"全部": invalid_period_count(rows)}
    report["unit 上限扫描"] = unit_ceiling(rows, contents)
    report["unit 候选扫描（表级）"] = unit_candidates_scan(docs)
    table_rows = [r for r in rows if type_of(r) == "table"]
    report["table_type 分布"] = distribution(table_rows, "table_type")
    report["table_scope 分布"] = distribution(table_rows, "table_scope")
    report["unit_source 分布"] = distribution(table_rows, "unit_source")
    report["unit_candidates 分桶"] = unit_candidate_distribution(table_rows)
    report["分层覆盖率_表格块"] = layered_table_coverage(
        table_rows, ["unit", "report_period", "publish_year", "metric_name", "stock_code"]
    )

    # 对照实验：把行业研报也放进「正文首段」推断，看假阳性代价
    original = splitter_mod.BODY_PERIOD_DOC_TYPES
    splitter_mod.BODY_PERIOD_DOC_TYPES = ("stock", "industry")
    _control_examples: List[Dict[str, str]] = []
    try:
        alt_chunks = load_chunks(config)
        alt_rows = [dict(c["metadata"]) for c in alt_chunks]
        _seen_docs = set()
        for c, r in zip(alt_chunks, alt_rows):
            if (r.get("doc_type") == "industry" and r.get("report_period_source") == "body"
                    and r.get("filename") not in _seen_docs):
                _seen_docs.add(r.get("filename"))
                _control_examples.append({
                    "period": str(r.get("report_period")),
                    "title": (r.get("title") or "")[:44],
                    "首段": (c.get("content") or "")[:110].replace("\n", " "),
                })
        report["对照_行业研报也启用正文推断"] = {
            "全部覆盖": coverage(alt_rows, "report_period"),
            "个股覆盖": coverage([r for r in alt_rows if r.get("doc_type") == "stock"], "report_period"),
            "行业覆盖": coverage([r for r in alt_rows if r.get("doc_type") == "industry"], "report_period"),
            "自检": invalid_period_count(alt_rows),
            "行业来自正文的块数": sum(1 for r in alt_rows
                                      if r.get("doc_type") == "industry"
                                      and r.get("report_period_source") == "body"),
            "行业正文来源样例（前 8，人工判断是否顺带提及）": _control_examples[:8],
        }
    finally:
        splitter_mod.BODY_PERIOD_DOC_TYPES = original

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== 元数据覆盖率（chunk 级）===")
    for name, sub in slices.items():
        n = len(sub)
        line = "  %-22s n=%5d | " % (name, n)
        line += " | ".join("%s %5.1f%%" % (f, 100.0 * coverage(sub, f)["覆盖率"])
                           for f in ("report_period", "unit", "metric_name", "stock_code"))
        print(line)
    print("\n=== 报告期来源分布 ===")
    for k, v in sorted(report["报告期来源"].items(), key=lambda x: -x[1]):
        print("  %-14s %6d  %5.1f%%" % (k, v, 100.0 * v / len(rows)))
    for dt, dist in report["报告期来源_by_doc_type"].items():
        tot = sum(dist.values())
        print("  [%s] %s" % (dt, {k: round(100.0 * v / tot, 1) for k, v in dist.items()}))
    print("\n=== unit 上限扫描（表格块）===")
    print("  ", json.dumps(report["unit 上限扫描"], ensure_ascii=False))
    print("\n=== unit 候选扫描（表级，解析器 unit_candidates）===")
    scan = report["unit 候选扫描（表级）"]
    print("  ", json.dumps({k: v for k, v in scan.items() if k != "表级分层覆盖率"},
                           ensure_ascii=False))
    print("\n=== 表级分层覆盖率（doc_type×table_type×table_scope）===")
    for key, item in scan["表级分层覆盖率"].items():
        print("  %-58s n=%5d | unit %5.1f%% | period %5.1f%% | year %5.1f%% | metric %5.1f%%" % (
            key, item["n"],
            100.0 * item["unit"]["覆盖率"],
            100.0 * item["report_period"]["覆盖率"],
            100.0 * item["publish_year"]["覆盖率"],
            100.0 * item["metric_name"]["覆盖率"],
        ))
    print("\n=== table_type / table_scope / unit_source 分布（表格块）===")
    print("  table_type :", json.dumps(report["table_type 分布"], ensure_ascii=False))
    print("  table_scope:", json.dumps(report["table_scope 分布"], ensure_ascii=False))
    print("  unit_source:", json.dumps(report["unit_source 分布"], ensure_ascii=False))
    print("  candidates :", json.dumps(report["unit_candidates 分桶"], ensure_ascii=False))
    print("\n=== 分层覆盖率（表格块，doc_type×table_type×table_scope）===")
    for key, item in report["分层覆盖率_表格块"].items():
        print("  %-58s n=%5d | unit %5.1f%% | period %5.1f%% | year %5.1f%% | metric %5.1f%%" % (
            key, item["n"],
            100.0 * item["unit"]["覆盖率"],
            100.0 * item["report_period"]["覆盖率"],
            100.0 * item["publish_year"]["覆盖率"],
            100.0 * item["metric_name"]["覆盖率"],
        ))
    print("\n=== 自检：报告期晚于发布年（应为 0）===")
    print("  ", json.dumps(report["自检"]["全部"], ensure_ascii=False))
    print("\n=== 对照：行业研报也启用正文推断 ===")
    print("  ", json.dumps(report["对照_行业研报也启用正文推断"], ensure_ascii=False))
    print("\nWROTE %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
