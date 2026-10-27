# -*- coding: utf-8 -*-
"""4 题检索遗漏归因：目标表在 K=50/100/1000 的 rank，以及原问题 vs 补实体后的差异。

只做诊断，不修改线上检索。输出 ``训练结果数据/stage5_measure/retrieval_miss_4q.json``。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "训练结果数据" / "stage5_measure"
LIVE = OUT_DIR / "faithfulness_a_live.json"
OUT = OUT_DIR / "retrieval_miss_4q.json"

# 目标表由「公司/领域锚点 + 关键数字」从 table chunk 中识别。
# min_hits 是命中的关键数字数；数字均为答案 claim 中缺失、且原文可核验的值。
TARGETS: Dict[str, Dict[str, Any]] = {
    "B2004|1": {
        "companies": ["云南白药"],
        "numbers": ["3348.44", "3286.65", "57.07", "57.73", "1011.68", "1023.38", "9.92"],
        "anchors": ["云南白药", "上证指数", "收盘价", "流通市值"],
        "min_hits": 1,
        "augment": "云南白药 000538 上证指数 收盘价 流通市值 2025年4月",
    },
    "B2006|2": {
        "companies": [],
        "numbers": ["2496.80", "56.62", "99.22", "15.51", "31.37", "43.24"],
        "anchors": ["涨跌幅", "涨幅", "跌幅", "医药"],
        "min_hits": 2,
        "augment": "医药生物 行业周报 涨跌幅 华北制药 国药现代 威高骨科 百利天恒 永安药业",
    },
    "B2007|2": {
        "companies": ["华润三九", "江中药业"],
        "numbers": ["16.92", "17.73", "17.4", "12.20", "14.15", "17.7", "19.0", "74.27", "73.26", "16.27", "3.91", "3.93", "276"],
        "anchors": ["ROE", "净资产收益率", "净利率", "毛利率"],
        "min_hits": 1,
        "augment": "华润三九 江中药业 ROE 净资产收益率 净利率 毛利率 2024A 2025E",
    },
    "B2080|1": {
        "companies": ["片仔癀", "同仁堂", "达仁堂"],
        "numbers": ["75.77", "67.55", "17.6", "515"],
        "anchors": ["中药材", "医药工业", "毛利率", "片仔癀", "同仁堂"],
        "min_hits": 1,
        "augment": "片仔癀 同仁堂 医药工业 毛利率 中药材价格指数 2024H1",
    },
    "B2080|2": {
        "companies": [],
        "numbers": ["2250", "1856", "3100", "2550", "401", "114", "51.02", "165"],
        "anchors": ["中药材", "康美", "综合200", "价格指数"],
        "min_hits": 2,
        "augment": "中药材价格指数 康美 综合200 价格指数 2024年7月 2025年3月",
    },
}


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _norm(text: str) -> str:
    return (
        str(text or "")
        .replace(",", "").replace("，", "").replace(" ", "").replace("\u3000", "")
        .replace("\t", "").replace("\r", "").replace("\n", "")
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
        for digits in range(decimals, 0, -1):
            text = f"{value:.{digits}f}"
            if digits == decimals:
                text = text.rstrip("0").rstrip(".")
            vals.add(text)
    except ValueError:
        pass
    return [v for v in vals if v]


def _has_num(text: str, num: str) -> bool:
    norm = _norm(text)
    return any(
        re.search(r"(?<![\dA-Za-z.])" + re.escape(v) + r"(?![\dA-Za-z.])", norm)
        for v in _num_variants(num)
    )


def _load_questions() -> Dict[str, str]:
    data = json.loads(LIVE.read_text(encoding="utf-8"))
    return {str(r.get("qid")): str(r.get("question") or "") for r in data.get("per_question") or []}


def _scan_tables(qd: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = None
    flt = {"must": [{"key": "chunk_type", "match": {"value": "table"}}]}
    while True:
        points, offset = qd.client.scroll(
            collection_name=str(qd.config.QDRANT_COLLECTION_NAME),
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=flt,
        )
        for p in points:
            pay = p.payload or {}
            rows.append({
                "table_id": str(pay.get("table_id") or ""),
                "content": str(pay.get("content") or ""),
                "file": str(pay.get("filename") or pay.get("file_path") or ""),
                "stock_name": str(pay.get("stock_name") or pay.get("stockName") or ""),
                "table_type": str(pay.get("table_type") or ""),
            })
        if offset is None:
            break
    return rows


def _match_target_tables(rows: Sequence[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        content = row["content"]
        stock = row["stock_name"]
        fname = row["file"]
        comp_ok = (not cfg["companies"]) or any(
            c in stock or c in fname or c in content[:800] for c in cfg["companies"]
        )
        anchor_ok = any(a in content for a in cfg.get("anchors") or [])
        if not (comp_ok and anchor_ok):
            continue
        matched = [n for n in cfg["numbers"] if _has_num(content, n)]
        if len(matched) >= int(cfg.get("min_hits") or 1):
            item = dict(row)
            item["matched"] = matched
            item["n_hits"] = len(matched)
            out.append(item)
    out.sort(key=lambda r: (-r["n_hits"], r["file"], r["table_id"]))
    return out


def _dedupe_order(hits: Sequence[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for hit in hits:
        payload = hit.get("payload") or {}
        tid = str(payload.get("table_id") or payload.get("parent_id") or "")
        if tid and tid not in seen:
            seen.append(tid)
    return seen


def _rank_of(targets: Sequence[str], order: Sequence[str]) -> Dict[str, Any]:
    positions = {tid: i + 1 for i, tid in enumerate(order)}
    best = min((positions[t] for t in targets if t in positions), default=None)
    return {
        "best_rank": best,
        "per_table": {t: positions.get(t) for t in targets},
        "in_top10": bool(best and best <= 10),
        "in_top20": bool(best and best <= 20),
        "in_top50": bool(best and best <= 50),
        "in_top100": bool(best and best <= 100),
    }


def _run_query(retriever: Any, emb: Any, query: str, targets: Sequence[str], coarse_k: int, row_k: int = 200, rerank_pool: int = 20) -> Dict[str, Any]:
    from core.multi_stage import combine_score, dedupe_table_hits, group_row_hits

    vec = emb.generate_embeddings([query], text_type="query")[0]
    coarse_hits = retriever._search(vec, retriever._merge_filter(None, "table"), coarse_k)
    coarse_map = dedupe_table_hits(coarse_hits)
    coarse_order = _dedupe_order(coarse_hits)
    constraints = __import__("filters.row_match", fromlist=["extract_query_constraints"]).extract_query_constraints(query)
    row_hits = retriever._search(
        vec,
        retriever._row_filter(None, constraints, list(coarse_map.keys())),
        row_k,
    )
    row_candidates = len(row_hits)
    row_hits = retriever._filter_row_hits(row_hits, constraints)
    row_groups = group_row_hits(row_hits)
    ranked_ids = sorted(
        coarse_map,
        key=lambda tid: combine_score(
            float(coarse_map[tid].get("score") or 0.0),
            row_groups.get(tid, []),
            retriever.coarse_weight,
        ),
        reverse=True,
    )[:rerank_pool]
    return {
        "query": query,
        "coarse_k": coarse_k,
        "coarse_hits": len(coarse_hits),
        "coarse_tables": len(coarse_map),
        "row_candidates": row_candidates,
        "row_hits": len(row_hits),
        "row_tables": len(row_groups),
        "query_metrics": list(constraints.metrics),
        "query_period": constraints.period,
        "coarse_rank": _rank_of(targets, coarse_order),
        "ranked_rank": _rank_of(targets, ranked_ids),
        "coarse_order_top50": coarse_order[:50],
        "target_row_counts": {
            t: len(row_groups.get(t, [])) for t in targets if t in coarse_map
        },
    }


def _build_bm25(rows: Sequence[Dict[str, Any]]) -> Any:
    """用表格块建 BM25；与线上 HybridRetriever 的 BM25 腿同口径。"""
    from core.retrievers import BM25Retriever

    docs = [
        {
            "id": row["table_id"],
            "payload": {"content": row["content"], "table_id": row["table_id"]},
        }
        for row in rows
    ]
    return BM25Retriever(docs)


def _bm25_run(bm25: Any, query: str, targets: Sequence[str], top_k: int = 200) -> Dict[str, Any]:
    hits = bm25.retrieve(query, top_k=top_k)
    order = _dedupe_order(hits)
    return {**_rank_of(targets, order), "order_top60": order[:60]}


def _rerank_run(
    retriever: Any,
    reranker: Any,
    query: str,
    table_ids: Sequence[str],
    targets: Sequence[str],
) -> Dict[str, Any]:
    """对候选池做一次 qwen3-rerank，返回目标表在精排结果中的 rank。"""
    if not table_ids or reranker is None:
        return {"best_rank": None, "pool": 0}
    tables = retriever._fetch_tables(list(table_ids))
    ordered = [tid for tid in table_ids if tid in tables]
    if not ordered:
        return {"best_rank": None, "pool": 0}
    docs = [str((tables[tid].get("payload") or {}).get("content") or "")[:4000] for tid in ordered]
    ranked = reranker.rerank(query=query, documents=docs, top_n=len(docs))
    order = [
        ordered[item["index"]]
        for item in sorted(ranked, key=lambda x: -float(x.get("relevance_score") or 0))
        if isinstance(item.get("index"), int) and 0 <= int(item["index"]) < len(ordered)
    ]
    return {**_rank_of(targets, order), "pool": len(ordered)}


def _diagnose(
    runs: Dict[str, Dict[str, Any]],
    augmented: Dict[str, Any],
    bm25_aug: Dict[str, Any],
    rerank: Dict[str, Dict[str, Any]],
    target_found: bool,
) -> str:
    if not target_found:
        return "evidence_not_in_table_index"  # 证据在正文/图表，表级检索天然召回不到
    rank50 = runs["k50"]["coarse_rank"]["best_rank"]
    rank1000 = runs["k1000"]["coarse_rank"]["best_rank"]
    rerank_best = min(
        (v.get("best_rank") for v in rerank.values() if v.get("best_rank")),
        default=None,
    )
    if rank50 is not None and rank50 <= 50:
        if runs["k50"]["ranked_rank"]["in_top10"]:
            return "returned"
        if runs["k50"]["ranked_rank"]["in_top20"]:
            return "final_cut_top20_to_top10"
        if rerank_best and rerank_best <= 10:
            return "rerank_pool_cutoff"
        return "row_rank_drop"
    if augmented["coarse_rank"]["in_top10"]:
        return "query_under_specified_entity_missing"
    if bm25_aug.get("best_rank") and bm25_aug["best_rank"] <= 10:
        return "query_under_specified_entity_missing"
    if rerank_best and rerank_best <= 10:
        return "vector_recall_weak_hybrid_fix"
    if rank1000 is None:
        return "vector_recall_miss_beyond_1000"
    return "coarse_depth_or_embedding"


def main(argv: Sequence[str] | None = None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=os.getenv("STAGE3_COLLECTION", "research_reports_v6_rowfacts"))
    parser.add_argument("--rerank", action="store_true", help="额外跑 qwen3-rerank 验证精排池/混合召回")
    args = parser.parse_args(argv)
    os.environ["STAGE3_COLLECTION"] = args.collection

    from eval.multi_stage_eval import _build_runtime

    cfg, emb, qd, reranker, retriever = _build_runtime()
    questions = _load_questions()
    rows = _scan_tables(qd)
    bm25 = _build_bm25(rows)
    print(f"[diag] table chunks={len(rows)} collection={args.collection}", flush=True)

    result: Dict[str, Any] = {"collection": args.collection, "per_question": []}
    for qid, tcfg in TARGETS.items():
        targets = _match_target_tables(rows, tcfg)
        target_ids = [r["table_id"] for r in targets]
        question = questions.get(qid) or ""
        print(f"\n[diag] {qid} question={question!r} target_tables={len(targets)}", flush=True)
        for t in targets[:8]:
            print(f"  target {t['table_id']} stock={t['stock_name']} hits={t['matched']}", flush=True)

        runs = {}
        for k in (50, 100, 200, 1000):
            runs[f"k{k}"] = _run_query(retriever, emb, question, target_ids, k)
        aug = _run_query(retriever, emb, question + " " + tcfg["augment"], target_ids, 100)
        bm25_orig = _bm25_run(bm25, question, target_ids)
        bm25_aug = _bm25_run(bm25, question + " " + tcfg["augment"], target_ids)
        rerank_results: Dict[str, Dict[str, Any]] = {}
        if args.rerank and targets:
            # 1) K=50 向量候选池：验证 rerank_pool=20 是否截断
            rerank_results["vector_k50"] = _rerank_run(
                retriever, reranker, question, runs["k50"].get("coarse_order_top50") or [], target_ids
            )
            # 2) BM25 补实体池：验证混合召回能否把数值表拉回
            rerank_results["bm25_aug"] = _rerank_run(
                retriever, reranker, question + " " + tcfg["augment"], bm25_aug.get("order_top60") or [], target_ids
            )
        diag = _diagnose(runs, aug, bm25_aug, rerank_results, bool(targets))
        result["per_question"].append({
            "qid": qid,
            "question": question,
            "target_tables": [
                {k: t[k] for k in ("table_id", "file", "stock_name", "table_type", "matched")}
                for t in targets
            ],
            "runs": runs,
            "augmented": aug,
            "bm25": {"orig": bm25_orig, "aug": bm25_aug},
            "rerank": rerank_results,
            "diagnosis": diag,
        })
        print(f"  diagnosis={diag} k50_rank={runs['k50']['coarse_rank']['best_rank']} "
              f"k100_rank={runs['k100']['coarse_rank']['best_rank']} "
              f"k1000_rank={runs['k1000']['coarse_rank']['best_rank']} "
              f"aug100_rank={aug['coarse_rank']['best_rank']} "
              f"bm25_orig={bm25_orig['best_rank']} bm25_aug={bm25_aug['best_rank']} "
              f"rerank={ {k: v.get('best_rank') for k, v in rerank_results.items()} }", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[diag] WROTE {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
