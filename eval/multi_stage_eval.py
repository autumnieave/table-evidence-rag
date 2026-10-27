# -*- coding: utf-8 -*-
"""阶段 3 多阶段检索评测（表级粗召回 → 行级精定位 → 整表返回）。

三个套件：
- ``entities``：27 条点名公司题 + 4 条严重题，对比单阶段整表 vs 多阶段；
- ``l5``：30 题 L5 标注的 table-level HR@10 / P@10 / nDCG@10 / MRR；
- ``latency``：12K vs 181K 点的 Qdrant 检索 P50/P95/P99（不含 rerank）。

用法::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m eval.multi_stage_eval --suite entities
    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m eval.multi_stage_eval --suite l5
    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m eval.multi_stage_eval --suite latency

边界：
- 返回单元是「整表」，不是 chunk；27 题真内容数不能与 chunk-level 的 4.93 做严格同分母对比。
- 精排输入截断 4000 字，返回仍是整表全文。
- 默认不做自动实体/期间抽取；A 类复跑可传 ``known_entities`` / ``context_queries``
  启用查询理解，27 题仍复用冻结的 entities108 缓存。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "训练结果数据" / "l5a_compare") not in sys.path:
    sys.path.insert(0, str(ROOT / "训练结果数据" / "l5a_compare"))

COLLECTION = os.getenv("STAGE3_COLLECTION", "research_reports_v5_rowfacts")
OLD_COLLECTION = "research_reports_v3_structured"
K = 10
RERANK_POOL = 50
FRAGMENT_MARKS = ("图表说明", "表格说明", "三大财务预测表", "[Table_", "Table_StockAndRan")
SHORT_CHARS = 300
TARGETS = ("B2065|1", "B2069|1", "B2015|1", "B2005|1")
OUT_DIR = ROOT / "训练结果数据" / "stage3_measure"


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _pct(values: Sequence[float], q: float) -> float:
    """最近秩分位数（毫秒）。"""
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(math.ceil(q * len(vals))) - 1))
    return round(vals[idx], 2)


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 2),
        "p50": _pct(values, 0.50),
        "p95": _pct(values, 0.95),
        "p99": _pct(values, 0.99),
        "max": round(max(values), 2),
    }


def _is_fragment(content: str) -> bool:
    c = (content or "").strip()
    if len(c) >= SHORT_CHARS:
        return False
    if any(m in c for m in FRAGMENT_MARKS):
        return True
    if c.startswith("#") and len(c) < 40:
        return True
    return bool(re.match(r"^《.{0,80}》\s*\(\d{4}-\d{2}-\d{2}\)?", c))


def _build_runtime(
    *,
    with_reranker: bool = False,
    with_bm25: bool = False,
    known_entities: Optional[Sequence[str]] = None,
    rerank_pool: int = RERANK_POOL,
) -> Tuple[Any, Any, Any, Any, Any]:
    """返回 (config, embedding, qdrant_wrapper, reranker, retriever)。

    Args:
        with_reranker: 是否把 ``RerankClient`` 接入多阶段检索器；
            默认 False，保持既有 single/multi 对照口径
        with_bm25: 是否构建表级 BM25 腿（评测 A 类时启用）
        known_entities: 公司名词表；传入后启用“缺实体 → clarify_query”
        rerank_pool: 每路进入精排的整表数
    """
    from config.rag_config import RAGConfig
    from embeddings import EmbeddingClient
    from vectorstore.qdrant_wrapper import QdrantClientWrapper
    from qdrant_client import QdrantClient
    from chains.rerank import RerankClient
    from core.multi_stage import MultiStageRetriever

    cfg = RAGConfig(QDRANT_COLLECTION_NAME=COLLECTION)
    emb = EmbeddingClient(cfg)
    qd = QdrantClientWrapper(cfg)
    qd.client = QdrantClient(url="http://localhost:6333", timeout=120.0)
    reranker = RerankClient(cfg)
    bm25 = None
    if with_bm25:
        from core.table_bm25 import build_table_bm25

        bm25 = build_table_bm25(qd, COLLECTION)
    retriever = MultiStageRetriever(
        emb,
        qd,
        reranker=reranker if with_reranker else None,
        bm25_retriever=bm25,
        known_entities=known_entities,
        coarse_k=100,
        row_k=200,
        rerank_pool=rerank_pool,
        final_k=rerank_pool,
        sibling_table_recall=bool(
            getattr(cfg, "SIBLING_TABLE_RECALL_ENABLED", False)
        ),
    )
    return cfg, emb, qd, reranker, retriever


def _finish_tables(
    hits: List[Dict[str, Any]],
    question: str,
    cfg: Any,
    reranker: Any,
    *,
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    vocab: Optional[Iterable[str]] = None,
    max_chars: int = 4000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
    """实体硬闸门 → 整表精排 → 每文件上限；返回 (top10, gate_meta, pool_real)。"""
    from core.rerankers import apply_file_diversity
    from filters.entity_gate import apply_entity_gate
    from filters.row_gate import is_lowinfo_row

    kept, gate = apply_entity_gate(
        hits,
        stock_name=stock_name,
        stock_code=stock_code,
        min_keep=0,
        keep_unknown_entity=True,
        known_names=vocab,
    )
    if not kept:
        return [], gate, 0

    resolved = None
    if stock_name or stock_code:
        from filters.entity_gate import resolve_names
        names = resolve_names(stock_name, vocab)
        resolved = names[0] if names else stock_name

    pool_real = sum(
        1 for h in kept
        if _same_entity(h, stock_name, stock_code, vocab)
        and len(str((h.get("payload") or {}).get("content") or "")) >= SHORT_CHARS
        and not is_lowinfo_row(str((h.get("payload") or {}).get("content") or ""), vocab)
        and not _is_fragment(str((h.get("payload") or {}).get("content") or ""))
    )

    docs = [str((h.get("payload") or {}).get("content") or "")[:max_chars] for h in kept]
    reranked = reranker.rerank(
        query=question,
        documents=docs,
        top_n=min(len(docs), cfg.RERANK_TOP_N * cfg.RERANK_OVERSAMPLE),
    )
    keys = [str((h.get("payload") or {}).get("file_path") or "") for h in kept]
    picked = apply_file_diversity(reranked, keys, cfg.RERANK_TOP_N, cfg.RERANK_MAX_PER_FILE)

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(picked[:K], start=1):
        idx = item.get("index")
        src = kept[idx] if isinstance(idx, int) and 0 <= idx < len(kept) else {}
        payload = dict((src or {}).get("payload") or {})
        out.append({
            "rank": i,
            "score": float(item.get("relevance_score") or 0.0),
            "payload": payload,
            "row_hits": list(payload.get("row_hits") or []),
            "_resolved": resolved,
        })
    return out, gate, pool_real



def _online_topk_filtered(retriever: Any, reranker: Any, cfg: Any, question: str, k: int, query_filter: Any) -> List[Dict[str, Any]]:
    """K=50 → 精排 → 每文件上限 → top-k；支持 Qdrant 过滤（用于排除 row_fact）。"""
    from core.rerankers import apply_file_diversity, file_keys_from_candidates

    hits = retriever.retrieve(question, query_filter=query_filter) or []
    if not hits:
        return []
    docs = [str((h.get("payload") or {}).get("content") or "") for h in hits]
    reranked = reranker.rerank(query=question, documents=docs,
                               top_n=cfg.RERANK_TOP_N * cfg.RERANK_OVERSAMPLE)
    selected = apply_file_diversity(reranked, file_keys_from_candidates(hits),
                                    cfg.RERANK_TOP_N, cfg.RERANK_MAX_PER_FILE)
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(selected[:k], start=1):
        idx = item.get("index")
        src = hits[idx] if isinstance(idx, int) and 0 <= idx < len(hits) else {}
        payload = (src or {}).get("payload") or {}
        out.append({"rank": i, "file_path": payload.get("file_path"), "content": payload.get("content") or ""})
    return out

def _same_entity(hit: Dict[str, Any], stock_name: Optional[str], stock_code: Optional[str], vocab: Any) -> bool:
    from filters.entity_gate import entity_hit
    if not stock_name and not stock_code:
        return False
    return entity_hit(hit.get("payload") or {}, stock_name, stock_code, vocab)


def _row_hit_audit(top10: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    """审计返回的 row_hits：指标命中 / 指标+期间同时命中。"""
    from filters.row_match import QueryConstraints, extract_query_constraints, row_fact_matches

    constraints = extract_query_constraints(question)
    metric_only = QueryConstraints(metrics=constraints.metrics, period=None)
    rows = [r for item in top10 for r in (item.get("row_hits") or [])]
    return {
        "row_hit_total": len(rows),
        "row_metric_hits": sum(1 for r in rows if row_fact_matches(r, metric_only)),
        "row_metric_period_hits": sum(1 for r in rows if row_fact_matches(r, constraints)),
        "query_metrics": list(constraints.metrics),
        "query_period": constraints.period,
    }


def _score_entity(top10: List[Dict[str, Any]], stock_name: Optional[str], stock_code: Optional[str], vocab: Any) -> Dict[str, Any]:
    from filters.row_gate import is_lowinfo_row

    hit = real = row = frag = mention = 0
    row_tables = 0
    first_same = None
    for item in top10:
        payload = item.get("payload") or {}
        content = str(payload.get("content") or "")
        same = _same_entity(item, stock_name, stock_code, vocab)
        if same:
            hit += 1
            if first_same is None:
                first_same = item["rank"]
            if len(content) >= SHORT_CHARS and not is_lowinfo_row(content, vocab) and not _is_fragment(content):
                real += 1
            if is_lowinfo_row(content, vocab):
                row += 1
            if _is_fragment(content):
                frag += 1
        resolved = item.get("_resolved") or stock_name or ""
        if resolved and str(resolved) in content:
            mention += 1
        if item.get("row_hits"):
            row_tables += 1
    return {
        "hit": hit,
        "real": real,
        "row": row,
        "frag": frag,
        "mention": mention,
        "row_tables": row_tables,
        "first_same": first_same,
        "empty": len(top10) == 0,
        "n_returned": len(top10),
    }


def _aggregate_entity(rows: Sequence[Dict[str, Any]], targets: Sequence[str]) -> Dict[str, Any]:
    n = len(rows)
    keys = ("hit", "real", "row", "frag", "mention", "row_tables", "n_returned")
    summary = {k: round(sum(float(r[k]) for r in rows) / n, 4) if n else 0.0 for k in keys}
    summary["空上下文题数"] = sum(1 for r in rows if r["empty"])
    summary["有行证据题数"] = sum(1 for r in rows if r["row_tables"] > 0)
    summary["题数"] = n
    detail = {qid: next((r for r in rows if r["qid"] == qid), None) for qid in targets}
    return {"summary": summary, "per_question": list(rows), "targets": detail}


def run_entities() -> int:
    cfg, emb, _qd, reranker, retriever = _build_runtime()
    here = ROOT / "训练结果数据" / "l5b_measure"
    entities = json.loads((here / "entities108.json").read_text(encoding="utf-8"))
    vocab = set(json.loads((here / "stockname_dist.json").read_text(encoding="utf-8"))["counts"])
    qs = {("%s|%s" % (v["bid"], v["sub_idx"])): v for v in entities.values()}
    order = sorted(k for k, v in qs.items() if v.get("stock_name") or v.get("stock_code"))
    print("[entities] 题数 %d" % len(order), flush=True)

    modes = ("single", "table_first")
    result: Dict[str, Any] = {"collection": COLLECTION, "modes": {}, "timings": {}}
    for mode in modes:
        rows: List[Dict[str, Any]] = []
        trace_times: List[float] = []
        for i, qid in enumerate(order, 1):
            v = qs[qid]
            q = v["子问题"]
            tower = time.perf_counter()
            vec = emb.generate_embeddings([q], text_type="query")[0]
            hits = retriever.retrieve(
                q,
                query_vector=vec,
                top_k=RERANK_POOL,
                multi_stage=(mode != "single"),
                mode="table_first",
                rerank=False,
            )
            tr = retriever.last_trace
            top10, gate, pool_real = _finish_tables(
                hits, q, cfg, reranker,
                stock_name=v.get("stock_name"), stock_code=v.get("stock_code"), vocab=vocab,
            )
            metrics = _score_entity(top10, v.get("stock_name"), v.get("stock_code"), vocab)
            metrics.update(_row_hit_audit(top10, q))
            metrics.update({
                "qid": qid,
                "entity": v.get("stock_name"),
                "gate": {k: gate.get(k) for k in ("applied", "reason", "n_in", "n_kept", "n_matched")},
                "pool_hit": sum(1 for h in hits if _same_entity(h, v.get("stock_name"), v.get("stock_code"), vocab)),
                "pool_real": pool_real,
                "trace": tr.as_dict() if tr else {},
                "elapsed_s": round(time.perf_counter() - tower, 3),
                "picked": [
                    {"file": Path(str((x.get("payload") or {}).get("file_path") or "")).name,
                     "table_id": (x.get("payload") or {}).get("table_id"),
                     "row_hits": len(x.get("row_hits") or [])}
                    for x in top10
                ],
            })
            rows.append(metrics)
            if tr:
                trace_times.append(float(tr.timings_ms.get("total") or 0.0))
            print("  [%s %d/%d] %s hit=%d real=%d rows=%d trace=%s" % (
                mode, i, len(order), qid, metrics["hit"], metrics["real"], metrics["row_tables"],
                json.dumps(tr.as_dict() if tr else {}, ensure_ascii=False)), flush=True)

        result["modes"][mode] = _aggregate_entity(rows, TARGETS)
        result["timings"][mode] = _summary(trace_times)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "entities.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["模式 | top10同主体 | 其中真内容 | 裸表行 | 碎片 | 正文提及 | 有行证据题数 | 空上下文 | trace P95(ms)"]
    for mode in modes:
        s = result["modes"][mode]["summary"]
        t = result["timings"][mode]
        lines.append("%s | %.2f | %.2f | %.2f | %.2f | %.2f | %d | %d | %.1f" % (
            mode, s["hit"], s["real"], s["row"], s["frag"], s["mention"], s["有行证据题数"],
            s["空上下文题数"], t["p95"]))
    lines.append("")
    lines.append("== 4 条严重题 ==")
    for qid in TARGETS:
        d1 = result["modes"]["single"]["targets"].get(qid) or {}
        d2 = result["modes"]["table_first"]["targets"].get(qid) or {}
        lines.append(
            "%s(%s) | 真内容 %s→%s | 同主体 %s→%s | 行证据表 %s→%s | "
            "行命中 %s→%s | 指标命中 %s→%s | 指标+期间 %s→%s | 空 %s→%s" % (
                qid, d1.get("entity"), d1.get("real"), d2.get("real"), d1.get("hit"), d2.get("hit"),
                d1.get("row_tables"), d2.get("row_tables"),
                d1.get("row_hit_total"), d2.get("row_hit_total"),
                d1.get("row_metric_hits"), d2.get("row_metric_hits"),
                d1.get("row_metric_period_hits"), d2.get("row_metric_period_hits"),
                d1.get("empty"), d2.get("empty"),
            )
        )
    (OUT_DIR / "entities_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print("[entities] WROTE %s" % out, flush=True)
    return 0


def run_l5() -> int:
    from l5a_compare import score_question
    from tools.data_scripts.retrieval_prelabel import load_sub_questions, stratified_sample

    cfg, emb, _qd, reranker, retriever = _build_runtime()
    rows = stratified_sample(load_sub_questions(), 30, 20260910)
    labels = json.loads((ROOT / "训练结果数据" / "retrieval_prelabel_20260910.json").read_text(encoding="utf-8"))
    by_qid = {("%s|%s" % (q["bid"], q["sub_idx"])): q for q in labels["questions"]}
    questions = []
    for r in rows:
        qid = "%s|%s" % (r["bid"], r["sub_idx"])
        lab = by_qid.get(qid)
        if lab is not None:
            questions.append((qid, r["子问题"], lab))
    print("[l5] 题数 %d（K=%d，table-level）" % (len(questions), K), flush=True)

    result: Dict[str, Any] = {"collection": COLLECTION, "k": K, "modes": {}, "timings": {}}
    for mode in ("single", "table_first"):
        per_q: List[Dict[str, Any]] = []
        trace_times: List[float] = []
        for i, (qid, q, lab) in enumerate(questions, 1):
            vec = emb.generate_embeddings([q], text_type="query")[0]
            hits = retriever.retrieve(
                q, query_vector=vec, top_k=RERANK_POOL,
                multi_stage=(mode != "single"), mode="table_first", rerank=False,
            )
            tr = retriever.last_trace
            top10, gate, pool_real = _finish_tables(hits, q, cfg, reranker, max_chars=4000)
            topk_rows = [
                {"rank": x["rank"], "file_path": (x.get("payload") or {}).get("file_path"),
                 "content": str((x.get("payload") or {}).get("content") or "")}
                for x in top10
            ]
            m = score_question(lab["chunks"], topk_rows, K)
            m["bid"] = qid
            m["gate"] = {k: gate.get(k) for k in ("applied", "reason", "n_in", "n_kept", "n_matched")}
            m["pool_real"] = pool_real
            m["trace"] = tr.as_dict() if tr else {}
            m["row_tables"] = sum(1 for x in top10 if x.get("row_hits"))
            per_q.append(m)
            if tr:
                trace_times.append(float(tr.timings_ms.get("total") or 0.0))
            print("  [%s %d/%d] %s HR=%.3f nDCG=%.3f MRR=%.3f rows=%d" % (
                mode, i, len(questions), qid, m["Hit Rate@%d" % K], m["nDCG@%d" % K], m["MRR"], m["row_tables"]),
                flush=True)

        n = len(per_q)
        def mean(key: str) -> float:
            return round(sum(float(x[key]) for x in per_q) / n, 4) if n else 0.0

        summary = {
            "题数": n,
            "Hit Rate@10": mean("Hit Rate@10"),
            "Precision@10": mean("Precision@10"),
            "nDCG@10": mean("nDCG@10"),
            "MRR": mean("MRR"),
            "命中片段总数": sum(x["命中片段数"] for x in per_q),
            "相关片段总数": sum(x["相关片段数"] for x in per_q),
            "有行证据题数": sum(1 for x in per_q if x["row_tables"] > 0),
        }
        result["modes"][mode] = {"summary": summary, "per_question": per_q}
        result["timings"][mode] = _summary(trace_times)
        print("[l5] %s -> %s" % (mode, json.dumps(summary, ensure_ascii=False)), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "l5.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[l5] WROTE %s" % out, flush=True)
    return 0


def run_latency() -> int:
    from config.rag_config import RAGConfig
    from embeddings import EmbeddingClient
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    from tools.data_scripts.retrieval_prelabel import load_sub_questions, stratified_sample

    cfg = RAGConfig()
    emb = EmbeddingClient(cfg)
    client = QdrantClient(url="http://localhost:6333", timeout=120.0)
    rows = stratified_sample(load_sub_questions(), 30, 20260910)
    questions = [r["子问题"] for r in rows]
    vectors: List[List[float]] = []
    for i in range(0, len(questions), 10):
        vectors.extend(emb.generate_embeddings(questions[i:i + 10], text_type="query"))
    print("[latency] 题数 %d，向量 %d" % (len(questions), len(vectors)), flush=True)

    configs = [
        ("v3_structured_all", OLD_COLLECTION, None),
        ("v5_all", COLLECTION, None),
        ("v5_table_only", COLLECTION, Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value="table"))])),
    ]
    result: Dict[str, Any] = {"points": {}, "configs": {}}
    for name, collection, flt in configs:
        info = client.get_collection(collection)
        result["points"][name] = info.points_count
        # warm-up：丢弃前 5 次
        for v in vectors[:5]:
            client.query_points(collection_name=collection, query=v, limit=50, with_payload=False, query_filter=flt)
        times: List[float] = []
        for rep in range(5):
            for v in vectors:
                t0 = time.perf_counter()
                client.query_points(collection_name=collection, query=v, limit=50, with_payload=False, query_filter=flt)
                times.append((time.perf_counter() - t0) * 1000.0)
        result["configs"][name] = _summary(times)
        print("[latency] %s points=%d -> %s" % (name, info.points_count, json.dumps(result["configs"][name], ensure_ascii=False)), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "latency.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["配置 | 点数 | P50(ms) | P95(ms) | P99(ms) | mean(ms)"]
    for name, _, _ in configs:
        s = result["configs"][name]
        lines.append("%s | %d | %.2f | %.2f | %.2f | %.2f" % (
            name, result["points"][name], s["p50"], s["p95"], s["p99"], s["mean"]))
    (OUT_DIR / "latency_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print("[latency] WROTE %s" % out, flush=True)
    return 0



def run_l5_full() -> int:
    """全索引 chunk-level L5 回归（不是多阶段表路由）。

    目的：检查加入 169,242 条 row_fact 后，线上同口径 HybridRetriever(50) → 精排 → top-10
    是否发生退化。基线为 `v3_structured` 的 30 题：HR@10 0.1072 / nDCG@10 0.1206 / MRR 0.1026。
    """
    from config.rag_config import RAGConfig
    from pipelines.rag_pipeline import RAGPipeline
    from core.retrievers import HandwrittenRetriever, HybridRetriever
    from core.rerankers import RerankerAdapter
    from chains.rerank import RerankClient
    from qdrant_client import QdrantClient
    from l5a_compare import online_topk, score_question
    from tools.data_scripts.retrieval_prelabel import load_sub_questions, stratified_sample

    cfg = RAGConfig(QDRANT_COLLECTION_NAME=COLLECTION)
    pipeline = RAGPipeline(cfg)
    pipeline.qdrant_client.client = QdrantClient(url="http://localhost:6333", timeout=120.0)
    bm25 = pipeline.build_bm25_index()
    vector_retriever = HandwrittenRetriever(
        embedding_client=pipeline.embedding_client,
        qdrant_client=pipeline.qdrant_client,
        top_k=cfg.HYBRID_TOPK_VECTOR,
    )
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        top_k=cfg.RETRIEVAL_K,
        rrf_k=cfg.HYBRID_RRF_K,
        topk_vector=cfg.HYBRID_TOPK_VECTOR,
        topk_bm25=cfg.HYBRID_TOPK_BM25,
        vector_floor_ratio=cfg.HYBRID_VECTOR_FLOOR_RATIO,
    )
    reranker = RerankerAdapter(RerankClient(cfg))

    rows = stratified_sample(load_sub_questions(), 30, 20260910)
    labels = json.loads((ROOT / "训练结果数据" / "retrieval_prelabel_20260910.json").read_text(encoding="utf-8"))
    by_qid = {("%s|%s" % (q["bid"], q["sub_idx"])): q for q in labels["questions"]}
    per_q: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, 1):
        qid = "%s|%s" % (r["bid"], r["sub_idx"])
        lab = by_qid.get(qid)
        if lab is None:
            continue
        topk = online_topk(retriever, reranker, cfg, r["子问题"], K)
        m = score_question(lab["chunks"], topk, K)
        m["bid"] = qid
        per_q.append(m)
        print("  [l5_full %d/%d] %s HR=%.3f nDCG=%.3f MRR=%.3f" % (
            i, len(rows), qid, m["Hit Rate@10"], m["nDCG@10"], m["MRR"]), flush=True)

    n = len(per_q)
    summary = {
        "题数": n,
        "Hit Rate@10": round(sum(x["Hit Rate@10"] for x in per_q) / n, 4),
        "Precision@10": round(sum(x["Precision@10"] for x in per_q) / n, 4),
        "nDCG@10": round(sum(x["nDCG@10"] for x in per_q) / n, 4),
        "MRR": round(sum(x["MRR"] for x in per_q) / n, 4),
        "命中片段总数": sum(x["命中片段数"] for x in per_q),
        "相关片段总数": sum(x["相关片段数"] for x in per_q),
    }
    result = {"collection": COLLECTION, "k": K, "summary": summary, "per_question": per_q,
              "baseline_v3_structured": {"Hit Rate@10": 0.1072, "nDCG@10": 0.1206, "MRR": 0.1026}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "l5_full.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[l5_full] -> %s" % json.dumps(summary, ensure_ascii=False), flush=True)
    print("[l5_full] WROTE %s" % out, flush=True)
    return 0


def run_l5_vector() -> int:
    """全索引 chunk-level L5 回归（向量-only，非混合检索）。

    对 `v3_structured`（12,301 点）与 `v5_rowfacts`（181,543 点）使用同一
    HandwrittenRetriever(K=50) → qwen3-rerank(20) → 每文件≤2 → top-10，
    检查 row_fact 是否在全索引召回中造成退化。
    """
    from config.rag_config import RAGConfig
    from embeddings import EmbeddingClient
    from vectorstore.qdrant_wrapper import QdrantClientWrapper
    from qdrant_client import QdrantClient
    from core.retrievers import HandwrittenRetriever
    from core.rerankers import RerankerAdapter
    from chains.rerank import RerankClient
    from l5a_compare import online_topk, score_question
    from tools.data_scripts.retrieval_prelabel import load_sub_questions, stratified_sample

    rows = stratified_sample(load_sub_questions(), 30, 20260910)
    labels = json.loads((ROOT / "训练结果数据" / "retrieval_prelabel_20260910.json").read_text(encoding="utf-8"))
    by_qid = {("%s|%s" % (q["bid"], q["sub_idx"])): q for q in labels["questions"]}
    questions = []
    for r in rows:
        qid = "%s|%s" % (r["bid"], r["sub_idx"])
        if qid in by_qid:
            questions.append((qid, r["子问题"], by_qid[qid]))

    from qdrant_client.http.models import FieldCondition, Filter, MatchAny

    no_rowfacts = Filter(must=[FieldCondition(key="chunk_type", match=MatchAny(any=["prose", "table", "boilerplate"]))])
    run_cfgs = [
        ("v3_structured", OLD_COLLECTION, None),
        ("v5_all", COLLECTION, None),
        ("v5_no_rowfacts", COLLECTION, no_rowfacts),
    ]
    result: Dict[str, Any] = {"k": K, "collections": {}}
    for name, collection, flt in run_cfgs:
        cfg = RAGConfig(QDRANT_COLLECTION_NAME=collection)
        emb = EmbeddingClient(cfg)
        qd = QdrantClientWrapper(cfg)
        qd.client = QdrantClient(url="http://localhost:6333", timeout=120.0)
        retriever = HandwrittenRetriever(emb, qd, top_k=cfg.RETRIEVAL_K)
        reranker = RerankerAdapter(RerankClient(cfg))
        per_q: List[Dict[str, Any]] = []
        print("[l5_vector] %s 开始（%d 题）" % (name, len(questions)), flush=True)
        for i, (qid, q, lab) in enumerate(questions, 1):
            topk = _online_topk_filtered(retriever, reranker, cfg, q, K, flt) if flt is not None else online_topk(retriever, reranker, cfg, q, K)
            m = score_question(lab["chunks"], topk, K)
            m["bid"] = qid
            per_q.append(m)
            print("  [%s %d/%d] %s HR=%.3f nDCG=%.3f MRR=%.3f" % (
                name, i, len(questions), qid, m["Hit Rate@10"], m["nDCG@10"], m["MRR"]), flush=True)
        n = len(per_q)
        summary = {
            "题数": n,
            "Hit Rate@10": round(sum(x["Hit Rate@10"] for x in per_q) / n, 4),
            "Precision@10": round(sum(x["Precision@10"] for x in per_q) / n, 4),
            "nDCG@10": round(sum(x["nDCG@10"] for x in per_q) / n, 4),
            "MRR": round(sum(x["MRR"] for x in per_q) / n, 4),
            "命中片段总数": sum(x["命中片段数"] for x in per_q),
            "相关片段总数": sum(x["相关片段数"] for x in per_q),
        }
        result["collections"][name] = {"summary": summary, "per_question": per_q}
        print("[l5_vector] %s -> %s" % (name, json.dumps(summary, ensure_ascii=False)), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "l5_vector.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["配置 | 集合 | 点数 | Hit Rate@10 | P@10 | nDCG@10 | MRR | 命中/相关"]
    for name, collection, _ in run_cfgs:
        ss = result["collections"][name]["summary"]
        pts = 12301 if collection == OLD_COLLECTION else 181543
        lines.append("%s | %s | %d | %.4f | %.4f | %.4f | %.4f | %d/%d" % (
            name, collection, pts, ss["Hit Rate@10"], ss["Precision@10"], ss["nDCG@10"], ss["MRR"],
            ss["命中片段总数"], ss["相关片段总数"]))
    (OUT_DIR / "l5_vector_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print("[l5_vector] WROTE %s" % out, flush=True)
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="阶段 3 多阶段检索评测")
    parser.add_argument("--suite", choices=("entities", "l5", "l5_full", "l5_vector", "latency"), required=True)
    args = parser.parse_args(argv)
    if args.suite == "entities":
        return run_entities()
    if args.suite == "l5":
        return run_l5()
    if args.suite == "l5_full":
        return run_l5_full()
    if args.suite == "l5_vector":
        return run_l5_vector()
    return run_latency()


if __name__ == "__main__":
    raise SystemExit(main())
