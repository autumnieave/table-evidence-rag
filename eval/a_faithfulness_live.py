# -*- coding: utf-8 -*-
"""A 类 12 题：答案 vs 当前多阶段检索整表全文的 claim faithfulness。

输出 ``训练结果数据/stage5_measure/faithfulness_a_live_v2.json``，保留：

- 每题返回的整表 `content`（供“返回表里有没有对应行”审计）；
- 每条 claim 的 numbers / missing_numbers / supported / evidence；
- refs 口径（引用卡片 + L1 上下文）与 live 口径（整表全文）的逐题对照；
- 每题的证据决策（``clarify_query`` / ``retry_entity`` / ``retry_table`` / ``route_to_sql`` / ``accept``）。

v2 相对 v1 的变化：启用表级 BM25 腿 + 精排 + ``rerank_pool=50``，并把同一
主问题下更早的子问题作为上下文，供“涨跌的原因是什么”这类追问补全实体。

用法::

    $env:PYTHONPATH="src"; $env:STAGE3_COLLECTION="research_reports_v6_rowfacts"; \
        .venv\\Scripts\\python.exe -m eval.a_faithfulness_live
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.claim_faithfulness import aggregate_faithfulness, evaluate_claim_faithfulness
from eval.question_ab_classify import A_SET

OUT_DIR = ROOT / "训练结果数据" / "stage5_measure"
ANSWERS = ROOT / "训练结果数据" / "golden_answer_regression.json"
ENTITIES = ROOT / "训练结果数据" / "l5b_measure" / "entities108.json"
REFS = OUT_DIR / "faithfulness_refs.json"
STOCKNAMES = ROOT / "训练结果数据" / "l5b_measure" / "stockname_dist.json"


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _load_answers() -> Dict[str, Dict[str, Any]]:
    data = json.loads(ANSWERS.read_text(encoding="utf-8"))
    items = list(data.get("items") or [])
    entities = json.loads(ENTITIES.read_text(encoding="utf-8"))
    qids = list(entities)
    if len(qids) != len(items):
        raise ValueError(f"answers {len(items)} != entities {len(qids)}")
    return dict(zip(qids, items))


def _load_stocknames() -> Sequence[str]:
    """读取语料内公司名词表；文件不存在时返回空表（只影响澄清判定）。"""
    if not STOCKNAMES.exists():
        return ()
    payload = json.loads(STOCKNAMES.read_text(encoding="utf-8"))
    return tuple((payload.get("counts") or {}).keys())


def _context_queries(qid: str, entities: Dict[str, Dict[str, Any]]) -> List[str]:
    """取同一主问题下更早的子问题，作为追问的实体/指标上下文。"""
    base, _, index = qid.partition("|")
    if not index.isdigit():
        return []
    out: List[str] = []
    for prev in range(1, int(index)):
        item = entities.get(f"{base}|{prev}") or {}
        question = str(item.get("子问题") or "").strip()
        if question:
            out.append(question)
    return out


def _run_one(
    qid: str,
    question: str,
    answer: str,
    emb: Any,
    retriever: Any,
    context_queries: Sequence[str] = (),
    rerank: bool = True,
) -> Dict[str, Any]:
    retrieval_query = " ".join([question, *[str(q) for q in context_queries]]).strip()
    vec = emb.generate_embeddings([retrieval_query], text_type="query")[0]
    hits = retriever.retrieve_with_iteration(
        question,
        query_vector=vec,
        top_k=10,
        max_rounds=2,
        rerank=rerank,
        context_queries=context_queries,
    )
    result = evaluate_claim_faithfulness(answer, hits)
    decision = (hits[0]["payload"].get("evidence_decision") or {}) if hits else {}
    if not decision and retriever.last_trace is not None:
        decision = {
            "action": retriever.last_trace.final_action,
            "mismatch_type": retriever.last_trace.final_mismatch,
        }
    return {
        "qid": qid,
        "question": question,
        "retrieval_query": retrieval_query,
        "context_queries": list(context_queries),
        "answer": answer,
        "decision": decision,
        "trace": retriever.last_trace.as_dict() if retriever.last_trace else {},
        "tables": [
            {
                "table_id": (h.get("payload") or {}).get("table_id"),
                "file_path": (h.get("payload") or {}).get("file_path"),
                "chunk_index": (h.get("payload") or {}).get("chunk_index"),
                "content": (h.get("payload") or {}).get("content") or "",
            }
            for h in hits
        ],
        "claims": result.get("claims") or [],
        "summary": result.get("summary") or {},
    }


def main(argv: Sequence[str] | None = None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="A 类 12 题整表 live faithfulness")
    parser.add_argument("--collection", default=os.getenv("STAGE3_COLLECTION", "research_reports_v6_rowfacts"))
    parser.add_argument("--out", default="faithfulness_a_live_v2.json")
    parser.add_argument("--no-bm25", action="store_true", help="关闭表级 BM25，做消融对照")
    parser.add_argument("--no-rerank", action="store_true", help="关闭精排，做消融对照")
    parser.add_argument("--no-context", action="store_true", help="关闭追加上下文，做消融对照")
    parser.add_argument("--rerank-pool", type=int, default=50)
    args = parser.parse_args(argv)
    os.environ["STAGE3_COLLECTION"] = args.collection

    from eval.multi_stage_eval import COLLECTION, _build_runtime

    answers = _load_answers()
    entities = json.loads(ENTITIES.read_text(encoding="utf-8"))
    refs = json.loads(REFS.read_text(encoding="utf-8"))
    refs_by_qid = {r["qid"]: r for r in refs.get("per_question") or []}
    stocknames = _load_stocknames()
    _cfg, emb, _qd, _reranker, retriever = _build_runtime(
        with_reranker=not args.no_rerank,
        with_bm25=not args.no_bm25,
        known_entities=stocknames,
        rerank_pool=args.rerank_pool,
    )

    targets = sorted(A_SET)
    rows: List[Dict[str, Any]] = []
    print(f"[a-live] collection={COLLECTION} targets={len(targets)}", flush=True)
    for qid in targets:
        entity = entities.get(qid) or {}
        item = answers.get(qid) or {}
        question = str(entity.get("子问题") or item.get("问题") or "")
        answer = str(item.get("答案") or "")
        context = [] if args.no_context else _context_queries(qid, entities)
        row = _run_one(qid, question, answer, emb, retriever, context, rerank=not args.no_rerank)
        refs_summary = (refs_by_qid.get(qid) or {}).get("summary") or {}
        row["refs_summary"] = refs_summary
        row["delta_support_rate"] = round(
            float(row["summary"].get("support_rate") or 0.0)
            - float(refs_summary.get("support_rate") or 0.0),
            4,
        )
        rows.append(row)
        print(
            "  %s decision=%s refs=%.3f live=%.3f claims=%d live_supported=%d clarify=%s"
            % (
                qid,
                (row.get("decision") or {}).get("action"),
                float(refs_summary.get("support_rate") or 0.0),
                float(row["summary"].get("support_rate") or 0.0),
                row["summary"].get("n_claims", 0),
                row["summary"].get("n_supported", 0),
                (row.get("decision") or {}).get("clarify_question") or "-",
            ),
            flush=True,
        )

    agg = aggregate_faithfulness(rows)
    components = ["向量"]
    if not args.no_bm25:
        components.append("表级BM25(门控)")
    if not args.no_rerank:
        components.append("精排")
    agg.update(
        {
            "suite": "a_live",
            "collection": COLLECTION,
            "materials": "+".join(components) + " → top-10 整表全文（不追加 SQL 行）",
            "retrieval_config": {
                "bm25": not args.no_bm25,
                "rerank": not args.no_rerank,
                "rerank_pool": args.rerank_pool,
                "context_queries": not args.no_context,
                "known_entities": len(stocknames),
            },
            "targets": targets,
            "refs_mean_support_rate": round(
                sum(float(r.get("refs_summary", {}).get("support_rate") or 0.0) for r in rows) / len(rows),
                6,
            ),
            "live_mean_support_rate": round(
                sum(float(r.get("summary", {}).get("support_rate") or 0.0) for r in rows) / len(rows),
                6,
            ),
            "clarify_qids": [
                r["qid"]
                for r in rows
                if (r.get("decision") or {}).get("action") == "clarify_query"
            ],
            "bm25_enabled_qids": [
                r["qid"]
                for r in rows
                if (r.get("trace") or {}).get("bm25_enabled")
            ],
        }
    )
    baseline_path = OUT_DIR / "faithfulness_a_live.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get("summary") or {}
        old_live = float(baseline.get("live_mean_support_rate") or 0.0)
        new_live = float(agg.get("live_mean_support_rate") or 0.0)
        agg["baseline_live_mean_support_rate"] = round(old_live, 6)
        agg["delta_vs_baseline"] = round(new_live - old_live, 6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = OUT_DIR / out
    out.write_text(json.dumps({"summary": agg, "per_question": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[a-live] %s" % json.dumps(agg, ensure_ascii=False), flush=True)
    print("[a-live] WROTE %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
