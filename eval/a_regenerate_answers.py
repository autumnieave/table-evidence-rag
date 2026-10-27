# -*- coding: utf-8 -*-
"""A 类 12 题：用当前检索证据重生成答案，再算 claim faithfulness。

对比三个口径：

1. 旧答案 + 旧证据（``faithfulness_a_live.json``，v1 基线）
2. 旧答案 + 新证据（``faithfulness_a_gated_bm25_no_rerank.json``，推荐检索）
3. **新答案 + 新证据**（本脚本）

检索配置与推荐口径一致：向量 + 门控表级 BM25 + 行证据排序，不接精排；
生成侧走 ``LLMGenerator.generate``（与线上同一 ``prompts.rag.build_prompt``），
只换证据，不换 prompt / 模型 / 温度。

用法::

    $env:PYTHONPATH="src"; $env:STAGE3_COLLECTION="research_reports_v6_rowfacts"; \
        .venv\\Scripts\\python.exe -m eval.a_regenerate_answers

    # 单题冒烟
    .venv\\Scripts\\python.exe -m eval.a_regenerate_answers --qid "B2006|2"
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
from core.entity_coverage import audit_entity_coverage, build_entity_constraint
from core.evidence_context import build_contexts_from_hits
from eval.a_faithfulness_live import (
    ENTITIES,
    OUT_DIR,
    _context_queries,
    _load_answers,
    _load_stocknames,
    _stdout_utf8,
)
from eval.question_ab_classify import A_SET

OLD_ANSWER_OLD_EVIDENCE = OUT_DIR / "faithfulness_a_live.json"
OLD_ANSWER_NEW_EVIDENCE = OUT_DIR / "faithfulness_a_gated_bm25_no_rerank.json"


def _load_per_question(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取已有评测产物，按 qid 建索引；文件不存在时返回空表。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["qid"]: row for row in payload.get("per_question") or []}


def _mean_summary(summaries: Sequence[Dict[str, Any]]) -> float:
    """对多题 summary 的 support_rate 取题均。"""
    values = [float((summary or {}).get("support_rate") or 0.0) for summary in summaries]
    return round(sum(values) / len(values), 6) if values else 0.0


def _regenerate_one(
    *,
    qid: str,
    question: str,
    emb: Any,
    retriever: Any,
    generator: Any,
    context_queries: Sequence[str] = (),
    known_entities: Sequence[str] = (),
    use_retrieval_query: bool = False,
) -> Dict[str, Any]:
    """跑一次检索 + 生成 + claim faithfulness，返回单题记录。"""
    retrieval_query = " ".join([question, *[str(q) for q in context_queries]]).strip()
    generation_query = retrieval_query if use_retrieval_query else question
    vec = emb.generate_embeddings([retrieval_query], text_type="query")[0]
    hits = retriever.retrieve_with_iteration(
        question,
        query_vector=vec,
        top_k=10,
        max_rounds=2,
        rerank=False,
        context_queries=context_queries,
    )
    contexts = build_contexts_from_hits(hits)
    first_answer = generator.generate(
        query=generation_query, contexts=contexts, history=None, stream=False
    )
    first_result = evaluate_claim_faithfulness(first_answer, hits)

    entity_aliases: Dict[str, str] = {}
    for hit in hits:
        payload = hit.get("payload") or {}
        code = str(payload.get("stock_code") or payload.get("stockCode") or "").strip()
        name = str(payload.get("stock_name") or payload.get("stockName") or "").strip()
        if code and name:
            entity_aliases[code] = name
    coverage_before = audit_entity_coverage(
        question,
        first_answer,
        known_entities=known_entities,
        context_queries=context_queries,
        entity_aliases=entity_aliases,
    )

    new_answer = first_answer
    result = first_result
    coverage_after = coverage_before
    repair_used = False
    if coverage_before.needs_repair:
        constraint = build_entity_constraint(coverage_before)
        if constraint:
            repaired = generator.generate(
                query=generation_query + "\n\n" + constraint,
                contexts=contexts,
                history=None,
                stream=False,
            )
            repaired_coverage = audit_entity_coverage(
                question,
                repaired,
                known_entities=known_entities,
                context_queries=context_queries,
                entity_aliases=entity_aliases,
            )
            before_bad = len(coverage_before.missing_query_entities) + len(
                coverage_before.unexpected_entities
            )
            after_bad = len(repaired_coverage.missing_query_entities) + len(
                repaired_coverage.unexpected_entities
            )
            if after_bad < before_bad or not repaired_coverage.needs_repair:
                new_answer = repaired
                result = evaluate_claim_faithfulness(new_answer, hits)
                coverage_after = repaired_coverage
                repair_used = True

    decision = (hits[0]["payload"].get("evidence_decision") or {}) if hits else {}
    if not decision and retriever.last_trace is not None:
        decision = {
            "action": retriever.last_trace.final_action,
            "mismatch_type": retriever.last_trace.final_mismatch,
        }
    return {
        "qid": qid,
        "question": question,
        "generation_query": generation_query,
        "retrieval_query": retrieval_query,
        "context_queries": list(context_queries),
        "decision": decision,
        "trace": retriever.last_trace.as_dict() if retriever.last_trace else {},
        "evidence_tables": [
            {
                "table_id": (h.get("payload") or {}).get("table_id"),
                "file_path": (h.get("payload") or {}).get("file_path"),
                "chunk_index": (h.get("payload") or {}).get("chunk_index"),
                "content_chars": len(str((h.get("payload") or {}).get("content") or "")),
            }
            for h in hits
        ],
        "context_chars": sum(len(c) for c in contexts),
        "entity_coverage_before": coverage_before.as_dict(),
        "entity_coverage_after": coverage_after.as_dict(),
        "entity_repair_used": repair_used,
        "first_answer": first_answer if repair_used else None,
        "first_answer_summary": first_result.get("summary") if repair_used else None,
        "new_answer": new_answer,
        "new_claims": result.get("claims") or [],
        "summary": result.get("summary") or {},
    }


def main(argv: Sequence[str] | None = None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="A 类 12 题新检索重生成 + faithfulness")
    parser.add_argument("--collection", default=os.getenv("STAGE3_COLLECTION", "research_reports_v6_rowfacts"))
    parser.add_argument("--out", default="faithfulness_a_regenerated.json")
    parser.add_argument("--qid", action="append", default=None, help="只跑指定 qid，可重复")
    parser.add_argument("--rerank-pool", type=int, default=50)
    parser.add_argument("--no-context", action="store_true", help="关闭追问上下文，做消融对照")
    parser.add_argument(
        "--retrieval-query",
        action="store_true",
        help="生成时使用 retrieval_query（问题+追问上下文），默认沿用 question",
    )
    args = parser.parse_args(argv)
    os.environ["STAGE3_COLLECTION"] = args.collection

    from eval.multi_stage_eval import COLLECTION, _build_runtime
    from llm.generator import LLMGenerator
    from prompts.rag import RAG_PROMPT_VERSION

    answers = _load_answers()
    entities = json.loads(ENTITIES.read_text(encoding="utf-8"))
    stocknames = _load_stocknames()
    cfg, emb, _qd, _reranker, retriever = _build_runtime(
        with_reranker=False,
        with_bm25=True,
        known_entities=stocknames,
        rerank_pool=args.rerank_pool,
    )
    generator = LLMGenerator(cfg)

    old_old = _load_per_question(OLD_ANSWER_OLD_EVIDENCE)
    old_new = _load_per_question(OLD_ANSWER_NEW_EVIDENCE)
    targets = sorted(A_SET)
    if args.qid:
        wanted = set(args.qid)
        targets = [qid for qid in targets if qid in wanted]
        if not targets:
            raise SystemExit(f"未找到指定 qid：{sorted(wanted)}")

    rows: List[Dict[str, Any]] = []
    print(f"[a-regen] collection={COLLECTION} targets={len(targets)} prompt={RAG_PROMPT_VERSION}", flush=True)
    for qid in targets:
        entity = entities.get(qid) or {}
        item = answers.get(qid) or {}
        question = str(entity.get("子问题") or item.get("问题") or "")
        context = [] if args.no_context else _context_queries(qid, entities)
        try:
            row = _regenerate_one(
                qid=qid,
                question=question,
                emb=emb,
                retriever=retriever,
                generator=generator,
                context_queries=context,
                known_entities=stocknames,
                use_retrieval_query=args.retrieval_query,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "qid": qid,
                "question": question,
                "error": f"{type(exc).__name__}: {exc}",
                "summary": {},
            }
        row["old_answer"] = str(item.get("答案") or "")
        row["old_answer_old_evidence_summary"] = (old_old.get(qid) or {}).get("summary") or {}
        row["old_answer_new_evidence_summary"] = (old_new.get(qid) or {}).get("summary") or {}
        if not row.get("error"):
            old_rate = float(row["old_answer_new_evidence_summary"].get("support_rate") or 0.0)
            new_rate = float(row["summary"].get("support_rate") or 0.0)
            row["delta_new_vs_old_answer"] = round(new_rate - old_rate, 4)
        rows.append(row)
        if row.get("error"):
            print(f"  {qid} ERROR {row['error']}", flush=True)
        else:
            print(
                "  %s old=%.3f old/new_ev=%.3f new=%.3f claims=%d supported=%d"
                % (
                    qid,
                    float(row["old_answer_old_evidence_summary"].get("support_rate") or 0.0),
                    float(row["old_answer_new_evidence_summary"].get("support_rate") or 0.0),
                    float(row["summary"].get("support_rate") or 0.0),
                    row["summary"].get("n_claims", 0),
                    row["summary"].get("n_supported", 0),
                ),
                flush=True,
            )

    good = [row for row in rows if not row.get("error")]
    good_with_claims = [
        row for row in good if int((row.get("summary") or {}).get("n_claims") or 0) > 0
    ]
    agg = aggregate_faithfulness(good)
    agg.update(
        {
            "suite": "a_regenerated",
            "collection": COLLECTION,
            "prompt_version": RAG_PROMPT_VERSION,
            "materials": "向量+门控BM25+行证据排序 → top-10 整表全文（不精排）",
            "retrieval_config": {
                "bm25": True,
                "gate": True,
                "rerank": False,
                "rerank_pool": args.rerank_pool,
                "context_queries": not args.no_context,
                "generation_query": "retrieval_query" if args.retrieval_query else "question",
                "known_entities": len(stocknames),
            },
            "targets": [row["qid"] for row in rows],
            "errors": [row["qid"] for row in rows if row.get("error")],
            "old_answer_old_evidence_mean": _mean_summary(
                [row.get("old_answer_old_evidence_summary") or {} for row in good]
            ),
            "old_answer_new_evidence_mean": _mean_summary(
                [row.get("old_answer_new_evidence_summary") or {} for row in good]
            ),
            "new_answer_new_evidence_mean": round(
                sum(float(row["summary"].get("support_rate") or 0.0) for row in good) / len(good), 6
            )
            if good
            else 0.0,
            "new_answer_new_evidence_mean_with_claims": round(
                sum(
                    float(row["summary"].get("support_rate") or 0.0)
                    for row in good_with_claims
                )
                / len(good_with_claims),
                6,
            )
            if good_with_claims
            else 0.0,
            "n_questions_with_claims": len(good_with_claims),
            "no_claim_qids": [
                row["qid"]
                for row in good
                if int((row.get("summary") or {}).get("n_claims") or 0) == 0
            ],
            "delta_new_vs_old_answer": round(
                sum(float(row.get("delta_new_vs_old_answer") or 0.0) for row in good) / len(good), 6
            )
            if good
            else 0.0,
            "clarify_qids": [
                row["qid"] for row in good if (row.get("decision") or {}).get("action") == "clarify_query"
            ],
            "bm25_enabled_qids": [
                row["qid"] for row in good if (row.get("trace") or {}).get("bm25_enabled")
            ],
            "entity_drift_before_qids": [
                row["qid"]
                for row in good
                if (row.get("entity_coverage_before") or {}).get("drift")
            ],
            "entity_drift_after_qids": [
                row["qid"]
                for row in good
                if (row.get("entity_coverage_after") or {}).get("drift")
            ],
            "entity_repair_qids": [row["qid"] for row in good if row.get("entity_repair_used")],
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = OUT_DIR / out
    out.write_text(
        json.dumps({"summary": agg, "per_question": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("[a-regen] %s" % json.dumps(agg, ensure_ascii=False), flush=True)
    print("[a-regen] WROTE %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
