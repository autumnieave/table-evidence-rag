# -*- coding: utf-8 -*-
"""阶段5：原子化 claim faithfulness 评测。

两个口径：

- ``refs``（默认）：108 条 agent 答案 vs **其自身引用卡片文本**（``引用[].text``）。
  这是可离线复现的基线，但引用卡片可能经过摘要/截断，不等于 full_text 全文，
  因此支持率是偏保守的下界，不能替代 claim-level 工作单。
- ``live``：4 条严重题，答案 vs **当前多阶段检索返回的整表全文**。需要 Qdrant +
  embedding API；默认集合 ``research_reports_v6_rowfacts``（可用 ``STAGE3_COLLECTION`` 覆盖）。

用法::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m eval.faithfulness_eval --suite refs
    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m eval.faithfulness_eval --suite live
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from core.claim_faithfulness import aggregate_faithfulness, evaluate_claim_faithfulness

OUT_DIR = ROOT / "训练结果数据" / "stage5_measure"
ANSWERS = ROOT / "训练结果数据" / "golden_answer_regression.json"
ENTITIES = ROOT / "训练结果数据" / "l5b_measure" / "entities108.json"
SQL_ROWS = ROOT / "训练结果数据" / "rows_full108_20261015.jsonl"
LIVE_TARGETS = ("B2065|1", "B2069|1", "B2015|1", "B2005|1")


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _load_answers() -> List[tuple[str, Dict[str, Any]]]:
    """按 entities108 的 108 个 qid 顺序与 golden 回归答案对齐。

    同一 BID 有多个子问题，不能按 ``编号`` 建字典（会覆盖成 80 条）。
    """
    data = json.loads(ANSWERS.read_text(encoding="utf-8"))
    items = list(data.get("items") or [])
    entities = json.loads(ENTITIES.read_text(encoding="utf-8"))
    qids = list(entities)
    if len(qids) != len(items):
        qids = ["%s|%d" % (item.get("编号") or "item", i + 1) for i, item in enumerate(items)]
    return list(zip(qids, items))


def _refs_to_hits(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """引用卡片文本 + L1 数字命中上下文（均来自源文件，不是 full_text 全文）。"""
    hits: List[Dict[str, Any]] = []
    for idx, ref in enumerate(item.get("引用") or []):
        text = str(ref.get("full_text") or ref.get("text") or "")
        citation = ref.get("citation") or {}
        contexts: List[str] = []
        for hit in citation.get("hits_context") or []:
            ctx = str((hit or {}).get("context") or "").strip()
            if ctx and ctx not in contexts:
                contexts.append(ctx)
        if contexts:
            text = text + "\n" + "\n".join(contexts)
        if not text.strip():
            continue
        hits.append(
            {
                "id": "%s#ref%d" % (item.get("编号"), idx),
                "score": 1.0,
                "payload": {
                    "content": text,
                    "file_path": ref.get("paper_path") or "",
                    "chunk_index": idx,
                    "table_id": "%s#ref%d" % (item.get("编号"), idx),
                },
            }
        )
    return hits


def _load_sql_rows() -> List[Dict[str, Any]]:
    """读取 80 条 SQL 结果（按问题匹配；同一 BID 多子问题取最相似者）。"""
    if not SQL_ROWS.exists():
        return []
    return [json.loads(line) for line in SQL_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _match_sql(question: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按问题文本相似度匹配 SQL 结果；阈值 0.45（只在 live 4 题上使用）。"""
    if not rows:
        return {}
    best = max(rows, key=lambda r: difflib.SequenceMatcher(None, question, str(r.get("问题") or "")).ratio())
    ratio = difflib.SequenceMatcher(None, question, str(best.get("问题") or "")).ratio()
    return best if ratio >= 0.45 else {}


def _compact_item(qid: str, result: Dict[str, Any]) -> Dict[str, Any]:
    claims = result.get("claims") or []
    unsupported = [c for c in claims if not c.get("supported")]
    return {
        "qid": qid,
        "summary": result.get("summary") or {},
        "unsupported_examples": [
            {
                "text": c.get("text"),
                "numbers": c.get("numbers"),
                "missing_numbers": c.get("missing_numbers"),
                "source_tables": c.get("source_tables"),
            }
            for c in unsupported[:5]
        ],
    }


def run_refs() -> int:
    answers = _load_answers()
    rows: List[Dict[str, Any]] = []
    print("[refs] 题数 %d" % len(answers), flush=True)
    for qid, item in answers:
        hits = _refs_to_hits(item)
        result = evaluate_claim_faithfulness(str(item.get("答案") or ""), hits)
        row = _compact_item(qid, result)
        rows.append(row)
        print("  %s claims=%d supported=%d rate=%.3f" % (
            qid, row["summary"].get("n_claims", 0), row["summary"].get("n_supported", 0),
            float(row["summary"].get("support_rate") or 0.0)), flush=True)

    agg = aggregate_faithfulness(rows)
    agg.update({
        "suite": "refs",
        "materials": "答案自身引用卡片文本 + L1 数字命中上下文（非 full_text）",
        "n_questions_with_claims": sum(1 for r in rows if (r["summary"].get("n_claims") or 0) > 0),
        "n_questions_support_rate_ge_0_8": sum(
            1 for r in rows if float(r["summary"].get("support_rate") or 0.0) >= 0.8
        ),
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "faithfulness_refs.json"
    out.write_text(json.dumps({"summary": agg, "per_question": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[refs] %s" % json.dumps(agg, ensure_ascii=False), flush=True)
    print("[refs] WROTE %s" % out, flush=True)
    return 0


def run_live() -> int:
    from eval.multi_stage_eval import COLLECTION, _build_runtime

    cfg, emb, _qd, _reranker, retriever = _build_runtime()
    answers = dict(_load_answers())
    entities = json.loads(ENTITIES.read_text(encoding="utf-8"))
    sql_rows = _load_sql_rows()
    rows: List[Dict[str, Any]] = []
    print("[live] collection=%s targets=%d" % (COLLECTION, len(LIVE_TARGETS)), flush=True)
    for qid in LIVE_TARGETS:
        entity = entities.get(qid) or {}
        item = answers.get(qid) or {}
        question = str(entity.get("子问题") or item.get("问题") or "")
        answer = str(item.get("答案") or "")
        matched_sql = _match_sql(question, sql_rows)
        vec = emb.generate_embeddings([question], text_type="query")[0]
        hits = retriever.retrieve_with_iteration(
            question, query_vector=vec, top_k=10, max_rounds=2, rerank=False
        )
        result = evaluate_claim_faithfulness(
            answer,
            hits,
            sql_rows=matched_sql.get("rows") or None,
            sql_text=str(matched_sql.get("SQL") or ""),
        )
        row = _compact_item(qid, result)
        row["question"] = question
        row["sql_matched"] = bool(matched_sql)
        row["sql_rows"] = len(matched_sql.get("rows") or [])
        row["evidence_decision"] = (hits[0]["payload"].get("evidence_decision") if hits else None)
        row["trace"] = retriever.last_trace.as_dict() if retriever.last_trace else {}
        rows.append(row)
        print("  %s decision=%s claims=%d supported=%d rate=%.3f" % (
            qid, (row["evidence_decision"] or {}).get("action"), row["summary"].get("n_claims", 0),
            row["summary"].get("n_supported", 0), float(row["summary"].get("support_rate") or 0.0)), flush=True)

    agg = aggregate_faithfulness(rows)
    agg.update({
        "suite": "live",
        "collection": COLLECTION,
        "materials": "当前多阶段检索整表全文",
        "targets": list(LIVE_TARGETS),
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "faithfulness_live.json"
    out.write_text(json.dumps({"summary": agg, "per_question": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[live] %s" % json.dumps(agg, ensure_ascii=False), flush=True)
    print("[live] WROTE %s" % out, flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="阶段5 claim faithfulness 评测")
    parser.add_argument("--suite", choices=("refs", "live"), default="refs")
    args = parser.parse_args(argv)
    return run_refs() if args.suite == "refs" else run_live()


if __name__ == "__main__":
    raise SystemExit(main())
