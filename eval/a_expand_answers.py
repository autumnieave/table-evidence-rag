# -*- coding: utf-8 -*-
"""A 类扩样：研报预测事实题的主口径 / v2 口径 faithfulness。

扩样题来自 ``训练结果数据/stage5_measure/a_expand_questions.json``：每题的
gold 行均为 forecast 表中的 2025E 营业收入（``unit_source=cell``），预测值只
存在于研报、不依赖财报实际值或 SQL，因此仍属 A 类（研报可答）。

两个口径：

- **主口径**：与 12 题冻结协议一致的多阶段 gated BM25 + 行证据排序（不精排），
  question 生成、不做实体定向重试；答案与同源 top-10 整表全文比对；
- **v2 口径**：多阶段表级检索 + 实体定向重试 + ``retrieval_query`` 生成，答案与
  top-10 整表全文比对。

两种口径都使用同一 strict claim faithfulness 评测器；输出逐题 strict 支持率、
bootstrap 95% CI（按题重采样）与 claim 级 Wilson 区间（参考值，不假设条间独立）。

用法::

    $env:PYTHONPATH="src"; $env:STAGE3_COLLECTION="research_reports_v6_rowfacts"; \
        $env:CITATION_FULL_TEXT_IN_PAYLOAD="1"; \
        .venv\\Scripts\\python.exe -m eval.a_expand_answers
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.a_faithfulness_live import OUT_DIR, _load_stocknames, _stdout_utf8
from eval.a_regenerate_answers import _regenerate_one

QUESTIONS = OUT_DIR / "a_expand_questions.json"
BASELINE_MAIN = OUT_DIR / "faithfulness_a_evaluator_fix.json"
BASELINE_V2 = OUT_DIR / "faithfulness_a_v2_evaluator_fix_reeval.json"


def _bootstrap_ci(values: Sequence[float], iterations: int = 10000, seed: int = 20261018) -> List[float]:
    """按题重采样的 percentile bootstrap 95% CI。"""
    values = [float(v) for v in values]
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[min(iterations - 1, int(0.975 * iterations))]
    return [round(lo, 6), round(hi, 6)]


def _wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """claim 级 Wilson 区间（参考值；同题 claim 不独立，区间会偏窄）。"""
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def _multi_run(
    item: Dict[str, Any],
    *,
    mode: str,
    emb: Any,
    retriever: Any,
    generator: Any,
    known_entities: Sequence[str],
    use_retrieval_query: bool,
) -> Dict[str, Any]:
    """多阶段 gated BM25 口径；main 与 v2 只在实体重试/生成输入上不同。"""
    row = _regenerate_one(
        qid=str(item["qid"]),
        question=str(item["question"]),
        emb=emb,
        retriever=retriever,
        generator=generator,
        context_queries=(),
        known_entities=known_entities,
        use_retrieval_query=use_retrieval_query,
    )
    gold = item.get("gold") or {}
    gold_table = str(gold.get("table_id") or "")
    tables = row.get("evidence_tables") or []
    row["mode"] = mode
    row["gold_table_in_top10"] = any(str(t.get("table_id") or "") == gold_table for t in tables)
    row["gold_table_rank"] = next(
        (i + 1 for i, t in enumerate(tables) if str(t.get("table_id") or "") == gold_table),
        None,
    )
    return row


def _load_baseline(path: Path, mode: str) -> Tuple[List[str], List[float]]:
    """读取 12 题冻结基线，返回 (qids, 逐题 strict 支持率)。"""
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    qids: List[str] = []
    rates: List[float] = []
    for row in payload.get("per_question") or []:
        qids.append(str(row.get("qid") or ""))
        if mode == "main":
            rates.append(float((row.get("strict") or {}).get("support_rate") or 0.0))
        else:
            rates.append(float(row.get("support_rate") or 0.0))
    return qids, rates


def _mode_summary(rows: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """汇总某一模式的逐题 strict 支持率、CI 与 claim 级参考区间。"""
    good = [r for r in rows if r.get("mode") == mode and not r.get("error")]
    rates = [float((r.get("summary") or {}).get("support_rate") or 0.0) for r in good]
    supported = sum(int((r.get("summary") or {}).get("n_supported") or 0) for r in good)
    claims = sum(int((r.get("summary") or {}).get("n_claims") or 0) for r in good)
    return {
        "n_questions": len(good),
        "mean_support_rate": round(sum(rates) / len(rates), 6) if rates else 0.0,
        "bootstrap_95_ci": _bootstrap_ci(rates),
        "n_claims": claims,
        "n_supported": supported,
        "claim_level_support_rate": round(supported / claims, 6) if claims else 0.0,
        "claim_level_wilson_95_ci": _wilson(supported, claims),
        "per_question_rates": [round(v, 6) for v in rates],
    }


def main(argv: Sequence[str] | None = None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="A 类扩样：主口径 / v2 口径 faithfulness")
    parser.add_argument("--collection", default=os.getenv("STAGE3_COLLECTION", "research_reports_v6_rowfacts"))
    parser.add_argument("--out", default="faithfulness_a_expand_20plus.json")
    parser.add_argument("--qid", action="append", default=None, help="只跑指定 qid，可重复")
    parser.add_argument("--mode", choices=("both", "main", "v2"), default="both")
    parser.add_argument("--no-baseline", action="store_true", help="不合并 12 题冻结基线")
    args = parser.parse_args(argv)
    os.environ["STAGE3_COLLECTION"] = args.collection
    os.environ.setdefault("CITATION_FULL_TEXT_IN_PAYLOAD", "1")

    payload = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    items = list(payload.get("items") or [])
    if args.qid:
        wanted = set(args.qid)
        items = [item for item in items if str(item.get("qid")) in wanted]
        if not items:
            raise SystemExit(f"未找到指定 qid：{sorted(wanted)}")

    from eval.multi_stage_eval import COLLECTION, _build_runtime
    from llm.generator import LLMGenerator
    stocknames = _load_stocknames()
    rows: List[Dict[str, Any]] = []
    main_runtime = None
    if args.mode in ("both", "main"):
        cfg_main, emb_main, _qd_main, _rr_main, retriever_main = _build_runtime(
            with_reranker=False,
            with_bm25=True,
            known_entities=stocknames,
            rerank_pool=50,
        )
        main_runtime = (emb_main, retriever_main, LLMGenerator(cfg_main))
    v2_runtime = None
    if args.mode in ("both", "v2"):
        cfg, emb, _qd, _reranker, retriever = _build_runtime(
            with_reranker=False,
            with_bm25=True,
            known_entities=stocknames,
            rerank_pool=50,
        )
        v2_runtime = (emb, retriever, LLMGenerator(cfg))

    print(
        f"[a-expand] collection={COLLECTION} items={len(items)} mode={args.mode}",
        flush=True,
    )
    for item in items:
        qid = str(item.get("qid") or "")
        if main_runtime is not None:
            emb, retriever, generator = main_runtime
            try:
                rows.append(
                    _multi_run(
                        item,
                        mode="main",
                        emb=emb,
                        retriever=retriever,
                        generator=generator,
                        known_entities=stocknames,
                        use_retrieval_query=False,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "qid": qid,
                        "mode": "main",
                        "question": item.get("question"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "summary": {},
                    }
                )
                print(f"  {qid} MAIN ERROR {type(exc).__name__}: {exc}", flush=True)
        if v2_runtime is not None:
            emb, retriever, generator = v2_runtime
            try:
                rows.append(
                    _multi_run(
                        item,
                        mode="v2",
                        emb=emb,
                        retriever=retriever,
                        generator=generator,
                        known_entities=stocknames,
                        use_retrieval_query=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "qid": qid,
                        "mode": "v2",
                        "question": item.get("question"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "summary": {},
                    }
                )
                print(f"  {qid} V2 ERROR {type(exc).__name__}: {exc}", flush=True)
        for mode in ("main", "v2"):
            hit = next((r for r in rows if r.get("qid") == qid and r.get("mode") == mode), None)
            if hit and not hit.get("error"):
                print(
                    "  %s %-4s rate=%.3f claims=%d supported=%d"
                    % (
                        qid,
                        mode,
                        float((hit.get("summary") or {}).get("support_rate") or 0.0),
                        int((hit.get("summary") or {}).get("n_claims") or 0),
                        int((hit.get("summary") or {}).get("n_supported") or 0),
                    ),
                    flush=True,
                )

    summary: Dict[str, Any] = {
        "suite": "a_expand_20plus",
        "collection": COLLECTION,
        "n_new_questions": len(items),
        "questions_file": str(QUESTIONS),
        "main_protocol": "多阶段 gated BM25+行证据（不精排），question 生成、无实体重试；答案 vs top-10 整表全文",
        "v2_protocol": "多阶段表级检索+实体定向重试+retrieval_query 答案 vs top-10 整表全文",
        "main": _mode_summary(rows, "main"),
        "v2": _mode_summary(rows, "v2"),
        "errors": [
            {"qid": r.get("qid"), "mode": r.get("mode"), "error": r.get("error")}
            for r in rows
            if r.get("error")
        ],
    }

    if not args.no_baseline:
        base_main_qids, base_main = _load_baseline(BASELINE_MAIN, "main")
        base_v2_qids, base_v2 = _load_baseline(BASELINE_V2, "v2")
        new_main = [
            float((r.get("summary") or {}).get("support_rate") or 0.0)
            for r in rows
            if r.get("mode") == "main" and not r.get("error")
        ]
        new_v2 = [
            float((r.get("summary") or {}).get("support_rate") or 0.0)
            for r in rows
            if r.get("mode") == "v2" and not r.get("error")
        ]
        combined_main = base_main + new_main
        combined_v2 = base_v2 + new_v2
        summary["baseline_12"] = {
            "main_mean": round(sum(base_main) / len(base_main), 6) if base_main else 0.0,
            "v2_mean": round(sum(base_v2) / len(base_v2), 6) if base_v2 else 0.0,
            "main_qids": base_main_qids,
            "v2_qids": base_v2_qids,
        }
        summary["combined_20plus"] = {
            "main": {
                "n_questions": len(combined_main),
                "mean_support_rate": round(sum(combined_main) / len(combined_main), 6)
                if combined_main
                else 0.0,
                "bootstrap_95_ci": _bootstrap_ci(combined_main),
            },
            "v2": {
                "n_questions": len(combined_v2),
                "mean_support_rate": round(sum(combined_v2) / len(combined_v2), 6)
                if combined_v2
                else 0.0,
                "bootstrap_95_ci": _bootstrap_ci(combined_v2),
            },
            "caveat": "主口径与 12 题冻结协议同为 gated BM25 证据；新增 10 题无历史答案，答案为本轮 question 生成，口径差异见报告。v2 口径协议一致。",
        }

    out = Path(args.out)
    if not out.is_absolute():
        out = OUT_DIR / out
    out.write_text(
        json.dumps({"summary": summary, "per_question": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("[a-expand] summary=%s" % json.dumps(summary, ensure_ascii=False), flush=True)
    print("[a-expand] WROTE %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
