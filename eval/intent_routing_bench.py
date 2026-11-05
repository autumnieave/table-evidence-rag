"""意图路由验证：SQL / RAG 分流准确率与 SQL 链路耗时。

两种口径：

1. ``keyword``：离线确定性分类器 ``LangGraphMultiAgentPlanner._classify_route_type``；
2. ``supervisor``：真实 supervisor LLM 拆任务后的派发结果（需 ``--run-supervisor``）。

``--run-sql`` 会真实执行 5 条 SQL 类问题（MySQL + LLM），记录平均耗时。

用法::

    $env:PYTHONPATH="src"
    python eval/intent_routing_bench.py --run-supervisor --run-sql
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


CASES: List[Dict[str, str]] = [
    {"id": "S1", "label": "sql", "question": "2024年利润最高的top10企业是哪些"},
    {"id": "S2", "label": "sql", "question": "片仔癀2025年第三季度的利润总额是多少"},
    {"id": "S3", "label": "sql", "question": "2025年前三季度收入超过200亿元的企业有哪些"},
    {"id": "S4", "label": "sql", "question": "江中药业2025年三季度净利润是多少"},
    {"id": "S5", "label": "sql", "question": "方盛制药2025年第三季度营业收入是多少"},
    {"id": "R1", "label": "rag", "question": "研报怎么评价中药行业的竞争格局"},
    {"id": "R2", "label": "rag", "question": "券商对片仔癀的目标价和评级"},
    {"id": "R3", "label": "rag", "question": "中药创新药政策对行业的影响"},
    {"id": "R4", "label": "rag", "question": "这些研报观点的来源可靠吗"},
    {"id": "R5", "label": "rag", "question": "分析一下中药产业链的竞争格局"},
]


def _binary(route_type: str) -> str:
    """把三分类压成 SQL / RAG 二分类；mixed 记为 mixed。"""
    if route_type == "financial":
        return "sql"
    if route_type == "research":
        return "rag"
    return "mixed"


def _tasks_to_binary(tasks: List[Dict[str, Any]], route_type: str) -> str:
    agents = {str((t or {}).get("agent") or "") for t in (tasks or [])}
    has_fin = "financial" in agents
    has_res = "research" in agents
    if has_fin and has_res:
        return "mixed"
    if has_fin:
        return "sql"
    if has_res:
        return "rag"
    return _binary(route_type)


def _keyword_predictions() -> List[Dict[str, Any]]:
    from agents.langgraph_multi_agent import LangGraphMultiAgentPlanner

    rows = []
    for case in CASES:
        route_type = LangGraphMultiAgentPlanner._classify_route_type(case["question"])
        pred = _binary(route_type)
        rows.append({**case, "route_type": route_type, "pred": pred, "correct": pred == case["label"]})
    return rows


def _supervisor_predictions() -> List[Dict[str, Any]]:
    from agents.langgraph_multi_agent import LangGraphMultiAgentPlanner
    from core.pipeline import get_pipeline
    from prompts.multi_agent import MULTI_AGENT_SUPERVISOR_PROMPT

    pipeline = get_pipeline()
    planner = LangGraphMultiAgentPlanner(
        llm_client=pipeline.llm_generator.client,
        config=pipeline.config,
        rag_pipeline=pipeline,
    )
    rows = []
    for case in CASES:
        state = {
            "messages": [
                {"role": "system", "content": MULTI_AGENT_SUPERVISOR_PROMPT},
                {"role": "user", "content": case["question"]},
            ],
            "user_query": case["question"],
            "user_id": "route-bench",
        }
        started = time.perf_counter()
        out = planner._supervisor(state)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        tasks = out.get("tasks") or []
        route_type = out.get("route_type") or ""
        pred = _tasks_to_binary(tasks, route_type)
        rows.append(
            {
                **case,
                "route_type": route_type,
                "tasks": tasks,
                "pred": pred,
                "correct": pred == case["label"],
                "supervisor_ms": elapsed_ms,
            }
        )
    return rows


def _run_sql() -> List[Dict[str, Any]]:
    from agents.planner import call_financial_chatflow
    from core.pipeline import get_pipeline

    pipeline = get_pipeline()
    rows = []
    for case in [c for c in CASES if c["label"] == "sql"]:
        started = time.perf_counter()
        raw = call_financial_chatflow(pipeline, case["question"], user_id=f"sql-bench-{case['id']}")
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            data = {"content": str(raw)}
        sql = str(data.get("sql") or "")
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "elapsed_ms": elapsed_ms,
                "has_sql": bool(sql.strip()),
                "content_len": len(str(data.get("content") or "")),
                "sql_chars": len(sql),
            }
        )
        print(f"{case['id']} {elapsed_ms:>9.1f}ms sql={bool(sql.strip())} len={len(str(data.get('content') or ''))}")
    return rows


def _accuracy(rows: List[Dict[str, Any]]) -> float:
    return round(sum(1 for r in rows if r.get("correct")) / len(rows), 4) if rows else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="意图路由验证")
    parser.add_argument("--run-supervisor", action="store_true")
    parser.add_argument("--run-sql", action="store_true")
    parser.add_argument("--out", default="训练结果数据/intent_routing_bench.json")
    args = parser.parse_args(argv)

    # 允许分多次跑：保留已有 section，只覆盖本次执行的 section
    out = Path(args.out)
    payload: Dict[str, Any] = {"cases": CASES}
    if out.exists():
        try:
            payload.update(json.loads(out.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            payload = {"cases": CASES}

    keyword_rows = _keyword_predictions()
    payload["keyword"] = {"rows": keyword_rows, "accuracy": _accuracy(keyword_rows)}
    print("keyword accuracy", payload["keyword"]["accuracy"])
    for row in keyword_rows:
        print(f"  {row['id']} label={row['label']:<3s} pred={row['pred']:<5s} route={row['route_type']:<9s} ok={row['correct']}")

    if args.run_supervisor:
        sup_rows = _supervisor_predictions()
        payload["supervisor"] = {"rows": sup_rows, "accuracy": _accuracy(sup_rows)}
        print("supervisor accuracy", payload["supervisor"]["accuracy"])
        for row in sup_rows:
            print(
                f"  {row['id']} label={row['label']:<3s} pred={row['pred']:<5s} "
                f"route={row['route_type']:<9s} ok={row['correct']} ms={row['supervisor_ms']} tasks={row['tasks']}"
            )

    if args.run_sql:
        sql_rows = _run_sql()
        values = [r["elapsed_ms"] for r in sql_rows]
        payload["sql_latency"] = {
            "rows": sql_rows,
            "n": len(values),
            "avg_ms": round(statistics.mean(values), 1) if values else None,
            "p50_ms": round(statistics.median(values), 1) if values else None,
            "has_sql_rate": round(sum(1 for r in sql_rows if r["has_sql"]) / len(sql_rows), 4) if sql_rows else 0.0,
        }
        print("sql latency", json.dumps(payload["sql_latency"], ensure_ascii=False))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
