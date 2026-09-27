# -*- coding: utf-8 -*-
"""B-16 动态 few-shot vs 静态 few-shot 对照实验（原生 SQL 生成环节，10 题冒烟收口）。

只测「指标标准化 + SQL 生成 + 三层防线编译」环节（tools.native_financial），
不跑 Agent/RAG/研报汇总链路，避免把编排与检索噪声混入 few-shot 对照。

口径与成本闸门（对齐任务 B-16「先规则后向量、golden 子集对照、开关可回退」）：
- 10 道财务子问题从 golden v1 抽取，覆盖 5 题型（single_period / cross_period_trend /
  ranking_compare / binding_relation / industry_mean）；
- 每题仅调用一次指标标准化（AGENT_METRIC_STANDARDIZE=true，B-12 第 1 步），
  结果在静态/动态两模式间复用，避免标准化噪声影响对照；
- 静态模式 AGENT_DYNAMIC_FEWSHOT=false（现状基线，走 SQL_GEN_SYSTEM_PROMPT 静态示例）；
  动态模式 =true（命中 prompts/examples 示例库则注入，未命中回退静态提示词）；
- QUERY_CACHE_ENABLED=false 强制真实重跑；retries=1（最多 2 次生成尝试）控制 LLM 成本；
- 编译通过 = _generate_sql 三层防线全过（静态校验 + MySQL 编译终审，MAX_EXECUTION_TIME=15s）；
- 题型命中率 = 动态模式下 FewShotRetriever 至少命中 1 条示例的问题占比（抽样统计归档）。

用法::

    python tools/data_scripts/few_shot_dynamic_compare.py            # 默认 10 题双模式
    python tools/data_scripts/few_shot_dynamic_compare.py --limit 3  # 冒烟

输出::

    训练结果数据/few_shot_dynamic_compare.jsonl   # 逐题明细
    训练结果数据/few_shot_dynamic_compare_summary.json
    汇总报告人工落 docs/评估报告/few_shot动态检索对照.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]  # 仓库根（数据路径）
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# 真实重跑：关闭查询缓存（config 实例化前生效）
os.environ["QUERY_CACHE_ENABLED"] = "false"

from config.rag_config import RAGConfig  # noqa: E402
from llm import LLMGenerator  # noqa: E402

GOLDEN_SNAPSHOT = REPO_ROOT / "database" / "golden" / "v1_2026-08-22.json"
OUT_JSONL = REPO_ROOT / "训练结果数据" / "few_shot_dynamic_compare.jsonl"
OUT_SUMMARY = REPO_ROOT / "训练结果数据" / "few_shot_dynamic_compare_summary.json"

#: 冒烟样本：golden 编号 + 子问题下标（0-based），覆盖 5 题型
SAMPLE = [
    ("B2001", 0),  # 多意图：top10 利润 + 同比 + 涨幅最大（rank/binding + trend）
    ("B2005", 0),  # 归因：片仔癀利润总额（single_period）
    ("B2003", 0),  # 归因：华润三九近三年主营业务收入可视化（cross_period_trend）
    ("B2006", 0),  # 归因：片仔癀近几年利润总额变化趋势（cross_period_trend）
    ("B2008", 0),  # 归因：收入超 200 亿企业（ranking_compare / 阈值筛选）
    ("B2010", 0),  # 归因：佐力药业 2025 收入情况（single_period）
    ("B2010", 1),  # 归因：近 3 年收入趋势图（cross_period_trend）
    ("B2012", 0),  # 归因：太极集团 2024 研发费用（single_period）
    ("B2036", 0),  # 数据校验：66 家公司资产负债率行业均值（industry_mean）
    ("B2074", 0),  # 归因：广誉远未分配利润为负（多指标 binding + 归因）
]


def _utf8() -> None:
    """Windows 控制台统一 UTF-8 输出。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class _MiniRag:
    """RAGPipeline 轻量替身：只带 config + llm_generator，供 native_financial 私有函数调用。"""

    def __init__(self, config: RAGConfig, generator: LLMGenerator) -> None:
        self.config = config
        self.llm_generator = generator


def _make_config(enable_fewshot: bool) -> RAGConfig:
    """按模式构造 config（env 在实例化前设置，default_factory 读取）。"""
    os.environ["AGENT_DYNAMIC_FEWSHOT"] = "true" if enable_fewshot else "false"
    return RAGConfig()


def _load_golden() -> Dict[str, Dict[str, Any]]:
    """golden v1 -> {编号: item}。"""
    data = json.loads(GOLDEN_SNAPSHOT.read_text(encoding="utf-8"))
    return {it["编号"]: it for it in data["items"]}


def _pick_row(it: Dict[str, Any], idx: int) -> Tuple[str, str]:
    """返回 (编号-Q序号, 问题文本)。"""
    subs = it.get("子问题") or []
    if idx >= len(subs):
        raise ValueError(f"{it.get('编号')} 子问题下标越界: idx={idx}, total={len(subs)}")
    return f"{it.get('编号')}-Q{idx + 1}", subs[idx]


def _run_sql_gen(stub: _MiniRag, question: str, schema: Dict, conn: Any,
                 retries: int, plan: Optional[Dict[str, Any]]) -> Tuple[bool, str, float]:
    """跑一次 _generate_sql，返回 (编译通过?, 错误摘要, 耗时秒)。"""
    from tools.native_financial import _generate_sql

    t0 = time.perf_counter()
    sql, errors = _generate_sql(stub, question, schema, conn, retries=retries, metric_plan=plan)
    cost = round(time.perf_counter() - t0, 2)
    if sql:
        return True, "", cost
    return False, "；".join(errors[:2])[:300], cost


def _probe_retrieval(question: str, plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """独立探测：该题在动态模式下是否命中示例库（与生产调用同一签名 question + metric_plan）。"""
    from utils.few_shot_retriever import FewShotRetriever, predict_type

    if not plan:
        return {"predicted_type": predict_type(question, None), "matched": 0, "reason": "no_metric_plan"}
    retriever = FewShotRetriever(k=2)
    examples = retriever.retrieve(question, plan)
    return {"predicted_type": predict_type(question, plan), "matched": len(examples), "reason": "ok"}


def _main() -> int:
    _utf8()
    parser = argparse.ArgumentParser(description="B-16 动态 few-shot 对照（10 题冒烟收口）")
    parser.add_argument("--limit", type=int, default=len(SAMPLE), help="只跑前 N 题")
    parser.add_argument("--retries", type=int, default=1, help="SQL 生成失败重试次数（默认 1）")
    args = parser.parse_args()

    sample = SAMPLE[: args.limit]
    golden = _load_golden()
    # 生成器与 config 解耦（native 调用取 stub.config.LLM_MODEL），共用 1 个 OpenAI client
    base_cfg = _make_config(enable_fewshot=False)
    generator = LLMGenerator(base_cfg)
    static_stub = _MiniRag(_make_config(enable_fewshot=False), generator)
    dynamic_stub = _MiniRag(_make_config(enable_fewshot=True), generator)

    try:
        from agents.planner import _load_schema

        schema, conn = _load_schema(static_stub.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[致命] MySQL schema/连接加载失败，无法做编译终审: {exc}", flush=True)
        return 2
    if conn is None or schema is None:
        print("[致命] MySQL schema/连接不可用（需本地 MySQL financial_database），退出", flush=True)
        return 2

    from tools.native_financial import _standardize_metrics

    print(f"样本 {len(sample)} 题 × 双模式，retries={args.retries}，QUERY_CACHE_ENABLED=false", flush=True)
    rows: List[Dict[str, Any]] = []
    try:
        for code, idx in sample:
            row_id, question = _pick_row(golden[code], idx)
            plan = _standardize_metrics(static_stub, question)
            probe = _probe_retrieval(question, plan)
            static_ok, static_err, static_cost = _run_sql_gen(
                static_stub, question, schema, conn, args.retries, plan
            )
            dynamic_ok, dynamic_err, dynamic_cost = _run_sql_gen(
                dynamic_stub, question, schema, conn, args.retries, plan
            )
            row = {
                "code": row_id,
                "question": question,
                "metric_plan_ok": plan is not None,
                **probe,
                "static_ok": static_ok,
                "static_error": static_err,
                "static_cost_s": static_cost,
                "dynamic_ok": dynamic_ok,
                "dynamic_error": dynamic_err,
                "dynamic_cost_s": dynamic_cost,
            }
            rows.append(row)
            print(
                f"{row_id} 命中={probe['matched']} 类型={probe['predicted_type']} | "
                f"静态={'PASS' if static_ok else 'FAIL'} 动态={'PASS' if dynamic_ok else 'FAIL'} "
                f"| {static_cost}s/{dynamic_cost}s",
                flush=True,
            )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    evaluated = len(rows)
    static_pass = sum(1 for r in rows if r["static_ok"])
    dynamic_pass = sum(1 for r in rows if r["dynamic_ok"])
    planned = [r for r in rows if r["metric_plan_ok"]]
    hit_q = [r for r in planned if r["matched"] > 0]
    summary = {
        "sample_size": evaluated,
        "static_compile_pass": static_pass,
        "static_compile_rate": round(static_pass / evaluated, 4) if evaluated else None,
        "dynamic_compile_pass": dynamic_pass,
        "dynamic_compile_rate": round(dynamic_pass / evaluated, 4) if evaluated else None,
        "metric_plan_ok": len(planned),
        "retrieval_hit_questions": len(hit_q),
        "retrieval_hit_rate": round(len(hit_q) / len(planned), 4) if planned else None,
        "retries": args.retries,
        "cache_enabled": False,
        "mode": "sql_generation_layer_smoke",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
