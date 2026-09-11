# -*- coding: utf-8 -*-
"""B-39 动态 few-shot 价值证明实验（三组对照：none / static / dynamic）。

与 B-16 的 10 题对照（无区分度）不同，本脚本按「题型」分组做三组对照，
目标是回答：**按题型检索示例（dynamic）是否比固定示例（static）/ 无示例（none）更有价值**。

口径与成本闸门：
- 只测「指标标准化 + SQL 生成 + 三层防线编译」环节（tools.native_financial），不跑 Agent/RAG；
- 每题只做 **1 次**指标标准化（三组共用同一 plan），消除标准化噪声对对照的干扰；
- 三组 = AGENT_FEWSHOT_MODE（none / static / dynamic），除示例注入外其余提示词完全一致；
- QUERY_CACHE_ENABLED=false 强制真实重跑；--retries 控制重试上限（默认 1）；
- 每题每组记录：首次生成成功率 / 最终编译通过率 / SQL 结构正确性 / 耗时（秒）；
- 首次生成成功率取自 `tools.native_financial._gen_stats()`（attempts / first_ok），不额外调用 LLM；
- **按题型分组**用运行时真实 metric_plan 的 `predict_type` 结果，不用字符规则近似。

用法::

    python tools/data_scripts/few_shot_value_experiment.py --limit 3      # 冒烟（3 题 × 3 组 = 9 次生成）
    python tools/data_scripts/few_shot_value_experiment.py                # 全量 20 题 × 3 组 = 60 次生成
    python tools/data_scripts/few_shot_value_experiment.py --modes none,static

输出::

    训练结果数据/few_shot_value_experiment.jsonl          # 逐题逐组明细
    训练结果数据/few_shot_value_experiment_summary.json   # 分组汇总（总表 + 按题型）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 真实重跑：关闭查询缓存（config 实例化前生效）
os.environ["QUERY_CACHE_ENABLED"] = "false"

from config.rag_config import RAGConfig  # noqa: E402
from llm import LLMGenerator  # noqa: E402

GOLDEN_SNAPSHOT = REPO_ROOT / "database" / "golden" / "v1_2026-08-22.json"
OUT_DIR = REPO_ROOT / "训练结果数据"
OUT_JSONL = OUT_DIR / "few_shot_value_experiment.jsonl"
OUT_SUMMARY = OUT_DIR / "few_shot_value_experiment_summary.json"

MODES = ("none", "static", "dynamic")

#: 20 题样本（golden 编号 + 子问题下标 0-based），覆盖三大 SQL 形态：
#: 跨期趋势 / 对比排名（含阈值筛选）/ 多指标绑定。最终分组以运行时 predict_type 为准。
#:
#: 2026-09-11 用户确认的两处替换（避开已知缺陷，避免污染 few-shot 结论）：
#:   B2032-Q1 → B2038-Q1（B2032 涉及 B-30 指标替换缺陷）
#:   B2053-Q1 → B2047-Q1（B2053 涉及 B-33/B-36/B-37 三重缺陷）
SAMPLE: List[Tuple[str, int]] = [
    # —— 跨期趋势（多期时间序列）——
    ("B2003", 0),  # 华润三九近三年主营业务收入（可视化）
    ("B2006", 0),  # 片仔癀近几年利润总额变化趋势
    ("B2010", 1),  # 佐力药业近 3 年收入趋势图
    ("B2055", 0),  # 2022-2025 连续四期增长 + 复合增长率
    ("B2058", 0),  # 行业龙头 2022-2025 营业总收入
    ("B2059", 0),  # 三九/白云山/云南白药/片仔癀 2022-2025 收入 + 同比
    ("B2067", 0),  # 东阿阿胶 2022-2025 收入复合增长率
    # —— 对比排名 / 阈值筛选 ——
    ("B2001", 0),  # 2024 利润 top10 + 同比 + 涨幅最大
    ("B2008", 0),  # 收入超 200 亿的企业（阈值）
    ("B2011", 0),  # 哪些企业是亏钱的（阈值为负）
    ("B2014", 0),  # 各企业上半年销售额对比、谁的增速更快
    ("B2028", 0),  # 老年病相关药品收入占比前五 + ROE
    ("B2038", 0),  # 净利润同比增长率超 100% 的公司（阈值；替换 B2032-Q1：该题涉 B-30 指标替换缺陷）
    ("B2054", 0),  # 货币资金占总资产比例前五 + 投资性现金流
    # —— 多指标绑定 ——
    ("B2047", 0),  # 存货金额前五 + 存货周转率 + 低于行业均值（替换 B2053-Q1：该题涉 B-33/B-36/B-37）
    ("B2056", 0),  # 营业总成本构成占比 + 营业成本占比最低
    ("B2057", 0),  # 未分配利润/净利润 比值 > 5 + 资产负债率
    ("B2068", 0),  # 扣非净利润与净利润差值超 1 亿
    ("B2071", 0),  # ROE 同比下降 5 个百分点（资产负债表 + 利润表）
    ("B2074", 0),  # 未分配利润为负但净利润为正
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


def _make_config(mode: str) -> RAGConfig:
    """按模式构造 config（env 在实例化前设置，default_factory 读取）。"""
    os.environ["AGENT_FEWSHOT_MODE"] = mode
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


def check_sql_structure(
    sql: str, metric_plan: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """规则口径的「SQL 结构正确性」核验（B-39）：只看形态是否契合题目意图，不看数值真值。

    检查项：非空 / 仅 SELECT / JOIN 带 ON 等值键 / rank 类含 ORDER BY + LIMIT /
    跨期类含 report_year 时间标签 / 无 10000 倍单位换算（B-33 回归护栏）/ SELECT 无除法列。

    Returns:
        {"ok": bool, "checks": {检查项: bool}, "failed": [失败检查项]}
    """
    text = (sql or "").strip()
    upper = text.upper()
    plan = metric_plan or {}
    kind = (plan.get("calculation") or {}).get("kind")
    mode = (plan.get("time_grain") or {}).get("mode")

    checks: Dict[str, bool] = {
        "non_empty": bool(text),
        "select_only": upper.startswith("SELECT") if text else False,
    }
    if text and re.search(r"\bJOIN\b", upper):
        checks["join_has_on"] = bool(re.search(r"\bON\b", upper))
        checks["join_equi_keys"] = bool(
            re.search(r"\bON\b[\s\S]{0,240}?stock_code\s*=\s*\w+\.?stock_code", text, re.I)
        )
    if kind == "rank":
        checks["rank_has_order_by"] = "ORDER BY" in upper
        checks["rank_has_limit"] = "LIMIT" in upper
    if kind == "multi_period_history" or mode in ("full_history", "annual_fy", "annual_fy_with_latest_q3"):
        checks["trend_has_year_label"] = "report_year" in text.lower()
    checks["no_10000_scaling"] = not bool(re.search(r"\*\s*10000|10000\s*\*", text))
    if text:
        select_part = text.split(" FROM ", 1)[0]
        checks["no_division_in_select"] = "/" not in select_part

    failed = [k for k, v in checks.items() if not v]
    return {"ok": not failed, "checks": checks, "failed": failed}


def _run_one_group(stub: _MiniRag, question: str, schema: Dict, conn: Any,
                   retries: int, plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """跑一组（一个 mode）的一次 _generate_sql，返回该组的四项指标。"""
    from tools.native_financial import _gen_stats, _generate_sql

    t0 = time.perf_counter()
    sql, errors = _generate_sql(stub, question, schema, conn, retries=retries, metric_plan=plan)
    cost = round(time.perf_counter() - t0, 2)
    stats = _gen_stats().last()
    raw_sql = sql or str(stats.get("sql") or "")
    structure = check_sql_structure(raw_sql, plan)
    return {
        "compile_ok": bool(sql),
        "first_ok": bool(stats.get("first_ok")) if stats else False,
        "attempts": int(stats.get("attempts") or 0),
        "injected": int(stats.get("injected") or 0),
        "mode_reported": stats.get("mode"),
        "structure_ok": structure["ok"],
        "structure_failed": structure["failed"],
        "cost_s": cost,
        "error": ("；".join(errors[:2])[:300] if errors else ""),
        "sql_head": raw_sql.replace("\n", " ")[:160],
    }


def _summarize(rows: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    """汇总：总表（按 mode）+ 按题型（predicted_type × mode）分组。"""

    def _agg(items: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
        sel = [r for r in items if r["mode"] == mode]
        n = len(sel)
        if not n:
            return {"questions": 0}
        return {
            "questions": n,
            "first_pass": sum(1 for r in sel if r["first_ok"]),
            "first_pass_rate": round(sum(1 for r in sel if r["first_ok"]) / n, 4),
            "compile_pass": sum(1 for r in sel if r["compile_ok"]),
            "compile_rate": round(sum(1 for r in sel if r["compile_ok"]) / n, 4),
            "structure_pass": sum(1 for r in sel if r["structure_ok"]),
            "structure_rate": round(sum(1 for r in sel if r["structure_ok"]) / n, 4),
            "avg_cost_s": round(sum(r["cost_s"] for r in sel) / n, 2),
            "avg_injected": round(sum(r["injected"] for r in sel) / n, 2),
        }

    types = sorted({r["predicted_type"] for r in rows})
    return {
        "overall": {m: _agg(rows, m) for m in modes},
        "by_type": {
            t: {m: _agg([r for r in rows if r["predicted_type"] == t], m) for m in modes}
            for t in types
        },
    }


def _main() -> int:
    _utf8()
    parser = argparse.ArgumentParser(description="B-39 动态 few-shot 三组对照（none/static/dynamic）")
    parser.add_argument("--limit", type=int, default=len(SAMPLE), help="只跑前 N 题")
    parser.add_argument("--retries", type=int, default=1, help="SQL 生成失败重试次数（默认 1）")
    parser.add_argument("--modes", type=str, default=",".join(MODES), help="要跑的组，逗号分隔")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            print(f"[致命] 未知模式 {m!r}，可选 {MODES}", flush=True)
            return 2

    sample = SAMPLE[: args.limit]
    golden = _load_golden()
    base_cfg = _make_config("none")
    generator = LLMGenerator(base_cfg)
    stubs = {m: _MiniRag(_make_config(m), generator) for m in modes}

    try:
        from agents.planner import _load_schema

        schema, conn = _load_schema(stubs[modes[0]].config)
    except Exception as exc:  # noqa: BLE001
        print(f"[致命] MySQL schema/连接加载失败，无法做编译终审: {exc}", flush=True)
        return 2
    if conn is None or schema is None:
        print("[致命] MySQL schema/连接不可用（需本地 MySQL financial_database），退出", flush=True)
        return 2

    from tools.native_financial import _standardize_metrics
    from utils.few_shot_retriever import predict_type

    print(
        f"样本 {len(sample)} 题 × {len(modes)} 组 = {len(sample) * len(modes)} 次生成，"
        f"modes={modes}，retries={args.retries}，QUERY_CACHE_ENABLED=false",
        flush=True,
    )
    rows: List[Dict[str, Any]] = []
    try:
        for code, idx in sample:
            row_id, question = _pick_row(golden[code], idx)
            plan = _standardize_metrics(stubs[modes[0]], question)
            ptype = predict_type(question, plan)
            for m in modes:
                res = _run_one_group(stubs[m], question, schema, conn, args.retries, plan)
                row = {
                    "code": row_id,
                    "question": question,
                    "predicted_type": ptype,
                    "metric_plan_ok": plan is not None,
                    "mode": m,
                    **res,
                }
                rows.append(row)
                print(
                    f"{row_id} [{ptype}] mode={m:<7} 首次={'PASS' if res['first_ok'] else 'FAIL'} "
                    f"编译={'PASS' if res['compile_ok'] else 'FAIL'} "
                    f"结构={'PASS' if res['structure_ok'] else 'FAIL'} "
                    f"注入={res['injected']} {res['cost_s']}s",
                    flush=True,
                )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    summary = {
        "sample_size": len(sample),
        "modes": modes,
        "retries": args.retries,
        "cache_enabled": False,
        "mode": "sql_generation_layer_three_group",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **_summarize(rows, modes),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2), flush=True)
    print(f"\n明细: {OUT_JSONL}\n汇总: {OUT_SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
