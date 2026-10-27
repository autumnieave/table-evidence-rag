"""Memory 会话记忆 R@5 对照。

对照两种检索输入：

- ``no_memory``：只把本轮追问送给检索器，例如“它的净利润呢”；
- ``memory``：把上一轮问题拼进检索 query，显式注入实体/指标/期间，
  例如“云南白药2025年三季度营业收入是多少？ 它的净利润呢”。

R@5 的严格命中定义为：top-5 中至少一条同时包含期望实体与期望指标
（若配置了期间，还要求期间命中）。实体/指标均未命中时不算。

真实运行依赖 Qdrant + Embedding；依赖不可用时报告会明确写 ``[SKIP]``，
不会用空结果冒充 0% 召回。

用法::

    .venv\\Scripts\\python.exe -m eval.memory_r5_compare --out docs/评估报告/Memory检索质量对照_20261023.md

    # 只验证查询构造与评分逻辑，不访问外部服务
    .venv\\Scripts\\python.exe -m eval.memory_r5_compare --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "评估报告" / "Memory检索质量对照_20261023.md"


@dataclass(frozen=True)
class FollowUp:
    """一条依赖上一轮的追问。"""

    qid: str
    history: str
    followup: str
    entity: str
    metrics: tuple[str, ...]
    period: Optional[str] = None


FOLLOWUPS: tuple[FollowUp, ...] = (
    FollowUp("M01", "云南白药2025年三季度营业收入是多少？", "它的净利润呢？", "云南白药", ("净利润", "归母净利润"), "2025"),
    FollowUp("M02", "片仔癀2024年营业收入是多少？", "那2024年呢？", "片仔癀", ("营业收入", "营业总收入"), "2024"),
    FollowUp("M03", "同仁堂2025年三季度毛利率是多少？", "研发费用呢？", "同仁堂", ("研发费用", "研发投入"), "2025"),
    FollowUp("M04", "江中药业2024年经营现金流是多少？", "它的ROE呢？", "江中药业", ("ROE", "净资产收益率"), "2024"),
    FollowUp("M05", "华润三九2025年三季度营业收入是多少？", "资产负债率呢？", "华润三九", ("资产负债率",), "2025"),
    FollowUp("M06", "东阿阿胶2024年归母净利润是多少？", "营收增速呢？", "东阿阿胶", ("营业收入", "营业总收入", "同比"), "2024"),
    FollowUp("M07", "桂林三金2025年一季度净利润是多少？", "每股收益呢？", "桂林三金", ("每股收益", "EPS"), "2025"),
    FollowUp("M08", "寿仙谷2024年营业收入是多少？", "它的毛利率呢？", "寿仙谷", ("毛利率",), "2024"),
    FollowUp("M09", "白云山2025年三季度经营现金流是多少？", "净利润呢？", "白云山", ("净利润", "归母净利润"), "2025"),
    FollowUp("M10", "以岭药业2024年研发费用是多少？", "那2024年呢？", "以岭药业", ("研发费用", "研发投入"), "2024"),
)


def build_query(item: FollowUp, with_memory: bool) -> str:
    """构造检索 query；memory 模式显式拼入上一轮问题。"""
    if not with_memory:
        return item.followup
    return f"{item.history} {item.followup}".strip()


def _payload_text(payload: Dict[str, Any]) -> str:
    """把 payload 的可检索字段拼成一个文本，用于实体/指标命中判断。"""
    parts: List[str] = []
    for key in ("content", "title", "file_path", "stock_name", "stock_code", "table_caption"):
        value = payload.get(key)
        if value:
            parts.append(str(value))
    if not parts:
        parts.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(parts)


def score_hits(hits: Sequence[Dict[str, Any]], item: FollowUp, k: int = 5) -> Dict[str, Any]:
    """计算单题 top-k 的实体/指标/期间命中。"""
    top = list(hits[:k])
    entity_hit = False
    metric_hit = False
    period_hit = False if item.period else True
    strict_hit = False
    for hit in top:
        payload = (hit or {}).get("payload") or {}
        text = _payload_text(payload)
        this_entity = item.entity in text
        this_metric = any(metric in text for metric in item.metrics)
        this_period = (item.period in text) if item.period else True
        entity_hit = entity_hit or this_entity
        metric_hit = metric_hit or this_metric
        period_hit = period_hit or this_period
        if this_entity and this_metric and this_period:
            strict_hit = True
    return {
        "qid": item.qid,
        "followup": item.followup,
        "entity": item.entity,
        "metrics": list(item.metrics),
        "period": item.period,
        "top_k": len(top),
        "entity_hit": entity_hit,
        "metric_hit": metric_hit,
        "period_hit": period_hit,
        "strict_hit": strict_hit,
    }


def summarize(rows: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """汇总某个模式的 R@5。"""
    n = len(rows)
    strict = sum(1 for row in rows if row.get("strict_hit"))
    entity = sum(1 for row in rows if row.get("entity_hit"))
    metric = sum(1 for row in rows if row.get("metric_hit"))
    return {
        "mode": mode,
        "n": n,
        "strict_hits": strict,
        "r_at_5": round(strict / n, 6) if n else None,
        "entity_hits": entity,
        "metric_hits": metric,
        "errors": sum(1 for row in rows if row.get("error")),
    }


def _run_mode(mode: str, limit: Optional[int], dry_run: bool) -> List[Dict[str, Any]]:
    """执行一个模式；dry_run 只返回查询，不访问检索器。"""
    items = list(FOLLOWUPS[:limit] if limit else FOLLOWUPS)
    if dry_run:
        return [
            {
                "qid": item.qid,
                "mode": mode,
                "query": build_query(item, with_memory=(mode == "memory")),
                "dry_run": True,
            }
            for item in items
        ]

    from core.pipeline import get_pipeline  # 延迟导入，dry-run 不依赖 src

    rows: List[Dict[str, Any]] = []
    try:
        pipeline = get_pipeline()
        retriever = pipeline._get_retriever()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        return [
            {
                "qid": item.qid,
                "mode": mode,
                "query": build_query(item, with_memory=(mode == "memory")),
                "followup": item.followup,
                "entity": item.entity,
                "metrics": list(item.metrics),
                "period": item.period,
                "error": error,
                "strict_hit": False,
                "entity_hit": False,
                "metric_hit": False,
                "period_hit": False,
                "top_k": 0,
            }
            for item in items
        ]
    for item in items:
        query = build_query(item, with_memory=(mode == "memory"))
        try:
            hits = retriever.retrieve(query, top_k=5)
            scored = score_hits(hits, item, k=5)
            scored.update({"mode": mode, "query": query, "error": None})
            rows.append(scored)
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "qid": item.qid,
                "mode": mode,
                "query": query,
                "error": f"{type(exc).__name__}: {exc}",
                "strict_hit": False,
                "entity_hit": False,
                "metric_hit": False,
                "period_hit": False,
                "top_k": 0,
            })
    return rows


def render_markdown(
    *,
    rows_no_memory: Sequence[Dict[str, Any]],
    rows_memory: Sequence[Dict[str, Any]],
    dry_run: bool,
    errors: Sequence[str],
) -> str:
    """渲染对照报告。"""
    combined = list(rows_no_memory) + list(rows_memory)
    all_errors = bool(combined) and all(row.get("error") for row in combined)
    lines = [
        "# Memory 会话记忆 R@5 对照（2026-10-23）",
        "",
        "## 结论",
        "",
    ]
    if dry_run:
        lines.append("- **[DRY-RUN]** 仅验证追问构造与评分逻辑，未访问 Qdrant / Embedding；无真实 R@5。")
    elif errors or all_errors:
        lines.append("- **[SKIP] 真实检索不可用**；错误见下表，未把空结果当 0% 召回。")
    else:
        s_no = summarize(rows_no_memory, "no_memory")
        s_mem = summarize(rows_memory, "memory")
        delta = (s_mem["r_at_5"] - s_no["r_at_5"]) if s_mem["r_at_5"] is not None and s_no["r_at_5"] is not None else None
        lines.append(f"- 无记忆 R@5：**{s_no['r_at_5']:.2%}**（{s_no['strict_hits']}/{s_no['n']}）。")
        lines.append(f"- 有记忆 R@5：**{s_mem['r_at_5']:.2%}**（{s_mem['strict_hits']}/{s_mem['n']}）。")
        lines.append(f"- 提升：**{delta:+.2%}**（严格命中口径）。")

    lines += [
        "",
        "## 一、口径",
        "",
        "- R@5 严格命中：top-5 中至少一条同时包含期望实体与期望指标；配置期间时还要求期间命中。",
        "- `no_memory`：只送本轮追问；`memory`：把上一轮问题拼进检索 query。",
        "- 失败题计入分母；空结果不当作命中，也不隐藏错误。",
        "",
    ]
    if dry_run or all_errors:
        lines += [
            "## 二、题集与查询构造",
            "",
            "| 题号 | 模式 | 检索 query |",
            "| --- | --- | --- |",
        ]
        for row in combined:
            lines.append(f"| {row.get('qid')} | {row.get('mode')} | {row.get('query') or row.get('followup') or '—'} |")
    else:
        lines += [
            "## 二、逐题结果",
            "",
            "| 题号 | 模式 | 追问 | 实体命中 | 指标命中 | 严格命中 | top-k | 错误 |",
            "| --- | --- | --- | :---: | :---: | :---: | ---: | --- |",
        ]
        for row in combined:
            lines.append(
                f"| {row.get('qid')} | {row.get('mode')} | {row.get('followup') or '—'} | "
                f"{'✅' if row.get('entity_hit') else '❌'} | {'✅' if row.get('metric_hit') else '❌'} | "
                f"{'✅' if row.get('strict_hit') else '❌'} | {row.get('top_k', 0)} | {row.get('error') or '—'} |"
            )

    if errors:
        lines += ["", "## 三、运行错误", ""]
        lines.extend(f"- {err}" for err in errors)
    else:
        lines += ["", "## 三、运行错误", "", "- 无。"]

    lines += [
        "",
        "## 四、边界",
        "",
        "- 这是“检索输入注入”对照，不是完整多轮生成对照；它衡量记忆是否把实体/指标带回检索 query。",
        "- 若 memory 组提升但绝对值仍低，应优先查语料覆盖与实体过滤，而不是继续加长 query。",
        "- 真实运行依赖 Qdrant + Embedding；Qdrant 不可达时报告保持 [SKIP]。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Memory 会话记忆 R@5 对照")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部 10 条）")
    parser.add_argument("--dry-run", action="store_true", help="只验证查询构造，不访问外部服务")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Markdown 输出路径")
    args = parser.parse_args(argv)

    errors: List[str] = []
    rows_no_memory = _run_mode("no_memory", args.limit or None, args.dry_run)
    rows_memory = _run_mode("memory", args.limit or None, args.dry_run)
    for row in list(rows_no_memory) + list(rows_memory):
        if row.get("error"):
            message = f"{row['mode']}: {row['error']}"
            if message not in errors:
                errors.append(message)

    report = render_markdown(
        rows_no_memory=rows_no_memory,
        rows_memory=rows_memory,
        dry_run=args.dry_run,
        errors=errors,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
