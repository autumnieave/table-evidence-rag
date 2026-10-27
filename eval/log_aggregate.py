"""聚合结构化请求日志：P50/P95/P99 延迟、token、成本与阶段占比。

用法::

    .venv\\Scripts\\python.exe -m eval.log_aggregate --date 20261023 \\
        --out docs/评估报告/结构化日志与成本_20261023.md

只依赖标准库；日志缺失时输出“无记录”，不编造数字。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "训练结果数据" / "logs"
LATENCY_FIELDS = ("total_ms", "retrieval_ms", "rerank_ms", "generate_ms", "query_total_ms", "agent_total_ms")
STAGE_FIELDS = ("retrieval_ms", "rerank_ms", "generate_ms")


def load_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """读取一个或多个 JSONL；坏行跳过并计数。"""
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
    return records


def filter_records(records: Iterable[Dict[str, Any]], path_prefix: str) -> List[Dict[str, Any]]:
    """Keep only records whose path starts with ``path_prefix``."""
    items = list(records)
    if not path_prefix:
        return items
    return [record for record in items if str(record.get("path") or "").startswith(path_prefix)]


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """线性插值分位数（p 取 0~1）；空序列返回 None。"""
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 3)
    pos = (len(clean) - 1) * float(p)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return round(clean[low], 3)
    weight = pos - low
    return round(clean[low] * (1.0 - weight) + clean[high] * weight, 3)


def _numbers(records: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _sum_optional(records: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = _numbers(records, key)
    if not values:
        return None
    return round(sum(values), 6)


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把日志记录聚合成可直接写报告的字典。"""
    n = len(records)
    statuses = Counter(str(record.get("status") or "unknown") for record in records)
    models = sorted({str(record.get("model_name")) for record in records if record.get("model_name")})

    latency: Dict[str, Dict[str, Optional[float]]] = {}
    for field in LATENCY_FIELDS:
        values = _numbers(records, field)
        latency[field] = {
            "n": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": round(max(values), 3) if values else None,
        }

    stage_totals = {field: sum(_numbers(records, field)) for field in STAGE_FIELDS}
    stage_denominator = sum(stage_totals.values())
    stage_share = {
        field: (round(total / stage_denominator * 100.0, 3) if stage_denominator > 0 else None)
        for field, total in stage_totals.items()
    }

    token_records = [
        record for record in records
        if record.get("total_tokens") is not None or record.get("prompt_tokens") is not None
    ]
    cost_values = [
        float(record["estimated_cost"])
        for record in token_records
        if record.get("estimated_cost") is not None
    ]
    if not token_records:
        cost_status = "no_token_data"
    elif len(cost_values) == len(token_records):
        cost_status = "estimated"
    elif cost_values:
        cost_status = "partial"
    else:
        cost_status = "unit_price_not_registered"

    usage_statuses = Counter(str(record.get("usage_status") or "unknown") for record in records)
    return {
        "n_requests": n,
        "status_counts": dict(statuses),
        "error_rate": round(statuses.get("error", 0) / n, 6) if n else None,
        "models": models,
        "cache_hits": sum(1 for record in records if record.get("cache_hit")),
        "latency_ms": latency,
        "stage_share_pct": stage_share,
        "tokens": {
            "prompt_tokens": _sum_optional(records, "prompt_tokens"),
            "completion_tokens": _sum_optional(records, "completion_tokens"),
            "total_tokens": _sum_optional(records, "total_tokens"),
        },
        "cost": {
            "total_estimated_cost": round(sum(cost_values), 8) if cost_values else None,
            "cost_status": cost_status,
            "n_costed_records": len(cost_values),
        },
        "usage_status_counts": dict(usage_statuses),
        "percentile_method": "linear_interpolation_over_non_null_values",
    }


def render_markdown(summary: Dict[str, Any], source_files: Sequence[Path]) -> str:
    """把聚合结果渲染为中文报告。"""
    lines = [
        "# 结构化日志与成本",
        "",
        f"- 数据源：{', '.join(f'`{p}`' for p in source_files) if source_files else '无'}",
        f"- 请求数：**{summary['n_requests']}**",
        f"- 状态分布：`{summary['status_counts']}`；错误率：**{summary['error_rate'] if summary['error_rate'] is not None else 'N/A'}**",
        f"- 模型：{', '.join(f'`{m}`' for m in summary['models']) if summary['models'] else '未记录'}",
        f"- 缓存命中：**{summary['cache_hits']}**",
        f"- 分位数方法：`{summary['percentile_method']}`（仅对非空值计算，不把缺失当 0）",
        "",
        "## 一、延迟",
        "",
        "| 指标 | n | P50 | P95 | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, item in summary["latency_ms"].items():
        lines.append(
            f"| `{field}` | {item['n']} | {item['p50'] if item['p50'] is not None else '—'} | "
            f"{item['p95'] if item['p95'] is not None else '—'} | {item['p99'] if item['p99'] is not None else '—'} | "
            f"{item['max'] if item['max'] is not None else '—'} |"
        )

    lines += [
        "",
        "## 二、阶段耗时占比",
        "",
        "| 阶段 | 累计耗时占比 |",
        "| --- | ---: |",
    ]
    for field, share in summary["stage_share_pct"].items():
        lines.append(f"| `{field}` | {share if share is not None else '—'}% |")

    tokens = summary["tokens"]
    cost = summary["cost"]
    lines += [
        "",
        "## 三、Token 与成本",
        "",
        f"- Prompt tokens：**{tokens['prompt_tokens'] if tokens['prompt_tokens'] is not None else '未记录'}**",
        f"- Completion tokens：**{tokens['completion_tokens'] if tokens['completion_tokens'] is not None else '未记录'}**",
        f"- Total tokens：**{tokens['total_tokens'] if tokens['total_tokens'] is not None else '未记录'}**",
        f"- 估算总成本：**{cost['total_estimated_cost'] if cost['total_estimated_cost'] is not None else 'unit_price_not_registered'}**",
        f"- 成本状态：`{cost['cost_status']}`（已计价记录 {cost['n_costed_records']} 条）",
        f"- usage 状态分布：`{summary['usage_status_counts']}`",
        "",
        "> 单价未在 `LLM_INPUT_PRICE_PER_1K` / `LLM_OUTPUT_PRICE_PER_1K` 登记时，成本列写 `unit_price_not_registered`，不编造金额。",
    ]
    return "\n".join(lines) + "\n"


def _default_paths(date: str, log_dir: Path) -> List[Path]:
    return [log_dir / f"{date}.jsonl"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="聚合结构化请求日志")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="日志日期 YYYYMMDD")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="日志目录")
    parser.add_argument("--path", action="append", help="显式 JSONL 路径（可重复；给出时忽略 --date）")
    parser.add_argument("--path-prefix", default="", help="只聚合 path 以该前缀开头的记录，如 /chat")
    parser.add_argument("--out", default="", help="Markdown 输出路径")
    parser.add_argument("--json-out", default="", help="JSON 聚合结果输出路径")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.path] if args.path else _default_paths(args.date, Path(args.log_dir))
    records = load_records(paths)
    records = filter_records(records, args.path_prefix)
    summary = aggregate(records)
    markdown = render_markdown(summary, paths)
    print(markdown)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
