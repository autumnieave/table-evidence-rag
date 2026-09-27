# -*- coding: utf-8 -*-
"""B-17 输出契约 10 题冒烟：对 golden v1 财务子问题跑真实指标标准化（B-12 第 1 步），
复用运行链路内嵌的契约事件（utils.output_contracts.ContractStats），输出各 kind 格式错误率。

只跑「指标标准化」真实 LLM 小调用（10 次，成本闸门），supervisor/aggregator/SQL 三类契约
以单测（tests/test_output_contracts.py 34 例）离线覆盖，事件落盘随 Agent 调用自动累计。

用法::

    python tools/data_scripts/contract_smoke.py [--limit 10]

输出::

    训练结果数据/contract_smoke_summary.json
    事件明细追加: 训练结果数据/output_contract_stats.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]  # 仓库根（数据路径）
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ["QUERY_CACHE_ENABLED"] = "false"

from config.rag_config import RAGConfig  # noqa: E402
from llm import LLMGenerator  # noqa: E402

GOLDEN_SNAPSHOT = REPO_ROOT / "database" / "golden" / "v1_2026-08-22.json"
OUT_SUMMARY = REPO_ROOT / "训练结果数据" / "contract_smoke_summary.json"

#: 与 B-16 对照实验同口径的 10 题样本（编号 + 子问题下标）
SAMPLE = [
    ("B2001", 0), ("B2005", 0), ("B2003", 0), ("B2006", 0), ("B2008", 0),
    ("B2010", 0), ("B2010", 1), ("B2012", 0), ("B2036", 0), ("B2074", 0),
]


class _MiniRag:
    def __init__(self, config: RAGConfig, generator: LLMGenerator) -> None:
        self.config = config
        self.llm_generator = generator


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _load_golden() -> Dict[str, Dict[str, Any]]:
    data = json.loads(GOLDEN_SNAPSHOT.read_text(encoding="utf-8"))
    return {it["编号"]: it for it in data["items"]}


def _main() -> int:
    _utf8()
    parser = argparse.ArgumentParser(description="B-17 输出契约 10 题冒烟")
    parser.add_argument("--limit", type=int, default=len(SAMPLE))
    args = parser.parse_args()

    config = RAGConfig()
    generator = LLMGenerator(config)
    stub = _MiniRag(config, generator)
    golden = _load_golden()

    from tools.native_financial import _standardize_metrics
    from utils.output_contracts import get_stats

    sample = SAMPLE[: args.limit]
    print(f"B-17 冒烟：{len(sample)} 题真实指标标准化（QUERY_CACHE_ENABLED=false）", flush=True)
    details = []
    for code, idx in sample:
        item = golden[code]
        subs = item.get("子问题") or []
        q = subs[idx] if idx < len(subs) else ""
        plan = _standardize_metrics(stub, q)
        details.append({"code": f"{code}-Q{idx + 1}", "metric_plan_ok": plan is not None})
        print(f"{code}-Q{idx + 1}  plan={'OK' if plan is not None else 'FAIL'}", flush=True)
    stats = get_stats().summary()
    summary = {
        "sample_size": len(sample),
        "per_question": details,
        "contract_stats": stats,
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "note": "真实冒烟仅覆盖 metric_plan（指标标准化）；supervisor_tasks/aggregate_result/sql_output 由 34 例单测离线覆盖 + 运行链路事件落盘累计",
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
