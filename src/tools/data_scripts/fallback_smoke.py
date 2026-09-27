# -*- coding: utf-8 -*-
"""B-18 兜底话术 10 题冒烟：对 golden v1 财务子问题 + 边界构造题跑真实原生财务链路
（tools.native_financial.native_financial_query），统计 refuse/suggest/human 三类兜底分支命中分布。

与 B-17 contract_smoke 同款成本闸门（10 题真实小调用）；事件经 log_fallback →
utils.output_contracts.ContractStats.record_fallback 落盘 训练结果数据/output_contract_stats.jsonl。

用法::

    python tools/data_scripts/fallback_smoke.py [--limit 10]

输出::
    训练结果数据/fallback_smoke_summary.json
    事件明细追加: 训练结果数据/output_contract_stats.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]  # 仓库根（数据路径）
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ["QUERY_CACHE_ENABLED"] = "false"

from config.rag_config import RAGConfig  # noqa: E402
from llm import LLMGenerator  # noqa: E402

GOLDEN_SNAPSHOT = REPO_ROOT / "database" / "golden" / "v1_2026-08-22.json"
OUT_SUMMARY = REPO_ROOT / "训练结果数据" / "fallback_smoke_summary.json"

#: golden 正常题（历史上 SQL 编译通过的样本，编号 + 子问题下标）
NORMAL_SAMPLE: List[Tuple[str, int]] = [
    ("B2005", 0), ("B2012", 0), ("B2006", 0), ("B2010", 0), ("B2036", 0), ("B2074", 0),
]

#: 边界构造题（期望命中 refuse 兜底分支）
BOUNDARY_QUESTIONS: List[Tuple[str, str]] = [
    ("year_2026_q1", "片仔癀2026年一季度的营业收入是多少？"),
    ("year_2026_annual", "白云山2026年的全年净利润是多少？"),
    ("company_not_in_db", "特斯拉2025年三季度的净利润是多少？"),
    ("metric_not_in_whitelist", "片仔癀2025年三季度的市盈率是多少？"),
]


class _MiniRag:
    """运行桩：只暴露 native_financial_query 需要的最小属性，query_cache 置空禁用缓存。"""

    def __init__(self, config: RAGConfig, generator: LLMGenerator) -> None:
        self.config = config
        self.llm_generator = generator
        self.query_cache = None


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _load_golden() -> Dict[str, Dict[str, Any]]:
    data = json.loads(io.open(GOLDEN_SNAPSHOT, encoding="utf-8").read())
    return {it["编号"]: it for it in data["items"]}


def _tail_jsonl(path: Path, kind: str = "fallback", limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for ln in lines[-limit:]:
        try:
            obj = json.loads(ln)
        except ValueError:
            continue
        if obj.get("kind") == kind:
            out.append(obj)
    return out


def _main() -> int:
    _utf8()
    parser = argparse.ArgumentParser(description="B-18 兜底话术 10 题冒烟")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    config = RAGConfig()
    generator = LLMGenerator(config)
    stub = _MiniRag(config, generator)
    golden = _load_golden()

    from tools.native_financial import native_financial_query

    questions: List[Tuple[str, str]] = []
    for code, idx in NORMAL_SAMPLE:
        subs = golden[code].get("子问题") or []
        questions.append((f"{code}-Q{idx + 1}", subs[idx] if idx < len(subs) else ""))
    for tag, q in BOUNDARY_QUESTIONS:
        questions.append((tag, q))

    questions = questions[: args.limit]
    print(f"B-18 冒烟：{len(questions)} 题真实原生财务链路（QUERY_CACHE_ENABLED=false）", flush=True)

    details: List[Dict[str, Any]] = []
    for code, q in questions:
        try:
            raw = native_financial_query(stub, q)
            obj = json.loads(raw)
            content = obj.get("content", "")
        except Exception as exc:  # noqa: BLE001
            details.append({"code": code, "question": q, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{code}  ERROR {type(exc).__name__}", flush=True)
            continue
        snippet = content.replace("\n", " ")[:90]
        details.append({"code": code, "question": q, "content_prefix": snippet, "has_sql": bool(obj.get("sql"))})
        print(f"{code}  {snippet}", flush=True)

    from utils.output_contracts import get_stats

    fallback = get_stats().fallback_summary()
    events = _tail_jsonl(REPO_ROOT / "训练结果数据" / "output_contract_stats.jsonl")
    summary = {
        "sample_size": len(details),
        "per_question": details,
        "fallback_distribution": fallback,
        "fallback_event_tail": events[-len(details):],
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
