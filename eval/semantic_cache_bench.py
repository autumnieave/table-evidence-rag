"""语义缓存端到端实测：5 组同义问法，顺序请求 /chat。

用法::

    $env:PYTHONPATH="src"
    python eval/semantic_cache_bench.py --base-url http://127.0.0.1:8010 --clear

输出：

- 控制台 Markdown 表；
- ``--out`` 指定的 JSON 明细（默认 ``训练结果数据/semantic_cache_bench.json``）。

说明：

- 每条请求记录客户端端到端耗时、``X-Cache-Status``、``X-Semantic-Similarity``、``X-Trace-Id``；
- token 从结构化日志 ``训练结果数据/logs/YYYYMMDD.jsonl`` 按 trace_id 读回；
- 语义命中的请求不调用生成模型，token 记为 0；**embedding 调用的 token 未计入**（服务端未上报）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


PAIRS: List[Dict[str, str]] = [
    {
        "id": "P1",
        "base": "华润三九2025年第三季度的主营业务收入是多少",
        "paraphrase": "华润三九2025年三季度营收多少",
    },
    {
        "id": "P2",
        "base": "片仔癀近几年的利润总额变化趋势",
        "paraphrase": "片仔癀近几年利润总额的变化情况",
    },
    {
        "id": "P3",
        "base": "江中药业2025年三季度的净利润是多少",
        "paraphrase": "江中药业2025年第三季度赚了多少净利润",
    },
    {
        "id": "P4",
        "base": "昆药集团2024年的营业收入和同比增长率",
        "paraphrase": "昆药集团2024年营收及同比增速",
    },
    {
        "id": "P5",
        "base": "方盛制药2025年第三季度营收增长的原因",
        "paraphrase": "方盛制药2025年三季度收入上涨的原因是什么",
    },
]


CONTROL: Dict[str, str] = {
    "id": "C1",
    "base": "马应龙2024年净利润",
    "paraphrase": "众生药业2024年营业收入",
}


def _log_path(log_dir: str) -> Path:
    return Path(log_dir) / (time.strftime("%Y%m%d") + ".jsonl")


def _read_trace(log_dir: str, trace_id: Optional[str], timeout: float = 10.0) -> Dict[str, Any]:
    """按 trace_id 从 JSONL 读回日志；日志异步落盘，最多等 timeout 秒。"""
    if not trace_id:
        return {}
    path = _log_path(log_dir)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("trace_id") == trace_id:
                    return item
        time.sleep(0.2)
    return {}


def _post(base_url: str, question: str, user_id: str, timeout: float) -> Dict[str, Any]:
    started = time.perf_counter()
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat",
        json={"question": question, "mode": "rag", "user_id": user_id},
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    return {
        "question": question,
        "elapsed_ms": round(elapsed_ms, 1),
        "cache_status": resp.headers.get("X-Cache-Status", ""),
        "similarity": resp.headers.get("X-Semantic-Similarity"),
        "trace_id": resp.headers.get("X-Trace-Id"),
        "content_len": len(str(data.get("content") or "")),
    }


def _clear_cache(redis_url: str, version: str = "v1") -> int:
    import redis  # noqa: PLC0415

    client = redis.from_url(redis_url, decode_responses=True)
    keys = list(client.scan_iter(match=f"semcache:{version}:*"))
    if keys:
        client.delete(*keys)
    return len(keys)


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    miss = [r for r in rows if r["cache_status"] != "semantic_hit"]
    hit = [r for r in rows if r["cache_status"] == "semantic_hit"]

    def _stats(items: List[Dict[str, Any]], key: str) -> Dict[str, Optional[float]]:
        values = [float(r[key]) for r in items if r.get(key) is not None]
        if not values:
            return {"n": 0, "avg": None, "p50": None}
        return {
            "n": len(values),
            "avg": round(statistics.mean(values), 1),
            "p50": round(statistics.median(values), 1),
        }

    miss_tokens = [r["total_tokens"] for r in miss if r.get("total_tokens") is not None]
    hit_tokens = [r["total_tokens"] for r in hit if r.get("total_tokens") is not None]
    return {
        "requests": len(rows),
        "hits": len(hit),
        "misses": len(miss),
        "hit_rate": round(len(hit) / len(rows), 4) if rows else 0.0,
        "miss_latency": _stats(miss, "elapsed_ms"),
        "hit_latency": _stats(hit, "elapsed_ms"),
        "miss_tokens_avg": round(statistics.mean(miss_tokens), 1) if miss_tokens else None,
        "hit_tokens_avg": round(statistics.mean(hit_tokens), 1) if hit_tokens else 0.0,
        "miss_tokens_total": sum(miss_tokens) if miss_tokens else 0,
        "hit_tokens_total": sum(hit_tokens) if hit_tokens else 0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="语义缓存端到端实测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--user-id", default="semcache-bench")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--log-dir", default="训练结果数据/logs")
    parser.add_argument("--out", default="训练结果数据/semantic_cache_bench.json")
    parser.add_argument("--redis-url", default="redis://localhost:6379/1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--clear", action="store_true", help="开跑前清空该版本语义缓存")
    parser.add_argument("--control", action="store_true", help="额外跑 1 组不相关对照")
    args = parser.parse_args(argv)

    if args.clear:
        removed = _clear_cache(args.redis_url, args.version)
        print(f"[clear] removed={removed}")

    rows: List[Dict[str, Any]] = []
    for pair in PAIRS:
        for kind in ("base", "paraphrase"):
            row = _post(args.base_url, pair[kind], args.user_id, args.timeout)
            row.update({"pair": pair["id"], "kind": kind})
            trace = _read_trace(args.log_dir, row["trace_id"])
            row["total_tokens"] = trace.get("total_tokens")
            row["prompt_tokens"] = trace.get("prompt_tokens")
            row["completion_tokens"] = trace.get("completion_tokens")
            rows.append(row)
            print(
                f"{pair['id']}-{kind:10s} {row['elapsed_ms']:>9.1f}ms "
                f"cache={row['cache_status']:<13s} sim={row['similarity']} "
                f"tokens={row['total_tokens']} len={row['content_len']}"
            )

    if args.control:
        for kind in ("base", "paraphrase"):
            row = _post(args.base_url, CONTROL[kind], args.user_id, args.timeout)
            row.update({"pair": CONTROL["id"], "kind": kind})
            trace = _read_trace(args.log_dir, row["trace_id"])
            row["total_tokens"] = trace.get("total_tokens")
            row["prompt_tokens"] = trace.get("prompt_tokens")
            row["completion_tokens"] = trace.get("completion_tokens")
            rows.append(row)
            print(
                f"{CONTROL['id']}-{kind:10s} {row['elapsed_ms']:>9.1f}ms "
                f"cache={row['cache_status']:<13s} sim={row['similarity']} "
                f"tokens={row['total_tokens']} len={row['content_len']}"
            )

    summary = _summary(rows)
    payload = {"pairs": PAIRS, "control": CONTROL if args.control else None, "rows": rows, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
