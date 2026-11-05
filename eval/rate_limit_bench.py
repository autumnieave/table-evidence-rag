# -*- coding: utf-8 -*-
"""Redis 分布式滑动窗口限流 50 并发压测。

用法::

    $env:PYTHONPATH="src"
    .\.venv\Scripts\python.exe eval\rate_limit_bench.py --concurrency 50 --limit 20

设计边界：

- **不调用真实 LLM**：把 ``app.api.pipeline`` 替换为即时返回的桩对象，
  只压测限流链路；限流判断、Redis Lua、429 响应头、滑动窗口共享状态
  全部走真实实现。
- 除 HTTP 并发外，另建一个独立的限流器实例（新 Redis 连接、同一 prefix），
  验证它能看到刚才那批请求占用的窗口——这就是“多节点共享同一窗口”的
  分布式语义，而不是进程内计数。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redis 限流 50 并发压测")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--redis-url", default=os.getenv("API_RATE_LIMIT_REDIS_URL", "redis://localhost:6379/2"))
    parser.add_argument(
        "--out",
        default=str(ROOT / "训练结果数据" / "stage5_measure" / "rate_limit_bench.json"),
    )
    return parser.parse_args()


ARGS = _parse_args()

# 环境变量必须在导入 app.api（其内部会构建全局 config 单例）之前设置。
os.environ["USER_RATE_LIMIT_ENABLED"] = "true"
os.environ["USER_RATE_LIMIT_PER_MINUTE"] = str(ARGS.limit)
os.environ["API_RATE_LIMIT_BACKEND"] = "redis"
os.environ["API_RATE_LIMIT_REDIS_URL"] = ARGS.redis_url
os.environ["API_AUTH_ENABLED"] = "false"
os.environ["QUERY_CACHE_ENABLED"] = "false"
os.environ["SEMANTIC_CACHE_ENABLED"] = "false"


class _StubPipeline:
    """最小桩：只让 /chat 快速返回，避免触发真实检索与生成。"""

    query_cache = None

    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        time.sleep(0.01)
        return {"content": "bench-stub", "references": [], "image": []}

    def agent_query(self, question: str, user_id: str | None = None) -> Dict[str, Any]:
        return {"content": "bench-stub", "references": [], "image": []}


def _start_server(port: int):
    import uvicorn

    import app.api as api_module

    api_module.pipeline = _StubPipeline()
    config = uvicorn.Config(
        api_module.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    if not server.started:
        raise RuntimeError("uvicorn 未能在 60s 内启动")
    return api_module, server


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    import requests

    api_module, server = _start_server(ARGS.port)
    base_url = f"http://127.0.0.1:{ARGS.port}"
    user_id = f"bench-{uuid.uuid4().hex[:8]}"
    print(f"[bench] user_id={user_id} limit={ARGS.limit}/min concurrency={ARGS.concurrency}")

    # 预热：另一个 user_id，避免占用被测窗口
    try:
        requests.post(
            f"{base_url}/chat",
            json={"question": "warmup", "user_id": "bench-warmup"},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] warmup failed: {exc}")

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _one(i: int) -> None:
        started = time.perf_counter()
        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={"question": f"限流压测 {i}", "user_id": user_id, "mode": "rag"},
                timeout=60,
            )
            record = {
                "status": resp.status_code,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "retry_after": resp.headers.get("Retry-After"),
                "detail": (resp.json() or {}).get("detail") if resp.status_code != 200 else None,
            }
        except Exception as exc:  # noqa: BLE001
            record = {
                "status": -1,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "retry_after": None,
                "detail": str(exc),
            }
        with lock:
            results.append(record)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=ARGS.concurrency) as pool:
        list(pool.map(_one, range(ARGS.concurrency)))
    wall_ms = round((time.perf_counter() - started) * 1000, 1)

    ok = [r for r in results if r["status"] == 200]
    limited = [r for r in results if r["status"] == 429]
    other = [r for r in results if r["status"] not in (200, 429)]
    ok_lat = [r["elapsed_ms"] for r in ok]
    limited_lat = [r["elapsed_ms"] for r in limited]
    retry_values = sorted({int(r["retry_after"]) for r in limited if r["retry_after"]})

    # 分布式语义：独立实例（新连接、同一 prefix）应立刻看到窗口已被占满
    import redis

    from utils.api_security import create_rate_limiter

    other_instance = create_rate_limiter(
        limit=ARGS.limit,
        backend="redis",
        redis_url=ARGS.redis_url,
        prefix="rl:user",
    )
    cross_allowed, cross_retry = other_instance.allow(user_id)

    client = redis.Redis.from_url(ARGS.redis_url, decode_responses=True)
    window_key = f"rl:user:{user_id}"
    window_size = client.zcard(window_key)
    ttl_ms = client.pttl(window_key)

    payload = {
        "user_id": user_id,
        "limit_per_minute": ARGS.limit,
        "concurrency": ARGS.concurrency,
        "wall_ms": wall_ms,
        "status_counts": {
            "200": len(ok),
            "429": len(limited),
            "other": len(other),
        },
        "retry_after_seconds": retry_values,
        "latency_ms": {
            "success_p50": _percentile(ok_lat, 50),
            "success_p95": _percentile(ok_lat, 95),
            "limited_p50": _percentile(limited_lat, 50),
            "limited_p95": _percentile(limited_lat, 95),
            "mean_ok": round(statistics.mean(ok_lat), 1) if ok_lat else 0.0,
            "mean_limited": round(statistics.mean(limited_lat), 1) if limited_lat else 0.0,
        },
        "redis": {
            "key": window_key,
            "zcard": window_size,
            "pttl_ms": ttl_ms,
        },
        "cross_instance": {
            "instance_backend": getattr(other_instance, "backend", "unknown"),
            "allowed": cross_allowed,
            "retry_after_seconds": cross_retry,
        },
        "other_errors": other[:5],
    }

    out_path = Path(ARGS.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| 指标 | 值 |")
    print("| --- | --- |")
    print(f"| 并发数 | {ARGS.concurrency} |")
    print(f"| 限额（user_id/分钟） | {ARGS.limit} |")
    print(f"| 200 通过 | {len(ok)} |")
    print(f"| 429 拦截 | {len(limited)} |")
    print(f"| 其他状态 | {len(other)} |")
    print(f"| 总墙钟耗时 | {wall_ms} ms |")
    print(f"| 成功请求 P50 / P95 | {payload['latency_ms']['success_p50']} / {payload['latency_ms']['success_p95']} ms |")
    print(f"| 429 请求 P50 / P95 | {payload['latency_ms']['limited_p50']} / {payload['latency_ms']['limited_p95']} ms |")
    print(f"| Retry-After 取值 | {retry_values} |")
    print(f"| Redis 窗口 zcard | {window_size} |")
    print(f"| 独立实例再请求 | allowed={cross_allowed} retry={cross_retry}s |")
    print(f"\n[bench] 明细写入 {out_path}")

    server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
