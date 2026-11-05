# -*- coding: utf-8 -*-
"""前后端契约联调检查：经 Nginx 前端同源入口验证 SSE / 缓存 / 429。

用法::

    python eval/frontend_contract_check.py --base-url http://localhost:8080

检查项：

1. ``GET /health`` 经 Nginx 可达；
2. ``POST /chat/stream`` 返回 meta/stage/content/done 事件；
3. 同义改写第二次提问命中 Semantic Cache（meta.cache_hit=true, cache_layer=semantic）；
4. user_id 触发限流时返回 HTTP 429 + ``Retry-After``（通过预填 Redis 窗口，避免额外 LLM 调用）。

说明：本脚本只读调用接口，不修改任何评测数据；第 4 项会向 Redis DB 2 写入一个测试用户的窗口。
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Dict, List, Tuple

import requests

DEFAULT_BASE = "http://localhost:8080"
DEFAULT_REDIS = "redis://localhost:6379/2"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="前后端 SSE / 缓存 / 限流契约检查")
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--redis-url", default=DEFAULT_REDIS)
    p.add_argument("--timeout", type=int, default=240)
    return p.parse_args()


def stream_chat(base_url: str, question: str, user_id: str, mode: str = "rag", timeout: int = 240) -> Dict[str, Any]:
    """发一次 SSE 请求并收集事件；返回 status/headers/events/elapsed。"""
    started = time.perf_counter()
    with requests.post(
        f"{base_url}/chat/stream",
        json={"question": question, "user_id": user_id, "mode": mode},
        stream=True,
        timeout=timeout,
    ) as resp:
        events: List[Dict[str, Any]] = []
        if resp.status_code == 200:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                payload = raw[6:]
                if payload == "[DONE]":
                    continue
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    events.append({"type": "unparsed", "raw": payload})
        return {
            "status": resp.status_code,
            "retry_after": resp.headers.get("Retry-After"),
            "events": events,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }


def _meta(result: Dict[str, Any]) -> Dict[str, Any]:
    for event in result.get("events") or []:
        if event.get("type") == "meta":
            return event
    return {}


def _text(result: Dict[str, Any]) -> str:
    return "".join(e.get("text", "") for e in result.get("events") or [] if e.get("type") == "content")


def _seed_window(redis_url: str, user_id: str, count: int) -> None:
    """把测试用户的滑动窗口预填到上限，避免触发真实生成。"""
    import redis

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    key = f"rl:user:{user_id}"
    client.delete(key)
    now_ms = int(time.time() * 1000)
    for i in range(count):
        client.zadd(key, {f"seed-{i}:{now_ms}": now_ms})
    client.pexpire(key, 60_000)


def main() -> int:
    args = _parse_args()
    base = args.base_url.rstrip("/")
    run_id = uuid.uuid4().hex[:6]
    report: Dict[str, Any] = {"base_url": base, "run_id": run_id}

    health = requests.get(f"{base}/health", timeout=10)
    report["health"] = {"status_code": health.status_code, "body": health.json() if health.ok else health.text}
    print(f"[1] GET /health -> {health.status_code} {report['health']['body']}")

    q1 = "片仔癀2024年营业收入是多少？"
    q2 = "片仔癀2024年营收有多少"
    user_cache = f"verify-cache-{run_id}"
    first = stream_chat(base, q1, user_cache, timeout=args.timeout)
    report["first"] = first
    m1 = _meta(first)
    print(f"[2] 首次提问 -> HTTP {first['status']} {first['elapsed_s']}s events={[e.get('type') for e in first['events']][:8]}")
    print(f"    cache_hit={m1.get('cache_hit')} layer={m1.get('cache_layer')} content_len={len(_text(first))}")

    second = stream_chat(base, q2, user_cache, timeout=args.timeout)
    report["second"] = second
    m2 = _meta(second)
    print(f"[3] 同义改写 -> HTTP {second['status']} {second['elapsed_s']}s")
    print(f"    cache_hit={m2.get('cache_hit')} layer={m2.get('cache_layer')} similarity={m2.get('semantic_similarity')}")

    # 429：预填窗口到限额（默认 3），再发一次
    user_limit = f"verify-limit-{run_id}"
    _seed_window(args.redis_url, user_limit, 3)
    limited = stream_chat(base, "限流验证", user_limit, timeout=30)
    report["limited"] = limited
    print(f"[4] 预填 3 次窗口后再请求 -> HTTP {limited['status']} Retry-After={limited['retry_after']}")

    checks = {
        "health_ok": health.status_code == 200,
        "sse_has_meta_content_done": (
            first["status"] == 200
            and {"meta", "content", "done"}.issubset({e.get("type") for e in first["events"]})
        ),
        "first_not_cached": m1.get("cache_hit") is False,
        "paraphrase_cache_hit": bool(m2.get("cache_hit")) and m2.get("cache_layer") == "semantic",
        "rate_limit_429_with_header": limited["status"] == 429 and bool(limited["retry_after"]),
    }
    report["checks"] = checks

    print("\n| 检查项 | 结果 |")
    print("| --- | --- |")
    for name, ok in checks.items():
        print(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    print(f"\n首次 {first['elapsed_s']}s -> 命中 {second['elapsed_s']}s；"
          f"内容长度 {len(_text(first))} / {len(_text(second))}")

    print(json.dumps({"checks": checks}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
