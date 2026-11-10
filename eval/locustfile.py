"""Locust 压测脚本：50 并发打 ``/chat``，验证 Redis+Lua 限流与 5xx 降级。

用法::

    .\.venv\Scripts\locust.exe -f eval/locustfile.py --host http://localhost:18000 `
        --headless -u 50 -r 10 -t 60s --csv 训练结果数据/locust_teds_stress

鉴权：设置 ``LOCUST_API_KEY``，或复用进程环境里的 ``API_KEYS`` 首个 key。
限流：默认让所有虚拟用户共用一个 ``user_id``，从而稳定触发
``USER_RATE_LIMIT_PER_MINUTE=20``；429 视为预期结果，不计入失败。

结束时会在 stdout 打印 ``LOCUST_SUMMARY <json>``；设置 ``LOCUST_SUMMARY_JSON``
可同时落盘 JSON。报告包含 QPS、P50/P95/P99、429 次数与 5xx 错误率。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from locust import HttpUser, between, events, task


DEFAULT_QUESTION = "片仔癀2023年肝病用药毛利率是多少？请结合研报简要说明。"
QUESTION = os.getenv("LOCUST_QUESTION", DEFAULT_QUESTION)
SHARED_USER_ID = os.getenv("LOCUST_USER_ID", "locust-shared-user")
SUMMARY_PATH = os.getenv("LOCUST_SUMMARY_JSON", "").strip()
WAIT_MIN_SECONDS = float(os.getenv("LOCUST_WAIT_MIN", "0.1"))
WAIT_MAX_SECONDS = float(os.getenv("LOCUST_WAIT_MAX", "0.4"))
if WAIT_MAX_SECONDS < WAIT_MIN_SECONDS:
    raise ValueError("LOCUST_WAIT_MAX 必须大于等于 LOCUST_WAIT_MIN")

_STARTED_AT = time.time()
_LOCK = threading.Lock()
_STATUS_COUNTS: Counter = Counter()
_ERRORS: Counter = Counter()
_LATENCIES: Dict[str, List[float]] = defaultdict(list)
_REQUEST_SECONDS: List[float] = []


def _percentile(values: List[float], ratio: float) -> float:
    """最近秩百分位，避免引入 numpy。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return round(ordered[index], 2)


@events.request.add_listener
def _record_request(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    start_time,
    url,
    **kwargs: Any,
) -> None:
    """收集 HTTP 状态、延迟与异常，供收尾报告使用。"""
    status = int(getattr(response, "status_code", 0) or 0)
    with _LOCK:
        _STATUS_COUNTS[str(status)] += 1
        _REQUEST_SECONDS.append(time.time())
        if status:
            _LATENCIES[str(status)].append(float(response_time))
            _LATENCIES["all"].append(float(response_time))
        if exception:
            _ERRORS[type(exception).__name__] += 1


@events.quitting.add_listener
def _write_summary(environment, **kwargs: Any) -> None:
    """输出可复核的压测摘要。"""
    elapsed = max(0.001, time.time() - _STARTED_AT)
    total = sum(_STATUS_COUNTS.values())
    status_200 = _STATUS_COUNTS.get("200", 0)
    status_429 = _STATUS_COUNTS.get("429", 0)
    server_errors = sum(
        count for code, count in _STATUS_COUNTS.items() if code.startswith("5")
    )
    other = total - status_200 - status_429 - server_errors

    per_second: Counter = Counter()
    for stamp in _REQUEST_SECONDS:
        per_second[int(stamp - _STARTED_AT)] += 1
    max_qps = max(per_second.values(), default=0)
    allowed = _LATENCIES.get("200", [])
    all_latencies = _LATENCIES.get("all", [])
    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "total_requests": total,
        "status_200": status_200,
        "status_429": status_429,
        "status_5xx": server_errors,
        "status_other": other,
        "error_rate_5xx": round(server_errors / total, 6) if total else 0.0,
        "avg_qps": round(total / elapsed, 2),
        "max_qps_1s": max_qps,
        "allowed_latency_ms": {
            "p50": _percentile(allowed, 0.50),
            "p95": _percentile(allowed, 0.95),
            "p99": _percentile(allowed, 0.99),
            "mean": round(sum(allowed) / len(allowed), 2) if allowed else 0.0,
            "n": len(allowed),
        },
        "all_latency_ms": {
            "p50": _percentile(all_latencies, 0.50),
            "p95": _percentile(all_latencies, 0.95),
            "p99": _percentile(all_latencies, 0.99),
            "mean": round(sum(all_latencies) / len(all_latencies), 2)
            if all_latencies
            else 0.0,
            "n": len(all_latencies),
        },
        "exceptions": dict(_ERRORS),
        "status_counts": dict(sorted(_STATUS_COUNTS.items())),
    }
    print("LOCUST_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    if SUMMARY_PATH:
        path = Path(SUMMARY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


class ChatUser(HttpUser):
    """共享 user_id 的并发用户，专门验证限流与错误隔离。"""

    wait_time = between(WAIT_MIN_SECONDS, WAIT_MAX_SECONDS)

    def on_start(self) -> None:
        self.headers = {"Content-Type": "application/json"}
        api_key = os.getenv("LOCUST_API_KEY", "").strip()
        if not api_key:
            configured = [part.strip() for part in os.getenv("API_KEYS", "").split(",")]
            api_key = next((part for part in configured if part), "")
        if api_key:
            self.headers["X-API-Key"] = api_key

    @task
    def chat(self) -> None:
        payload = {
            "question": QUESTION,
            "mode": "rag",
            "user_id": SHARED_USER_ID,
        }
        with self.client.post(
            "/chat",
            json=payload,
            headers=self.headers,
            catch_response=True,
            timeout=180,
        ) as response:
            if response.status_code == 429:
                response.success()
                return
            if response.status_code >= 500:
                response.failure(f"5xx: {response.text[:200]}")
                return
            if response.status_code != 200:
                response.failure(f"http_{response.status_code}: {response.text[:200]}")
                return
            response.success()
