"""Locust 压测脚本：50 并发持续 2 分钟打 /chat/stream。

用法（需先安装 locust，并确保 API 与 Qdrant 可用）::

    .\.venv\Scripts\python.exe -m pip install locust
    $env:LOCUST_HOST="http://localhost:8000"
    $env:LOCUST_API_KEY="your-key"   # API_KEYS 启用时必填
    .\.venv\Scripts\locust.exe -f eval/locustfile.py --headless -u 50 -r 5 -t 2m `
        --host $env:LOCUST_HOST --csv 训练结果数据/locust_20261023

判定：

- HTTP 200 且 SSE 流里出现 ``done`` 才算成功；
- 出现 ``error`` 事件、无 ``done``、连接异常都记失败；
- 失败请求也计入 Locust 延迟与错误率分母。
"""

from __future__ import annotations

import os
import time

from locust import HttpUser, between, task


DEFAULT_QUESTION = "云南白药2025年三季度营业收入和净利润是多少？请给出研报观点。"


class ChatStreamUser(HttpUser):
    """模拟前端流式问答用户。"""

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.headers = {"Content-Type": "application/json"}
        api_key = os.getenv("LOCUST_API_KEY", "").strip()
        if api_key:
            self.headers["X-API-Key"] = api_key

    @task
    def chat_stream(self) -> None:
        payload = {
            "question": DEFAULT_QUESTION,
            "mode": "rag",
            "user_id": f"locust-{id(self)}",
        }
        started = time.perf_counter()
        with self.client.post(
            "/chat/stream",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True,
            timeout=180,
        ) as response:
            if response.status_code != 200:
                response.failure(f"http_{response.status_code}")
                return

            saw_done = False
            try:
                for raw in response.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if '"type": "error"' in data:
                        response.failure(f"sse_error: {data[:200]}")
                        return
                    if '"type": "done"' in data:
                        saw_done = True
                        break
            except Exception as exc:  # noqa: BLE001
                response.failure(f"stream_exception: {type(exc).__name__}: {exc}")
                return

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not saw_done:
                response.failure("stream_ended_without_done")
                return
            response.success()
            # 自定义字段会出现在 Locust CSV 的 request 统计之外；这里只打印便于人工排查
            if elapsed_ms > 30000:
                print(f"[locust] slow_stream elapsed_ms={elapsed_ms:.1f}")
