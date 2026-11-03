# -*- coding: utf-8 -*-
"""顺序回放 A 类问题到 ``/chat``，用于产生延迟分位统计所需的结构化日志。

脚本只负责发请求，不修改检索/生成链路。服务端需已开启
``OBSERVABILITY_LOG_ENABLED=true``，并在统计延迟时关闭查询缓存。

用法::

    # 新终端启动 API（关闭缓存，避免命中旧结果）
    $env:PYTHONPATH="src"; $env:QUERY_CACHE_ENABLED="false"; \
        .\.venv\Scripts\python.exe -m uvicorn app.api:app --app-dir src --port 8000

    # 顺序回放 10 题
    .\.venv\Scripts\python.exe eval\latency_batch.py --limit 10

如果服务端开启鉴权，追加 ``--api-key $env:API_KEYS.Split(',')[0]``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "训练结果数据" / "stage5_measure" / "a_expand_questions.json"


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """读取题目 JSON；兼容 ``{"items": [...]}`` 与裸数组两种格式。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("题目文件必须是数组，或包含 items 数组")

    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        rows.append({"qid": str(item.get("qid") or ""), "question": question})
    return rows


def post_chat(
    base_url: str,
    endpoint: str,
    question: str,
    mode: str,
    timeout: float,
    api_key: str = "",
) -> tuple[int, str]:
    """发送一次非流式请求，返回 ``(status_code, response_text)``。"""
    body = json.dumps({"question": question, "mode": mode}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(
        base_url.rstrip("/") + "/" + endpoint.lstrip("/"),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="顺序回放 A 类问题到 /chat")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/chat")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", default="rag")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--dry-run", action="store_true", help="只打印待发题目，不发请求")
    args = parser.parse_args(argv)

    path = Path(args.questions)
    if not path.exists():
        raise SystemExit(f"题目文件不存在：{path}")
    items = load_questions(path)[: max(0, args.limit)]
    if not items:
        raise SystemExit("没有可回放的题目")

    failures = 0
    for index, item in enumerate(items, start=1):
        if args.dry_run:
            print(f"[{index}/{len(items)}] {item['qid']} {item['question']}")
            continue
        started = time.perf_counter()
        status_code, text = post_chat(
            args.base_url,
            args.endpoint,
            item["question"],
            args.mode,
            args.timeout,
            args.api_key,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if status_code != 200:
            failures += 1
        print(
            f"[{index}/{len(items)}] {item['qid']} status={status_code} "
            f"client_ms={elapsed_ms:.1f} response_chars={len(text)}"
        )

    if args.dry_run:
        return 0
    print(f"completed={len(items) - failures}/{len(items)} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
