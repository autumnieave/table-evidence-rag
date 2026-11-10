"""Live Redis Stack checkpoint smoke test (not collected by pytest tests/)."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import uuid
from typing import TypedDict

REDIS_URL = os.getenv("REDIS_E2E_URL", "redis://127.0.0.1:6379/0")


class CounterState(TypedDict, total=False):
    count: int


def _redis_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6379), timeout=0.5):
            return True
    except OSError:
        return False


def _build_graph(saver):
    from langgraph.graph import END, START, StateGraph

    def inc(state: CounterState) -> CounterState:
        return {"count": int(state.get("count", 0)) + 1}

    builder = StateGraph(CounterState)
    builder.add_node("inc", inc)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile(checkpointer=saver)


def run() -> int:
    if not _redis_available():
        print(f"SKIP: Redis Stack is not reachable at {REDIS_URL}")
        return 0

    from langgraph.checkpoint.redis import RedisSaver

    thread_id = f"e2e-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    first = RedisSaver(redis_url=REDIS_URL)
    first.setup()
    graph1 = _build_graph(first)
    assert graph1.invoke({"count": 0}, config)["count"] == 1
    assert graph1.invoke({}, config)["count"] == 2
    first._redis.close()

    second = RedisSaver(redis_url=REDIS_URL)
    second.setup()
    graph2 = _build_graph(second)
    assert graph2.invoke({}, config)["count"] == 3
    second._redis.close()

    print(f"PASS: Redis checkpoint restored thread {thread_id} across instances")
    return 0


if __name__ == "__main__":
    sys.exit(run())
