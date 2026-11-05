"""Redis Stack checkpoint 跨实例恢复测试（无 Redis Stack 时自动跳过）。"""

from __future__ import annotations

import socket
import uuid
from typing import TypedDict

import pytest


REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"


def _stack_available() -> bool:
    try:
        with socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _stack_available(), reason="Redis Stack 不可用")


class CounterState(TypedDict, total=False):
    count: int


def _build_graph(saver):
    from langgraph.graph import END, START, StateGraph

    def inc(state):
        return {"count": int(state.get("count", 0)) + 1}

    builder = StateGraph(CounterState)
    builder.add_node("inc", inc)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile(checkpointer=saver)


def test_redis_checkpoint_restores_across_instances() -> None:
    from langgraph.checkpoint.redis import RedisSaver

    thread_id = f"test-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    first = RedisSaver(redis_url=REDIS_URL)
    first.setup()
    graph1 = _build_graph(first)
    assert graph1.invoke({"count": 0}, config)["count"] == 1
    assert graph1.invoke({}, config)["count"] == 2
    first._redis.close()

    # 新实例（模拟另一进程）读回同一 thread_id 的状态
    second = RedisSaver(redis_url=REDIS_URL)
    second.setup()
    graph2 = _build_graph(second)
    assert graph2.invoke({}, config)["count"] == 3
    second._redis.close()


def test_planner_selects_redis_checkpointer() -> None:
    from agents.langgraph_planner import LangGraphPlanner

    class Cfg:
        AGENT_LANGGRAPH_CHECKPOINT = True
        AGENT_LANGGRAPH_CHECKPOINT_BACKEND = "redis"
        AGENT_LANGGRAPH_CHECKPOINT_REDIS_URL = REDIS_URL

    planner = LangGraphPlanner.__new__(LangGraphPlanner)
    planner.config = Cfg()
    planner._sqlite_conn = None
    planner._redis_saver = None
    saver = planner._build_checkpointer()
    try:
        assert saver.__class__.__name__ == "RedisSaver"
    finally:
        planner.close()


def test_multi_agent_selects_redis_checkpointer() -> None:
    from agents.langgraph_multi_agent import LangGraphMultiAgentPlanner

    class Cfg:
        AGENT_LANGGRAPH_CHECKPOINT = True
        AGENT_LANGGRAPH_CHECKPOINT_BACKEND = "redis"
        AGENT_LANGGRAPH_CHECKPOINT_REDIS_URL = REDIS_URL

    planner = LangGraphMultiAgentPlanner.__new__(LangGraphMultiAgentPlanner)
    planner.config = Cfg()
    planner._sqlite_conn = None
    planner._redis_saver = None
    saver = planner._build_checkpointer()
    try:
        assert saver.__class__.__name__ == "RedisSaver"
    finally:
        planner.close()
