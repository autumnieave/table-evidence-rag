"""Offline Redis checkpoint tests; live Redis Stack E2E lives under eval/integration/."""

from __future__ import annotations

import importlib
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver


REDIS_URL = "redis://127.0.0.1:6379/0"


class CounterState(TypedDict, total=False):
    count: int


class _SharedMemorySaver(InMemorySaver):
    """Two saver objects share one store to emulate a shared persistent backend."""

    def __init__(self, shared: InMemorySaver) -> None:
        super().__init__()
        self.storage = shared.storage
        self.writes = shared.writes
        self.blobs = shared.blobs


class _FakeRedisConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeRedisSaver:
    """Only verifies the planner selection branch; no real Redis connection."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self.setup_called = False
        self._redis = _FakeRedisConnection()

    def setup(self) -> None:
        self.setup_called = True


def _build_graph(saver: InMemorySaver):
    from langgraph.graph import END, START, StateGraph

    def inc(state: CounterState) -> CounterState:
        return {"count": int(state.get("count", 0)) + 1}

    builder = StateGraph(CounterState)
    builder.add_node("inc", inc)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile(checkpointer=saver)


def test_checkpoint_restores_across_saver_instances_with_shared_store() -> None:
    """State for one thread_id continues to be readable from a second saver/graph."""
    shared = InMemorySaver()
    first = _SharedMemorySaver(shared)
    config = {"configurable": {"thread_id": "offline-shared-checkpoint"}}

    graph1 = _build_graph(first)
    assert graph1.invoke({"count": 0}, config)["count"] == 1
    assert graph1.invoke({}, config)["count"] == 2

    second = _SharedMemorySaver(shared)
    graph2 = _build_graph(second)
    assert graph2.invoke({}, config)["count"] == 3


def test_planner_selects_redis_checkpointer_with_mocked_saver(monkeypatch) -> None:
    """LangGraphPlanner selects RedisSaver for the redis backend and calls setup."""
    from agents.langgraph_planner import LangGraphPlanner

    redis_module = importlib.import_module("langgraph.checkpoint.redis")
    monkeypatch.setattr(redis_module, "RedisSaver", _FakeRedisSaver)

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
        assert isinstance(saver, _FakeRedisSaver)
        assert saver.redis_url == REDIS_URL
        assert saver.setup_called is True
    finally:
        planner.close()
        assert saver._redis.closed is True


def test_multi_agent_selects_redis_checkpointer_with_mocked_saver(monkeypatch) -> None:
    """The multi-agent planner follows the same Redis selection contract."""
    from agents.langgraph_multi_agent import LangGraphMultiAgentPlanner

    redis_module = importlib.import_module("langgraph.checkpoint.redis")
    monkeypatch.setattr(redis_module, "RedisSaver", _FakeRedisSaver)

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
        assert isinstance(saver, _FakeRedisSaver)
        assert saver.redis_url == REDIS_URL
        assert saver.setup_called is True
    finally:
        planner.close()
        assert saver._redis.closed is True
