# -*- coding: utf-8 -*-
"""B-19 API/接口层契约测试（FastAPI TestClient，零外部依赖）。

mock 策略（对齐 tests/ 既有 FakeClient / StubRag 模式）：
- 导入 app.api 前替换 core.pipeline.get_config / get_pipeline，返回 StubConfig + FakePipeline，
  避免 Qdrant / MySQL / LLM / BM25 / 引用语料初始化；
- QUERY_CACHE_ENABLED=false：不走 SQLite 缓存；
- result/ 静态目录在 CI 不存在，fixture 确保导入前创建、结束清理。

覆盖：/health、/chat（rag/agent/字符串兼容/图片归一化/500/422）、
/chat/stream（SSE：content/meta/done/final/error/stage 事件序列）、
/chat/clarify（成功与 500）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class StubConfig:
    """app.api 运行所需的配置子集（其余属性不会在 stub 路径被读取）。"""

    QUERY_CACHE_ENABLED = False
    API_RATE_LIMIT_BACKEND = "memory"
    USER_RATE_LIMIT_ENABLED = False
    SSE_RESUME_ENABLED = True
    SSE_RESUME_BACKEND = "memory"
    SSE_RESUME_REDIS_URL = ""
    SSE_RESUME_TTL_SECONDS = 1800
    QDRANT_COLLECTION_NAME = "test_collection"
    CITATION_CORPUS_ROOT = "__missing_corpus__"
    CITATION_MATCH_MODE = "smart"


class FakePipeline:
    """stub RAGPipeline：记录调用并返回固定结果；可通过标志注入故障/混合流。"""

    def __init__(self) -> None:
        self.query_cache = None
        self.citation_validator = None
        self.query_calls: List[str] = []
        self.agent_calls: List[Dict[str, Any]] = []
        self.clarify_calls: List[Dict[str, Any]] = []
        self.rag_fail: Optional[Exception] = None
        self.agent_fail: Optional[Exception] = None
        self.clarify_fail: Optional[Exception] = None
        self.rag_stream_text: str = "RAG 答案"      # 流式回调文本
        self.rag_content: str = "RAG 答案"          # 最终 content
        self.agent_stream_text: str = "Agent 答案"
        self.agent_content: str = "Agent 答案"
        self.agent_route_type: Optional[str] = None
        self.agent_stages: List[str] = ["thinking"]

    def build_bm25_index(self) -> None:
        pass

    def query(self, question: str, verbose: bool = False, stream_callback=None) -> Dict[str, Any]:
        if self.rag_fail is not None:
            raise self.rag_fail
        self.query_calls.append(question)
        if stream_callback is not None:
            stream_callback(self.rag_stream_text)
        return {
            "content": self.rag_content,
            "image": ["result/fig.png", "http://cdn.example.com/ext.png"],
            "references": [{"paper_path": "stub.md", "text": "研报原文"}],
            "chart_json": {"type": "line"},
        }

    def agent_query(self, question: str, user_id: Optional[str] = None,
                    on_stage=None, on_chunk=None) -> Dict[str, Any]:
        if self.agent_fail is not None:
            raise self.agent_fail
        self.agent_calls.append({"question": question, "user_id": user_id})
        if on_stage is not None:
            for stage in self.agent_stages:
                on_stage(stage)
        if on_chunk is not None:
            on_chunk(self.agent_stream_text)
        result = {
            "content": self.agent_content,
            "image": [],
            "references": [],
            "chart_json": None,
        }
        if self.agent_route_type:
            result["route_type"] = self.agent_route_type
        return result

    def conversational_query(self, input: str, user_id: Optional[str] = None):
        if self.clarify_fail is not None:
            raise self.clarify_fail
        self.clarify_calls.append({"input": input, "user_id": user_id})
        return ("需要补充报告期（2025 年 Q3？）", False)


class FakeSemanticCache:
    """stub 语义缓存：验证 /chat/stream 的命中回放与未命中写回。"""

    def __init__(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self.payload = payload
        self.last_similarity = 0.97
        self.last_lookup_ms = 1.5
        self.get_calls = 0
        self.set_calls = 0

    def get(self, question: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        self.get_calls += 1
        return dict(self.payload) if self.payload is not None else None

    def set(self, question: str, payload: Dict[str, Any], namespace: str = "default", embedding: Any = None) -> bool:
        self.set_calls += 1
        self.payload = dict(payload)
        return True

    def stats(self) -> Dict[str, Any]:
        return {}


@pytest.fixture(scope="module")
def api_env():
    """装配 app.api（首导一次）：替换 core.pipeline 装配函数 + 保证 result/ 存在。"""
    import core.pipeline as core_pipeline
    from fastapi.testclient import TestClient

    orig_config, orig_pipeline = core_pipeline.get_config, core_pipeline.get_pipeline
    stub_cfg = StubConfig()
    stub_pipe = FakePipeline()
    core_pipeline.get_config = lambda: stub_cfg
    core_pipeline.get_pipeline = lambda: stub_pipe

    result_dir = REPO_ROOT / "result"
    existed = result_dir.is_dir()
    result_dir.mkdir(exist_ok=True)

    import app.api as api_module  # noqa: PLC0415

    with TestClient(api_module.app) as client:
        yield client, api_module, stub_cfg, stub_pipe

    core_pipeline.get_config, core_pipeline.get_pipeline = orig_config, orig_pipeline
    if not existed:
        try:
            result_dir.rmdir()
        except OSError:
            pass


def _sse_events(body: str) -> List[Dict[str, Any]]:
    """SSE body -> event dict 列表（忽略 id:/空行）。"""
    events: List[Dict[str, Any]] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except ValueError:
            continue
    return events


def _sse_frames(body: str) -> List[Dict[str, Any]]:
    """SSE body -> [{id, data}]；用于断言自增 id 与断点续传。"""
    frames: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in body.splitlines():
        if line.startswith("id: "):
            current["id"] = int(line[4:].strip())
        elif line.startswith("data: "):
            current["data"] = json.loads(line[6:])
        elif not line.strip() and current:
            if "data" in current:
                frames.append(current)
            current = {}
    if current and "data" in current:
        frames.append(current)
    return frames


# ── /health ───────────────────────────────────────────────────────────
def test_health_ok(api_env) -> None:
    client, _, stub_cfg, _ = api_env
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "collection": stub_cfg.QDRANT_COLLECTION_NAME}


# ── /chat ─────────────────────────────────────────────────────────────
def test_chat_rag_mode(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat", json={"question": "云南白药净利润？", "mode": "rag"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "RAG 答案"
    assert body["image"] == ["/result/fig.png", "http://cdn.example.com/ext.png"]
    assert body["references"] == [{"paper_path": "stub.md", "text": "研报原文"}]
    assert stub_pipe.query_calls == ["云南白药净利润？"]


def test_chat_agent_mode(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat", json={"question": "分析营收下滑原因", "mode": "agent", "user_id": "u1"})
    assert resp.status_code == 200
    assert stub_pipe.agent_calls[-1] == {"question": "分析营收下滑原因", "user_id": "u1"}


def test_chat_default_mode_is_rag(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat", json={"question": "默认模式问题"})
    assert resp.status_code == 200
    assert stub_pipe.query_calls[-1] == "默认模式问题"


def test_chat_null_mode_422(api_env) -> None:
    client, *_ = api_env
    resp = client.post("/chat", json={"question": "空模式", "mode": None})
    assert resp.status_code == 422


def test_chat_invalid_mode_422(api_env) -> None:
    client, *_ = api_env
    resp = client.post("/chat", json={"question": "非法模式", "mode": "Agent"})
    assert resp.status_code == 422


def test_chat_string_result_wrapped(api_env) -> None:
    client, _, _, stub_pipe = api_env
    orig_query = stub_pipe.query
    stub_pipe.query = lambda question, verbose=False, stream_callback=None: "纯文本答案"
    try:
        resp = client.post("/chat", json={"question": "x"})
        assert resp.status_code == 200
        assert resp.json()["content"] == "纯文本答案"
    finally:
        stub_pipe.query = orig_query

def test_chat_error_500(api_env) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_fail = RuntimeError("MySQL 连接失败")
    try:
        resp = client.post("/chat", json={"question": "出错问题"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "MySQL 连接失败"
    finally:
        stub_pipe.rag_fail = None


def test_chat_missing_question_422(api_env) -> None:
    client, *_ = api_env
    resp = client.post("/chat", json={"mode": "rag"})
    assert resp.status_code == 422


def test_chat_wrong_question_type_422(api_env) -> None:
    client, *_ = api_env
    resp = client.post("/chat", json={"question": 123})
    assert resp.status_code == 422


def test_chat_user_rate_limit_429(api_env, monkeypatch) -> None:
    """超过 user_id 限额返回 429 + Retry-After。"""
    client, api_module, stub_cfg, _ = api_env
    monkeypatch.setattr(stub_cfg, "USER_RATE_LIMIT_ENABLED", True, raising=False)

    class FakeUserLimiter:
        limit = 2

        def __init__(self) -> None:
            self.calls = 0

        def allow(self, key: str, now=None):
            self.calls += 1
            return (self.calls <= self.limit, 5.0 if self.calls > self.limit else 0.0)

    fake = FakeUserLimiter()
    monkeypatch.setattr(api_module, "_get_user_rate_limiter", lambda: fake)

    ok1 = client.post("/chat", json={"question": "限流1", "user_id": "rl-user"})
    ok2 = client.post("/chat", json={"question": "限流2", "user_id": "rl-user"})
    limited = client.post("/chat", json={"question": "限流3", "user_id": "rl-user"})

    assert ok1.status_code == 200 and ok2.status_code == 200
    assert limited.status_code == 429
    assert limited.headers.get("Retry-After") == "5"


# ── /chat/stream（SSE）───────────────────────────────────────────────
def test_stream_rag_content_meta_done(api_env) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_stream_text = "RAG 答案"
    stub_pipe.rag_content = "RAG 答案"
    resp = client.post("/chat/stream", json={"question": "流式RAG", "mode": "rag"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    assert "content" in types and "meta" in types and "done" in types
    assert "final" not in types
    assert events[-1]["type"] == "done"


def test_stream_rag_mixed_final_event(api_env) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_stream_text = "研报草稿"
    stub_pipe.rag_content = "RAG 终稿内容"
    resp = client.post("/chat/stream", json={"question": "混合题", "mode": "rag"})
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    assert "final" in types
    assert types[-1] == "done"
    # final 事件之后重发的 content 片段拼接 = 终稿（此前草稿由前端在 final 时重置）
    last_final = max(i for i, e in enumerate(events) if e["type"] == "final")
    content_payloads = [e.get("text", "") for e in events[last_final + 1:] if e["type"] == "content"]
    assert "".join(content_payloads) == "RAG 终稿内容"


def test_stream_agent_stage_and_done(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat/stream", json={"question": "Agent流式", "mode": "agent"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    assert "stage" in types
    assert any(e.get("stage") == "thinking" for e in events if e["type"] == "stage")
    assert "content" in types and "meta" in types and "done" in types


def test_stream_agent_final_when_content_differs(api_env) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.agent_stream_text = "Agent 草稿"
    stub_pipe.agent_content = "Agent 终稿"
    resp = client.post("/chat/stream", json={"question": "Agent汇总", "mode": "agent"})
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    assert "final" in types and types[-1] == "done"


def test_stream_error_event(api_env) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_fail = RuntimeError("生成超时")
    try:
        resp = client.post("/chat/stream", json={"question": "会失败", "mode": "rag"})
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        assert events[-1]["type"] == "error"
        assert events[-1]["message"] == "生成超时"
    finally:
        stub_pipe.rag_fail = None


def test_stream_agent_kwargs_passed(api_env) -> None:
    client, _, _, stub_pipe = api_env
    client.post("/chat/stream", json={"question": "Agentkw", "mode": "agent", "user_id": "u9"})
    assert stub_pipe.agent_calls[-1] == {"question": "Agentkw", "user_id": "u9"}


def test_stream_missing_question_422(api_env) -> None:
    client, *_ = api_env
    resp = client.post("/chat/stream", json={"mode": "agent"})
    assert resp.status_code == 422


def test_stream_semantic_cache_hit(api_env, monkeypatch) -> None:
    """语义命中时直接回放，不进入真实检索/生成。"""
    client, api_module, stub_cfg, stub_pipe = api_env
    monkeypatch.setattr(stub_cfg, "SEMANTIC_CACHE_ENABLED", True, raising=False)
    fake = FakeSemanticCache(
        {"content": "语义缓存答案", "image": [], "references": [], "chart_json": None}
    )
    monkeypatch.setattr(api_module, "_semantic_cache", fake, raising=False)
    before = len(stub_pipe.query_calls)

    resp = client.post("/chat/stream", json={"question": "同义问法", "mode": "rag"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert "".join(e.get("text", "") for e in events if e["type"] == "content") == "语义缓存答案"
    assert events[-1]["type"] == "done"
    assert len(stub_pipe.query_calls) == before
    assert fake.get_calls == 1


def test_stream_semantic_cache_miss_then_set(api_env, monkeypatch) -> None:
    """语义未命中时走真实链路，并把结果写回缓存。"""
    client, api_module, stub_cfg, stub_pipe = api_env
    monkeypatch.setattr(stub_cfg, "SEMANTIC_CACHE_ENABLED", True, raising=False)
    fake = FakeSemanticCache(None)
    monkeypatch.setattr(api_module, "_semantic_cache", fake, raising=False)
    stub_pipe.rag_stream_text = "RAG 答案"
    stub_pipe.rag_content = "RAG 答案"

    resp = client.post("/chat/stream", json={"question": "首次问法", "mode": "rag"})
    assert resp.status_code == 200
    assert fake.get_calls == 1
    assert fake.set_calls == 1
    assert fake.payload is not None and fake.payload.get("content") == "RAG 答案"


@pytest.fixture
def sse_store(api_env):
    _, api_module, _, _ = api_env
    api_module.reset_sse_resume_store()
    yield api_module
    api_module.reset_sse_resume_store()


def test_stream_events_have_incrementing_ids(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_stream_text = "草稿"
    stub_pipe.rag_content = "终稿内容XYZ"
    resp = client.post("/chat/stream", json={"question": "id递增", "mode": "rag"})
    frames = _sse_frames(resp.text)
    assert frames
    assert [frame["id"] for frame in frames] == list(range(1, len(frames) + 1))
    assert all("id" not in frame["data"] for frame in frames)


def test_stream_meta_route_type_rag(api_env, sse_store) -> None:
    client, _, _, _ = api_env
    resp = client.post("/chat/stream", json={"question": "RAG路由", "mode": "rag"})
    metas = [e for e in _sse_events(resp.text) if e["type"] == "meta"]
    assert metas and metas[0]["route_type"] == "research"


def test_stream_meta_route_type_agent_from_result(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.agent_route_type = "financial"
    stub_pipe.agent_stages = ["query_financial"]
    try:
        resp = client.post("/chat/stream", json={"question": "财务路由", "mode": "agent"})
        metas = [e for e in _sse_events(resp.text) if e["type"] == "meta"]
        assert metas and metas[0]["route_type"] == "financial"
    finally:
        stub_pipe.agent_route_type = None
        stub_pipe.agent_stages = ["thinking"]


def test_stream_error_retryable_true_for_timeout(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_fail = RuntimeError("生成超时")
    try:
        resp = client.post("/chat/stream", json={"question": "超时错误", "mode": "rag"})
        events = _sse_events(resp.text)
        assert events[-1]["type"] == "error"
        assert events[-1]["retryable"] is True
    finally:
        stub_pipe.rag_fail = None


def test_stream_error_retryable_false_for_bad_request(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_fail = ValueError("invalid parameter: question")
    try:
        resp = client.post("/chat/stream", json={"question": "参数错误", "mode": "rag"})
        events = _sse_events(resp.text)
        assert events[-1]["type"] == "error"
        assert events[-1]["retryable"] is False
    finally:
        stub_pipe.rag_fail = None


def test_stream_resume_replays_after_last_event_id(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.rag_stream_text = "草稿"
    stub_pipe.rag_content = "终稿内容XYZ"
    payload = {"question": "断点题", "mode": "rag", "user_id": "resume-user-1"}
    first = client.post("/chat/stream", json=payload)
    first_frames = _sse_frames(first.text)
    assert [f["data"]["type"] for f in first_frames] == [
        "content", "meta", "final", "content", "content", "done",
    ]

    before_calls = len(stub_pipe.query_calls)
    resumed = client.post(
        "/chat/stream",
        json=payload,
        headers={"Last-Event-ID": "2"},
    )
    frames = _sse_frames(resumed.text)
    assert [f["id"] for f in frames] == [3, 4, 5, 6]
    assert [f["data"]["type"] for f in frames] == ["final", "content", "content", "done"]
    assert "".join(f["data"].get("text", "") for f in frames if f["data"]["type"] == "content") == "终稿内容XYZ"
    assert len(stub_pipe.query_calls) == before_calls


def test_stream_resume_new_question_starts_fresh(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    payload = {"question": "断点题A", "mode": "rag", "user_id": "resume-user-2"}
    first = client.post("/chat/stream", json=payload)
    assert _sse_frames(first.text)

    before_calls = len(stub_pipe.query_calls)
    second = client.post(
        "/chat/stream",
        json={"question": "断点题B", "mode": "rag", "user_id": "resume-user-2"},
        headers={"Last-Event-ID": "2"},
    )
    frames = _sse_frames(second.text)
    assert frames[0]["id"] == 1
    assert len(stub_pipe.query_calls) == before_calls + 1


def test_stream_resume_invalid_header_starts_fresh(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    payload = {"question": "非法断点", "mode": "rag", "user_id": "resume-user-3"}
    client.post("/chat/stream", json=payload)

    before_calls = len(stub_pipe.query_calls)
    second = client.post(
        "/chat/stream",
        json=payload,
        headers={"Last-Event-ID": "not-a-number"},
    )
    frames = _sse_frames(second.text)
    assert frames[0]["id"] == 1
    assert len(stub_pipe.query_calls) == before_calls + 1


def test_stream_resume_id_beyond_record_starts_fresh(api_env, sse_store) -> None:
    client, _, _, stub_pipe = api_env
    payload = {"question": "越界断点", "mode": "rag", "user_id": "resume-user-4"}
    client.post("/chat/stream", json=payload)

    before_calls = len(stub_pipe.query_calls)
    second = client.post(
        "/chat/stream",
        json=payload,
        headers={"Last-Event-ID": "999"},
    )
    frames = _sse_frames(second.text)
    assert frames[0]["id"] == 1
    assert len(stub_pipe.query_calls) == before_calls + 1


def test_openapi_chat_and_stream_contract(api_env) -> None:
    client, *_ = api_env
    schema = client.get("/openapi.json").json()
    chat_200 = schema["paths"]["/chat"]["post"]["responses"]["200"]
    assert chat_200["content"]["application/json"]["schema"]["$ref"].endswith("/ChatResponse")
    stream_200 = schema["paths"]["/chat/stream"]["post"]["responses"]["200"]
    assert "text/event-stream" in stream_200["content"]


# ── /chat/clarify ─────────────────────────────────────────────────────
def test_clarify_ok_returns_done_false(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat/clarify", json={"input": "净利润趋势", "user_id": "u2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "需要补充报告期（2025 年 Q3？）"
    assert body["clarify_done"] is False
    assert stub_pipe.clarify_calls[-1] == {"input": "净利润趋势", "user_id": "u2"}


def test_clarify_error_500(api_env) -> None:
    client, _, _, stub_pipe = api_env
    stub_pipe.clarify_fail = RuntimeError("会话存储不可用")
    try:
        resp = client.post("/chat/clarify", json={"input": "x"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "会话存储不可用"
    finally:
        stub_pipe.clarify_fail = None


def test_clarify_missing_input_422(api_env) -> None:
    client, *_ = api_env
    resp = client.post("/chat/clarify", json={"user_id": "u3"})
    assert resp.status_code == 422


# ── 契约：请求模型兼容性与响应形状 ─────────────────────────────────────
def test_chat_response_shape_fields(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat", json={"question": "形状校验", "mode": "agent"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"content", "image", "references", "chart_json", "route_type"}
    assert body["route_type"] is None
    assert isinstance(body["image"], list) and isinstance(body["references"], list)


def test_stream_meta_contains_references(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat/stream", json={"question": "引用校验", "mode": "rag"})
    events = _sse_events(resp.text)
    metas = [e for e in events if e["type"] == "meta"]
    assert metas
    assert metas[0]["references"] == [{"paper_path": "stub.md", "text": "研报原文"}]
    assert metas[0]["chart_json"] == {"type": "line"}


# ── API key 鉴权 / 限流 ───────────────────────────────────────────────
def _restore_attr(obj: Any, name: str, value: Any) -> None:
    if value is None:
        if hasattr(obj, name):
            delattr(obj, name)
    else:
        setattr(obj, name, value)


def test_api_auth_missing_key_401(api_env) -> None:
    client, api_module, stub_cfg, _ = api_env
    old_keys = getattr(stub_cfg, "API_KEYS", None)
    old_enabled = getattr(stub_cfg, "API_AUTH_ENABLED", None)
    stub_cfg.API_KEYS = "night-key"
    stub_cfg.API_AUTH_ENABLED = True
    api_module.reset_api_rate_limiter()
    try:
        resp = client.post("/chat", json={"question": "无 key"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "missing api key"
    finally:
        _restore_attr(stub_cfg, "API_KEYS", old_keys)
        _restore_attr(stub_cfg, "API_AUTH_ENABLED", old_enabled)


def test_api_auth_invalid_key_403(api_env) -> None:
    client, api_module, stub_cfg, _ = api_env
    old_keys, old_enabled = getattr(stub_cfg, "API_KEYS", None), getattr(stub_cfg, "API_AUTH_ENABLED", None)
    stub_cfg.API_KEYS = "night-key"
    stub_cfg.API_AUTH_ENABLED = True
    api_module.reset_api_rate_limiter()
    try:
        resp = client.post("/chat", json={"question": "错 key"}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "invalid api key"
    finally:
        _restore_attr(stub_cfg, "API_KEYS", old_keys)
        _restore_attr(stub_cfg, "API_AUTH_ENABLED", old_enabled)


def test_api_auth_valid_key_passes(api_env) -> None:
    client, api_module, stub_cfg, _ = api_env
    old_keys, old_enabled = getattr(stub_cfg, "API_KEYS", None), getattr(stub_cfg, "API_AUTH_ENABLED", None)
    stub_cfg.API_KEYS = "night-key"
    stub_cfg.API_AUTH_ENABLED = True
    api_module.reset_api_rate_limiter()
    try:
        resp = client.post("/chat", json={"question": "有 key"}, headers={"X-API-Key": "night-key"})
        assert resp.status_code == 200
    finally:
        _restore_attr(stub_cfg, "API_KEYS", old_keys)
        _restore_attr(stub_cfg, "API_AUTH_ENABLED", old_enabled)


def test_api_rate_limit_429(api_env) -> None:
    client, api_module, stub_cfg, _ = api_env
    old_keys = getattr(stub_cfg, "API_KEYS", None)
    old_enabled = getattr(stub_cfg, "API_AUTH_ENABLED", None)
    old_limit = getattr(stub_cfg, "API_RATE_LIMIT_PER_MINUTE", None)
    stub_cfg.API_KEYS = "rate-key"
    stub_cfg.API_AUTH_ENABLED = True
    stub_cfg.API_RATE_LIMIT_PER_MINUTE = 2
    api_module.reset_api_rate_limiter()
    try:
        headers = {"Authorization": "Bearer rate-key"}
        assert client.post("/chat", json={"question": "r1"}, headers=headers).status_code == 200
        assert client.post("/chat", json={"question": "r2"}, headers=headers).status_code == 200
        resp = client.post("/chat", json={"question": "r3"}, headers=headers)
        assert resp.status_code == 429
        assert resp.json()["detail"] == "rate limit exceeded"
        assert "Retry-After" in resp.headers
    finally:
        _restore_attr(stub_cfg, "API_KEYS", old_keys)
        _restore_attr(stub_cfg, "API_AUTH_ENABLED", old_enabled)
        _restore_attr(stub_cfg, "API_RATE_LIMIT_PER_MINUTE", old_limit)
        api_module.reset_api_rate_limiter()


def test_health_is_exempt_from_api_key(api_env) -> None:
    client, _, stub_cfg, _ = api_env
    old_keys, old_enabled = getattr(stub_cfg, "API_KEYS", None), getattr(stub_cfg, "API_AUTH_ENABLED", None)
    stub_cfg.API_KEYS = "night-key"
    stub_cfg.API_AUTH_ENABLED = True
    try:
        assert client.get("/health").status_code == 200
    finally:
        _restore_attr(stub_cfg, "API_KEYS", old_keys)
        _restore_attr(stub_cfg, "API_AUTH_ENABLED", old_enabled)

# ── 结构化日志埋点 ────────────────────────────────────────────────────
def test_chat_injects_trace_and_records_stages(api_env, monkeypatch, tmp_path) -> None:
    """成功路径：/chat 必须把 RequestTrace 注入 pipeline 上下文，阶段与 token 能落盘。"""
    client, _, stub_cfg, stub_pipe = api_env
    monkeypatch.setattr(stub_cfg, "OBSERVABILITY_LOG_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(stub_cfg, "OBSERVABILITY_LOG_ENABLED", True, raising=False)
    seen = {}

    def fake_query(question, verbose=False, stream_callback=None):
        from utils.observability import get_trace

        trace = get_trace()
        seen["trace"] = trace
        if trace is not None:
            trace.record_stage("retrieval_ms", 12.0)
            trace.record_usage(prompt_tokens=10, completion_tokens=5)
        return {"content": "ok", "image": [], "references": []}

    monkeypatch.setattr(stub_pipe, "query", fake_query)
    resp = client.post("/chat", json={"question": "trace 注入测试", "mode": "rag"})
    assert resp.status_code == 200
    assert seen["trace"] is not None

    log = tmp_path / f"{datetime.now():%Y%m%d}.jsonl"
    row = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()][-1]
    assert row["retrieval_ms"] == 12.0
    assert row["total_tokens"] == 15
    assert row["usage_status"] == "recorded"


def test_chat_error_path_logs_error_status(api_env, monkeypatch, tmp_path) -> None:
    """异常路径：500 响应必须落 status=error / status_code=500，字段缺失不填 0。"""
    client, _, stub_cfg, stub_pipe = api_env
    monkeypatch.setattr(stub_cfg, "OBSERVABILITY_LOG_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(stub_cfg, "OBSERVABILITY_LOG_ENABLED", True, raising=False)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(stub_pipe, "query", boom)
    resp = client.post("/chat", json={"question": "错误路径测试", "mode": "rag"})
    assert resp.status_code == 500

    log = tmp_path / f"{datetime.now():%Y%m%d}.jsonl"
    row = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()][-1]
    assert row["status"] == "error"
    assert row["status_code"] == 500
    assert row["retrieval_ms"] is None
