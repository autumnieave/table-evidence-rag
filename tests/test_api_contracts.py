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
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class StubConfig:
    """app.api 运行所需的配置子集（其余属性不会在 stub 路径被读取）。"""

    QUERY_CACHE_ENABLED = False
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
            on_stage("thinking")
        if on_chunk is not None:
            on_chunk(self.agent_stream_text)
        return {
            "content": self.agent_content,
            "image": [],
            "references": [],
            "chart_json": None,
        }

    def conversational_query(self, input: str, user_id: Optional[str] = None):
        if self.clarify_fail is not None:
            raise self.clarify_fail
        self.clarify_calls.append({"input": input, "user_id": user_id})
        return ("需要补充报告期（2025 年 Q3？）", False)


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
    """SSE body -> event dict 列表（过滤注释行/空行）。"""
    events: List[Dict[str, Any]] = []
    for raw in body.split("data: "):
        line = raw.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


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


def test_chat_null_mode_is_rag(api_env) -> None:
    client, _, _, stub_pipe = api_env
    resp = client.post("/chat", json={"question": "空模式", "mode": None})
    assert resp.status_code == 200
    assert stub_pipe.query_calls[-1] == "空模式"


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
    assert set(body) == {"content", "image", "references", "chart_json"}
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
