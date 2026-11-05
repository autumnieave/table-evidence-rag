# -*- coding: utf-8 -*-
"""SSE 断点续传存储与帧编码单元测试（零外部依赖）。"""

from utils.sse_stream import (
    SSEResumeStore,
    format_sse,
    parse_last_event_id,
    question_fingerprint,
)


def test_format_sse_keeps_id_out_of_payload():
    frame = format_sse({"id": 7, "type": "content", "text": "答案"})
    assert frame.startswith("id: 7\ndata: ")
    assert frame.endswith("\n\n")
    assert '"id"' not in frame
    assert '"type": "content"' in frame


def test_parse_last_event_id():
    assert parse_last_event_id("12") == 12
    assert parse_last_event_id(" 0 ") == 0
    assert parse_last_event_id("") is None
    assert parse_last_event_id("abc") is None
    assert parse_last_event_id("-1") is None
    assert parse_last_event_id(None) is None


def test_fingerprint_changes_with_question_and_mode():
    a = question_fingerprint("同一问题", "rag")
    assert a == question_fingerprint("同一问题", "rag")
    assert a != question_fingerprint("同一问题", "agent")
    assert a != question_fingerprint("另一问题", "rag")


def test_store_append_load_and_done():
    store = SSEResumeStore()
    stream_id = store.start("u1", "问题", "rag")
    first = store.append("u1", stream_id, {"type": "content", "text": "a"})
    second = store.append("u1", stream_id, {"type": "done"})
    assert first["id"] == 1 and second["id"] == 2

    meta = store.get_meta("u1")
    assert meta["stream_id"] == stream_id
    assert meta["last_id"] == 2
    assert meta["done"] is False

    assert [e["id"] for e in store.load_events("u1", 0)] == [1, 2]
    assert [e["id"] for e in store.load_events("u1", 1)] == [2]
    assert store.load_events("u1", 2) == []

    store.mark_done("u1", stream_id)
    assert store.get_meta("u1")["done"] is True


def test_store_new_stream_replaces_old_record():
    store = SSEResumeStore()
    old = store.start("u2", "旧问题", "rag")
    store.append("u2", old, {"type": "content", "text": "old"})
    new = store.start("u2", "新问题", "rag")
    assert new != old
    assert store.get_meta("u2")["stream_id"] == new
    assert store.load_events("u2", 0) == []
    assert store.append("u2", old, {"type": "content", "text": "stale"}) is None


def test_store_clear_and_reset():
    store = SSEResumeStore()
    sid = store.start("u3", "问题", "rag")
    store.append("u3", sid, {"type": "done"})
    store.clear("u3")
    assert store.get_meta("u3") is None

    sid2 = store.start("u4", "问题", "rag")
    store.append("u4", sid2, {"type": "done"})
    store.reset()
    assert store.get_meta("u4") is None


class _BrokenRedis:
    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise RuntimeError("redis down")
        return _raise


def test_store_redis_failure_falls_back_to_memory():
    store = SSEResumeStore(redis_client=_BrokenRedis())
    sid = store.start("u5", "问题", "rag")
    event = store.append("u5", sid, {"type": "content", "text": "x"})
    assert event["id"] == 1
    assert store.degraded is True
    assert store.load_events("u5", 0)[0]["text"] == "x"
