# -*- coding: utf-8 -*-
"""结构化请求 trace / JSONL / 聚合脚本的离线单测。"""

from __future__ import annotations

import json
from pathlib import Path

from eval.log_aggregate import aggregate, filter_records, load_records, percentile
from utils.observability import RequestTrace, append_trace


def test_trace_usage_without_registered_price() -> None:
    trace = RequestTrace(trace_id="t1", path="/chat")
    trace.record_stage("retrieval_ms", 12.3456)
    trace.record_usage(prompt_tokens=10, completion_tokens=5, model_name="qwen-test")
    payload = trace.as_dict()

    assert payload["retrieval_ms"] == 12.346
    assert payload["total_tokens"] == 15
    assert payload["model_name"] == "qwen-test"
    assert payload["estimated_cost"] is None
    assert payload["cost_status"] == "unit_price_not_registered"
    assert payload["usage_status"] == "recorded"


def test_trace_usage_with_registered_price() -> None:
    trace = RequestTrace(trace_id="t2")
    trace.record_usage(
        prompt_tokens=1000,
        completion_tokens=500,
        input_price_per_1k=0.002,
        output_price_per_1k=0.006,
    )
    assert trace.estimated_cost == 0.005
    assert trace.cost_status == "estimated"


def test_append_trace_writes_jsonl(tmp_path: Path) -> None:
    trace = RequestTrace(trace_id="t3", path="/health", method="GET")
    trace.record_stage("retrieval_ms", 1.0)
    trace.finalize(total_ms=3.0, status="ok", status_code=200)
    path = append_trace(trace, log_dir=str(tmp_path))

    assert path is not None and path.exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["trace_id"] == "t3"
    assert rows[-1]["total_ms"] == 3.0
    assert rows[-1]["status_code"] == 200


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == 3.85
    assert percentile([], 0.95) is None


def test_aggregate_cost_and_stage_share() -> None:
    records = [
        {
            "status": "ok",
            "total_ms": 100.0,
            "retrieval_ms": 20.0,
            "rerank_ms": 30.0,
            "generate_ms": 50.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "estimated_cost": None,
            "usage_status": "recorded",
        },
        {
            "status": "ok",
            "total_ms": 200.0,
            "retrieval_ms": 40.0,
            "rerank_ms": 60.0,
            "generate_ms": 100.0,
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
            "estimated_cost": None,
            "usage_status": "recorded",
        },
    ]
    summary = aggregate(records)
    assert summary["n_requests"] == 2
    assert summary["tokens"]["total_tokens"] == 45
    assert summary["cost"]["cost_status"] == "unit_price_not_registered"
    assert summary["cost"]["total_estimated_cost"] is None
    assert summary["stage_share_pct"] == {
        "retrieval_ms": 20.0,
        "rerank_ms": 30.0,
        "generate_ms": 50.0,
    }
    assert summary["latency_ms"]["total_ms"]["p95"] is not None


def test_aggregate_empty_is_honest() -> None:
    summary = aggregate([])
    assert summary["n_requests"] == 0
    assert summary["error_rate"] is None
    assert summary["latency_ms"]["total_ms"]["p95"] is None
    assert summary["tokens"]["total_tokens"] is None
    assert summary["cost"]["cost_status"] == "no_token_data"


def test_load_records_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "20261023.jsonl"
    path.write_text('{"trace_id":"a"}\nnot-json\n{"trace_id":"b"}\n', encoding="utf-8")
    records = load_records([path])
    assert [row["trace_id"] for row in records] == ["a", "b"]


def test_filter_records_by_path_prefix() -> None:
    records = [
        {"path": "/chat", "total_ms": 100.0},
        {"path": "/health", "total_ms": 1.0},
        {"path": "/chat/stream", "total_ms": 200.0},
    ]
    assert [row["total_ms"] for row in filter_records(records, "/chat")] == [100.0, 200.0]
    assert len(filter_records(records, "")) == 3
