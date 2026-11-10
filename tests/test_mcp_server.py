# -*- coding: utf-8 -*-
"""MCP Server 协议适配层离线单测。

覆盖：
- search_financial_reports 对 RAGPipeline.query 的 Mock 调用；
- query_financial_data 对原始非 SELECT SQL 输入的生成前拦截；
- 底层工具异常返回恶意 SQL 时的 MCP 边界二次拦截；
- 网络/超时异常的结构化 retryable 映射；
- search_tools 的渐进式发现。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import mcp_server


class _FakePipeline:
    """只实现 MCP 适配层需要的 query 方法，不连接 Qdrant/LLM。"""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def query(
        self,
        question: str,
        verbose: bool = True,
        stream_callback: Any = None,
    ) -> Dict[str, Any]:
        self.calls.append({"question": question, "verbose": verbose, "stream_callback": stream_callback})
        return {
            "content": "mock answer",
            "image": [],
            "references": [{"chunk_type": "table", "markdown": "| 项目 | 值 |"}],
        }


def test_search_financial_reports_uses_pipeline_and_keeps_references(monkeypatch) -> None:
    pipeline = _FakePipeline()
    monkeypatch.setattr(mcp_server, "_get_pipeline", lambda: pipeline)

    result = mcp_server.search_financial_reports("片仔癀2023年肝病用药毛利率")

    assert result["content"] == "mock answer"
    assert result["references"][0]["chunk_type"] == "table"
    assert pipeline.calls == [
        {
            "question": "片仔癀2023年肝病用药毛利率",
            "verbose": False,
            "stream_callback": None,
        }
    ]


def test_query_financial_data_blocks_raw_non_select_before_upstream(monkeypatch) -> None:
    called = {"native": 0}

    def fake_native(*args: Any, **kwargs: Any) -> str:
        called["native"] += 1
        return json.dumps({"content": "should not run", "sql": ""})

    monkeypatch.setattr(mcp_server, "_get_pipeline", lambda: object())
    monkeypatch.setattr(mcp_server, "native_financial_query", fake_native)

    result = mcp_server.query_financial_data("DROP TABLE income_sheet;")

    assert result == {
        "error_type": "SecurityValidationError",
        "message": "SQL blocked by gatekeeper",
        "retryable": False,
    }
    assert called["native"] == 0


def test_query_financial_data_blocks_malicious_sql_returned_by_upstream(monkeypatch) -> None:
    def fake_native(*args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {
                "content": "compromised upstream",
                "sql": "DROP TABLE income_sheet;",
                "image": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(mcp_server, "_get_pipeline", lambda: object())
    monkeypatch.setattr(mcp_server, "native_financial_query", fake_native)

    result = mcp_server.query_financial_data("片仔癀2023年净利润")

    assert result["error_type"] == "SecurityValidationError"
    assert result["retryable"] is False
    assert result["message"] == "SQL blocked by gatekeeper"


def test_query_financial_data_maps_timeout_to_retryable_network_error(monkeypatch) -> None:
    def fake_native(*args: Any, **kwargs: Any) -> str:
        raise TimeoutError("upstream timed out")

    monkeypatch.setattr(mcp_server, "_get_pipeline", lambda: object())
    monkeypatch.setattr(mcp_server, "native_financial_query", fake_native)

    result = mcp_server.query_financial_data("片仔癀2023年净利润")

    assert result == {
        "error_type": "NetworkTimeout",
        "message": "upstream unavailable",
        "retryable": True,
    }


def test_search_tools_returns_only_matching_tool() -> None:
    result = mcp_server.search_tools("SQL 查数 利润")

    assert result
    assert result[0]["name"] == "query_financial_data"
    assert result[0]["risk_level"] == "high (executes SQL)"
    assert all(item["name"] != "search_financial_reports" for item in result)


def test_query_financial_data_maps_format_gate_refusal_to_security_error(monkeypatch) -> None:
    def fake_native(*args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {
                "content": "抱歉，暂时无法完成该查询，请稍后重试或换个问法。 原因：查询语句未通过库内格式校验（已自动重试）",
                "image": [],
                "sql": "",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(mcp_server, "_get_pipeline", lambda: object())
    monkeypatch.setattr(mcp_server, "native_financial_query", fake_native)

    result = mcp_server.query_financial_data("片仔癀2023年净利润")

    assert result["error_type"] == "SecurityValidationError"
    assert result["retryable"] is False


def test_query_financial_data_maps_mysql_connection_refusal_to_network_error(monkeypatch) -> None:
    def fake_native(*args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {
                "content": "抱歉，暂时无法完成该查询，请稍后重试或换个问法。 原因：MySQL 连接失败：connection refused",
                "image": [],
                "sql": "",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(mcp_server, "_get_pipeline", lambda: object())
    monkeypatch.setattr(mcp_server, "native_financial_query", fake_native)

    result = mcp_server.query_financial_data("片仔癀2023年净利润")

    assert result == {
        "error_type": "NetworkTimeout",
        "message": "upstream unavailable",
        "retryable": True,
    }
