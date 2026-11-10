# -*- coding: utf-8 -*-
"""MCP Server：将研报 RAG 与受控财务 SQL 能力标准化暴露给外部 Agent。

本模块只做协议适配，不替换 FastAPI / LangGraph 主链路：

- ``search_financial_reports`` 直接复用 ``RAGPipeline.query``；
- ``query_financial_data`` 直接复用 ``native_financial_query``，自然语言查数仍会
  经过 ``_generate_sql`` 的既有 SQL 闸门；
- ``search_tools`` 提供关键词级渐进式发现，避免外部 Agent 一次性加载全部工具定义。

传输层使用官方 SDK v1 的 ``FastMCP`` + stdio。工具内部所有异常都在此边界捕获，
客户端只会收到结构化 JSON 错误，不会看到 Python traceback。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from tools.native_financial import native_financial_query
from tools.sql_validator import enforce_select_limit
from utils.output_contracts import validate_sql_output


logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """可安全返回给 MCP 客户端的业务异常。

    Args:
        error_type: 稳定错误类型，供客户端做程序化分支。
        message: 脱敏后的用户可读错误信息。
        retryable: 是否建议客户端重试。
    """

    def __init__(self, error_type: str, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.retryable = retryable


#: 原始 SQL 语句起始关键字：MCP 的自然语言查数工具不接受直接注入 SQL。
_RAW_SQL_HEAD_RE = re.compile(
    r"^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/\s*))*"
    r"(select|with|insert|update|delete|drop|alter|truncate|create|grant|revoke|"
    r"replace|merge|call|exec|execute|set|load)\b",
    re.IGNORECASE | re.DOTALL,
)

#: 多语句拼接中的 SQL 变更/危险语句。
_RAW_SQL_TAIL_RE = re.compile(
    r";\s*(select|insert|update|delete|drop|alter|truncate|create|grant|revoke|"
    r"replace|merge|call|exec|execute)\b",
    re.IGNORECASE | re.DOTALL,
)


_pipeline: Optional[Any] = None
_pipeline_lock = threading.Lock()


mcp = FastMCP(
    "table-evidence-rag",
    instructions=(
        "财务研报 Agentic RAG MCP Server。工具调用前可先使用 search_tools 做渐进式发现；"
        "SQL 工具风险等级为 high，安全策略由服务端闸门强制接管。"
    ),
)


def _get_pipeline() -> Any:
    """懒加载并复用 RAGPipeline（避免 MCP 导入阶段触发外部连接）。"""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from config.rag_config import get_config
                from pipelines.rag_pipeline import RAGPipeline

                _pipeline = RAGPipeline(get_config())
    return _pipeline


def _error_payload(error_type: str, message: str, retryable: bool) -> Dict[str, Any]:
    """构造 MCP 工具的结构化错误 JSON。"""
    return {
        "error_type": error_type,
        "message": message,
        "retryable": retryable,
    }


def _security_error() -> MCPToolError:
    """SQL 安全边界拒绝。"""
    return MCPToolError(
        error_type="SecurityValidationError",
        message="SQL blocked by gatekeeper",
        retryable=False,
    )


def _is_network_failure(exc: Exception) -> bool:
    """判断异常是否更像瞬时网络/上游不可用，而不是业务或参数错误。"""
    message = str(exc).lower()
    network_markers = (
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "connection aborted",
        "connection error",
        "network is unreachable",
        "temporary failure",
        "upstream",
        "429",
        "502",
        "503",
        "504",
    )
    if any(marker in message for marker in network_markers):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def _classify_exception(exc: Exception) -> MCPToolError:
    """把内部异常映射为稳定的 MCP 错误语义，不向客户端泄露堆栈。"""
    if isinstance(exc, MCPToolError):
        return exc
    if _is_network_failure(exc):
        return MCPToolError("NetworkTimeout", "upstream unavailable", True)
    logger.warning("MCP tool internal error: %s", exc, exc_info=True)
    return MCPToolError("InternalError", "internal server error", False)


def _reject_raw_sql(text: str) -> None:
    """拒绝把原始 SQL 当作自然语言问题传入，避免绕过生成前意图与闸门。"""
    if _RAW_SQL_HEAD_RE.search(text) or _RAW_SQL_TAIL_RE.search(text):
        raise _security_error()


def _assert_returned_sql_safe(sql: str) -> None:
    """对底层工具返回的 SQL 再做一次 Server 侧契约校验。

    这是纵深防御：即使底层工具因重构/异常返回了非 SELECT，也不能让它穿过 MCP
    边界。自然语言查数的第一道闸门仍在 ``native_financial._generate_sql`` 内。
    """
    contract = validate_sql_output(sql)
    if not contract.ok:
        logger.warning("MCP 拒绝底层返回的非法 SQL: %s", contract.errors[:3])
        raise _security_error()
    try:
        enforce_select_limit(sql)
    except ValueError as exc:
        logger.warning("MCP 拒绝底层返回的非 SELECT SQL: %s", str(exc)[:200])
        raise _security_error() from exc


def _classify_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把底层工具的结构化拒答映射为 MCP 可重试语义。

    ``native_financial_query`` 为了兼容 CLI/API，会把连接失败与 SQL 格式拒绝
    包装成自然语言 JSON。MCP 客户端需要可编程的 ``error_type``，因此在这里做
    一次无堆栈、无 SQL 明细的语义归类。
    """
    content = str(payload.get("content") or "")
    if "查询语句未通过库内格式校验" in content or "格式契约" in content:
        raise _security_error()
    if any(
        marker in content
        for marker in (
            "MySQL 连接失败",
            "MySQL schema/连接加载失败",
            "连接超时",
            "timed out",
            "Timeout",
        )
    ):
        raise MCPToolError("NetworkTimeout", "upstream unavailable", True)
    return payload


def _normalise_rag_result(result: Any) -> Dict[str, Any]:
    """把 RAGPipeline.query 的返回值归一为 MCP 可序列化对象。"""
    if isinstance(result, dict):
        return result
    return {"content": str(result), "image": [], "references": []}


@mcp.tool(
    name="search_financial_reports",
    title="Search financial reports",
    description=(
        "从公司研报与财报表格中做证据检索并生成带引用的答案。"
        "适用于毛利率、营收、净利润、期间对比、表格事实等研究型问题。"
    ),
)
def search_financial_reports(query: str) -> Dict[str, Any]:
    """调用现有 RAGPipeline.query，返回答案、图片与引用。"""
    question = str(query or "").strip()
    if not question:
        return _error_payload("InvalidRequest", "query must not be empty", False)
    try:
        result = _get_pipeline().query(question, verbose=False)
        return _normalise_rag_result(result)
    except Exception as exc:  # noqa: BLE001 - MCP 协议边界必须兜底
        error = _classify_exception(exc)
        return _error_payload(error.error_type, error.message, error.retryable)


@mcp.tool(
    name="query_financial_data",
    title="Query financial data (SQL, high risk)",
    description=(
        "用自然语言查询结构化财务数据库。服务端会强制执行只读账号、SELECT-only、"
        "危险关键字、静态字段校验、LIMIT 100 兜底、MySQL 编译终审等 7 道硬闸门，"
        '并保留 2 类柔性审计。risk_level: "high (executes SQL)"。'
    ),
)
def query_financial_data(question: str) -> Dict[str, Any]:
    """调用现有原生财务 SQL Tool，并在 MCP 边界做输入/输出双重校验。"""
    text = str(question or "").strip()
    if not text:
        return _error_payload("InvalidRequest", "question must not be empty", False)
    try:
        _reject_raw_sql(text)
        raw = native_financial_query(_get_pipeline(), text, user_id="mcp")
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            raise MCPToolError("InternalError", "invalid upstream response", False)
        sql = str(payload.get("sql") or "").strip()
        if sql:
            _assert_returned_sql_safe(sql)
        return _classify_payload(payload)
    except Exception as exc:  # noqa: BLE001 - MCP 协议边界必须兜底
        error = _classify_exception(exc)
        return _error_payload(error.error_type, error.message, error.retryable)


_TOOL_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "search_financial_reports",
        "description": "研报/财报证据检索与带引用回答；适合毛利率、营收、期间对比等研究型问题。",
        "keywords": (
            "研报",
            "财报",
            "报告",
            "检索",
            "证据",
            "引用",
            "表格",
            "毛利率",
            "营收",
            "研究",
            "分析",
            "rag",
            "report",
            "research",
            "table",
            "citation",
        ),
    },
    {
        "name": "query_financial_data",
        "description": "自然语言财务查数；服务端强制只读 SELECT 与多层 SQL 安全闸门。",
        "keywords": (
            "查数",
            "数据库",
            "sql",
            "财务",
            "数字",
            "指标",
            "利润",
            "收入",
            "同比",
            "环比",
            "精确",
            "金额",
            "query",
            "data",
            "financial",
            "metric",
        ),
        "risk_level": "high (executes SQL)",
    },
]


@mcp.tool(
    name="search_tools",
    title="Search available MCP tools",
    description=(
        "当外部 Agent 连接多个 MCP Server 时，先用关键词过滤本 Server 的工具，"
        "再只加载命中的工具定义，避免上下文预算被上百个工具描述占满。"
    ),
)
def search_tools(query: str) -> List[Dict[str, Any]]:
    """基于关键词返回匹配的工具名称、描述与风险等级。"""
    text = str(query or "").strip().lower()
    if not text:
        return [
            {
                "name": item["name"],
                "description": item["description"],
                **({"risk_level": item["risk_level"]} if "risk_level" in item else {}),
            }
            for item in _TOOL_REGISTRY
        ]

    scored: List[tuple[int, Dict[str, Any]]] = []
    for item in _TOOL_REGISTRY:
        haystack = " ".join(
            [str(item.get("name", "")), str(item.get("description", "")), *item.get("keywords", ())]
        ).lower()
        score = sum(1 for keyword in item.get("keywords", ()) if str(keyword).lower() in text)
        if text in haystack:
            score += 2
        if score:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "name": item["name"],
            "description": item["description"],
            **({"risk_level": item["risk_level"]} if "risk_level" in item else {}),
        }
        for _, item in scored
    ]


def main() -> None:
    """以 stdio 传输启动 MCP Server。"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
