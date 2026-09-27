"""
agents/planner.py
Agent 规划器 - 负责多工具调用和任务规划

提供 AgentPlanner 类，用于在 RAG 流程中执行多步骤智能体任务。
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Dict, Optional

from tools.tools_registry import get_agent_tools
from prompts.agent import AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_schema_cache: Optional[Dict[str, Dict[str, str]]] = None
_schema_conn = None


def _load_schema(config) -> tuple:
    """懒加载 MySQL schema + 连接（供 SQL 校验/编译终审），失败返回 (None, None)。

    模块级缓存；连接断开会由 compile_check 抛错兜底（按无编译终审处理）。
    """
    global _schema_cache, _schema_conn
    if _schema_cache is not None:
        return _schema_cache, _schema_conn
    if config is None or not getattr(config, "AGENT_SQL_VALIDATE", True):
        _schema_cache, _schema_conn = None, None
        return None, None
    try:
        import pymysql
        from tools.sql_validator import load_schema

        conn = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE, charset="utf8mb4", connect_timeout=5,
        )
        schema = load_schema(conn)
        _schema_cache, _schema_conn = schema, conn
        logger.info("SQL 守卫：MySQL schema 已加载（%d 张表）", len(schema))
        return schema, conn
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQL 守卫：MySQL schema 加载失败，跳过校验重试: %s", exc)
        _schema_cache, _schema_conn = None, None
        return None, None


def _merge_chart_json(result: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把财务工具返回的 chart_json 合并进最终结果（LLM 输出格式不含该字段）。

    Args:
        result: Agent 最终结果字典（content/image/references）
        messages: 本轮全部消息（含 role=tool 的工具返回，取第一个非空 chart_json）

    Returns:
        合并 chart_json 后的结果字典
    """
    if not isinstance(result, dict) or result.get("chart_json"):
        return result
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        try:
            parsed = json.loads(m.get("content") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed.get("chart_json"):
            result["chart_json"] = parsed["chart_json"]
            break
    return result


def _append_sql(rag, sql: str) -> None:
    """线程安全地累积 SQL 到会话状态（Agent 并行工具执行时防止丢失/覆盖）。"""
    if not sql:
        return
    lock = getattr(rag, "_conversation_lock", None)
    if lock is not None:
        with lock:
            rag.conversation_state.sql += "\n" + sql
    else:
        rag.conversation_state.sql += "\n" + sql


def call_financial_chatflow(rag, user_query: str, user_id: str = "default") -> str:
    """财务查询统一入口（原生 SQL 链路，路线 3），返回 JSON 字符串。

    内部走 tools.native_financial 的「SQL 生成（三层防线重试）→ MySQL 执行 → 分析 → 图表」闭环，
    自带查询缓存（key 含 FINANCIAL_PROMPT_VERSION），SQL 累积到会话状态（_append_sql）。
    异常时返回查询失败提示 JSON，不抛异常。

    Args:
        rag: RAGPipeline 实例（读取/写入 conversation_state.sql）
        user_query: 用户的自然语言财务问题
        user_id: 会话用户标识

    Returns:
        JSON 字符串：{"content": ..., "image": [], "sql": ..., "chart_json": {...}|null}
    """
    try:
        from tools.native_financial import native_financial_query

        return native_financial_query(rag, user_query, user_id=user_id)
    except Exception as e:  # noqa: BLE001
        rag.conversation_state.sql = ""
        return json.dumps({"content": f"查询失败: {e}", "image": []})


class AgentPlanner:
    def __init__(self, llm_client, config, rag_pipeline):
        self.client = llm_client
        self.config = config
        self.rag = rag_pipeline

    def execute(self, user_query: str, history: List[Dict] = None, user_id: str = "default", verbose: bool = True, on_stage: Optional[Callable[[str], None]] = None, on_chunk: Optional[Callable[[str], None]] = None) -> str:
        system_prompt = AGENT_SYSTEM_PROMPT
        if on_stage:
            on_stage("parse")
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_query})

        tools = get_agent_tools()

        for _ in range(10):
            if on_stage:
                on_stage("thinking")
            response = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                # qwen3.5-plus 推理模型：统一关闭思考避免耗尽 max_tokens（RAGConfig.AGENT_ENABLE_THINKING 可开）
                extra_body={"enable_thinking": self.config.AGENT_ENABLE_THINKING},
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                content = msg.content
                if on_stage:
                    on_stage("generate")
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    result = {
                        "content": content,
                        "image": [],
                        "references": [],
                    }
                return _merge_chart_json(result, messages)

            # 路线 1：同一轮多个工具调用并行执行（AGENT_PARALLEL_TOOLS=false 可关）
            def _run_tool(tool_call):
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name == "search_reports":
                    if on_stage:
                        on_stage("search_reports")
                    if verbose:
                        print(f"[Agent] 执行 search_reports，查询 {args['query']}")
                    result_dict = (
                        self.rag.query(args["query"], verbose=False, stream_callback=on_chunk)
                        if on_chunk is not None
                        else self.rag.query(args["query"], verbose=False)
                    )
                    result = json.dumps(result_dict, ensure_ascii=False)
                elif func_name == "query_financial_and_visualize":
                    if on_stage:
                        on_stage("query_financial")
                    if verbose:
                        print(f"[Agent] 执行 query_financial_and_visualize，查询 {args['query']}")
                    result = self._call_financial_chatflow(args["query"], user_id=user_id)
                else:
                    result = f"未知工具: {func_name}"

                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                }

            if len(msg.tool_calls) > 1 and getattr(self.config, "AGENT_PARALLEL_TOOLS", True):
                with ThreadPoolExecutor(max_workers=min(len(msg.tool_calls), 4)) as executor:
                    tool_messages = list(executor.map(_run_tool, msg.tool_calls))
            else:
                tool_messages = [_run_tool(tc) for tc in msg.tool_calls]
            messages.extend(tool_messages)
        return {"content": "抱歉，处理超时，请简化问题后重试。", "image": [], "references": []}

    def _call_financial_chatflow(self, user_query: str, user_id: str = "default", num: int = 0) -> str:
        # 委托模块级共享函数（与 LangGraphPlanner 同源，避免工具行为漂移）
        return call_financial_chatflow(self.rag, user_query, user_id=user_id)
