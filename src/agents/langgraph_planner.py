"""
agents/langgraph_planner.py
LangGraph 版 Agent 规划器（实验）—— 与 agents/planner.py 的 AgentPlanner 同接口、同输出契约。

设计对照（详见 docs/LangGraph对照.md）：
- 自研 AgentPlanner：while 循环 + Function Calling（最多 10 轮），状态隐式保存在 messages 列表。
- LangGraphPlanner：同一 prompt / 同一 tools / 同一工具执行，改为 StateGraph 显式状态机
  （call_model → 条件边 → tools → 条件边 → finalize），状态显式、可打断、可可视化。

默认后端仍为自研 AgentPlanner（RAGConfig.AGENT_PLANNER_BACKEND=handwritten），本实现标实验。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.planner import _merge_chart_json, call_financial_chatflow
from prompts.agent import AGENT_SYSTEM_PROMPT
from tools.tools_registry import get_agent_tools

logger = logging.getLogger(__name__)

MAX_ROUNDS = 10


class AgentState(TypedDict, total=False):
    """LangGraph 状态：消息列表 / 工具轮次 / 用户标识 / 最终结果。"""

    messages: List[Dict[str, Any]]
    rounds: int
    user_id: str
    result: Optional[Dict[str, Any]]
    last_active: float  # checkpoint 记忆新鲜度（超时视为新话题）


def _assistant_to_dict(message: Any) -> Dict[str, Any]:
    """把 OpenAI SDK 的 assistant message 规范化为 dict（tool_calls 保留完整结构）。

    Args:
        message: OpenAI SDK 返回的 assistant message 对象

    Returns:
        标准 OpenAI 对话消息 dict（可直接回传给 API 或存入状态）
    """
    tool_calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        tool_calls.append({
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        })
    return {"role": "assistant", "content": message.content, "tool_calls": tool_calls}


class LangGraphPlanner:
    """LangGraph 版 Agent 规划器（实验）：StateGraph 编排，与 AgentPlanner 输出契约一致。

    Attributes:
        client: OpenAI 兼容客户端（LLMGenerator.client，须含 chat.completions.create）
        config: RAGConfig（LLM_MODEL 等）
        rag: RAGPipeline（search_reports 工具执行入口）
    """

    def __init__(self, llm_client: Any, config: Any, rag_pipeline: Any) -> None:
        self.client = llm_client
        self.config = config
        self.rag = rag_pipeline
        self._sqlite_conn = None
        self._checkpointer = self._build_checkpointer()
        self._graph = self._build_graph()

    # ── Checkpoint（会话记忆持久化）─────────────────────────────────────
    def _build_checkpointer(self) -> Any:
        """按配置构建 LangGraph checkpointer：sqlite=落盘（默认）/ memory=进程内存 / none=关闭。

        sqlite 后端依赖 langgraph-checkpoint-sqlite 包；初始化失败降级为内存，保证主流程可用。
        """
        if not getattr(self.config, "AGENT_LANGGRAPH_CHECKPOINT", True):
            return None
        backend = str(getattr(self.config, "AGENT_LANGGRAPH_CHECKPOINT_BACKEND", "sqlite")).strip().lower()
        if backend == "sqlite":
            try:
                import sqlite3  # noqa: PLC0415
                from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

                path = str(getattr(self.config, "AGENT_LANGGRAPH_CHECKPOINT_PATH", "database/langgraph_checkpoints.sqlite"))
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(path, check_same_thread=False)
                self._sqlite_conn = conn
                logger.info("LangGraph checkpoint 已启用（sqlite: %s）", path)
                return SqliteSaver(conn)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LangGraph sqlite checkpoint 初始化失败，降级为内存: %s", exc)
        if backend == "none":
            return None
        from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415

        logger.info("LangGraph checkpoint 已启用（memory）")
        return MemorySaver()

    def close(self) -> None:
        """释放 sqlite checkpoint 连接（进程退出/测试清理时调用）。"""
        conn = getattr(self, "_sqlite_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._sqlite_conn = None

    # ── 图构建 ─────────────────────────────────────────────────────────
    def _build_graph(self) -> Any:
        """构建 LangGraph 状态机：call_model → (tools) → finalize。"""
        graph = StateGraph(AgentState)
        graph.add_node("call_model", self._call_model)
        graph.add_node("tools", self._execute_tools)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "call_model")
        graph.add_conditional_edges(
            "call_model",
            self._route_after_model,
            {"tools": "tools", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"call_model": "call_model", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self._checkpointer)

    # ── 节点 ───────────────────────────────────────────────────────────
    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """模型节点：携带历史调用工具 API，把 assistant 回复追加进状态。"""
        self._emit("thinking")
        response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=state["messages"],
            tools=get_agent_tools(),
            tool_choice="auto",
            # qwen3.5-plus 推理模型：统一关闭思考避免耗尽 max_tokens（RAGConfig.AGENT_ENABLE_THINKING 可开）
            extra_body={"enable_thinking": self.config.AGENT_ENABLE_THINKING},
        )
        assistant = response.choices[0].message
        return {"messages": state["messages"] + [_assistant_to_dict(assistant)]}

    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """工具节点：执行上一条 assistant 消息中的全部 tool_calls，结果回填为 tool 消息。"""
        messages = list(state["messages"])
        last = messages[-1]
        for tool_call in last.get("tool_calls") or []:
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"] or "{}")
            result = self._run_tool(name, args, state.get("user_id", "default"))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(result),
            })
        return {"messages": messages, "rounds": int(state["rounds"]) + 1}

    def _run_tool(self, func_name: str, args: Dict[str, Any], user_id: str) -> str:
        """工具分发：与 AgentPlanner 保持同一工具集与执行行为。"""
        if func_name == "search_reports":
            self._emit("search_reports")
            if getattr(self, "verbose", True):
                print(f"[LangGraph-Agent] 执行 search_reports，查询 {args.get('query')}")
            result_dict = (
                self.rag.query(args.get("query", ""), verbose=False, stream_callback=self.on_chunk)
                if getattr(self, "on_chunk", None) is not None
                else self.rag.query(args.get("query", ""), verbose=False)
            )
            return json.dumps(result_dict, ensure_ascii=False)
        if func_name == "query_financial_and_visualize":
            self._emit("query_financial")
            if getattr(self, "verbose", True):
                print(f"[LangGraph-Agent] 执行 query_financial_and_visualize，查询 {args.get('query')}")
            return call_financial_chatflow(self.rag, args.get("query", ""), user_id=user_id)
        return f"未知工具: {func_name}"

    # ── 路由与收尾 ─────────────────────────────────────────────────────
    def _route_after_model(self, state: AgentState) -> str:
        """模型回复含工具调用则进 tools，否则 finalize。"""
        last = state["messages"][-1]
        return "tools" if last.get("tool_calls") else "finalize"

    def _route_after_tools(self, state: AgentState) -> str:
        """轮次未超限则回模型，否则 finalize（超时兜底）。"""
        return "call_model" if int(state["rounds"]) < MAX_ROUNDS else "finalize"

    def _finalize(self, state: AgentState) -> Dict[str, Any]:
        """收尾节点：解析模型 JSON 输出；超时/非 JSON 时给出兜底结果。"""
        self._emit("generate")
        last = state["messages"][-1]
        if last.get("role") == "tool" or last.get("tool_calls"):
            return {"result": {"content": "抱歉，处理超时，请简化问题后重试。", "image": [], "references": []}}
        content = (last.get("content") or "").strip()
        if not content:
            return {"result": {"content": "抱歉，处理超时，请简化问题后重试。", "image": [], "references": []}}
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"content": content, "image": [], "references": []}
        return {"result": _merge_chart_json(result, state["messages"])}

    def _emit(self, stage: str) -> None:
        """发射前端阶段事件（stage 回调，未注入时忽略）。"""
        cb = getattr(self, "on_stage", None)
        if cb:
            cb(stage)

    def _load_memory(self, history: Optional[List[Dict[str, Any]]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从 checkpoint 读回该 user_id 上次会话作为记忆；超时/不存在则退回传入 history。

        Args:
            history: RAGPipeline 传入的多轮历史（checkpoint 可用时被忽略）
            config: LangGraph 运行配置（含 thread_id=user_id）

        Returns:
            本轮回话基础消息列表（可能为空）
        """
        if self._checkpointer is None:
            return list(history or [])
        try:
            snap = self._graph.get_state(config)
            values = (snap.values or {}) if snap else {}
            msgs = values.get("messages")
            if not msgs:
                return list(history or [])
            last_active = values.get("last_active") or 0
            timeout = int(getattr(self.config, "CONVERSATION_TIMEOUT_SECONDS", 1800))
            if time.time() - last_active > timeout:
                return []  # 超时视为新话题，不携带历史
            return self._trim_messages(msgs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangGraph checkpoint 读取失败，退回传入历史: %s", exc)
            return list(history or [])

    def _trim_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """截断历史消息（保留首条 system），并避免截断后首条为 tool 导致引用悬空。"""
        max_len = int(getattr(self.config, "AGENT_LANGGRAPH_MAX_HISTORY", 40))
        msgs = list(messages or [])
        if len(msgs) <= max_len:
            return msgs
        head = msgs[:1] if msgs and (msgs[0] or {}).get("role") == "system" else []
        body = msgs[len(head):]
        keep = body[-(max_len - len(head)):]
        while keep and keep[0].get("role") == "tool":
            idx = len(body) - len(keep) - 1
            if idx < 0:
                break
            keep.insert(0, body[idx])
        return head + keep

    # ── 入口 ───────────────────────────────────────────────────────────
    def execute(
        self,
        user_query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_id: str = "default",
        verbose: bool = True,
        on_stage: Optional[Callable[[str], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """执行 Agent 任务，返回 {content, image, references} 字典（与 AgentPlanner 一致）。

        Args:
            user_query: 用户问题
            history: 多轮历史（取最近 12 条）
            user_id: 会话用户标识
            verbose: 是否打印工具执行日志

        Returns:
            最终结果字典
        """
        config = {"configurable": {"thread_id": user_id}}
        base = self._load_memory(history, config)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        if base:
            # checkpoint 历史已含 system 提示词则直接续用；传入 history 保持原 12 条截断口径
            if (base[0] or {}).get("role") == "system":
                messages = list(base)
            else:
                messages.extend(base[-12:])
        messages.append({"role": "user", "content": user_query})
        self.verbose = verbose
        self.on_stage = on_stage
        self.on_chunk = on_chunk
        if on_stage:
            on_stage("parse")
        output = self._graph.invoke(
            {
                "messages": messages,
                "rounds": 0,
                "user_id": user_id,
                "result": None,
            },
            config,
        )
        # 记录本轮活跃时间，供下次 execute 判断 checkpoint 是否超时
        if self._checkpointer is not None:
            try:
                self._graph.update_state(config, {"last_active": time.time()})
            except Exception as exc:  # noqa: BLE001
                logger.warning("LangGraph checkpoint 写回 last_active 失败: %s", exc)
        return output["result"]
