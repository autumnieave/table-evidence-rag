"""
tools/data_scripts/agent_planner_compare.py
自研 Agent 循环 vs LangGraph StateGraph 对照（真实 LLM，工具可 stub）

用法:
  python tools/data_scripts/agent_planner_compare.py --query "马应龙成本控制优势" [--stub] [--backend both|handwritten|langgraph]

- --stub  : 用 stub 替代研报检索（无需 Qdrant，推荐离线验证工具循环）；默认尝试真实检索（需 Qdrant 运行）。
- 输出：两版 Agent 的耗时、工具调用轮次、结果摘要与 LangGraph 图结构。
"""

import argparse
import json
import sys
import time
import types
from typing import Any, Dict, List

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from agents.langgraph_planner import LangGraphPlanner
from agents.planner import AgentPlanner
from config.rag_config import RAGConfig

DASHSCOPE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class StubRag:
    """stub RAG：记录检索调用并返回固定结果（无需 Qdrant）。"""

    def __init__(self) -> None:
        self.conversation_state = types.SimpleNamespace(sql="")
        self.tool_calls: List[str] = []

    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        self.tool_calls.append(question)
        return {
            "content": f"[STUB 研报检索结果] 关于「{question}」的摘要（供工具循环验证）。",
            "image": [],
            "references": [{"paper_path": "stub.md", "text": question, "paper_image": ""}],
        }


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": (result.get("content") or "")[:180],
        "image数": len(result.get("image") or []),
        "references数": len(result.get("references") or []),
    }


def main(argv: List[str] | None = None) -> int:
    _utf8()
    parser = argparse.ArgumentParser(description="自研 Agent vs LangGraph 对照")
    parser.add_argument("--query", default="结合研报分析马应龙的成本控制优势")
    parser.add_argument("--stub", action="store_true", help="用 stub 替代真实研报检索（无需 Qdrant）")
    parser.add_argument("--backend", choices=["both", "handwritten", "langgraph"], default="both")
    args = parser.parse_args(argv)

    cfg = RAGConfig()
    if args.stub:
        rag: Any = StubRag()
        from openai import OpenAI
        client = OpenAI(api_key=cfg.DASHSCOPE_API_KEY, base_url=DASHSCOPE_COMPAT_URL)
    else:
        from pipelines.rag_pipeline import RAGPipeline
        rag = RAGPipeline(cfg)
        client = rag.llm_generator.client

    def run(planner: Any, label: str) -> None:
        rag.tool_calls.clear()
        t0 = time.time()
        result = planner.execute(user_query=args.query, verbose=False)
        elapsed = time.time() - t0
        print(f"\n===== {label} =====")
        print(f"耗时: {elapsed:.1f}s | 检索工具调用: {len(rag.tool_calls)} 次")
        print(f"结果: {json.dumps(summarize(result), ensure_ascii=False)}")

    if args.backend in ("both", "handwritten"):
        run(AgentPlanner(llm_client=client, config=cfg, rag_pipeline=rag), "自研 AgentPlanner（while 循环）")
    if args.backend in ("both", "langgraph"):
        lg = LangGraphPlanner(llm_client=client, config=cfg, rag_pipeline=rag)
        graph = lg._graph.get_graph()
        print("\n[LangGraph 图结构] nodes:", sorted(graph.nodes))
        run(lg, "LangGraphPlanner（StateGraph）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
