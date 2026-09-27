"""RAG 问答 Prompt —— 手写链路与 LCEL 链路共用同一模板。

背景：原 chains/prompt_templates.build_prompt（手写链路）与
chains/rag_chain.py 内联 ChatPromptTemplate（LCEL 链路）各存一份模板，
存在措辞/字段漂移风险；2026-08-22 统一为 RAG_PROMPT_TEMPLATE，
见 docs/ARCHITECTURE.md 差距清单 #2。
"""
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

# 模块级版本号：修改本文件任一 Prompt 文本后递增（prompts/registry.json 同步登记）
RAG_PROMPT_VERSION = "2026-09-09-v1"

# 唯一模板：手写链路（build_prompt）与 LCEL 链路（build_rag_chat_prompt）同源
RAG_PROMPT_TEMPLATE = chr(10).join([
    "你是一个专业的金融研究助手，请基于以下参考资料和对话历史回答用户的问题。回答时请保持专业、客观，并在适当位置引用来源。",
    "## 参考资料",
    "{context}",
    "",
    "## 对话历史",
    "{history}",
    "",
    "## 用户问题",
    "{question}",
    "",
    "## 回答要求",
    "1. 请仔细阅读所有参考资料，从参考资料中整合分散的信息，并从中提取、整合所有相关信息来回答。",
    "2. 如果参考资料提供了部分信息但不够完整，请基于已有信息作答，并指出信息的局限性。",
    "3. 回答时请保持专业、客观。",
    "4. 如果参考资料不足以回答问题，请明确告知。",
    "5. 回答结构清晰，必要时使用要点或段落。",
    "",
    "## 回答",
    "",
])

# 手写链路上下文总长度上限（保留历史行为，防止超长上下文）
MAX_CONTEXT_LEN = 256000

# 上下文分隔符（与历史行为一致：换行换行---换行换行）
CONTEXT_SEP = chr(10) * 2 + "---" + chr(10) * 2


def format_history(history: Optional[List[Dict]] = None, max_rounds: int = 6) -> str:
    """将对话历史格式化为文本；无历史时返回“无历史对话”"""
    if not history:
        return "无历史对话"
    return chr(10).join(msg["role"] + ": " + msg["content"] for msg in history[-max_rounds:])


def _truncate_contexts(contexts: List[str], max_len: int = MAX_CONTEXT_LEN) -> List[str]:
    """按总长度截断上下文，避免上下文超过模型窗口"""
    truncated: List[str] = []
    total = 0
    for ctx in contexts:
        if total + len(ctx) > max_len:
            remaining = max_len - total
            if remaining > 200:
                truncated.append(ctx[:remaining] + "...")
            break
        truncated.append(ctx)
        total += len(ctx)
    return truncated


def build_prompt(query: str, contexts: List[str], history: Optional[List[Dict]] = None) -> str:
    """构建手写链路 RAG 问答 Prompt（行为与原 chains.prompt_templates.build_prompt 一致）"""
    context_text = CONTEXT_SEP.join(_truncate_contexts(contexts))
    return RAG_PROMPT_TEMPLATE.format(
        context=context_text,
        history=format_history(history),
        question=query,
    )


def build_rag_chat_prompt() -> ChatPromptTemplate:
    """构建 LCEL 链路 ChatPromptTemplate（与手写链路同一模板）"""
    return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
