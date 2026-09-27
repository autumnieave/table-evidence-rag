"""prompts 包 —— 项目唯一 Prompt 目录（2026-08-22 收敛）
统一所有业务 Prompt 的来源，避免散落内联导致漂移：
- prompts/rag.py       RAG 问答（手写链路与 LCEL 链路同源模板）
- prompts/pipeline.py  多轮澄清字段提取 / 摘要 / 图片检测
- prompts/agent.py     Agent 工具调用 System Prompt
- prompts/fallback.py  兜底话术模板（refuse/suggest/human，B-18）

新增 Prompt 一律放本目录；修改 Prompt 文本后请更新 PROMPT_VERSION，
并在 docs/ARCHITECTURE.md 记录变更。
"""
from prompts.rag import (
    RAG_PROMPT_TEMPLATE,
    build_prompt,
    build_rag_chat_prompt,
    format_history,
)
from prompts.pipeline import (
    FILTER_EXTRACT_PROMPT_TEMPLATE,
    IMAGE_DETECT_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
)
from prompts.agent import AGENT_SYSTEM_PROMPT

# 模板版本：每次修改 prompts 下任意模板文本时递增/更新
PROMPT_VERSION = "2026-09-10-v6"  # B-33: financial 补齐金额列单位口径（万元/元）+ 禁止 10000 倍换算

__all__ = [
    "PROMPT_VERSION",
    "RAG_PROMPT_TEMPLATE",
    "build_prompt",
    "build_rag_chat_prompt",
    "format_history",
    "FILTER_EXTRACT_PROMPT_TEMPLATE",
    "IMAGE_DETECT_PROMPT_TEMPLATE",
    "SUMMARY_PROMPT_TEMPLATE",
    "AGENT_SYSTEM_PROMPT",
]
