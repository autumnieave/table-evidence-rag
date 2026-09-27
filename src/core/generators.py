"""生成接口适配层（2026-08-22 链路收敛 #3）

LLMGenerator 已与 IGenerator 结构化兼容；适配层用于统一 query() 的依赖边界，
后续可替换为其他生成实现（如 vLLM / 其他云厂商）而不改调用方。
"""
from typing import Any, Callable, Dict, List, Optional

from core.interfaces import IGenerator


class GeneratorAdapter(IGenerator):
    """将 LLMGenerator 包装为 IGenerator 接口"""

    def __init__(self, llm_generator: Any):
        self.llm_generator = llm_generator

    def generate(
        self,
        query: str,
        contexts: List[str],
        history: Optional[List[Dict]] = None,
        stream: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """委托 LLMGenerator（支持流式）"""
        return self.llm_generator.generate(
            query=query,
            contexts=contexts,
            history=history,
            stream=stream,
            on_chunk=on_chunk,
        )
