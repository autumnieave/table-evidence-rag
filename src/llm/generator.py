"""llm/generator.py"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
from config.rag_config import RAGConfig
from prompts.rag import build_prompt

logger = logging.getLogger(__name__)

class LLMGenerator:
    """
    大模型生成客户端封装

    支持流式和非流式生成，自动构建带上下文的Prompt
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.api_key = config.DASHSCOPE_API_KEY

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY未设置")

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=self.config.LLM_TIMEOUT,
            )
        except ImportError:
            logger.error("请先安装openai: pip install openai")
            raise

    def _build_prompt(self, query: str, contexts: List[str], history: List[Dict] = None) -> str:
        """构建包含检索上下文的Prompt（委托给 prompts.rag.build_prompt）"""
        return build_prompt(query, contexts, history)
    def generate(
        self,
        query: str,
        contexts: List[str],
        history: Optional[List[Dict]] = None,   # 新增参数
        stream: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        生成回答（支持流式；on_chunk 收到逐 token 文本回调，供 SSE 真流式使用）

        Args:
            query: 用户问题
            contexts: 上下文列表
            stream: 是否流式输出，默认使用配置值
            on_chunk: 逐 token 回调（stream=True 时生效）

        Returns:
            生成的回答文本
        """
        if stream is None:
            stream = self.config.STREAM

        prompt = self._build_prompt(query, contexts, history)

        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    stream=stream,
                    # qwen3.5-plus 是推理模型：必须关闭思考，否则思考过程会拖慢首 token / 耗尽 max_tokens
                    extra_body={"enable_thinking": False},
                )

                if stream:
                    collected_content = []
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            if on_chunk is None:
                                print(content, end="", flush=True)
                            if on_chunk is not None:
                                on_chunk(content)
                            collected_content.append(content)
                    if on_chunk is None:
                        print()  # 换行
                    return "".join(collected_content)
                else:
                    return response.choices[0].message.content

            except Exception as e:
                logger.error(f"LLM生成失败 (尝试 {attempt+1}/{self.config.MAX_RETRIES}): {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return "抱歉，生成回答时发生错误，请稍后重试。"


# ==================== 主流程与交互模块 ====================

if __name__ == "__main__":
    from scripts.interactive import main as interactive_main

    interactive_main()
