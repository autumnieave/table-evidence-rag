"""llm/generator.py"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
from config.rag_config import RAGConfig
from prompts.rag import build_prompt
from utils.observability import get_trace

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

    @staticmethod
    def _usage_value(usage: Any, *names: str) -> Optional[int]:
        """兼容 dict / pydantic 两种 usage 结构读取 token 字段。"""
        if usage is None:
            return None
        for name in names:
            if isinstance(usage, dict) and usage.get(name) is not None:
                return int(usage[name])
            value = getattr(usage, name, None)
            if value is not None:
                return int(value)
        return None

    def _record_usage(self, usage: Any) -> None:
        """把一次调用的 usage 写入当前请求 trace；缺失时明确标记。"""
        trace = get_trace()
        if trace is None:
            return
        if usage is None:
            trace.mark_usage_unavailable("usage_missing")
            return
        trace.record_usage(
            prompt_tokens=self._usage_value(usage, "prompt_tokens", "input_tokens"),
            completion_tokens=self._usage_value(usage, "completion_tokens", "output_tokens"),
            total_tokens=self._usage_value(usage, "total_tokens"),
            model_name=self.config.LLM_MODEL,
            input_price_per_1k=getattr(self.config, "LLM_INPUT_PRICE_PER_1K", None),
            output_price_per_1k=getattr(self.config, "LLM_OUTPUT_PRICE_PER_1K", None),
        )

    @staticmethod
    def _is_stream_options_unsupported(exc: Exception) -> bool:
        """识别供应商不支持 stream_options 的错误，避免因此中断生成。"""
        message = str(exc).lower()
        return "stream_options" in message and any(
            token in message
            for token in ("unsupported", "unrecognized", "unknown", "invalid", "not support")
        )

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

        create_kwargs: Dict[str, Any] = {
            "model": self.config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.TEMPERATURE,
            "max_tokens": self.config.MAX_TOKENS,
            "stream": stream,
            # qwen3.5-plus 是推理模型：必须关闭思考，否则思考过程会拖慢首 token / 耗尽 max_tokens
            "extra_body": {"enable_thinking": False},
        }
        if stream and getattr(self.config, "LLM_STREAM_INCLUDE_USAGE", True):
            # OpenAI 兼容接口只有在显式请求时才在流末尾返回 usage
            create_kwargs["stream_options"] = {"include_usage": True}

        for attempt in range(self.config.MAX_RETRIES):
            try:
                try:
                    response = self.client.chat.completions.create(**create_kwargs)
                except Exception as exc:  # noqa: BLE001
                    if "stream_options" in create_kwargs and self._is_stream_options_unsupported(exc):
                        logger.warning("供应商不支持 stream_options，回退为不请求流式 usage: %s", exc)
                        create_kwargs.pop("stream_options", None)
                        trace = get_trace()
                        if trace is not None:
                            trace.mark_usage_unavailable("stream_options_unsupported")
                        response = self.client.chat.completions.create(**create_kwargs)
                    else:
                        raise

                if stream:
                    collected_content = []
                    stream_usage = None
                    for chunk in response:
                        if getattr(chunk, "usage", None) is not None:
                            stream_usage = chunk.usage
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            if on_chunk is None:
                                print(content, end="", flush=True)
                            if on_chunk is not None:
                                on_chunk(content)
                            collected_content.append(content)
                    if on_chunk is None:
                        print()  # 换行
                    self._record_usage(stream_usage)
                    return "".join(collected_content)
                else:
                    self._record_usage(getattr(response, "usage", None))
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
