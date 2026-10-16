"""embeddings/client.py"""
import logging
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from config.rag_config import RAGConfig

logger = logging.getLogger(__name__)


def cap_embedding_texts(texts: List[str], max_chars: int) -> Tuple[List[str], int]:
    """嵌入前的**字符兜底**：超过 ``max_chars`` 的文本截断后再送 API，返回 ``(新列表, 截断条数)``。

    DashScope ``text-embedding-v2`` 单条输入上限 2048 token，**一个超长块会让整批 400**
    （实测结构化口径下有整表块 13,239 字）。正常由分块的 ``TABLE_SLICE_MAX_CHARS`` 保证不触发，
    这里是**最后一道防线**；**payload 里的正文不受影响**（只影响该点的向量语义，做多长截断见 docstring）。
    """
    if max_chars <= 0:
        return list(texts), 0
    capped: List[str] = []
    n_capped = 0
    for text in texts:
        if text is not None and len(text) > max_chars:
            capped.append(text[:max_chars])
            n_capped += 1
        else:
            capped.append(text)
    return capped, n_capped

class EmbeddingClient:
    """
    Embedding客户端封装类

    封装对阿里云百炼text-embedding-2模型的调用，支持批量处理和错误重试

    Attributes:
        config: RAG配置对象
        api_key: API密钥
        dashscope: DashScope SDK模块
    """

    def __init__(self, config: RAGConfig):
        """
        初始化Embedding客户端

        Args:
            config: RAG配置对象

        Raises:
            ValueError: API Key未设置时抛出
            ImportError: dashscope未安装时抛出
        """
        self.config = config
        self.api_key = config.DASHSCOPE_API_KEY

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY未设置，无法初始化Embedding客户端")

        # 导入dashscope
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.dashscope = dashscope
            logger.info("DashScope SDK初始化成功")
        except ImportError:
            logger.error("请先安装dashscope: pip install dashscope")
            raise

    def generate_embeddings(
        self,
        texts: List[str],
        text_type: str = "document"
    ) -> List[List[float]]:
        """
        生成文本的Embedding向量

        批量生成文本的向量表示，自动处理分批和重试

        Args:
            texts: 待编码的文本列表
            text_type: 文本类型，"document"(存储)或"query"(查询)

        Returns:
            List[List[float]]: 向量列表

        Raises:
            Exception: API调用失败时抛出

        Note:
            text_type参数对检索效果有重要影响，存储时必须设为"document"，
            查询时必须设为"query"
        """
        if not texts:
            return []

        logger.info(f"开始生成Embedding: {len(texts)} 条文本, text_type={text_type}")

        all_embeddings = []
        batch_size = self.config.EMBEDDING_BATCH_SIZE

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch, n_capped = cap_embedding_texts(
                texts[i:i + batch_size], getattr(self.config, "EMBEDDING_MAX_CHARS", 0)
            )
            if n_capped:
                logger.warning(
                    "Embedding 输入超长已截断：%d/%d 条（上限 %d 字符；正文不受影响）",
                    n_capped, len(batch), self.config.EMBEDDING_MAX_CHARS,
                )
            batch_embeddings = self._call_embedding_api(batch, text_type)
            all_embeddings.extend(batch_embeddings)

            logger.info(f"Embedding进度: {min(i + batch_size, len(texts))}/{len(texts)}")
            
        return all_embeddings

    def _call_embedding_api(self, texts: List[str], text_type: str) -> List[List[float]]:
        """
        调用 DashScope Embedding HTTP API，包含重试机制

        Args:
            texts: 文本批次
            text_type: "document" 或 "query"

        Returns:
            向量列表
        """
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(self.config.MAX_RETRIES):
            try:
                payload = {
                    "model": self.config.EMBEDDING_MODEL,
                    "input": {"texts": texts},  # DashScope v2 API: input.texts ?????
                    "parameters": {"text_type": text_type},
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=self.config.REQUEST_TIMEOUT)
                data = resp.json()

                if resp.status_code == 200 and "output" in data and "embeddings" in data["output"]:
                    embeddings = [item["embedding"] for item in data["output"]["embeddings"]]
                    return embeddings
                else:
                    logger.warning(f"Embedding API返回错误 (尝试 {attempt+1}/{self.config.MAX_RETRIES}): {data.get('message')}")
                    if attempt < self.config.MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)  # 指数退避

            except Exception as e:
                logger.error(f"Embedding API调用异常 (尝试 {attempt+1}/{self.config.MAX_RETRIES}): {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError(f"Embedding API调用失败，已重试{self.config.MAX_RETRIES}次")
# ==================== Qdrant向量数据库封装模块 ====================

