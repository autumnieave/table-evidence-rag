"""chains/rerank.py"""
import time
import logging
from typing import List, Dict, Any, Optional
from config.rag_config import RAGConfig
import dashscope

logger = logging.getLogger(__name__)

class RerankClient:
    """
    重排序客户端封装

    调用阿里云百炼qwen3-rerank模型对候选文档进行精排
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.api_key = config.DASHSCOPE_API_KEY

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY未设置")

        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.dashscope = dashscope
        except ImportError:
            logger.error("请先安装dashscope: pip install dashscope")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        对文档列表进行重排序

        Args:
            query: 用户查询
            documents: 候选文档内容列表
            top_n: 返回前N个结果，默认使用配置值
            return_documents: 是否在结果中包含文档内容

        Returns:
            排序后的结果列表，每项包含index、relevance_score、document等
        """
        if top_n is None:
            top_n = self.config.RERANK_TOP_N

        logger.info(f"开始Rerank: 查询长度 {len(query)}, 文档数 {len(documents)}")

        for attempt in range(self.config.MAX_RETRIES):
            try:
                resp = self.dashscope.TextReRank.call(
                    model=self.config.RERANK_MODEL,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    return_documents=return_documents,
                    api_key=self.api_key,
                    timeout=30.0
                )

                if resp.status_code == 200:
                    results = []
                    for item in resp.output.results:
                        result = {
                            "index": item.index,
                            "relevance_score": item.relevance_score
                        }
                        if return_documents and hasattr(item, 'document'):
                            result["document"] = item["document"]['text']
                            logger.debug("Rerank结果 index=%s score=%s doc=%s",
                                         result["index"], result["relevance_score"],
                                         result["document"][:50])
                        results.append(result)
                    logger.info(f"Rerank完成，返回 {len(results)} 个结果")
                    return results
                else:
                    logger.warning(f"Rerank API返回错误: {resp.message}")
                    if attempt < self.config.MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Rerank API调用异常: {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError(f"Rerank API调用失败，已重试{self.config.MAX_RETRIES}次")

# ==================== 多轮对话 ======================
