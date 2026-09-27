"""检索器实现（2026-08-22 链路收敛 #3）

- HandwrittenRetriever（默认）：EmbeddingClient 生成查询向量 → QdrantClientWrapper.search_similar
- LangChainRetriever（实验）：QdrantVectorStore.similarity_search_with_score → 转统一格式
- BM25Retriever / HybridRetriever（实验，#8）：纯 Python BM25 关键词召回 + 向量召回 RRF 融合

统一返回结构：[{"score": float, "payload": {"content": str, ...}}, ...]，
与 QdrantClientWrapper.search_similar 的返回兼容。
"""
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces import IRetriever


class HandwrittenRetriever(IRetriever):
    """默认检索器（手写链路）：向量化 + Qdrant 检索"""

    def __init__(self, embedding_client: Any, qdrant_client: Any, top_k: int):
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """向量化查询 → Qdrant 相似检索，返回统一候选结构"""
        query_vector = self.embedding_client.generate_embeddings([query], text_type="query")[0]
        return self.qdrant_client.search_similar(
            query_vector,
            limit=top_k or self.top_k,
            query_filter=query_filter,
        )


class LangChainRetriever(IRetriever):
    """实验检索器（LangChain QdrantVectorStore）：similarity_search_with_score 转统一格式"""

    def __init__(self, vectorstore: Any, top_k: int):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """调用带分数的相似检索并过滤空内容文档"""
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k or self.top_k,
        )
        search_results: List[Dict[str, Any]] = []
        for doc, score in docs_with_scores:
            if not doc.page_content or not doc.page_content.strip():
                continue
            search_results.append({
                "score": score,
                "payload": {"content": doc.page_content, **doc.metadata},
            })
        return search_results


class BM25Retriever(IRetriever):
    """纯 Python BM25 检索器（#8 混合检索实验）。

    不依赖外部检索库：分词采用英文/数字单词 + 中文单字，适合研报类中英混合文本；
    检索返回统一候选结构 [{"id", "score", "payload"}, ...]。
    支持 pickle 持久化（部署重启复用）。
    """

    K1: float = 1.5
    B: float = 0.75

    def __init__(self, docs: List[Dict[str, Any]]):
        """构建 BM25 索引。

        Args:
            docs: 文档块列表，每项含 id 与 payload（payload["content"] 为正文）
        """
        self.docs: List[Dict[str, Any]] = docs
        self._tokenized: List[List[str]] = []
        self._doc_len: List[int] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._build()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """简单分词：英文/数字单词 + 中文单字（小写归一）。

        Args:
            text: 原始文本

        Returns:
            词元列表，如 "贵州茅台2024" → ["2024", "贵", "州", "茅", "台"]
        """
        text = (text or "").lower()
        words = re.findall(r"[a-z0-9_]+", text)
        cjk = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        return words + cjk

    def _build(self) -> None:
        """统计词频 / 文档频率 / 平均长度。"""
        for doc in self.docs:
            tokens = self.tokenize(doc.get("payload", {}).get("content", ""))
            self._tokenized.append(tokens)
            self._doc_len.append(len(tokens))
            for token in set(tokens):
                self._df[token] += 1
        total = sum(self._doc_len) or 1
        self._avgdl = total / max(len(self._doc_len), 1)

    def _idf(self, token: str, n_docs: int) -> float:
        """BM25 逆文档频率：ln(1 + (N - df + 0.5) / (df + 0.5))"""
        df = self._df.get(token, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def retrieve(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 检索，返回统一候选结构；无命中时返回空列表。

        Args:
            query: 查询文本
            query_filter: 兼容接口签名，BM25 不消费（保持与其他检索器一致）
            top_k: 返回条数，默认返回全部命中

        Returns:
            [{"id", "score", "payload"}, ...]，按 BM25 得分降序
        """
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []
        n_docs = len(self.docs)
        scored: List[Tuple[float, int]] = []
        for idx, tokens in enumerate(self._tokenized):
            freq = Counter(tokens)
            score = 0.0
            for token in set(q_tokens):
                tf = freq.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.K1 * (1.0 - self.B + self.B * self._doc_len[idx] / self._avgdl)
                score += self._idf(token, n_docs) * (tf * (self.K1 + 1.0)) / denom
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[Dict[str, Any]] = []
        for score, idx in scored[: (top_k or len(scored))]:
            doc = self.docs[idx]
            results.append({"id": doc.get("id"), "score": score, "payload": doc.get("payload", {})})
        return results


class HybridRetriever(IRetriever):
    """混合检索（#8 实验）：向量召回 + BM25 召回 → RRF 融合。

    RRF 融合公式：score(doc) = Σ 1 / (RRF_K + rank)，对两路排序结果按文档去重合并；
    两路都命中的文档天然排前，兼顾语义相关与关键词精确命中。

    vector_floor_ratio：融合结果保底策略（#8 改进）。RRF 纯融合时，BM25 关键词路
    可能挤掉向量路的高质量文档，导致"引用文件覆盖"回撤。ratio=r 时，最终 top-K
    中至少 round(K*r) 条来自向量路（按 RRF 序取向量路最优 N 条），
    BM25 独有文档最多占 K-N 条；r=0 时保持纯 RRF 行为（默认）。
    """

    def __init__(
        self,
        vector_retriever: IRetriever,
        bm25_retriever: BM25Retriever,
        top_k: int = 50,
        rrf_k: int = 60,
        topk_vector: int = 50,
        topk_bm25: int = 50,
        vector_floor_ratio: float = 0.0,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.topk_vector = topk_vector
        self.topk_bm25 = topk_bm25
        self.vector_floor_ratio = vector_floor_ratio

    @staticmethod
    def _doc_key(item: Dict[str, Any]) -> Any:
        """文档去重键：优先点 id，缺失时退回正文内容。"""
        if item.get("id") is not None:
            return ("id", item["id"])
        content = item.get("payload", {}).get("content", "")
        return ("content", content)

    def retrieve(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """双路召回 + RRF 融合，返回统一候选结构（按融合分降序）。

        Args:
            query: 查询文本
            query_filter: 透传给向量路（Qdrant 软过滤）
            top_k: 融合后返回条数

        Returns:
            [{"id", "score", "payload"}, ...]

        双腿召回量取各自配置值与 limit 的较大者，top_k 加深时候选池不会被
        topk_vector / topk_bm25 卡死（#8 调优发现：K=100/200 时混合路结果一致）。
        """
        limit = top_k if top_k is not None else self.top_k
        vector_results = self.vector_retriever.retrieve(
            query, query_filter=query_filter, top_k=max(self.topk_vector, limit)
        )
        bm25_results = self.bm25_retriever.retrieve(query, top_k=max(self.topk_bm25, limit))
        merged: Dict[Any, Dict[str, Any]] = {}
        for rank, item in enumerate(vector_results):
            key = self._doc_key(item)
            entry = merged.setdefault(
                key,
                {"id": item.get("id"), "score": 0.0, "payload": item.get("payload", {})},
            )
            entry["score"] += 1.0 / (self.rrf_k + rank + 1)
        for rank, item in enumerate(bm25_results):
            key = self._doc_key(item)
            entry = merged.setdefault(
                key,
                {"id": item.get("id"), "score": 0.0, "payload": item.get("payload", {})},
            )
            entry["score"] += 1.0 / (self.rrf_k + rank + 1)
        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        floor = min(limit, int(round(limit * self.vector_floor_ratio))) if self.vector_floor_ratio > 0 else 0
        if floor > 0:
            vector_keys = {self._doc_key(item) for item in vector_results}
            vec_docs = [d for d in results if self._doc_key(d) in vector_keys]
            if len(vec_docs) >= floor:
                # 保底：向量路最优 floor 条（按 RRF 序）置顶，其余按 RRF 序补足到 limit
                ordered = vec_docs[:floor]
                seen = {self._doc_key(d) for d in ordered}
                for d in results:
                    if self._doc_key(d) in seen:
                        continue
                    ordered.append(d)
                    if len(ordered) >= limit:
                        break
                return ordered[:limit]
        return results[:limit]
