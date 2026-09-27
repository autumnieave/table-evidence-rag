"""核心抽象接口 —— 检索 / 精排 / 生成（2026-08-22 链路收敛 #3）

背景：项目曾有三条问答链路（手写 / LangChain 检索器 / LCEL）靠开关切换，
各链路内部直接调用具体类，改动容易漏。抽公共接口后，RAGPipeline.query() 只依赖本模块：

- IRetriever：问题 → 候选文档（统一返回 ["{"score", "payload"}", ...] 结构）
- IReranker：问题 + 候选文档 → 精排结果（[{index, relevance_score, document?}, ...]）
- IGenerator：问题 + 上下文 + 历史 → 答案文本（支持流式）

约定：手写链路为默认实现，LangChain 检索器 / LCEL 为实验链路（见 docs/ARCHITECTURE.md 差距清单 #3）。
"""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class IRetriever(Protocol):
    """检索接口：将用户问题转换为候选文档列表"""

    def retrieve(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """检索候选文档，返回 [{"score": float, "payload": {"content": str, ...}}, ...]"""
        ...


@runtime_checkable
class IReranker(Protocol):
    """精排接口：对候选文档按相关性重排序"""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """返回 [{"index": int, "relevance_score": float, "document": str, ...}, ...]"""
        ...


@runtime_checkable
class IGenerator(Protocol):
    """生成接口：基于上下文与历史生成回答"""

    def generate(
        self,
        query: str,
        contexts: List[str],
        history: Optional[List[Dict]] = None,
        stream: Optional[bool] = None,
    ) -> str:
        """生成回答文本（stream=True 时逐字输出）"""
        ...
