# -*- coding: utf-8 -*-
"""多阶段检索运行时：表级 BM25 常驻 + 词表自动加载 + 启动预热。

该模块把评测阶段验证过的 ``MultiStageRetriever`` 组装成可复用的运行时，
供 API/CLI/评测共用，避免三处各自加载 BM25 与公司词表：

- **表级 BM25 常驻**：优先读 ``TABLE_BM25_INDEX_PATH`` 缓存；缓存缺失时滚动
  ``chunk_type=table`` 建索引并落盘。不可用时记录 ``bm25_unavailable_fallback``，
  由 ``MultiStageRetriever`` 回退旧的扩召回逻辑，不静默失败。
- **词表自动加载**：优先读 Qdrant 实际语料的 ``distinct_stock_names``（与集合自洽），
  失败再降级到 ``EXCEL_METADATA_PATH``；不需要调用方手动传 ``known_entities``。
- **启动预热**：启动时跑一次轻量检索，把 Qdrant 连接、表级/行级查询与 BM25 索引
  预热到热态，避免首个业务请求承担冷启动。

注意：本模块只提供检索运行时，不改变 ``RAGPipeline.query`` 的默认链路；
是否把多阶段检索接入线上由 ``MULTI_STAGE_RETRIEVAL_ENABLED`` 与后续路由改造决定。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.multi_stage import MultiStageRetriever
from core.table_bm25 import build_table_bm25_cached
from data.loader import load_stock_vocabulary
from embeddings import EmbeddingClient
from vectorstore.qdrant_wrapper import QdrantClientWrapper

logger = logging.getLogger(__name__)

BM25_FALLBACK_TOKEN = "bm25_unavailable_fallback"


@dataclass
class MultiStageRuntime:
    """多阶段检索运行时的装配结果。"""

    config: Any
    qdrant: Any
    embedding: Any
    retriever: MultiStageRetriever
    known_entities: Tuple[str, ...]
    vocab_source: str
    bm25_status: str
    bm25_load_ms: float = 0.0
    warmup_ms: float = 0.0
    warmup_status: str = "not_run"

    def retrieve(
        self,
        query: str,
        *,
        context_queries: Sequence[str] = (),
        top_k: int = 10,
        max_rounds: int = 2,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """执行一次查询：自动 embedding + 带审计/实体定向重试的检索。"""
        retrieval_query = " ".join(
            [query, *[str(item) for item in context_queries if str(item).strip()]]
        ).strip()
        vector = self.embedding.generate_embeddings([retrieval_query], text_type="query")[0]
        return self.retriever.retrieve_with_iteration(
            query,
            query_vector=vector,
            top_k=top_k,
            max_rounds=max_rounds,
            rerank=False,
            context_queries=context_queries,
            **kwargs,
        )

    def warmup(
        self,
        query: str = "贵州茅台 营业收入",
        *,
        rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """用真实检索形态预热 Qdrant / BM25 / 行证据 / 实体定向链路。

        轻量 ``top_k=1,row_k=10`` 预热不到最终链路：线上首查会走
        ``row_k=200`` 的行事实过滤 + 整表取回 + 实体 BM25。这里按真实形态跑
        多轮（通用 / 实体 / 短查询三种形态），并显式预热一次实体 BM25，避免
        首请求承担 Qdrant/BM25 的冷页。查询向量只算一次并在轮次间复用，避免
        预热本身消耗多次 embedding 调用。
        """
        t0 = time.perf_counter()
        try:
            queries = (query, "华润三九 白云山 收益率", "预热")
            vectors = [
                self.embedding.generate_embeddings([item], text_type="query")[0]
                for item in queries
            ]
            repeat = int(
                rounds
                if rounds is not None
                else getattr(self.config, "MULTI_STAGE_WARMUP_ROUNDS", 3)
            )
            for _ in range(max(1, repeat)):
                for item, vector in zip(queries, vectors):
                    self.retriever.retrieve(
                        item,
                        query_vector=vector,
                        top_k=10,
                        rerank=False,
                        coarse_k=100,
                        row_k=200,
                    )
            bm25 = getattr(self.retriever, "bm25_retriever", None)
            if bm25 is not None:
                bm25.retrieve("贵州茅台", top_k=60)
            self.warmup_status = "ok"
        except Exception as exc:  # noqa: BLE001
            self.warmup_status = "failed"
            logger.warning("multi_stage_warmup_failed query=%s error=%s", query, exc)
        self.warmup_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": self.warmup_status,
            "warmup_ms": self.warmup_ms,
            "bm25_status": self.bm25_status,
            "vocab_source": self.vocab_source,
            "known_entities": len(self.known_entities),
        }

    def as_dict(self) -> Dict[str, Any]:
        """返回可落盘/上报的运行时状态。"""
        return {
            "collection": getattr(self.config, "QDRANT_COLLECTION_NAME", ""),
            "known_entities": len(self.known_entities),
            "vocab_source": self.vocab_source,
            "bm25_status": self.bm25_status,
            "bm25_load_ms": self.bm25_load_ms,
            "warmup_ms": self.warmup_ms,
            "warmup_status": self.warmup_status,
        }


def load_known_entities(
    config: Any,
    qdrant: Any,
) -> Tuple[Tuple[str, ...], str]:
    """自动加载公司/指数简称词表。

    优先 Qdrant 实际语料（``distinct_stock_names``），失败降级到元数据 Excel。
    返回 ``(entities, source)``，source 为 ``qdrant`` / ``excel`` / ``none``。
    """
    try:
        names = sorted(str(item).strip() for item in qdrant.distinct_stock_names() if str(item).strip())
        if names:
            return tuple(names), "qdrant"
    except Exception as exc:  # noqa: BLE001
        logger.warning("词表从 Qdrant 加载失败，降级到元数据 Excel：%s", exc)
    try:
        names = sorted(str(item).strip() for item in load_stock_vocabulary(config.EXCEL_METADATA_PATH) if str(item).strip())
        if names:
            return tuple(names), "excel"
    except Exception as exc:  # noqa: BLE001
        logger.warning("词表从 Excel 加载失败（实体审计关闭）：%s", exc)
    logger.warning("known_entities 为空：多阶段检索将不启用实体审计/定向重试")
    return (), "none"


def build_multi_stage_runtime(
    config: Any,
    *,
    qdrant_client: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
    reranker: Optional[Any] = None,
    known_entities: Optional[Sequence[str]] = None,
    use_bm25: bool = True,
) -> MultiStageRuntime:
    """组装多阶段检索运行时（BM25 常驻 + 词表自动加载）。"""
    qdrant = qdrant_client or QdrantClientWrapper(config)
    embedding = embedding_client or EmbeddingClient(config)
    if known_entities is None:
        entities, vocab_source = load_known_entities(config, qdrant)
    else:
        entities = tuple(str(item).strip() for item in known_entities if str(item).strip())
        vocab_source = "explicit"

    bm25 = None
    bm25_status = "disabled"
    bm25_load_ms = 0.0
    if use_bm25:
        t0 = time.perf_counter()
        try:
            bm25 = build_table_bm25_cached(
                qdrant,
                str(getattr(config, "QDRANT_COLLECTION_NAME", "")),
                cache_path=str(
                    getattr(config, "TABLE_BM25_INDEX_PATH", "database/table_bm25_index.pkl")
                ),
            )
            bm25_status = "ready"
        except Exception as exc:  # noqa: BLE001
            bm25_status = "unavailable"
            logger.warning(
                "%s collection=%s error=%s",
                BM25_FALLBACK_TOKEN,
                getattr(config, "QDRANT_COLLECTION_NAME", ""),
                exc,
            )
        bm25_load_ms = round((time.perf_counter() - t0) * 1000, 2)

    retriever = MultiStageRetriever(
        embedding,
        qdrant,
        reranker=reranker,
        bm25_retriever=bm25,
        known_entities=list(entities),
        coarse_k=100,
        row_k=200,
        rerank_pool=50,
        final_k=10,
        sibling_table_recall=bool(
            getattr(config, "SIBLING_TABLE_RECALL_ENABLED", False)
        ),
    )
    return MultiStageRuntime(
        config=config,
        qdrant=qdrant,
        embedding=embedding,
        retriever=retriever,
        known_entities=entities,
        vocab_source=vocab_source,
        bm25_status=bm25_status,
        bm25_load_ms=bm25_load_ms,
    )
