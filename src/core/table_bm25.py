# -*- coding: utf-8 -*-
"""表级 BM25 索引：把 ``chunk_type=table`` 的切片按 ``table_id`` 合成整表后建索引。

用途
----
向量召回对“数值密集、措辞与问题差异大”的表格块经常失效（B2006/B2080|1）。
本模块只提供**表级 BM25 腿**，由 :class:`core.multi_stage.MultiStageRetriever`
与向量候选做并集/归一化排序（不做 RRF 直接截断）；不替代向量检索，也不改变
最终整表返回口径。

边界
----
- 只读 Qdrant，不写索引、不调用 LLM；
- 按 ``table_id`` 合并全部切片，避免同一张表的多个切片稀释 BM25 词频；
- 调用方负责决定是否启用（评测/离线可开，线上默认不启用）。
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


def scroll_table_docs(
    qdrant_client: Any,
    collection_name: str,
    *,
    batch_size: int = 512,
) -> List[Dict[str, Any]]:
    """滚动读取全部表格切片，按 ``table_id`` 合并为整表文档。

    Args:
        qdrant_client: ``QdrantClientWrapper``（含 ``.client``）
        collection_name: Qdrant 集合名
        batch_size: 单次 scroll 条数

    Returns:
        ``[{"id": table_id, "payload": {...content...}}, ...]``，每个 table_id 一条。
    """
    client = getattr(qdrant_client, "client", qdrant_client)
    flt = {"must": [{"key": "chunk_type", "match": {"value": "table"}}]}
    grouped: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    offset: Optional[Any] = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=flt,
        )
        for point in points:
            payload = dict(point.payload or {})
            table_id = str(payload.get("table_id") or "")
            if not table_id:
                continue
            grouped.setdefault(table_id, []).append(
                (
                    int(payload.get("table_slice_index") or 0),
                    int(payload.get("chunk_index") or 0),
                    payload,
                )
            )
        if offset is None:
            break

    docs: List[Dict[str, Any]] = []
    for table_id, items in grouped.items():
        items.sort(key=lambda item: (item[0], item[1]))
        payload = dict(items[0][2])
        # 表级元数据可能只挂在某个切片上（例如 stock_name 只在首片或尾片）；
        # 按“首个非空值”补齐，避免合并后丢元数据。
        for _, _, extra in items[1:]:
            for key, value in extra.items():
                if value not in (None, "", [], {}) and not payload.get(key):
                    payload[key] = value
        content = "\n".join(str(item[2].get("content") or "") for item in items).strip()
        if not content:
            continue
        payload["content"] = content
        payload["table_id"] = table_id
        payload["slice_count"] = len(items)
        docs.append({"id": table_id, "payload": payload})
    return docs


def build_table_bm25(
    qdrant_client: Any,
    collection_name: str,
    *,
    batch_size: int = 512,
) -> BM25Retriever:
    """构建表级 BM25 检索器。"""
    return BM25Retriever(scroll_table_docs(qdrant_client, collection_name, batch_size=batch_size))


def table_bm25_cache_target(
    cache_path: str,
    collection_name: str,
    count: int,
) -> Path:
    """返回表级 BM25 缓存文件路径（按集合名 + 点数失效）。"""
    path = Path(cache_path)
    return path.with_name(f"{path.stem}.{collection_name}.{count}.pkl")


def build_table_bm25_cached(
    qdrant_client: Any,
    collection_name: str,
    *,
    cache_path: str = "database/table_bm25_index.pkl",
    force: bool = False,
    batch_size: int = 512,
) -> BM25Retriever:
    """构建表级 BM25，并按“集合名 + 点数”落盘缓存。

    启动时优先加载缓存，避免每次冷启动都滚动全库；点数变化（增量插入/重建）
    会自动换缓存文件。缓存读写失败不吞掉：由调用方决定是否降级并记录
    ``bm25_unavailable_fallback``。

    Args:
        qdrant_client: ``QdrantClientWrapper``（含 ``count`` / ``client``）
        collection_name: Qdrant 集合名
        cache_path: 缓存文件基名
        force: 强制重建
        batch_size: scroll 批大小

    Returns:
        ``BM25Retriever``
    """
    raw_client = getattr(qdrant_client, "client", qdrant_client)
    count_fn = getattr(qdrant_client, "count", None)
    if callable(count_fn):
        count = int(count_fn())
    else:
        count = int(raw_client.count(collection_name=collection_name).count)
    target = table_bm25_cache_target(cache_path, collection_name, count)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not force and target.exists():
        with open(target, "rb") as fh:
            retriever = pickle.load(fh)
        logger.info("表级 BM25 从缓存加载：%s", target.name)
        return retriever
    docs = scroll_table_docs(qdrant_client, collection_name, batch_size=batch_size)
    if not docs:
        raise RuntimeError("没有可用的表格切片，无法构建表级 BM25 索引")
    retriever = BM25Retriever(docs=docs)
    with open(target, "wb") as fh:
        pickle.dump(retriever, fh)
    logger.info("表级 BM25 构建完成：%d 张表 → %s", len(docs), target.name)
    return retriever
