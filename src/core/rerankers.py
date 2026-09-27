"""精排接口适配层（2026-08-22 链路收敛 #3）

RerankClient 已与 IReranker 结构化兼容；适配层用于统一 query() 的依赖边界，
后续可替换为其他精排实现（如 Cohere Rerank）而不改调用方。
"""
import os
from typing import Any, Dict, List, Optional

from core.interfaces import IReranker


class RerankerAdapter(IReranker):
    """将 RerankClient 包装为 IReranker 接口"""

    def __init__(self, rerank_client: Any):
        self.rerank_client = rerank_client

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """委托 RerankClient（top_n=None 时由其按配置兜底）"""
        return self.rerank_client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
        )


def apply_file_diversity(
    reranked: List[Dict[str, Any]],
    file_keys: List[str],
    top_n: int,
    max_per_file: int,
) -> List[Dict[str, Any]]:
    """按精排分降序逐条选择，同一文件最多保留 max_per_file 条，直到凑满 top_n。

    用于缓解精排把上下文压缩到少数文件导致的引用文件/数字覆盖下降；
    max_per_file <= 0 时退化为取前 top_n 条。
    """
    if max_per_file <= 0:
        return reranked[:top_n]
    counts: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []
    for item in reranked:
        idx = item.get("index")
        key = file_keys[idx] if (idx is not None and 0 <= idx < len(file_keys)) else f"__idx{idx}"
        if counts.get(key, 0) >= max_per_file:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(item)
        if len(selected) >= top_n:
            break
    return selected


def file_keys_from_candidates(candidates: List[Dict[str, Any]]) -> List[str]:
    """提取候选片段的文件键（basename；缺失时退回空串，使每条独立成组）。"""
    return [os.path.basename(str(r.get("payload", {}).get("file_path", "") or "")) for r in candidates]
