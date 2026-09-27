"""vectorstore/qdrant_wrapper.py"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from config.rag_config import RAGConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class QdrantClientWrapper:
    """
    Qdrant向量数据库客户端封装

    支持本地持久化存储和远程服务两种模式
    """

    def __init__(self, config: RAGConfig):
        """
        初始化Qdrant客户端

        Args:
            config: RAG配置对象
        """
        self.config = config
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            self.models = models
            self.QdrantClient = QdrantClient
        except ImportError:
            logger.error("请先安装qdrant-client: pip install qdrant-client")
            raise


        # 远程服务模式
        logger.info(f"连接远程Qdrant服务: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        self.client = self.QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=10.0)

        # 确保集合存在
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """检查并创建集合"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.config.QDRANT_COLLECTION_NAME not in collection_names:
            logger.info(f"创建Qdrant集合: {self.config.QDRANT_COLLECTION_NAME}")

            # 距离度量映射
            distance_map = {
                "COSINE": self.models.Distance.COSINE,
                "DOT": self.models.Distance.DOT,
                "EUCLIDEAN": self.models.Distance.EUCLID,
                "MANHATTAN": self.models.Distance.MANHATTAN
            }
            distance = distance_map.get(
                self.config.DISTANCE_METRIC.upper(),
                self.models.Distance.COSINE
            )

            self.client.create_collection(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                vectors_config=self.models.VectorParams(
                    size=self.config.VECTOR_DIMENSION,
                    distance=distance
                )
            )
            logger.info("集合创建成功")
        else:
            logger.info(f"集合 {self.config.QDRANT_COLLECTION_NAME} 已存在")

    def insert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[int]] = None
    ) -> None:
        """
        批量插入向量及对应的元数据

        Args:
            vectors: 向量列表
            payloads: 每个向量对应的payload（元数据与文本内容）
            ids: 自定义ID列表，若不提供则自动生成
        """
        if ids is None:
            # 获取当前最大ID
            try:
                count_resp = self.client.count(collection_name=self.config.QDRANT_COLLECTION_NAME)
                start_id = count_resp.count
            except:
                start_id = 0
            ids = list(range(start_id, start_id + len(vectors)))

        points = [
            self.models.PointStruct(
                id=idx,
                vector=vec,
                payload=payload
            )
            for idx, vec, payload in zip(ids, vectors, payloads)
        ]

        # 分批上传（避免单次请求过大）
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                points=batch,
                wait=True
            )
            logger.info(f"已插入向量批次 {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

        logger.info(f"成功插入 {len(vectors)} 个向量")

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Any] = None   # 新增参数
    ) -> List[Dict[str, Any]]:
        """
        检索与查询向量最相似的Top-K个点

        Args:
            query_vector: 查询向量
            limit: 返回结果数量

        Returns:
            检索结果列表，每项包含id、score、payload
        """
        results = self.client.query_points(
            collection_name=self.config.QDRANT_COLLECTION_NAME,
            query=query_vector,  # 参数名从 query_vector 变为 query
            limit=limit,
            with_payload=True,
            query_filter=query_filter
        )

        formatted_results = []
        for res in results.points:
            formatted_results.append({
                "id": res.id,
                "score": res.score,
                "payload": res.payload
            })

        logger.info(f"检索返回 {len(formatted_results)} 个结果")
        return formatted_results

    def clear_collection(self):
        """清空集合数据（用于重建索引）"""
        self.client.delete_collection(self.config.QDRANT_COLLECTION_NAME)
        self._ensure_collection_exists()
        logger.info("集合已清空并重建")

    def count(self) -> int:
        """返回集合当前文档块总数（用于 BM25 索引一致性校验）"""
        return self.client.count(collection_name=self.config.QDRANT_COLLECTION_NAME).count

    def scroll_all(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """滚动读取集合全部点（id + payload），用于构建 BM25 等非向量索引。

        Args:
            batch_size: 每次滚动拉取的点数

        Returns:
            [{"id": ..., "payload": {...}}, ...]
        """
        results: List[Dict[str, Any]] = []
        offset: Optional[Any] = None
        while True:
            resp = self.client.scroll(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points = resp.points if hasattr(resp, "points") else resp[0]
            next_offset = resp.next_page_offset if hasattr(resp, "next_page_offset") else resp[1]
            for point in points:
                results.append({"id": point.id, "payload": point.payload})
            if next_offset is None or not points:
                break
            offset = next_offset
        logger.info("scroll_all 拉取全部点：%d 个", len(results))
        return results

# ==================== 元数据过滤模块 ====================
