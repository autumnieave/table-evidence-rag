# -*- coding: utf-8 -*-
"""多阶段检索（阶段 3）：表级粗召回 → 行级精定位 → 整表返回。

设计边界
--------
- **表级粗召回**只查 ``chunk_type=table``，避免 169,242 条 ``row_fact`` 把向量召回淹没
  （row_fact : table chunk ≈ 29.5 : 1）。
- **行级精定位**只查 ``chunk_type=row_fact``，并用候选 ``table_id`` 做硬过滤；
  行分只用于给候选表排序，不直接作为最终返回单元。
- **整表返回**：按 ``table_id`` 取回全部表切片（``#s{slice_index}``），按切片序拼接；
  一条结果 = 一张表，保留命中行 ``row_hits`` 作为证据。
- 不猜实体/期间：``entity`` / ``period`` 仅做**可选**后置过滤与打分，不做硬丢弃；
  调用方可传 ``known_entities`` / ``context_queries`` 启用查询理解与 ``clarify_query``，
  未传时保持旧行为（不因缺实体改变动作）。
- 所有中间计数写入 :class:`MultiStageTrace`，便于归因「表没召回 / 行没定位 / 精排丢了」。

返回结构与 ``IRetriever`` 一致：``[{"id", "score", "payload"}, ...]``；
额外在 payload 内带 ``row_hits`` / ``stage_scores`` / ``slice_count``。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.interfaces import IRetriever
from core.rerankers import apply_file_diversity, file_keys_from_candidates
from core.evidence_iteration import (
    RETRY_ENTITY,
    RETRY_TABLE,
    audit_evidence,
    hit_covers_entity,
)
from filters.row_match import (
    extract_query_constraints,
    metric_matches,
    period_matches,
    row_fact_matches,
)
from filters.query_understanding import understand_query
from filters.entity_gate import code_matches, name_matches, resolve_names


@dataclass
class MultiStageTrace:
    """一次多阶段检索的中间计数与耗时（毫秒）。"""

    coarse_hits: int = 0
    coarse_tables: int = 0
    row_candidates: int = 0
    row_hits: int = 0
    row_tables: int = 0
    bm25_hits: int = 0
    bm25_tables: int = 0
    bm25_enabled: bool = False
    bm25_gate: str = ""
    entity_retry_used: bool = False
    entity_targeted_hits: int = 0
    entity_targeted_tables: int = 0
    sibling_recall_enabled: bool = False
    sibling_candidates: int = 0
    sibling_replaced: int = 0
    sibling_files: List[str] = field(default_factory=list)
    sibling_table_ids: List[str] = field(default_factory=list)
    merged_tables: int = 0
    selected_tables: int = 0
    rerank_input: int = 0
    query_metrics: List[str] = field(default_factory=list)
    query_period: Optional[str] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    final_action: str = ""
    final_mismatch: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """转为可落盘的字典。"""
        return {
            "coarse_hits": self.coarse_hits,
            "coarse_tables": self.coarse_tables,
            "row_candidates": self.row_candidates,
            "row_hits": self.row_hits,
            "row_tables": self.row_tables,
            "bm25_hits": self.bm25_hits,
            "bm25_tables": self.bm25_tables,
            "bm25_enabled": self.bm25_enabled,
            "bm25_gate": self.bm25_gate,
            "entity_retry_used": self.entity_retry_used,
            "entity_targeted_hits": self.entity_targeted_hits,
            "entity_targeted_tables": self.entity_targeted_tables,
            "sibling_recall_enabled": self.sibling_recall_enabled,
            "sibling_candidates": self.sibling_candidates,
            "sibling_replaced": self.sibling_replaced,
            "sibling_files": list(self.sibling_files),
            "sibling_table_ids": list(self.sibling_table_ids),
            "merged_tables": self.merged_tables,
            "selected_tables": self.selected_tables,
            "rerank_input": self.rerank_input,
            "query_metrics": list(self.query_metrics),
            "query_period": self.query_period,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "iterations": list(self.iterations),
            "final_action": self.final_action,
            "final_mismatch": self.final_mismatch,
        }


def table_id_of(hit: Dict[str, Any]) -> str:
    """取候选的 table_id（缺失返回空串）。"""
    payload = (hit or {}).get("payload") or {}
    return str(payload.get("table_id") or payload.get("parent_id") or "")


_UNIT_HINTS: tuple[str, ...] = ("百万元", "亿元", "万元", "元/股", "%")
_FORECAST_TERMS: tuple[str, ...] = ("预计", "预测", "盈利预测", "展望", "forecast")


def _query_unit_hint(query: str) -> Optional[str]:
    """从问题里抽金额/百分比单位提示；只做字面匹配，不猜换算。"""
    text = str(query or "")
    for unit in _UNIT_HINTS:
        if unit in text:
            return unit
    # 裸“元”只接受“多少元 / 100元”这类单位位置，避免公司名里的“元”误判。
    if re.search(r"(?:多少|[\d.])\s*元(?![／/股])", text):
        return "元"
    return None


def _query_forecast_period(query: str) -> Optional[str]:
    """抽取预测年份（``2025E``）。

    ``extract_query_constraints`` 的强口径只认 ``2025Q3`` / ``2024A`` 这类
    报告期，不认预测年份；研报预测题（“2025年营业收入预计”）需要单独识别。
    """
    text = str(query or "")
    match = re.search(r"(20\d{2})\s*[Ee](?![A-Za-z0-9])", text)
    if match:
        return "{}E".format(match.group(1))
    if not any(term in text for term in _FORECAST_TERMS):
        return None
    match = re.search(r"(20\d{2})\s*年", text)
    return "{}E".format(match.group(1)) if match else None


def _unit_matches(left: Any, right: Any) -> bool:
    """单位是否同一口径；不做“元”包含“百万元”这类子串匹配。"""
    a = str(left or "").strip().replace("人民币", "").replace("RMB", "").upper()
    b = str(right or "").strip().replace("人民币", "").replace("RMB", "").upper()
    return bool(a and b and a == b)


def _anchor_entries(payload: Dict[str, Any]) -> List[tuple[str, str, str]]:
    """读取表 payload 的 (指标, 期间, 单位) 三元组。

    优先用 ``anchor_cells``（预测表每个年份/指标都有锚点）；旧索引没有锚点时，
    退回 ``metric_names`` + 表级 ``report_period`` / ``unit``。
    """
    payload = payload or {}
    entries: List[tuple[str, str, str]] = []
    for anchor in payload.get("anchor_cells") or []:
        if isinstance(anchor, dict):
            entries.append(
                (
                    str(anchor.get("metric") or ""),
                    str(anchor.get("period") or ""),
                    str(anchor.get("unit") or ""),
                )
            )
    if entries:
        return entries
    names: List[str] = [str(item) for item in (payload.get("metric_names") or [])]
    metric_name = str(payload.get("metric_name") or "")
    if metric_name:
        names.extend(part.strip() for part in metric_name.split(","))
    period = str(payload.get("report_period") or payload.get("period") or "")
    unit = str(payload.get("unit") or "")
    return [(name, period, unit) for name in names if name]


def _payload_covers(
    payload: Dict[str, Any],
    metrics: Sequence[str],
    period: Optional[str],
    unit: Optional[str],
) -> bool:
    """表 payload 是否覆盖查询的指标 / 期间 / 单位。"""
    if not metrics:
        return True
    for metric, anchor_period, anchor_unit in _anchor_entries(payload):
        if not any(metric_matches(metric, query_metric) for query_metric in metrics):
            continue
        if period and not period_matches(anchor_period, period):
            continue
        if unit and not _unit_matches(anchor_unit, unit):
            continue
        return True
    return False


def _sibling_score(
    payload: Dict[str, Any],
    metrics: Sequence[str],
    period: Optional[str],
    unit: Optional[str],
) -> Optional[float]:
    """同文件兄弟表的相关性打分；不满足硬约束返回 ``None``。"""
    if not _payload_covers(payload, metrics, period, unit):
        return None
    score = 0.0
    table_type = str((payload or {}).get("table_type") or "")
    if period and str(period).upper().endswith("E") and table_type == "forecast":
        score += 4.0
    elif period and table_type == "financial_statement":
        score += 3.0
    if unit:
        score += 2.0
    if period:
        score += 1.0
    content = str((payload or {}).get("content") or "")
    for metric in metrics:
        if metric and metric in content:
            score += 1.0
    if period and str(period).upper() in content.upper():
        score += 1.0
    return score


def _file_of(hit: Dict[str, Any]) -> str:
    """取候选的文件路径（缺失返回空串）。"""
    return str(((hit or {}).get("payload") or {}).get("file_path") or "")


def _payload_matches_entity(payload: Dict[str, Any], entities: Sequence[str]) -> bool:
    """表 payload 的 stockName / stockCode 是否命中点名实体。"""
    payload = payload or {}
    name = str(payload.get("stock_name") or payload.get("stockName") or "").strip()
    code = str(payload.get("stock_code") or payload.get("stockCode") or "").strip()
    for entity in entities:
        if name and name_matches(name, entity):
            return True
        if entity.isdigit() and code_matches(code, entity):
            return True
    return False


def dedupe_table_hits(hits: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """同一张表可能有多个切片；每个 table_id 只保留分数最高的切片。"""
    best: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        tid = table_id_of(hit)
        if not tid:
            continue
        score = float(hit.get("score") or 0.0)
        if tid not in best or score > float(best[tid].get("score") or 0.0):
            best[tid] = hit
    return best


def group_row_hits(hits: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 table_id 聚合行事实，组内按分数降序。"""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for hit in hits:
        payload = (hit or {}).get("payload") or {}
        tid = str(payload.get("table_id") or payload.get("parent_id") or "")
        if not tid:
            continue
        groups.setdefault(tid, []).append(hit)
    for rows in groups.values():
        rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return groups


def combine_score(coarse_score: float, rows: Sequence[Dict[str, Any]], coarse_weight: float = 0.3) -> float:
    """表级粗排分 + 行级最高分合成；没有行证据时保留粗排分的 ``coarse_weight``。"""
    if not rows:
        return float(coarse_score) * coarse_weight
    row_score = float(rows[0].get("score") or 0.0)
    return (1.0 - coarse_weight) * row_score + coarse_weight * float(coarse_score)


def rrf_fuse(
    *maps: Dict[str, Dict[str, Any]],
    rrf_k: int = 60,
) -> Dict[str, float]:
    """对多路召回按倒数排名融合（RRF）。

    Args:
        *maps: 每路 ``table_id -> hit``；各路分数量纲可以不同
        rrf_k: RRF 平滑常数

    Returns:
        ``table_id -> fusion_score``；未出现的路不贡献分数。
    """
    scores: Dict[str, float] = {}
    for mapping in maps:
        ordered = sorted(
            (mapping or {}),
            key=lambda tid: float((mapping or {}).get(tid, {}).get("score") or 0.0),
            reverse=True,
        )
        for rank, table_id in enumerate(ordered, 1):
            scores[table_id] = scores.get(table_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


class MultiStageRetriever(IRetriever):
    """表级粗召回 → 行事实精定位 → 整表返回。

    Args:
        embedding_client: ``EmbeddingClient``（查询向量）
        qdrant_client: ``QdrantClientWrapper``（含 ``.client`` 与 ``.config``）
        reranker: 可选 ``IReranker``；不传则按行/表合成分排序
        coarse_k: 表级粗召回条数（表切片，去重前）
        row_k: 行事实候选条数（限定候选表）
        rerank_pool: 每路进入精排的整表数（向量 / BM25 各取 top-N；
            无 BM25 时即向量 top-N）
        final_k: 最终返回表数
        max_per_file: 每文件最多保留几张表（0=不限）
        max_table_chars: 精排输入的表文本截断长度（返回仍为全文）
    """

    def __init__(
        self,
        embedding_client: Any,
        qdrant_client: Any,
        reranker: Optional[Any] = None,
        *,
        coarse_k: int = 100,
        row_k: int = 200,
        rerank_pool: int = 50,
        final_k: int = 10,
        max_per_file: int = 2,
        max_table_chars: int = 4000,
        coarse_weight: float = 0.3,
        known_entities: Optional[Sequence[str]] = None,
        bm25_retriever: Optional[Any] = None,
        bm25_k: int = 60,
        sibling_table_recall: bool = False,
    ):
        self.embedding_client = embedding_client
        self.qdrant = qdrant_client
        self.reranker = reranker
        self.coarse_k = coarse_k
        self.row_k = row_k
        self.rerank_pool = rerank_pool
        self.final_k = final_k
        self.max_per_file = max_per_file
        self.max_table_chars = max_table_chars
        self.coarse_weight = coarse_weight
        self.known_entities = known_entities
        self.bm25_retriever = bm25_retriever
        self.bm25_k = bm25_k
        self.sibling_table_recall = bool(sibling_table_recall)
        self.last_trace: Optional[MultiStageTrace] = None
        self._stock_doc_indices: Dict[str, List[int]] = {}
        self._code_doc_indices: Dict[str, List[int]] = {}
        self._file_doc_indices: Dict[str, List[int]] = {}
        self._table_docs: List[Dict[str, Any]] = []
        self._build_entity_index()
        self._build_file_index()

    # ---------- 基础检索 ----------
    def _collection(self) -> str:
        return str(self.qdrant.config.QDRANT_COLLECTION_NAME)

    def _build_entity_index(self) -> None:
        """按 ``stockName`` / ``stockCode`` 建表级倒排，供实体定向重试走子集。

        ``hit_covers_entity`` 只认这两个字段（不看正文），所以这里不需要扫全文；
        3,148 张表的元数据建索引在启动期是一次性开销。
        """
        docs = getattr(self.bm25_retriever, "docs", None) or []
        for idx, doc in enumerate(docs):
            payload = (doc or {}).get("payload") or {}
            name = str(payload.get("stockName") or "").strip()
            code = str(payload.get("stockCode") or "").strip()
            if name:
                self._stock_doc_indices.setdefault(name, []).append(idx)
            if code:
                self._code_doc_indices.setdefault(code, []).append(idx)

    def _build_file_index(self) -> None:
        """按 ``file_path`` / ``table_id`` 建表级倒排，供同文档兄弟表召回。

        只用表级 BM25 文档（每张表已按 ``table_id`` 合并全部切片），不额外查
        Qdrant；索引在启动期一次性建立，3,148 张表的开销可忽略。
        """
        docs = getattr(self.bm25_retriever, "docs", None) or []
        self._table_docs = list(docs)
        for idx, doc in enumerate(self._table_docs):
            payload = (doc or {}).get("payload") or {}
            file_path = str(payload.get("file_path") or "").strip()
            if file_path:
                self._file_doc_indices.setdefault(file_path, []).append(idx)

    def _entity_candidate_indices(self, names: Sequence[str]) -> Optional[List[int]]:
        """返回覆盖点名实体的文档下标；无实体索引时返回 ``None``（走全库）。"""
        if not self._stock_doc_indices and not self._code_doc_indices:
            return None
        chosen = set()
        for name in names:
            for resolved in resolve_names(name, self.known_entities):
                for payload_name, indices in self._stock_doc_indices.items():
                    if name_matches(payload_name, resolved):
                        chosen.update(indices)
            for payload_code, indices in self._code_doc_indices.items():
                if code_matches(payload_code, name):
                    chosen.update(indices)
        return sorted(chosen)

    def _score_candidates(
        self,
        ranked_ids: Sequence[str],
        coarse_map: Dict[str, Dict[str, Any]],
        bm25_map: Dict[str, Dict[str, Any]],
        row_groups: Dict[str, List[Dict[str, Any]]],
        *,
        multi_stage: bool,
    ) -> List[tuple[float, str]]:
        """按「向量腿优先 + BM25 归一 + 行证据加成」给候选表打分。"""
        bm25_max = max(
            (float(hit.get("score") or 0.0) for hit in bm25_map.values()),
            default=0.0,
        )
        scored: List[tuple[float, str]] = []
        for tid in ranked_ids:
            vector_score = float((coarse_map.get(tid) or {}).get("score") or 0.0)
            bm25_score = float((bm25_map.get(tid) or {}).get("score") or 0.0)
            if bm25_max > 0:
                bm25_score = bm25_score / bm25_max
            base_score = vector_score if tid in coarse_map else bm25_score
            score = (
                combine_score(base_score, row_groups.get(tid, []), self.coarse_weight)
                if multi_stage
                else base_score
            )
            scored.append((score, tid))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _fetch_for_output(
        self,
        ranked_ids: Sequence[str],
        coarse_map: Dict[str, Dict[str, Any]],
        bm25_map: Dict[str, Dict[str, Any]],
        row_groups: Dict[str, List[Dict[str, Any]]],
        *,
        multi_stage: bool,
        use_rerank: bool,
        k: int,
    ) -> Dict[str, Dict[str, Any]]:
        """取回最终要返回的整表。

        不精排时只取最终 top-k（精排才需要完整 ``rerank_pool``）。整表取回是
        scroll + ``table_id any``，50 张表的过滤成本远高于 10 张；而最终排序
        只依赖粗排分/行证据（都已在内存），因此先打分再取表不改变输出。
        """
        if use_rerank:
            fetch_ids = list(ranked_ids)
        else:
            scored = self._score_candidates(
                ranked_ids, coarse_map, bm25_map, row_groups, multi_stage=multi_stage
            )
            fetch_ids = [tid for _, tid in scored[:k]]
        return self._fetch_tables(fetch_ids)

    def _search(self, vector: List[float], query_filter: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        resp = self.qdrant.client.query_points(
            collection_name=self._collection(),
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return [
            {"id": p.id, "score": p.score, "payload": p.payload or {}}
            for p in resp.points
        ]

    @staticmethod
    def _merge_filter(query_filter: Optional[Any], chunk_type: str) -> Dict[str, Any]:
        """把 ``chunk_type`` 条件并入调用方 filter。

        Qdrant 的 ``Filter`` 对象与 dict 都支持；**不能原样返回 Filter 对象**，否则会
        静默丢掉本阶段的 ``chunk_type`` 条件，让表级召回混入 row_fact（或反之）。
        """
        condition = {"key": "chunk_type", "match": {"value": chunk_type}}
        if not query_filter:
            return {"must": [condition]}
        if isinstance(query_filter, dict):
            merged = dict(query_filter)
        else:
            dump = getattr(query_filter, "model_dump", None)
            if callable(dump):
                merged = dump(exclude_none=True)
            else:
                legacy_dump = getattr(query_filter, "dict", None)
                if not callable(legacy_dump):
                    raise TypeError("query_filter 必须是 dict 或 Qdrant Filter")
                merged = legacy_dump(exclude_none=True)
        must = merged.get("must") or []
        merged["must"] = ([must] if isinstance(must, dict) else list(must)) + [condition]
        return merged

    def _filter_row_hits(self, hits: Sequence[Dict[str, Any]], constraints: Any) -> List[Dict[str, Any]]:
        """按查询 metric/period 硬匹配过滤行事实；不匹配的行不进入候选。"""
        return [
            hit for hit in hits
            if row_fact_matches(hit.get("payload") or {}, constraints)
        ]

    def _row_filter(self, query_filter: Optional[Any], constraints: Any,
                    table_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """构建行事实检索 filter：额外带候选 table_id 和可确定的期间。"""
        merged = self._merge_filter(query_filter, "row_fact")
        must = merged.get("must") or []
        must = [must] if isinstance(must, dict) else list(must)
        if table_ids:
            must.append({"key": "table_id", "match": {"any": list(table_ids)}})
        if constraints.period:
            must.append({"key": "period", "match": {"value": constraints.period}})
        merged["must"] = must
        return merged

    def _fetch_tables(self, table_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """按 table_id 取回整表：拼接全部切片，按 ``table_slice_index`` 排序。"""
        if not table_ids:
            return {}
        flt = {
            "must": [
                {"key": "chunk_type", "match": {"value": "table"}},
                {"key": "table_id", "match": {"any": list(table_ids)}},
            ]
        }
        slices: Dict[str, List[Dict[str, Any]]] = {}
        offset = None
        while True:
            points, offset = self.qdrant.client.scroll(
                collection_name=self._collection(),
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=flt,
            )
            for p in points:
                payload = p.payload or {}
                tid = str(payload.get("table_id") or "")
                if tid:
                    slices.setdefault(tid, []).append({"id": p.id, "payload": payload})
            if offset is None:
                break

        assembled: Dict[str, Dict[str, Any]] = {}
        for tid, items in slices.items():
            items.sort(key=lambda x: (
                int((x["payload"] or {}).get("table_slice_index") or 0),
                int((x["payload"] or {}).get("chunk_index") or 0),
            ))
            content = "\n".join(str((x["payload"] or {}).get("content") or "") for x in items).strip()
            if not content:
                continue
            payload = dict(items[0]["payload"] or {})
            payload["content"] = content
            payload["slice_count"] = len(items)
            assembled[tid] = {"id": items[0]["id"], "payload": payload, "score": 0.0}
        return assembled

    def _rerank_document(self, table: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> str:
        """构造精排输入：命中行证据置顶，再拼整表正文（截断到 ``max_table_chars``）。

        表格正文往往以表头/元数据开头，关键数据行可能在 4000 字之后；只截正文会把
        行证据截掉。把 ``row_hits`` 的 content 放在前面，保证精排能看到“问题命中的
        那一行”，同时保留整表上下文。
        """
        content = str((table.get("payload") or {}).get("content") or "")
        row_text = "\n".join(
            str((row.get("payload") or {}).get("content") or "")
            for row in list(rows)[:5]
        ).strip()
        if row_text:
            return (row_text + "\n" + content)[: self.max_table_chars]
        return content[: self.max_table_chars]

    @staticmethod
    def _pick_replacement_index(
        selected: Sequence[Dict[str, Any]],
        sibling_file: str,
        used: set[int],
    ) -> Optional[int]:
        """从最低排名往上找可替换位置；优先不动兄弟表所在文件的现有表。"""
        for idx in range(len(selected) - 1, -1, -1):
            if idx in used:
                continue
            if _file_of(selected[idx]) != sibling_file:
                return idx
        for idx in range(len(selected) - 1, -1, -1):
            if idx not in used:
                return idx
        return None

    def _apply_sibling_recall(
        self,
        selected: Sequence[Dict[str, Any]],
        query: str,
        constraints: Any,
        trace: MultiStageTrace,
    ) -> List[Dict[str, Any]]:
        """同文档兄弟表召回：只替换最终 top-k，不追加、不扩大全局召回。

        触发条件（全部满足）：

        1. 开关打开；
        2. 查询能抽出指标 + 公司实体；
        3. 某个已选中的同主体文件里，所有已选表都不覆盖查询指标/期间/单位；
        4. 同文件存在满足指标/期间/单位硬约束的兄弟表。

        期间优先用查询硬约束（``2025Q3`` / ``2024A``）；研报预测题额外识别
        ``2025年...预计`` → ``2025E``。单位只做字面同口径匹配，不做换算。
        """
        trace.sibling_recall_enabled = True
        if not selected or self.bm25_retriever is None or not constraints.metrics:
            return list(selected)
        entities: Sequence[str] = ()
        if self.known_entities:
            entities = understand_query(query, known_entities=self.known_entities).entities
        if not entities:
            return list(selected)

        period = constraints.period or _query_forecast_period(query)
        unit = _query_unit_hint(query)
        selected_ids = {
            str((hit.get("payload") or {}).get("table_id") or "")
            for hit in selected
        }
        files: List[str] = []
        for hit in selected:
            file_path = _file_of(hit)
            if file_path and file_path not in files:
                files.append(file_path)

        replacements: List[tuple[int, str, Dict[str, Any], float]] = []
        used_indices: set[int] = set()
        considered = 0
        matched_files: List[str] = []
        for file_path in files:
            file_hits = [hit for hit in selected if _file_of(hit) == file_path]
            if not any(
                _payload_matches_entity(hit.get("payload") or {}, entities)
                for hit in file_hits
            ):
                continue
            if any(
                _payload_covers(
                    hit.get("payload") or {},
                    constraints.metrics,
                    period,
                    unit,
                )
                for hit in file_hits
            ):
                continue

            candidates: List[tuple[float, str, Dict[str, Any]]] = []
            for doc_idx in self._file_doc_indices.get(file_path, []):
                doc = self._table_docs[doc_idx]
                payload = (doc or {}).get("payload") or {}
                table_id = str(payload.get("table_id") or (doc or {}).get("id") or "")
                if not table_id or table_id in selected_ids:
                    continue
                if not _payload_matches_entity(payload, entities):
                    continue
                score = _sibling_score(payload, constraints.metrics, period, unit)
                if score is None:
                    continue
                candidates.append((score, table_id, doc))
            if not candidates:
                continue

            considered += len(candidates)
            matched_files.append(file_path)
            candidates.sort(key=lambda item: (-item[0], item[1]))
            score, table_id, doc = candidates[0]
            replacement_index = self._pick_replacement_index(
                selected, file_path, used_indices
            )
            if replacement_index is None:
                continue
            used_indices.add(replacement_index)
            replacements.append((replacement_index, table_id, doc, score))

        if not replacements:
            trace.sibling_candidates = considered
            return list(selected)

        output = list(selected)
        for idx, table_id, doc, score in sorted(replacements):
            replacement_score = float((selected[idx] or {}).get("score") or 0.0)
            result = self._as_result(table_id, doc, replacement_score, None, [])
            result["payload"]["sibling_recall"] = True
            result["payload"]["sibling_score"] = float(score)
            output[idx] = result
            selected_ids.add(table_id)
        trace.sibling_candidates = considered
        trace.sibling_replaced = len(replacements)
        trace.sibling_files = matched_files
        trace.sibling_table_ids = [item[1] for item in replacements]
        return output

    # ---------- 主流程 ----------
    def retrieve(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: Optional[int] = None,
        *,
        multi_stage: bool = True,
        mode: str = "table_first",
        query_vector: Optional[List[float]] = None,
        rerank: Optional[bool] = None,
        trace: Optional[MultiStageTrace] = None,
        coarse_k: Optional[int] = None,
        row_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """执行检索。

        Args:
            query: 用户问题
            query_filter: 额外 Qdrant 过滤条件（dict）
            top_k: 最终表数（默认 ``final_k``）
            multi_stage: False = 单阶段整表检索（对照）
            mode: ``table_first``（默认）/ ``row_first``（诊断：行事实先召回再回表）
            query_vector: 复用查询向量（评测缓存）
            rerank: 覆盖是否精排（None=有 reranker 就精排）
            trace: 外部传入的 trace 对象
            coarse_k: 覆盖本轮表级召回深度（迭代重试用）
            row_k: 覆盖本轮行级召回深度（迭代重试用）

        Returns:
            [{"id", "score", "payload"}]，一条 = 一张整表。
        """
        t_all = time.perf_counter()
        tr = trace or MultiStageTrace()
        tr.sibling_recall_enabled = self.sibling_table_recall
        k = top_k or self.final_k
        coarse_limit = coarse_k if coarse_k is not None else self.coarse_k
        row_limit = row_k if row_k is not None else self.row_k
        use_rerank = (self.reranker is not None) if rerank is None else bool(rerank)
        coarse_map: Dict[str, Dict[str, Any]] = {}
        bm25_map: Dict[str, Dict[str, Any]] = {}

        t0 = time.perf_counter()
        if query_vector is None:
            query_vector = self.embedding_client.generate_embeddings([query], text_type="query")[0]
        tr.timings_ms["embed"] = (time.perf_counter() - t0) * 1000
        constraints = extract_query_constraints(query)
        tr.query_metrics = list(constraints.metrics)
        tr.query_period = constraints.period

        # 第一阶段：表级粗召回 / 或诊断模式行事实先召回
        t0 = time.perf_counter()
        if multi_stage and mode == "row_first":
            row_hits = self._search(
                query_vector,
                self._row_filter(query_filter, constraints),
                row_limit,
            )
            tr.row_candidates = len(row_hits)
            row_hits = self._filter_row_hits(row_hits, constraints)
            tr.row_hits = len(row_hits)
            row_groups = group_row_hits(row_hits)
            tr.row_tables = len(row_groups)
            ranked_ids = sorted(
                row_groups,
                key=lambda tid: float(row_groups[tid][0].get("score") or 0.0),
                reverse=True,
            )[: self.rerank_pool]
            coarse_map: Dict[str, Dict[str, Any]] = {}
            tr.coarse_tables = len(ranked_ids)
            t_fetch = time.perf_counter()
            table_list = self._fetch_for_output(
                ranked_ids,
                coarse_map,
                bm25_map,
                row_groups,
                multi_stage=multi_stage,
                use_rerank=use_rerank,
                k=k,
            )
            tr.timings_ms["fetch"] = (time.perf_counter() - t_fetch) * 1000
        else:
            coarse_hits = self._search(query_vector, self._merge_filter(query_filter, "table"), coarse_limit)
            tr.coarse_hits = len(coarse_hits)
            coarse_map = dedupe_table_hits(coarse_hits)
            tr.coarse_tables = len(coarse_map)
            use_bm25 = self.bm25_retriever is not None and query_filter is None
            if use_bm25 and self.known_entities:
                understanding = understand_query(query, known_entities=self.known_entities)
                # 只在“实体 + 指标/期间”都可定位时开 BM25：
                # 行业级/无约束问题用 BM25 会引入大量同词不同主体的表（B2002/B2078 实测回退）。
                use_bm25 = bool(understanding.entities) and bool(
                    constraints.metrics or constraints.period
                )
                tr.bm25_gate = "entity+metric_or_period" if use_bm25 else "gate_off"
            elif use_bm25:
                tr.bm25_gate = "explicit_no_vocab"
            tr.bm25_enabled = use_bm25
            if use_bm25:
                t_bm25 = time.perf_counter()
                bm25_hits = self.bm25_retriever.retrieve(query, top_k=self.bm25_k)
                bm25_map = dedupe_table_hits(bm25_hits)
                tr.bm25_hits = len(bm25_hits)
                tr.bm25_tables = len(bm25_map)
                tr.timings_ms["bm25"] = (time.perf_counter() - t_bm25) * 1000
            row_groups: Dict[str, List[Dict[str, Any]]] = {}
            candidate_ids = list(dict.fromkeys([*coarse_map.keys(), *bm25_map.keys()]))
            tr.merged_tables = len(candidate_ids)
            if multi_stage and candidate_ids:
                row_hits = self._search(
                    query_vector,
                    self._row_filter(query_filter, constraints, candidate_ids),
                    row_limit,
                )
                tr.row_candidates = len(row_hits)
                row_hits = self._filter_row_hits(row_hits, constraints)
                tr.row_hits = len(row_hits)
                row_groups = group_row_hits(row_hits)
                tr.row_tables = len(row_groups)
            if bm25_map:
                # 两路各取 top-N 后并集，而不是按融合分直接截断：
                # 直接按 RRF 排序会让“BM25 中段”挤掉“向量 rank 30”这类目标表，
                # 与“K=50 全池精排”的诊断口径不一致。
                vector_ids = sorted(
                    coarse_map,
                    key=lambda tid: float(coarse_map[tid].get("score") or 0.0),
                    reverse=True,
                )[: self.rerank_pool]
                bm25_ids = sorted(
                    bm25_map,
                    key=lambda tid: float(bm25_map[tid].get("score") or 0.0),
                    reverse=True,
                )[: self.rerank_pool]
                ranked_ids = list(dict.fromkeys([*vector_ids, *bm25_ids]))
            elif multi_stage:
                ranked_ids = sorted(
                    coarse_map,
                    key=lambda tid: combine_score(
                        float(coarse_map[tid].get("score") or 0.0),
                        row_groups.get(tid, []),
                        self.coarse_weight,
                    ),
                    reverse=True,
                )[: self.rerank_pool]
            else:
                ranked_ids = sorted(
                    coarse_map,
                    key=lambda tid: float(coarse_map[tid].get("score") or 0.0),
                    reverse=True,
                )[: self.rerank_pool]
            t_fetch = time.perf_counter()
            table_list = self._fetch_for_output(
                ranked_ids,
                coarse_map,
                bm25_map,
                row_groups,
                multi_stage=multi_stage,
                use_rerank=use_rerank,
                k=k,
            )
            tr.timings_ms["fetch"] = (time.perf_counter() - t_fetch) * 1000
        tr.timings_ms["recall"] = (time.perf_counter() - t0) * 1000

        if not table_list:
            tr.selected_tables = 0
            tr.timings_ms["total"] = (time.perf_counter() - t_all) * 1000
            self.last_trace = tr
            return []

        # 第二阶段：整表精排（输入为整表文本，命中行只做证据）
        t0 = time.perf_counter()
        ordered_ids = [tid for tid in ranked_ids if tid in table_list]
        if not ordered_ids:
            ordered_ids = list(table_list.keys())
        coarse_lookup: Dict[str, Dict[str, Any]] = dict(bm25_map)
        coarse_lookup.update(coarse_map)
        docs = (
            [
                self._rerank_document(table_list[tid], row_groups.get(tid, []))
                for tid in ordered_ids
            ]
            if use_rerank
            else []
        )
        selected: List[Dict[str, Any]] = []
        if use_rerank and docs:
            reranked = self.reranker.rerank(query=query, documents=docs, top_n=len(docs))
            tr.rerank_input = len(docs)
            keys = [str((table_list[tid]["payload"] or {}).get("file_path") or "") for tid in ordered_ids]
            picked = apply_file_diversity(reranked, keys, k, self.max_per_file)
            for item in picked[:k]:
                idx = item.get("index")
                if not isinstance(idx, int) or not (0 <= idx < len(ordered_ids)):
                    continue
                tid = ordered_ids[idx]
                table = table_list[tid]
                score = float(item.get("relevance_score") or 0.0)
                selected.append(self._as_result(tid, table, score, coarse_lookup.get(tid), row_groups.get(tid, [])))
        else:
            scored = self._score_candidates(
                ordered_ids, coarse_map, bm25_map, row_groups, multi_stage=multi_stage
            )
            for score, tid in scored[:k]:
                table = table_list[tid]
                rows = row_groups.get(tid, [])
                coarse = coarse_lookup.get(tid) or table
                selected.append(self._as_result(tid, table, score, coarse, rows))
        if self.sibling_table_recall:
            selected = self._apply_sibling_recall(selected, query, constraints, tr)
        tr.timings_ms["rerank"] = (time.perf_counter() - t0) * 1000
        tr.selected_tables = len(selected)
        tr.timings_ms["total"] = (time.perf_counter() - t_all) * 1000
        self.last_trace = tr
        return selected

    @staticmethod
    def _merge_hits(
        primary: Sequence[Dict[str, Any]],
        secondary: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """合并两批整表 hits，按 table_id 去重并保留 primary 优先。"""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for hit in [*(primary or []), *(secondary or [])]:
            payload = (hit or {}).get("payload") or {}
            table_id = str(payload.get("table_id") or (hit or {}).get("id") or "")
            if not table_id or table_id in seen:
                continue
            seen.add(table_id)
            merged.append(hit)
            if len(merged) >= top_k:
                break
        return merged

    def _bm25_entity_hits(
        self,
        entities: Sequence[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """用内存 BM25 索引按点名主体取表；不查 Qdrant，不做全局扩召回。

        先保证每个缺失主体至少一张表，再用其余命中补足 top_k。BM25 文档本身
        带整表正文，可直接作为整表候选返回。
        """
        if self.bm25_retriever is None:
            return []
        names = [str(item or "").strip() for item in entities if str(item or "").strip()]
        if not names:
            return []
        pool_k = max(int(top_k), int(self.bm25_k), 50)
        # 只对“元数据覆盖点名主体”的表打分：BM25 的 df/avgdl 仍按全库统计，
        # 因此分值与全库扫描一致；但候选从 3,148 张降到几十张，实体重试从
        # ~300ms 降到 ~20ms。索引缺失时退回全库，保证行为不变。
        candidate_indices = self._entity_candidate_indices(names)
        hits = self.bm25_retriever.retrieve(
            " ".join(names),
            top_k=pool_k,
            candidate_indices=candidate_indices,
        )
        selected: List[Dict[str, Any]] = []
        seen = set()

        def _add(hit: Dict[str, Any]) -> None:
            payload = (hit or {}).get("payload") or {}
            table_id = str(payload.get("table_id") or (hit or {}).get("id") or "")
            if not table_id or table_id in seen:
                return
            seen.add(table_id)
            selected.append(hit)

        # 第一轮：每个缺失主体至少取一张，避免 BM25 排序把某一主体挤掉。
        for name in names:
            for hit in hits:
                if hit_covers_entity(hit, name, self.known_entities):
                    _add(hit)
                    break
        # 第二轮：补足其余命中。
        for hit in hits:
            if any(hit_covers_entity(hit, name, self.known_entities) for name in names):
                _add(hit)
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    def retrieve_with_iteration(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: Optional[int] = None,
        *,
        max_rounds: int = 2,
        context_queries: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """带证据审计的多阶段检索：最多重试 ``max_rounds`` 轮。

        首轮使用当前 ``coarse_k/row_k``。审计返回 ``retry_entity`` 时，只按缺失主体
        用内存 BM25 定向取表并合并，不扩大全局 ``coarse_k``；返回 ``retry_table`` 时才
        按决策扩大召回深度。``route_to_sql`` / ``reject_with_reason`` 不再重试，只把
        决策写入 trace 和每条结果的 payload，交给上层 Agent/Router 处理。

        Args:
            query: 用户问题
            query_filter: 额外 Qdrant 过滤条件
            top_k: 最终表数
            max_rounds: 最大重试轮数（不含首轮；默认 2）
            context_queries: 同一会话更早的子问题；用于补全实体/指标/期间，
                并让“涨跌的原因是什么”这类追问能复用前文主体
            **kwargs: 透传给 :meth:`retrieve`（multi_stage/mode/query_vector/rerank 等）

        Returns:
            最后一轮的整表结果；payload 内额外带 ``evidence_decision``。
        """
        if max_rounds < 0:
            raise ValueError("max_rounds 不能为负数")
        t_all = time.perf_counter()
        decisions: List[Dict[str, Any]] = []
        last_hits: List[Dict[str, Any]] = []
        last_trace: Optional[MultiStageTrace] = None
        coarse_limit = self.coarse_k
        row_limit = self.row_k
        context = [str(item or "") for item in (context_queries or []) if str(item or "").strip()]
        effective_query = " ".join([query, *context]).strip() if context else query
        constraints = extract_query_constraints(effective_query)
        entity_retry_used = False
        entity_hits: List[Dict[str, Any]] = []

        for round_no in range(max_rounds + 1):
            tr = MultiStageTrace()
            last_hits = self.retrieve(
                effective_query,
                query_filter=query_filter,
                top_k=top_k,
                trace=tr,
                coarse_k=coarse_limit,
                row_k=row_limit,
                **kwargs,
            )
            if entity_hits:
                last_hits = self._merge_hits(entity_hits, last_hits, top_k or self.final_k)
                if self.sibling_table_recall:
                    last_hits = self._apply_sibling_recall(
                        last_hits, effective_query, constraints, tr
                    )
                tr.entity_retry_used = True
                tr.entity_targeted_hits = len(entity_hits)
                tr.entity_targeted_tables = len({
                    str((hit.get("payload") or {}).get("table_id") or hit.get("id") or "")
                    for hit in entity_hits
                })
            decision = audit_evidence(
                query,
                last_hits,
                round_no=round_no,
                max_rounds=max_rounds,
                coarse_k=coarse_limit,
                row_k=row_limit,
                known_entities=self.known_entities,
                context_queries=context,
                entity_retry_used=entity_retry_used,
            )
            decisions.append({"round": round_no, **decision.as_dict()})
            last_trace = tr
            if decision.action == RETRY_ENTITY and decision.missing_entities:
                if self.bm25_retriever is None:
                    # 无 BM25 的部署回退旧行为：扩大表级召回。
                    coarse_limit = min(max(coarse_limit * 2, coarse_limit + 50), 400)
                    row_limit = min(max(row_limit * 2, row_limit + 50), 400)
                    continue
                t_entity = time.perf_counter()
                entity_hits = self._bm25_entity_hits(
                    decision.missing_entities,
                    top_k or self.final_k,
                )
                tr.timings_ms["entity_retry"] = (time.perf_counter() - t_entity) * 1000
                entity_retry_used = True
                if entity_hits:
                    last_hits = self._merge_hits(entity_hits, last_hits, top_k or self.final_k)
                    if self.sibling_table_recall:
                        last_hits = self._apply_sibling_recall(
                            last_hits, effective_query, constraints, tr
                        )
                    tr.entity_retry_used = True
                    tr.entity_targeted_hits = len(entity_hits)
                    tr.entity_targeted_tables = len({
                        str((hit.get("payload") or {}).get("table_id") or hit.get("id") or "")
                        for hit in entity_hits
                    })
                decision = audit_evidence(
                    query,
                    last_hits,
                    round_no=round_no + 1,
                    max_rounds=max_rounds,
                    coarse_k=coarse_limit,
                    row_k=row_limit,
                    known_entities=self.known_entities,
                    context_queries=context,
                    entity_retry_used=True,
                )
                decisions.append({"round": round_no + 1, **decision.as_dict()})
            if decision.action != RETRY_TABLE:
                break
            coarse_limit = int(decision.next_coarse_k or coarse_limit)
            row_limit = int(decision.next_row_k or row_limit)

        if last_trace is not None:
            last_trace.iterations = list(decisions)
            last_trace.final_action = str(decisions[-1].get("action") or "") if decisions else ""
            last_trace.final_mismatch = str(decisions[-1].get("mismatch_type") or "") if decisions else ""
            last_trace.timings_ms["iteration_total"] = (time.perf_counter() - t_all) * 1000
            self.last_trace = last_trace
        final = decisions[-1] if decisions else {}
        for hit in last_hits:
            payload = hit.setdefault("payload", {})
            payload["evidence_decision"] = final
        return last_hits

    def _as_result(
        self,
        table_id: str,
        table: Dict[str, Any],
        score: float,
        coarse: Optional[Dict[str, Any]],
        rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """组装单条结果（整表全文 + 命中行证据）。"""
        payload = dict(table.get("payload") or {})
        # 整表回取来自切片 payload；表级 metadata（指标/期间等）可能只挂在粗召回命中上。
        # 这些字段是同一张表的属性，缺失时从 coarse 继承，避免证据审计误判 schema。
        coarse_payload = (coarse or {}).get("payload") or {}
        for key in ("metric_names", "metric_name", "report_period", "anchor_cells"):
            if not payload.get(key) and coarse_payload.get(key):
                payload[key] = coarse_payload[key]
        payload["table_id"] = table_id
        row_hits = []
        for r in list(rows)[:5]:
            row_payload = r.get("payload") or {}
            anchors = row_payload.get("anchor_cells") or []
            anchor = anchors[0] if anchors and isinstance(anchors[0], dict) else None
            anchor = anchor or {}
            row_hits.append(
                {
                    "chunk_id": row_payload.get("chunk_id"),
                    "content": row_payload.get("content"),
                    "score": r.get("score"),
                    "metric": row_payload.get("metric") or anchor.get("metric"),
                    "period": row_payload.get("period") or anchor.get("period"),
                    "unit": row_payload.get("unit") or anchor.get("unit"),
                    "unit_source": row_payload.get("unit_source") or anchor.get("unit_source"),
                    "anchor": anchor or None,
                }
            )
        payload["row_hits"] = row_hits
        payload["stage_scores"] = {
            "coarse": float((coarse or {}).get("score") or 0.0),
            "row": float(rows[0].get("score") or 0.0) if rows else 0.0,
            "final": float(score),
        }
        return {"id": table.get("id"), "score": float(score), "payload": payload}
