# -*- coding: utf-8 -*-
"""24 题构造型表级盲测：Reranker 精排 + Metadata 前置过滤复算。

输入
----
- ``训练结果数据/evidence_20261103/blind_set_v1.json``：题目与表级 GT；
- ``训练结果数据/evidence_20261103/blind_recall_mrr.json``：原始 Top-50 表 ID；
- Qdrant ``research_reports_v6_rowfacts``：按 ``table_id`` 取整表正文。

两组实验
--------
1. **Reranker**：对原始 Top-50 用 ``BAAI/bge-reranker-v2-m3`` 重排，算 MRR@10 /
   Hit Rate@10；Top-50 集合不变，所以 Recall@50 不变。
2. **Metadata 前置过滤**：从问题解析公司实体与期间（如 ``2024年`` → ``2024A``），
   用 Qdrant filter 只在 ``stock_name`` / ``anchor_cells.period`` 命中的表池内重跑
   向量召回，算 Recall@50 / MRR@10。另跑“仅实体过滤”作为消融。

边界
----
- 查询向量复用项目现有 DashScope embedding（只编码 24 条 query，不调用 LLM 生成）；
  首次运行会把向量缓存到 ``训练结果数据/evidence_20261103/query_vectors.json``。
- 本脚本只做检索/重排，不生成答案。
- 24 题是构造型盲测，不是真实用户分布；结果只作方向性参考。

用法::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m eval.rag_retrieval_optimize
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Nested,
    NestedCondition,
)

from config.rag_config import get_config
from core.table_bm25 import scroll_table_docs
from embeddings.client import EmbeddingClient
from filters.query_understanding import understand_query


DEFAULT_BLIND = ROOT / "训练结果数据" / "evidence_20261103" / "blind_set_v1.json"
DEFAULT_BASELINE = ROOT / "训练结果数据" / "evidence_20261103" / "blind_recall_mrr.json"
DEFAULT_OUT = ROOT / "训练结果数据" / "evidence_20261103" / "retrieval_optimization.json"
DEFAULT_QUERY_CACHE = ROOT / "训练结果数据" / "evidence_20261103" / "query_vectors.json"
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def _stdout_utf8() -> None:
    """Windows 控制台统一 UTF-8，避免中文表名/题面乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def parse_period(query: str) -> Optional[str]:
    """从问题抽“2024年/2027年”并判断 A/E；抽不到返回 None。"""
    match = re.search(r"(20\d{2})\s*年", str(query or ""))
    if not match:
        return None
    year = match.group(1)
    forecast = any(
        token in str(query)
        for token in ("预测", "预计", "forecast", "展望", "有望")
    )
    return f"{year}{'E' if forecast else 'A'}"


def metrics_at_k(ranked_ids: Sequence[str], gt_ids: Sequence[str], k: int = 10) -> Dict[str, Any]:
    """算 Recall@50 / Recall@10 / MRR@10；返回首个 GT 排名。"""
    gt = {str(item) for item in gt_ids}
    first_rank: Optional[int] = None
    for index, table_id in enumerate(ranked_ids, start=1):
        if table_id in gt:
            first_rank = index
            break
    recall50 = 1.0 if any(table_id in gt for table_id in ranked_ids[:50]) else 0.0
    recall10 = 1.0 if any(table_id in gt for table_id in ranked_ids[:10]) else 0.0
    mrr10 = (1.0 / first_rank) if first_rank and first_rank <= k else 0.0
    return {
        "recall50": round(recall50, 4),
        "recall10": round(recall10, 4),
        "mrr10": round(mrr10, 4),
        "first_rank": first_rank,
        "n_returned": len(ranked_ids),
    }


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _dedupe_points(points: Sequence[Any], limit: int = 50) -> List[Dict[str, Any]]:
    """Qdrant 返回的是表切片；按 table_id 去重，保留最高分切片。"""
    best: Dict[str, Dict[str, Any]] = {}
    for point in points:
        payload = dict(point.payload or {})
        table_id = str(payload.get("table_id") or payload.get("parent_id") or "")
        if not table_id:
            continue
        score = float(point.score or 0.0)
        current = best.get(table_id)
        if current is None or score > current["score"]:
            best[table_id] = {
                "table_id": table_id,
                "score": score,
                "payload": payload,
            }
    ranked = sorted(best.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def _load_query_vectors(
    blind_items: Sequence[Dict[str, Any]],
    cache_path: Path,
    *,
    offline: bool = False,
) -> Dict[str, List[float]]:
    """读取/生成 24 条查询向量；已缓存的同题直接复用。"""
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        try:
            raw = _load_json(cache_path)
            if isinstance(raw, dict):
                cache = raw
        except Exception:
            cache = {}

    missing = [
        item for item in blind_items
        if not (
            item["qid"] in cache
            and cache[item["qid"]].get("question") == item["question"]
            and cache[item["qid"]].get("vector")
        )
    ]
    if missing:
        if offline:
            raise RuntimeError(
                f"query cache missing {len(missing)} 条且 --offline；"
                f"请先联网生成 {cache_path}"
            )
        client = EmbeddingClient(get_config())
        vectors = client.generate_embeddings(
            [item["question"] for item in missing],
            text_type="query",
        )
        for item, vector in zip(missing, vectors):
            cache[item["qid"]] = {
                "question": item["question"],
                "vector": vector,
            }
        _save_json(cache_path, cache)
    return {qid: list(payload["vector"]) for qid, payload in cache.items()}


def _reranker(model_path: str, max_length: int, batch_size: int) -> Dict[str, Any]:
    """懒加载 BGE reranker；CPU 推理，返回可调用的打分函数。"""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少可选依赖 torch/transformers") from exc

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    def score(query: str, texts: Sequence[str]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(texts), batch_size):
            batch = [str(text or "")[:4000] for text in texts[start:start + batch_size]]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits.view(-1)
            scores.extend(float(value) for value in logits)
        return scores

    return {"score": score, "model": model, "tokenizer": tokenizer}


def _rerank_ids(
    query: str,
    ids: Sequence[str],
    docs: Dict[str, Dict[str, Any]],
    reranker: Dict[str, Any],
    *,
    limit: int,
) -> Tuple[List[str], List[float], float]:
    """对候选表重排；缺失正文的候选保留原序并给低分。"""
    pairs: List[Tuple[str, str]] = []
    for table_id in ids[:limit]:
        payload = docs.get(table_id) or {}
        pairs.append((table_id, str(payload.get("content") or "")))
    started = time.perf_counter()
    scores = reranker["score"](query, [text for _, text in pairs])
    elapsed_ms = (time.perf_counter() - started) * 1000
    scored = list(zip([table_id for table_id, _ in pairs], scores))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in scored], [float(item[1]) for item in scored], elapsed_ms


def _filtered_search(
    client: QdrantClient,
    collection: str,
    query_vector: Sequence[float],
    query_filter: Filter,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    response = client.query_points(
        collection_name=collection,
        query=list(query_vector),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return _dedupe_points(response.points, limit=50)


def _pool_size(
    docs: Dict[str, Dict[str, Any]],
    *,
    stock: str,
    period: Optional[str] = None,
) -> int:
    count = 0
    for payload in docs.values():
        if str(payload.get("stock_name") or "") != stock:
            continue
        if period and not any(
            str(cell.get("period") or "") == period
            for cell in (payload.get("anchor_cells") or [])
        ):
            continue
        count += 1
    return count


def main() -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="RAG 检索层重排 + 元数据过滤复算")
    parser.add_argument("--blind-set", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--collection", default="research_reports_v6_rowfacts")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rerank-limit", type=int, default=50)
    parser.add_argument("--search-limit", type=int, default=400)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--offline", action="store_true", help="只允许读 query 向量缓存")
    args = parser.parse_args()

    blind = _load_json(args.blind_set)
    baseline = _load_json(args.baseline)
    items = list(blind.get("items") or [])
    baseline_by_qid = {row["qid"]: row for row in baseline.get("rows") or []}
    if len(items) != len(baseline_by_qid):
        raise RuntimeError("blind_set 与 baseline 行数不一致")

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    table_docs = scroll_table_docs(client, args.collection)
    docs = {str(doc["id"]): dict(doc["payload"] or {}) for doc in table_docs}
    stock_names = sorted({
        str(payload.get("stock_name") or "")
        for payload in docs.values()
        if payload.get("stock_name")
    })
    print(f"tables={len(docs)} stocks={len(stock_names)}")

    result: Dict[str, Any] = {
        "meta": {
            "blind_set": str(args.blind_set),
            "baseline": str(args.baseline),
            "collection": args.collection,
            "n": len(items),
            "model": args.model,
            "max_length": args.max_length,
            "note": "构造型表级盲测；只做检索/重排，不生成答案",
        },
        "baseline": {},
        "rerank": None,
        "entity_filter": None,
        "entity_period_filter": None,
        "rerank_entity_period": None,
    }

    per_question: List[Dict[str, Any]] = []
    for item in items:
        row = baseline_by_qid[item["qid"]]
        ids = [str(value) for value in (row.get("base_ids") or [])]
        per_question.append({
            "qid": item["qid"],
            "question": item["question"],
            "stock": item["stock"],
            "period": item["period"],
            "gt_tables": item["gt_tables"],
            "baseline_ids": ids,
            "baseline": metrics_at_k(ids, item["gt_tables"]),
        })

    result["baseline"] = {
        "recall50": _mean([row["baseline"]["recall50"] for row in per_question]),
        "recall10": _mean([row["baseline"]["recall10"] for row in per_question]),
        "mrr10": _mean([row["baseline"]["mrr10"] for row in per_question]),
    }
    print("baseline", result["baseline"])

    reranker: Optional[Dict[str, Any]] = None
    if not args.skip_rerank:
        reranker = _reranker(args.model, args.max_length, args.batch_size)
        rerank_rows: List[Dict[str, Any]] = []
        for row in per_question:
            ranked, scores, elapsed_ms = _rerank_ids(
                row["question"], row["baseline_ids"], docs, reranker,
                limit=args.rerank_limit,
            )
            metrics = metrics_at_k(ranked, row["gt_tables"])
            metrics["rerank_ms"] = round(elapsed_ms, 2)
            rerank_rows.append({
                "qid": row["qid"],
                "ranked_ids": ranked,
                "scores": [round(value, 4) for value in scores],
                **metrics,
            })
        result["rerank"] = {
            "recall50": _mean([row["recall50"] for row in rerank_rows]),
            "recall10": _mean([row["recall10"] for row in rerank_rows]),
            "mrr10": _mean([row["mrr10"] for row in rerank_rows]),
            "mean_rerank_ms": _mean([row["rerank_ms"] for row in rerank_rows]),
            "per_question": rerank_rows,
        }
        print("rerank", {key: result["rerank"][key] for key in ("recall50", "recall10", "mrr10", "mean_rerank_ms")})

    if not args.skip_filter:
        vectors = _load_query_vectors(items, args.query_cache, offline=args.offline)
        entity_rows: List[Dict[str, Any]] = []
        period_rows: List[Dict[str, Any]] = []
        combined_rows: List[Dict[str, Any]] = []
        for row in per_question:
            item = next(value for value in items if value["qid"] == row["qid"])
            understanding = understand_query(item["question"], known_entities=stock_names)
            extracted = list(understanding.entities)
            stock = extracted[0] if extracted else item["stock"]
            period = parse_period(item["question"])
            entity_filter = Filter(must=[
                FieldCondition(key="chunk_type", match=MatchValue(value="table")),
                FieldCondition(key="stock_name", match=MatchValue(value=stock)),
            ])
            entity_hits = _filtered_search(
                client, args.collection, vectors[row["qid"]], entity_filter,
                limit=args.search_limit,
            )
            entity_ids = [hit["table_id"] for hit in entity_hits]
            entity_metrics = metrics_at_k(entity_ids, row["gt_tables"])
            entity_rows.append({
                "qid": row["qid"],
                "extracted_entities": extracted,
                "stock_used": stock,
                "period_parsed": period,
                "pool_tables": _pool_size(docs, stock=stock),
                "ids": entity_ids,
                **entity_metrics,
            })

            period_filter = Filter(must=[
                FieldCondition(key="chunk_type", match=MatchValue(value="table")),
                FieldCondition(key="stock_name", match=MatchValue(value=stock)),
                NestedCondition(nested=Nested(
                    key="anchor_cells",
                    filter=Filter(must=[
                        FieldCondition(key="period", match=MatchValue(value=period)),
                    ]),
                )),
            ])
            period_hits = _filtered_search(
                client, args.collection, vectors[row["qid"]], period_filter,
                limit=args.search_limit,
            )
            period_ids = [hit["table_id"] for hit in period_hits]
            period_metrics = metrics_at_k(period_ids, row["gt_tables"])
            period_rows.append({
                "qid": row["qid"],
                "extracted_entities": extracted,
                "stock_used": stock,
                "period_parsed": period,
                "pool_tables": _pool_size(docs, stock=stock, period=period),
                "ids": period_ids,
                **period_metrics,
            })

            if reranker is not None:
                ranked, scores, elapsed_ms = _rerank_ids(
                    row["question"], period_ids, docs, reranker,
                    limit=args.rerank_limit,
                )
                combined_metrics = metrics_at_k(ranked, row["gt_tables"])
                combined_metrics["rerank_ms"] = round(elapsed_ms, 2)
                combined_rows.append({
                    "qid": row["qid"],
                    "ranked_ids": ranked,
                    "scores": [round(value, 4) for value in scores],
                    **combined_metrics,
                })

        result["entity_filter"] = {
            "recall50": _mean([row["recall50"] for row in entity_rows]),
            "recall10": _mean([row["recall10"] for row in entity_rows]),
            "mrr10": _mean([row["mrr10"] for row in entity_rows]),
            "mean_pool_tables": _mean([row["pool_tables"] for row in entity_rows]),
            "per_question": entity_rows,
        }
        result["entity_period_filter"] = {
            "recall50": _mean([row["recall50"] for row in period_rows]),
            "recall10": _mean([row["recall10"] for row in period_rows]),
            "mrr10": _mean([row["mrr10"] for row in period_rows]),
            "mean_pool_tables": _mean([row["pool_tables"] for row in period_rows]),
            "per_question": period_rows,
        }
        print("entity_filter", {key: result["entity_filter"][key] for key in ("recall50", "recall10", "mrr10", "mean_pool_tables")})
        print("entity_period_filter", {key: result["entity_period_filter"][key] for key in ("recall50", "recall10", "mrr10", "mean_pool_tables")})
        if combined_rows:
            result["rerank_entity_period"] = {
                "recall50": _mean([row["recall50"] for row in combined_rows]),
                "recall10": _mean([row["recall10"] for row in combined_rows]),
                "mrr10": _mean([row["mrr10"] for row in combined_rows]),
                "mean_rerank_ms": _mean([row["rerank_ms"] for row in combined_rows]),
                "per_question": combined_rows,
            }
            print("rerank_entity_period", {key: result["rerank_entity_period"][key] for key in ("recall50", "recall10", "mrr10", "mean_rerank_ms")})

    result["per_question"] = per_question
    _save_json(args.out, result)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
