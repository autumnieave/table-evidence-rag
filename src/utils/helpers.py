import time
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from prompts.pipeline import IMAGE_DETECT_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE


def _should_aggregate_table(query: str) -> bool:
    """根据查询关键词判断是否需要将表格行合并为完整表格"""
    keywords = ["趋势", "变化", "对比", "历年", "过去", "逐年", "增长趋势", "下降趋势", "近三年", "近五年"]
    query_lower = query.lower()
    return True  # if any(kw in query_lower for kw in keywords) else False


def _aggregate_parent_table(
    qdrant_client,
    collection_name: str,
    search_results: List[Dict[str, Any]],
    candidate_docs: List[str],
    max_table_len: int = 25000
) -> Tuple[List[str], List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    聚合检索结果中的表格行：命中任意一行，则拉取整个表格。
    返回：新的文档内容列表、新的元数据列表（用于后续可能追溯）
    """
    table_metadata = {}
    parent_ids = set()
    row_indices_map = {}
    for i, res in enumerate(search_results):
        payload = res["payload"]
        if payload.get("is_table_row") and payload.get("parent_id"):
            pid = payload["parent_id"]
            parent_ids.add(pid)
            if pid not in row_indices_map:
                row_indices_map[pid] = []
            row_indices_map[pid].append(i)

    if not parent_ids:
        return candidate_docs, search_results, table_metadata

    aggregated_tables = {}
    for pid in parent_ids:
        points, _ = qdrant_client.client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_client.models.Filter(
                must=[
                    qdrant_client.models.FieldCondition(
                        key="parent_id",
                        match=qdrant_client.models.MatchValue(value=pid)
                    )
                ]
            ),
            limit=200,
            with_payload=True,
            with_vectors=False
        )
        if not points:
            continue
        sorted_points = sorted(points, key=lambda p: p.payload.get("row_index", 0))
        table_text = "\n".join([p.payload["content"] for p in sorted_points])
        if len(table_text) > max_table_len:
            table_text = table_text[:max_table_len] + "\n...(表格内容过长，已截断)"
        aggregated_tables[pid] = table_text
        sources = list(set(p.payload.get("source", "") for p in points))
        table_metadata[pid] = {
            "paper_path": "; ".join(sources) if sources else "聚合表格/多源",
        }

    new_candidate_docs = []
    used_parents = set()
    aggregated_meta = {}
    for i, doc in enumerate(candidate_docs):
        payload = search_results[i]["payload"]
        pid = payload.get("parent_id")
        if pid and pid in aggregated_tables:
            if pid not in used_parents:
                new_idx = len(new_candidate_docs)
                new_candidate_docs.append(aggregated_tables[pid])
                aggregated_meta[new_idx] = table_metadata[pid]
                used_parents.add(pid)
        else:
            new_candidate_docs.append(doc)

    return new_candidate_docs, search_results, aggregated_meta


def _generate_summaries(llm_client, texts: List[str], batch_size: int = 20) -> List[str]:
    """
    批量生成文本摘要。对于空文本或过短文本，直接返回原文前100字。
    使用 qwen-turbo 模型以降低成本。
    """
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_summaries = []
        for text in batch:
            if not text or len(text) < 100:
                batch_summaries.append(text[:100] + ("..." if len(text) > 100 else ""))
                continue

            prompt = SUMMARY_PROMPT_TEMPLATE.format(text=text)

            try:
                response = llm_client.chat.completions.create(
                    model="qwen-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=80
                )
                summary = response.choices[0].message.content.strip()
                batch_summaries.append(summary)
            except Exception as e:
                batch_summaries.append(text[:100] + "...")
        summaries.extend(batch_summaries)
        time.sleep(0.5)
    return summaries


def _extract_image_title_with_llm(llm_client, text: str) -> str:
    """
    使用 LLM 判断文本块中是否包含图片，并返回图片的标题/说明。
    如果没有图片或无法提取，返回空字符串。
    """
    if '![]' not in text:
        return ""

    prompt = IMAGE_DETECT_PROMPT_TEMPLATE.format(text=text[:4000])
    try:
        response = llm_client.chat.completions.create(
            model="qwen3-max",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000
        )
        title = response.choices[0].message.content.strip()
        if title == "无图片" or not title:
            return ""
        return title
    except Exception:
        return ""


def _build_reference_for_doc(
    idx: int,
    search_results: List[Dict],
    candidate_docs: List[str],
    aggregated_meta: Dict[int, Dict[str, Any]] = None,
    llm_client=None
) -> Dict[str, str]:
    """根据候选文档索引构建引用条目，提取图表标题作为 paper_image"""
    if aggregated_meta is None:
        aggregated_meta = {}

    if idx in aggregated_meta:
        meta = aggregated_meta[idx]
        paper_path = meta.get("paper_path", "聚合表格/多源")
        full_text = candidate_docs[idx]
        summary_text = '这是一个表格'
    elif idx < len(search_results):
        payload = search_results[idx]["payload"]
        paper_path = payload.get("source", "")
        full_text = candidate_docs[idx] if idx < len(candidate_docs) else payload.get("content", "")
        summary_text = payload.get("summary")
        if not summary_text:
            summary_text = full_text[:200]
    else:
        paper_path = "未知来源"
        full_text = candidate_docs[idx] if idx < len(candidate_docs) else ""
        summary_text = full_text[:200]

    paper_image = ""
    chart_pattern = r'图表\s*\d+\s*[：:]\s*[^\n]+'
    match = re.search(chart_pattern, full_text)
    if match:
        paper_image = match.group(0).strip()

    if not paper_image and llm_client is not None:
        paper_image = _extract_image_title_with_llm(llm_client, full_text)

    return {
        "paper_path": paper_path,
        "text": summary_text,
        "paper_image": paper_image
    }
