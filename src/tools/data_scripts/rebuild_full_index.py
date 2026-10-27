# -*- coding: utf-8 -*-
"""tools/data_scripts/rebuild_full_index.py —— 全量语料索引重建（正式数据 附件5）

背景：线上集合 research_reports_v3 由旧 测试数据 语料（164 篇）构建，
与参考答案引用的正式数据语料（data_corpus/.../附件5：研报数据，473 篇）零重叠，
导致"引用可溯源"无法被检索层支撑。本脚本将正式数据语料重建为独立集合
（默认 research_reports_v3_full），不动线上集合，供检索评测与后续切换。

说明：
- 复用 data/loader、data/splitter、embeddings、vectorstore，与 build_index 同链路；
- 为控制成本/耗时，跳过逐块 LLM 摘要（qwen-turbo）；摘要缺失时引用展示回退正文前 200 字（pipelines 已兼容）；
- 向量化并发执行（--workers，默认 4；batch=25），实测 3 并发无限流限制；
- 边算边传：每累计 FLUSH_POINTS 条向量立即 upsert 并清空缓冲，避免全量向量常驻内存；
- 支持 --force 清空重建；非 force 且集合已有数据时直接跳过（防重复嵌入）；
- 分块参数 --chunk-size / --chunk-overlap / --min-chars（L5-A 碎片合并：-1=取配置默认、0=关闭）。

用法：
  python -m tools.data_scripts.rebuild_full_index [--collection research_reports_v3_full] [--workers 4] [--force]
  python -m tools.data_scripts.rebuild_full_index --collection research_reports_v3_l5a --chunk-size 1024 --min-chars 200 --force
  python -m tools.data_scripts.rebuild_full_index --collection research_reports_v3_structured --table-mode structured --strip-boilerplate --workers 4 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config.rag_config import RAGConfig
from data.loader import load_industry_documents, load_markdown_documents
from data.splitter import split_documents
from embeddings import EmbeddingClient
from vectorstore import QdrantClientWrapper

logger = logging.getLogger("rebuild_full_index")

#: 正式数据 附件5 语料根目录（参考答案引用来源）
CORPUS_ROOT = Path("data_corpus/全部数据/正式数据/附件5：研报数据")
#: 单请求嵌入条数（DashScope v2 上限 25）
EMBED_BATCH = 25
#: 内存缓冲上限：累计到该条数即上传并释放（500 × 1536 维约 25 MB）
FLUSH_POINTS = 500


def build_config(collection: str) -> RAGConfig:
    """构造指向正式数据语料的配置"""
    if not CORPUS_ROOT.is_dir():
        raise FileNotFoundError(f"正式数据语料不存在: {CORPUS_ROOT}")
    return RAGConfig(
        QDRANT_COLLECTION_NAME=collection,
        MARKDOWN_DIR=str(CORPUS_ROOT / "个股研报-解析结果-2.0"),
        EXCEL_METADATA_PATH=str(CORPUS_ROOT / "个股_研报信息.xlsx"),
        INDUSTRY_MARKDOWN_DIR=str(CORPUS_ROOT / "行业研报-解析结果-2.0"),
        INDUSTRY_EXCEL_PATH=str(CORPUS_ROOT / "行业_研报信息.xlsx"),
        EMBEDDING_BATCH_SIZE=EMBED_BATCH,
    )


def build_config_with_split(
    collection: str,
    chunk_size: int = 0,
    chunk_overlap: int = -1,
    min_chars: int = -1,
    table_mode: str = "legacy",
    table_slice_rows: int = -1,
    table_slice_max_chars: int = -1,
    strip_boilerplate: bool = False,
    row_fact_mode: str = "off",
) -> RAGConfig:
    """构造指向正式数据语料的配置，并可选覆盖分块参数。

    取配置默认：chunk_size=0 / overlap=-1 / min_chars=-1 / table_slice_rows=-1；
    ``table_mode``（``legacy`` / ``structured``）与 ``strip_boilerplate``（§3.2 锅炉板闸门）直通入库口径。
    """
    cfg = build_config(collection)
    if chunk_size > 0:
        cfg.CHUNK_SIZE = chunk_size
    if chunk_overlap >= 0:
        cfg.CHUNK_OVERLAP = chunk_overlap
    if min_chars >= 0:
        cfg.CHUNK_MIN_CHARS = min_chars
    cfg.TABLE_CHUNK_MODE = table_mode
    cfg.TABLE_ROW_FACT_MODE = row_fact_mode
    cfg.BOILERPLATE_STRIP_ENABLED = strip_boilerplate
    if table_slice_rows > 0:
        cfg.TABLE_SLICE_ROWS = table_slice_rows
    if table_slice_max_chars >= 0:
        cfg.TABLE_SLICE_MAX_CHARS = table_slice_max_chars
    return cfg


def main(argv=None) -> int:
    """重建入口"""
    parser = argparse.ArgumentParser(prog="rebuild_full_index", description="全量语料索引重建（正式数据 附件5）")
    parser.add_argument("--collection", default="research_reports_v3_full", help="目标集合名（默认 research_reports_v3_full）")
    parser.add_argument("--workers", type=int, default=4, help="并发嵌入线程数（默认 4）")
    parser.add_argument("--force", action="store_true", help="清空目标集合后重建")
    parser.add_argument("--resume", action="store_true", help="续跑：按 chunk_id 跳过集合中已存在的块")
    parser.add_argument("--chunk-size", type=int, default=0, help="分块大小（0=取配置默认 1024）")
    parser.add_argument("--chunk-overlap", type=int, default=-1, help="分块重叠字符数（-1=取配置默认；overlap 对比实验用）")
    parser.add_argument("--min-chars", type=int, default=-1, help="文本块最小长度（-1=取配置默认 200；0=关闭碎片合并，L5-A）")
    parser.add_argument("--table-mode", choices=["legacy", "structured"], default="legacy", help="表格入库口径（structured = 一张表一个/若干 chunk，B-60 重构）")
    parser.add_argument("--table-slice-rows", type=int, default=-1, help="结构化口径下单表切片行数上限（-1=取配置默认 40）")
    parser.add_argument("--table-slice-max-chars", type=int, default=-1, help="结构化口径下单表切片字符上限（-1=取配置默认 1200；0=不限）")
    parser.add_argument("--strip-boilerplate", action="store_true", help="剥离文档尾部声明/签名区（§3.2 锅炉板闸门）")
    parser.add_argument("--row-fact-mode", choices=["off", "chunks"], default="off", help="行级事实口径（chunks=生成 row_fact 子块，阶段 2/3）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = build_config_with_split(
        args.collection,
        args.chunk_size,
        args.chunk_overlap,
        args.min_chars,
        table_mode=args.table_mode,
        table_slice_rows=args.table_slice_rows,
        table_slice_max_chars=args.table_slice_max_chars,
        strip_boilerplate=args.strip_boilerplate,
        row_fact_mode=args.row_fact_mode,
    )
    logger.info(
        "分块参数：chunk_size=%d, chunk_overlap=%d, min_chars=%d, table_mode=%s, strip_boilerplate=%s, "
        "table_slice_rows=%d, table_slice_max_chars=%d, row_fact_mode=%s",
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
        config.CHUNK_MIN_CHARS,
        config.TABLE_CHUNK_MODE,
        config.BOILERPLATE_STRIP_ENABLED,
        config.TABLE_SLICE_ROWS,
        config.TABLE_SLICE_MAX_CHARS,
        config.TABLE_ROW_FACT_MODE,
    )
    qdrant = QdrantClientWrapper(config)

    if qdrant.count() > 0 and not args.force and not args.resume:
        logger.warning("集合 %s 已有 %d 个点，跳过（如需重建请加 --force）", args.collection, qdrant.count())
        return 0
    if args.force and qdrant.count() > 0:
        logger.warning("清空集合 %s（%d 个点）后重建", args.collection, qdrant.count())
        qdrant.clear_collection()

    t0 = time.time()
    stock_docs = load_markdown_documents(config.MARKDOWN_DIR, excel_metadata_path=config.EXCEL_METADATA_PATH)
    industry_docs = load_industry_documents(config.INDUSTRY_MARKDOWN_DIR, excel_metadata_path=config.INDUSTRY_EXCEL_PATH)
    all_docs = stock_docs + industry_docs
    logger.info("加载研报 %d 篇（个股 %d / 行业 %d）", len(all_docs), len(stock_docs), len(industry_docs))
    if not all_docs:
        logger.error("未加载到任何文档，退出")
        return 1

    chunks = split_documents(
        all_docs,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        min_chars=config.CHUNK_MIN_CHARS,
        table_mode=config.TABLE_CHUNK_MODE,
        table_slice_rows=config.TABLE_SLICE_ROWS,
        table_slice_max_chars=config.TABLE_SLICE_MAX_CHARS,
        strip_boilerplate=config.BOILERPLATE_STRIP_ENABLED,
        row_fact_mode=config.TABLE_ROW_FACT_MODE,
    )
    n_table = sum(1 for c in chunks if c["metadata"].get("chunk_type") == "table")
    n_row_fact = sum(1 for c in chunks if c["metadata"].get("chunk_type") == "row_fact")
    n_bp = sum(1 for c in chunks if c["metadata"].get("chunk_type") == "boilerplate")
    logger.info(
        "分块完成：%d 块（表格 %d / 行事实 %d / 锅炉板 %d / 正文 %d，耗时 %.1fs）",
        len(chunks),
        n_table,
        n_row_fact,
        n_bp,
        len(chunks) - n_table - n_row_fact - n_bp,
        time.time() - t0,
    )
    if not chunks:
        logger.error("分块结果为空，退出")
        return 1

    if args.resume and qdrant.count() > 0:
        seen = set()
        offset = None
        while True:
            points, offset = qdrant.client.scroll(
                collection_name=config.QDRANT_COLLECTION_NAME,
                limit=4096,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            for point in points:
                cid = (point.payload or {}).get("chunk_id")
                if cid:
                    seen.add(cid)
            if offset is None:
                break
        before = len(chunks)
        chunks = [c for c in chunks if c["metadata"].get("chunk_id") not in seen]
        logger.info("续跑：集合已有 %d 个 chunk_id，跳过 %d 块，剩余 %d 块", len(seen), before - len(chunks), len(chunks))
        if not chunks:
            logger.info("续跑：无剩余块，直接建 payload index")
            qdrant.ensure_payload_indexes({
                "chunk_type": "keyword",
                "table_id": "keyword",
                "chunk_id": "keyword",
                "source_file": "keyword",
                "row_index": "integer",
                "col_index": "integer",
                "metric": "keyword",
                "period": "keyword",
                "unit_source": "keyword",
                "entity": "keyword",
            })
            return 0

    embedding_client = EmbeddingClient(config)
    slices: List[List[Dict[str, Any]]] = [chunks[i : i + EMBED_BATCH] for i in range(0, len(chunks), EMBED_BATCH)]
    pending_vectors: List[List[float]] = []
    pending_payloads: List[Dict[str, Any]] = []
    uploaded = 0

    def embed_slice(sl: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        texts = [c["content"] for c in sl]
        emb = embedding_client.generate_embeddings(texts, text_type="document")
        pl = [{"content": c["content"], **c["metadata"]} for c in sl]
        return emb, pl

    done = 0
    last_report = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(embed_slice, sl): si for si, sl in enumerate(slices)}
        for fut in as_completed(futures):
            emb, pl = fut.result()
            pending_vectors.extend(emb)
            pending_payloads.extend(pl)
            done += 1
            if len(pending_vectors) >= FLUSH_POINTS:
                qdrant.insert_vectors(pending_vectors, pending_payloads)
                uploaded += len(pending_vectors)
                pending_vectors.clear()
                pending_payloads.clear()
            now = time.time()
            if now - last_report > 20:
                last_report = now
                n_done = done * EMBED_BATCH
                elapsed = now - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - n_done) / rate if rate > 0 else -1
                logger.info(
                    "进度 %d/%d 块（%.1f%%），%.1f 块/s，已上传 %d，已用 %.0fs，ETA %.0fs",
                    n_done, len(chunks), n_done * 100.0 / len(chunks), rate, uploaded, elapsed, eta,
                )

    if pending_vectors:
        qdrant.insert_vectors(pending_vectors, pending_payloads)
        uploaded += len(pending_vectors)
        pending_vectors.clear()
        pending_payloads.clear()
    if uploaded != len(chunks):
        logger.error("向量数量不一致：%d != %d（chunks），退出", uploaded, len(chunks))
        return 1
    logger.info("向量化 + 上传完成：%d 块（耗时 %.1fs）", uploaded, time.time() - t0)
    qdrant.ensure_payload_indexes({
        "chunk_type": "keyword",
        "table_id": "keyword",
        "chunk_id": "keyword",
        "source_file": "keyword",
        "row_index": "integer",
        "col_index": "integer",
        "metric": "keyword",
        "period": "keyword",
        "unit_source": "keyword",
        "entity": "keyword",
    })
    logger.info("索引构建完成：集合 %s，共 %d 个点，总耗时 %.1fs", args.collection, qdrant.count(), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
