# -*- coding: utf-8 -*-
"""流式输出首字延迟（TTFT）验证脚本。

目的：在**不改动** ``RAGPipeline.query()`` 同步逻辑的前提下，用项目当前的
``LLMGenerator``（qwen3.5-plus）对同一道 A 类题、同一组生成上下文分别跑
非流式与流式，比较：

- 同步：总耗时（完整回答返回）；
- 流式：首字延迟（第一个非空 delta）+ 总耗时（流结束）。

用法（项目根目录）::

    .\.venv\Scripts\python.exe eval\test_stream_ttft.py --qid "B2007|2" --runs 3
    .\.venv\Scripts\python.exe eval\test_stream_ttft.py --dry-run

说明：上下文从 ``训练结果数据/stage5_measure/faithfulness_a_regenerated.json``
读取该题的 ``evidence_tables``，再按 ``table_id`` 从 ``data_corpus`` 原文重新解析，
因此不依赖 Qdrant / Embedding 服务；生成本身仍走线上同一个 DashScope 客户端。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.rag_config import RAGConfig  # noqa: E402
from core.evidence_context import build_evidence_context  # noqa: E402
from data.table_struct import scan_tables  # noqa: E402
from llm.generator import LLMGenerator  # noqa: E402

DEFAULT_CASE_JSON = os.path.join(
    ROOT, "训练结果数据", "stage5_measure", "faithfulness_a_regenerated.json"
)
DEFAULT_OUT_DIR = os.path.join(ROOT, "训练结果数据", "stage5_measure")


def _load_case(path: str, qid: str) -> Dict[str, Any]:
    """从既有 A 类重生成结果中取出题目 + 证据表清单。"""
    with io.open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("per_question", []):
        if item.get("qid") == qid:
            return item
    raise SystemExit("未找到 qid={!r}；可选：{}".format(
        qid, ", ".join(x.get("qid", "") for x in data.get("per_question", []))
    ))


def _corpus_root() -> str:
    """定位研报语料根目录（正式数据优先）。"""
    candidates = [
        os.path.join(ROOT, "data_corpus", "全部数据", "正式数据", "附件5：研报数据"),
        os.path.join(ROOT, "测试数据", "附件5：研报数据"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise SystemExit("未找到研报语料目录，请检查 data_corpus/测试数据")


def _index_markdown(root: str) -> Dict[str, str]:
    """建立 basename -> 绝对路径 索引（473 篇量级，一次即可）。"""
    index: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".md"):
                index.setdefault(name, os.path.join(dirpath, name))
    return index


def build_contexts(case: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """按 evidence_tables 的 table_id 从原文重建生成上下文。

    Returns:
        ``(contexts, stats)``；stats 用于核对每张表的重建长度。
    """
    root = _corpus_root()
    file_index = _index_markdown(root)
    parsed_cache: Dict[str, Dict[str, Any]] = {}
    contexts: List[str] = []
    stats: List[Dict[str, Any]] = []

    for ev in case.get("evidence_tables", []):
        file_path = ev["file_path"]
        table_id = ev["table_id"]
        abs_path = file_index.get(file_path)
        if not abs_path:
            raise SystemExit("语料中未找到文件：{}".format(file_path))
        if abs_path not in parsed_cache:
            with io.open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc_type = "industry" if "行业研报" in abs_path else "stock"
            _body, tables = scan_tables(
                content,
                source_file=file_path,
                title=os.path.splitext(file_path)[0],
                doc_type=doc_type,
            )
            parsed_cache[abs_path] = {t.table_id: t for t in tables}
        table = parsed_cache[abs_path].get(table_id)
        if table is None:
            raise SystemExit("文件已找到但表不存在：{}".format(table_id))

        text = table.to_text()
        payload: Dict[str, Any] = {
            "table_id": table.table_id,
            "report_period": getattr(table, "report_period", None),
        }
        codes = getattr(table, "stock_codes", None) or []
        if codes:
            payload["stock_code"] = codes[0]
        contexts.append(build_evidence_context(text, payload))
        expected = ev.get("content_chars") or 0
        delta = len(text) - expected
        stats.append({
            "table_id": table_id,
            "rebuilt_chars": len(text),
            "expected_chars": ev.get("content_chars"),
            "delta_chars": delta,
            # 历史 content_chars 可能含元数据前缀或来自切片，允许 200 字以内差异；
            # 超过 200 字才视为重建异常。
            "match": abs(delta) <= 200,
        })
    return contexts, stats


def run_once(
    generator: LLMGenerator,
    question: str,
    contexts: List[str],
    stream: bool,
) -> Dict[str, Any]:
    """跑一次生成，返回 TTFT / 总耗时 / 回答长度。"""
    first_chunk_at: List[Optional[float]] = [None]
    started = time.perf_counter()

    if stream:
        def _on_chunk(text: str) -> None:
            if first_chunk_at[0] is None and text and text.strip():
                first_chunk_at[0] = time.perf_counter()

        answer = generator.generate(
            query=question,
            contexts=contexts,
            history=None,
            stream=True,
            on_chunk=_on_chunk,
        )
    else:
        answer = generator.generate(
            query=question,
            contexts=contexts,
            history=None,
            stream=False,
        )

    finished = time.perf_counter()
    ttft = None
    if stream and first_chunk_at[0] is not None:
        ttft = first_chunk_at[0] - started
    return {
        "stream": stream,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(finished - started, 3),
        "answer_chars": len(answer or ""),
        "answer_head": (answer or "")[:80].replace("\n", " "),
    }


def _median(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(statistics.median(clean), 3) if clean else None


def main() -> int:
    parser = argparse.ArgumentParser(description="流式输出 TTFT 验证（不改 query() 同步逻辑）")
    parser.add_argument("--qid", default="B2007|2", help="A 类题号，默认 B2007|2（历史生成 35.4s / 3,232 字长回答）")
    parser.add_argument("--case-json", default=DEFAULT_CASE_JSON, help="A 类重生成结果 JSON")
    parser.add_argument("--runs", type=int, default=3, help="每种模式的运行次数（默认 3，取中位数）")
    parser.add_argument("--sync-runs", type=int, default=None, help="非流式运行次数；默认与 --runs 相同")
    parser.add_argument("--skip-sync", action="store_true", help="跳过非流式重跑，仅测流式")
    parser.add_argument("--dry-run", action="store_true", help="只重建上下文并打印统计，不调用 LLM")
    parser.add_argument("--out", default="", help="结果 JSON 路径；默认写入 训练结果数据/stage5_measure/")
    args = parser.parse_args()

    case = _load_case(args.case_json, args.qid)
    question = case["question"]
    contexts, stats = build_contexts(case)

    total_chars = sum(len(c) for c in contexts)
    print("[case] qid={} question={}".format(args.qid, question))
    print("[context] tables={} chars={} (expected sum={})".format(
        len(contexts), total_chars, sum((s.get("expected_chars") or 0) for s in stats)
    ))
    mismatched = [s for s in stats if not s["match"]]
    if mismatched:
        print("[warn] 重建长度与历史 content_chars 差异超过 200 字：{}".format(mismatched))
    else:
        diffs = [s for s in stats if s["delta_chars"]]
        if diffs:
            print("[context] {}/{} 表重建长度与历史 chunk 有差异（切片/元数据前缀），最大 {} 字；其余一致".format(
                len(diffs), len(stats), max(abs(s["delta_chars"]) for s in diffs)
            ))
        else:
            print("[context] {}/{} 表重建长度与历史 content_chars 完全一致".format(len(stats), len(stats)))

    if args.dry_run:
        for s in stats:
            print("  - {table_id} rebuilt={rebuilt_chars} expected={expected_chars}".format(**s))
        return 0

    config = RAGConfig()
    if not config.DASHSCOPE_API_KEY:
        raise SystemExit("DASHSCOPE_API_KEY 未设置，无法调用模型")
    generator = LLMGenerator(config)

    # 先用同一 generator 做一次“连接/客户端预热”，不计入中位数；避免首个请求把
    # 建连成本算进 TTFT。预热内容与正式上下文相同，保证模型侧缓存条件一致。
    print("[warmup] 流式预热 1 次（不计入中位数，但单独记录冷启动）...")
    warmup_result = run_once(generator, question, contexts, stream=True)

    sync_runs = 0 if args.skip_sync else (args.sync_runs if args.sync_runs is not None else args.runs)
    sync_results: List[Dict[str, Any]] = []
    stream_results: List[Dict[str, Any]] = []
    # 交错执行，避免“同步先跑、流式后跑”带来的服务端时段偏差。
    for i in range(max(sync_runs, args.runs)):
        if i < sync_runs:
            print("[sync] run {}/{} ...".format(i + 1, sync_runs))
            sync_results.append(run_once(generator, question, contexts, stream=False))
        if i < args.runs:
            print("[stream] run {}/{} ...".format(i + 1, args.runs))
            stream_results.append(run_once(generator, question, contexts, stream=True))

    result = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "qid": args.qid,
        "question": question,
        "model": config.LLM_MODEL,
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
        "context_count": len(contexts),
        "context_chars": total_chars,
        "context_stats": stats,
        "warmup": warmup_result,
        "sync": {
            "runs": sync_results,
            "median_total_s": _median([r["total_s"] for r in sync_results]),
        },
        "stream": {
            "runs": stream_results,
            "median_ttft_s": _median([r["ttft_s"] for r in stream_results]),
            "median_total_s": _median([r["total_s"] for r in stream_results]),
        },
        "note": "TTFT = 请求发出到第一个非空 delta；总耗时 = 请求发出到流结束。同步模式无 TTFT。",
    }

    out_path = args.out
    if not out_path:
        out_path = os.path.join(
            DEFAULT_OUT_DIR,
            "stream_ttft_{}.json".format(_dt.datetime.now().strftime("%Y%m%d_%H%M%S")),
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    sync_total = result["sync"]["median_total_s"]
    stream_total = result["stream"]["median_total_s"]
    stream_ttft = result["stream"]["median_ttft_s"]

    print("\n=== 汇总 ===")
    if sync_total is not None:
        print("同步生成总耗时约 {:.1f}s；流式生成下，总耗时 {:.1f}s，但首字延迟仅为 {:.1f}s".format(
            sync_total, stream_total, stream_ttft
        ))
    else:
        print("（跳过同步重跑）流式生成下，总耗时 {:.1f}s，首字延迟仅为 {:.1f}s".format(
            stream_total, stream_ttft
        ))
    print("明细：{}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())