# -*- coding: utf-8 -*-
"""tools/data_scripts/retrieval_k50_probe.py —— L5 线上口径探针（2026-09-30）

背景：L5（代理指标层）现有基线 `训练结果数据/retrieval_metrics_baseline_20260927.json`
测的是 `HybridRetriever.retrieve(top_k=10)` 的**召回层**（未经精排、未按每文件上限收敛），
而线上主链路是 `RETRIEVAL_K=50 → qwen3-rerank → RERANK_TOP_N=10（oversample×2、每文件 ≤2）`
（`src/pipelines/rag_pipeline.py:1185-1214`）。两者不同源，所以旧基线不能对外当"检索质量"。

本脚本对同一批 30 题（seed=20260910，与 B-24A/B 同题）导出三档候选，供归因分析：
  - `k50`      ：混合检索原始顺序（＝现基线测的那一层）
  - `rerank20` ：过 `qwen3-rerank` 后的顺序（请求 top_n＝RERANK_TOP_N×RERANK_OVERSAMPLE）
  - `online10` ：再按「每文件 ≤ RERANK_MAX_PER_FILE」收敛到 RERANK_TOP_N（＝线上实际上下文）

硬门禁：`k50` 的 rank 1-10 必须与 B-24A/B 标注的 top-10 **逐条同键同序**。
对不上说明集合 `research_reports_v3_full` 已重建 → 旧人工标注作废，必须停下。

用法：
  python -m tools.data_scripts.retrieval_k50_probe --limit 1        # 冒烟（1 题）
  python -m tools.data_scripts.retrieval_k50_probe                  # 全量 30 题
  python -m tools.data_scripts.retrieval_k50_probe --analyze        # 只做阶段 2 归因（读已有产物）
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = PROJECT_ROOT / "训练结果数据" / "retrieval_prelabel_20260910.json"
DEFAULT_KEYS = PROJECT_ROOT / "训练结果数据" / "retrieval_keys_20260930.json"
DEFAULT_OUT = PROJECT_ROOT / "训练结果数据" / "retrieval_k50_probe_20260930.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "评估报告" / "retrieval线上口径归因_20260930.md"


def _stdout_utf8() -> None:
    """Windows 控制台统一 UTF-8 输出"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def payload_key(payload: Dict[str, Any]) -> str:
    """片段稳定键（与 retrieval_key_index.stable_key 同定义）"""
    h = payload.get("content_hash")
    if h:
        return str(h)
    fp = str(payload.get("file_path") or "")
    text = str(payload.get("content") or "")
    return "sha1:" + hashlib.sha1((fp + "\n" + text).encode("utf-8")).hexdigest()


def build_stack(collection: str, retrieval_k: int):
    """构造与线上同口径的检索栈：混合检索器（向量 + BM25 RRF）+ 精排客户端"""
    from config.rag_config import RAGConfig
    from core.rerankers import RerankerAdapter
    from core.retrievers import HandwrittenRetriever, HybridRetriever
    from chains.rerank import RerankClient
    from pipelines.rag_pipeline import RAGPipeline

    config = RAGConfig(QDRANT_COLLECTION_NAME=collection)
    pipeline = RAGPipeline(config)
    bm25 = pipeline.build_bm25_index()
    vector_retriever = HandwrittenRetriever(
        embedding_client=pipeline.embedding_client,
        qdrant_client=pipeline.qdrant_client,
        top_k=config.HYBRID_TOPK_VECTOR,
    )
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        top_k=retrieval_k,
        rrf_k=config.HYBRID_RRF_K,
        topk_vector=config.HYBRID_TOPK_VECTOR,
        topk_bm25=config.HYBRID_TOPK_BM25,
        vector_floor_ratio=config.HYBRID_VECTOR_FLOOR_RATIO,
    )
    reranker = RerankerAdapter(RerankClient(config))
    return config, retriever, reranker


def probe_one(
    question: str,
    retriever: Any,
    reranker: Any,
    config: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """对单题导出三档候选（严格复刻 `rag_pipeline.py:1185-1214` 的后处理顺序）"""
    from core.rerankers import apply_file_diversity, file_keys_from_candidates

    hits = retriever.retrieve(question)
    k50 = [
        {
            "rank": i,
            "key": payload_key(r.get("payload") or {}),
            "chunk_index": (r.get("payload") or {}).get("chunk_index"),
            "file_path": (r.get("payload") or {}).get("file_path", ""),
            "score": r.get("score"),
            "id": r.get("id"),
        }
        for i, r in enumerate(hits, start=1)
    ]
    docs = [str((r.get("payload") or {}).get("content") or "") for r in hits]
    if not docs:
        return {"k50": [], "rerank20": [], "online10": []}

    reranked = reranker.rerank(
        query=question,
        documents=docs,
        top_n=config.RERANK_TOP_N * config.RERANK_OVERSAMPLE,
    )
    rerank20 = []
    for i, item in enumerate(reranked, start=1):
        idx = item.get("index")
        src = k50[idx] if isinstance(idx, int) and 0 <= idx < len(k50) else {}
        rerank20.append(
            {
                "rank": i,
                "stage_index": idx,
                "relevance_score": item.get("relevance_score"),
                "key": src.get("key"),
                "file_path": src.get("file_path", ""),
                "chunk_index": src.get("chunk_index"),
                "recall_rank": (src.get("rank") if src else None),
            }
        )
    selected = apply_file_diversity(
        reranked,
        file_keys_from_candidates(hits),
        config.RERANK_TOP_N,
        config.RERANK_MAX_PER_FILE,
    )
    online10 = []
    for i, item in enumerate(selected, start=1):
        idx = item.get("index")
        src = k50[idx] if isinstance(idx, int) and 0 <= idx < len(k50) else {}
        online10.append(
            {
                "rank": i,
                "stage_index": idx,
                "relevance_score": item.get("relevance_score"),
                "key": src.get("key"),
                "file_path": src.get("file_path", ""),
                "chunk_index": src.get("chunk_index"),
                "recall_rank": (src.get("rank") if src else None),
            }
        )
    return {"k50": k50, "rerank20": rerank20, "online10": online10}


def run_probe(args: argparse.Namespace) -> int:
    """阶段 1：跑三档探针 + 硬门禁校验"""
    from tools.data_scripts.retrieval_prelabel import load_sub_questions, stratified_sample

    rows = stratified_sample(load_sub_questions(), args.count, args.seed)
    if args.limit:
        rows = rows[: args.limit]
    print("[probe] 抽样 %d 题（seed=%s）" % (len(rows), args.seed))

    keys_data = json.loads(Path(args.keys).read_text(encoding="utf-8"))
    key_by_q: Dict[str, List[str]] = collections.defaultdict(list)
    for it in sorted(keys_data["items"], key=lambda x: (x["bid"], x["sub_idx"], x["rank"])):
        key_by_q["%s|%s" % (it["bid"], it["sub_idx"])].append(it["key"])

    config, retriever, reranker = build_stack(args.collection, args.retrieval_k)
    print(
        "[probe] 配置：RETRIEVAL_K=%d RERANK_TOP_N=%d OVERSAMPLE=%d MAX_PER_FILE=%d RRF_K=%d FLOOR=%.2f"
        % (
            config.RETRIEVAL_K, config.RERANK_TOP_N, config.RERANK_OVERSAMPLE,
            config.RERANK_MAX_PER_FILE, config.HYBRID_RRF_K, config.HYBRID_VECTOR_FLOOR_RATIO,
        )
    )

    questions: List[Dict[str, Any]] = []
    gate_fail: List[Dict[str, Any]] = []
    no_hash = 0
    t_all = time.time()
    for r in rows:
        t0 = time.time()
        stages = probe_one(r["子问题"], retriever, reranker, config)
        qid = "%s|%s" % (r["bid"], r["sub_idx"])
        expect = key_by_q.get(qid, [])
        got = [c["key"] for c in stages["k50"][:10]]
        gate_ok = bool(expect) and got[: len(expect)] == expect
        if not gate_ok:
            gate_fail.append({"qid": qid, "expected": expect, "got": got})
        no_hash += sum(1 for c in stages["k50"] if not c["key"])
        questions.append(
            {
                "bid": r["bid"],
                "sub_idx": r["sub_idx"],
                "问题类型": r.get("问题类型", ""),
                "子问题": r["子问题"],
                "gate_ok": gate_ok,
                "k50_count": len(stages["k50"]),
                "online10_files": len({c["file_path"] for c in stages["online10"]}),
                **stages,
            }
        )
        print(
            "  [%s] 召回 %d ｜ rerank %d ｜ 线上 %d 条 / %d 文件 ｜ 门禁 %s ｜ %.1fs"
            % (qid, len(stages["k50"]), len(stages["rerank20"]), len(stages["online10"]),
               len({c["file_path"] for c in stages["online10"]}),
               "OK" if gate_ok else "**FAIL**", time.time() - t0)
        )

    data = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "collection": args.collection,
        "labels": str(DEFAULT_LABELS.relative_to(PROJECT_ROOT)),
        "keys": (str(Path(args.keys).relative_to(PROJECT_ROOT))
                if str(args.keys).startswith(str(PROJECT_ROOT)) else str(args.keys)),
        "口径": {
            "k50": "HybridRetriever.retrieve() 原始融合顺序（现 L5 基线测的召回层）",
            "rerank20": "qwen3-rerank 精排后顺序（请求 RERANK_TOP_N×RERANK_OVERSAMPLE 条）",
            "online10": "精排后再按每文件 ≤ RERANK_MAX_PER_FILE 收敛到 RERANK_TOP_N（线上实际上下文）",
            "未复刻": "线上在检索后还有一次「软过滤重排」(rag_pipeline.py:1176-1185，α=0.8，依赖 LLM 解析过滤条件)"
                       "；该步非确定性，本探针未复刻，故 online10 与线上运行结果可能有小幅顺序差",
        },
        "config": {
            "RETRIEVAL_K": config.RETRIEVAL_K,
            "RERANK_TOP_N": config.RERANK_TOP_N,
            "RERANK_OVERSAMPLE": config.RERANK_OVERSAMPLE,
            "RERANK_MAX_PER_FILE": config.RERANK_MAX_PER_FILE,
            "HYBRID_RRF_K": config.HYBRID_RRF_K,
            "HYBRID_TOPK_VECTOR": config.HYBRID_TOPK_VECTOR,
            "HYBRID_TOPK_BM25": config.HYBRID_TOPK_BM25,
            "HYBRID_VECTOR_FLOOR_RATIO": config.HYBRID_VECTOR_FLOOR_RATIO,
            "RERANK_MODEL": config.RERANK_MODEL,
        },
        "gate": {
            "通过题数": len(questions) - len(gate_fail),
            "失败题数": len(gate_fail),
            "失败明细": gate_fail,
            "无稳定键的 k50 条目数": no_hash,
        },
        "questions": questions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[probe] 输出 %s ｜ 总耗时 %.1fs" % (out, time.time() - t_all))
    print("[probe] 硬门禁：%d/%d 通过" % (data["gate"]["通过题数"], len(questions)))
    if gate_fail:
        print("[probe] ⚠️ 门禁失败 → 集合可能已重建，旧标注作废，请停下排查")
        for g in gate_fail[:3]:
            print("   %s\n     expected=%s\n     got     =%s" % (g["qid"], g["expected"], g["got"]))
        return 2
    return 0


def analyze(args: argparse.Namespace) -> int:
    """阶段 2：Layer A 归因（零新标注）——已标注片段在三档里的去留"""
    probe = json.loads(Path(args.out).read_text(encoding="utf-8"))
    keys_data = json.loads(Path(args.keys).read_text(encoding="utf-8"))

    rel_by_q: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    for it in keys_data["items"]:
        qid = "%s|%s" % (it["bid"], it["sub_idx"])
        state = it.get("人工判定") or "未判定"
        rel_by_q[qid][it["key"]] = state

    rows: List[Dict[str, Any]] = []
    agg = collections.Counter()
    for q in probe["questions"]:
        qid = "%s|%s" % (q["bid"], q["sub_idx"])
        states = rel_by_q.get(qid, {})
        online_pos = {c["key"]: c["rank"] for c in q["online10"] if c["key"]}
        rerank_pos = {c["key"]: c["rank"] for c in q["rerank20"] if c["key"]}
        recall_pos = {c["key"]: c["rank"] for c in q["k50"] if c["key"]}
        for c in q["k50"][:10]:
            key = c["key"]
            state = states.get(key, "未判定")
            in_recall = key in recall_pos
            in_rerank = key in rerank_pos
            in_online = key in online_pos
            if not in_recall:
                bucket = "召回层缺失"
            elif not in_rerank:
                bucket = "精排后落出 top20"
            elif in_online:
                bucket = "线上保留"
            elif (rerank_pos.get(key) or 99) <= 10:
                # 精排名次在 10 名内仍未被选中 ⇒ 按序选取时只能是被「每文件 ≤ max_per_file」跳过
                bucket = "精排≤10名却被每文件上限挤掉"
            else:
                bucket = "精排名次>10（未进窗口）"
            agg[(state, bucket)] += 1
            rows.append(
                {
                    "qid": qid, "标注rank": c["rank"], "key": key,
                    "file_path": c["file_path"], "人工判定": state,
                    "召回rank": recall_pos.get(key), "精排rank": rerank_pos.get(key),
                    "线上rank": online_pos.get(key), "去向": bucket,
                }
            )

    lines: List[str] = []
    lines.append("# L5 线上口径归因（Layer A，2026-09-30）")
    lines.append("")
    lines.append("> 范围：30 题 top-10 共 300 条**已人工标注**片段（B-24B 终稿口径）。")
    lines.append("> 方法：把每条已标注片段在 `k50 / rerank20 / online10` 三档里定位，看它到哪一档为止还在。")
    lines.append("> 口径：`k50`＝混合检索原始顺序（现基线测的召回层）；`rerank20`＝qwen3-rerank 后；`online10`＝再按每文件 ≤2 收敛（线上实际上下文）。")
    lines.append("> 局限：线上在检索后还有一次非确定性的「软过滤重排」（`rag_pipeline.py:1176-1185`），本探针未复刻。")
    lines.append("")
    lines.append("## 一、主表：按人工判定 × 去向")
    lines.append("")
    states_order = ["完全相关", "部分相关", "不相关", "未判定"]
    buckets_order = [
        "线上保留",
        "精排≤10名却被每文件上限挤掉",
        "精排名次>10（未进窗口）",
        "精排后落出 top20",
        "召回层缺失",
    ]
    lines.append("| 人工判定 | " + " | ".join(buckets_order) + " | 小计 |")
    lines.append("| :--- | " + " | ".join(["---:"] * len(buckets_order)) + " | ---: |")
    for st in states_order:
        counts = [agg.get((st, b), 0) for b in buckets_order]
        total = sum(counts)
        if total == 0 and st == "未判定":
            continue
        lines.append("| %s | %s | %d |" % (st, " | ".join(str(x) for x in counts), total))
    lines.append("")
    lines.append("## 二、相关片段的存活率与流失归因")
    lines.append("")
    lines.append("口径：只统计**召回层 top-10 中已被人工判为相关**的片段，看它们能穿过精排与每文件上限进入线上上下文的还有几条。")
    lines.append("")
    lines.append("| 人工判定 | 总数 | 线上保留 | 留存率 | 每文件上限挤掉 | 精排名次>10 | 精排落出 top20 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for st in ("完全相关", "部分相关"):
        tot = sum(agg.get((st, b), 0) for b in buckets_order)
        keep = agg.get((st, "线上保留"), 0)
        cap = agg.get((st, "精排≤10名却被每文件上限挤掉"), 0)
        rank = agg.get((st, "精排名次>10（未进窗口）"), 0)
        out = agg.get((st, "精排后落出 top20"), 0)
        lines.append("| %s | %d | %d | %.1f%% | %d | %d | %d |" % (
            st, tot, keep, (100.0 * keep / tot) if tot else 0.0, cap, rank, out))
    lines.append("")
    lines.append("**读法（重要）**：流失主要来自**精排把相关片段排到 10 名之外**（>10 名 ＋ 落出 top20），"
                 "「每文件 ≤2」这道上限只额外挤掉少数几条。**但不能据此判线上更差**——线上 top-10 里 "
                 "77.4% 是召回 rank 11-50 的新片段（未标注，见第四节），可能换进了新的相关片段；"
                 "线上口径的真实指标要等阶段 3/4 的增量标注才能算。")
    lines.append("")
    lines.append("## 三、同文件扎堆在线上口径下是否被压住")
    lines.append("")
    lines.append("| 题 | 召回 top-10 内最大同文件条数 | 线上 top-10 内最大同文件条数 | 线上覆盖文件数 |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for q in probe["questions"]:
        c_recall = collections.Counter(c["file_path"] for c in q["k50"][:10])
        c_online = collections.Counter(c["file_path"] for c in q["online10"])
        qid = "%s|%s" % (q["bid"], q["sub_idx"])
        lines.append("| %s | %d | %d | %d |" % (qid, max(c_recall.values()) if c_recall else 0,
                                                max(c_online.values()) if c_online else 0, len(c_online)))
    lines.append("")
    # 线上上下文槽位构成：多少条是已有标注（可判定）、多少条是未标注（下一阶段要标）
    slot_total = 0
    slot_known = collections.Counter()
    slot_unknown = 0
    unknown_per_q: List[Tuple[str, int]] = []
    for q in probe["questions"]:
        qid = "%s|%s" % (q["bid"], q["sub_idx"])
        states = rel_by_q.get(qid, {})
        u = 0
        for c in q["online10"]:
            slot_total += 1
            if c["key"] in states:
                slot_known[states[c["key"]]] += 1
            else:
                slot_unknown += 1
                u += 1
        unknown_per_q.append((qid, u))

    lines.append("## 四、线上上下文槽位构成（决定下一阶段标注量）")
    lines.append("")
    lines.append("线上 `online10` 共 **%d** 个槽位（30 题，个别题因每文件上限收敛不足 10 条）：" % slot_total)
    lines.append("")
    lines.append("| 槽位来源 | 条数 | 占比 |")
    lines.append("| :--- | ---: | ---: |")
    for st in states_order:
        n = slot_known.get(st, 0)
        if n == 0 and st == "未判定":
            continue
        lines.append("| 已有标注 · %s | %d | %.1f%% |" % (st, n, 100.0 * n / slot_total))
    lines.append("| **未标注（下一阶段需人工判）** | **%d** | **%.1f%%** |" % (slot_unknown, 100.0 * slot_unknown / slot_total))
    lines.append("")
    lines.append("> 结论：线上口径的指标**不能只用 B-24B 的 300 条算**——%.1f%% 的槽位是召回 rank 11-50 的新片段，"
                 "落在阶段 3（Layer B，k50 rank 11-50 共 560 条）的标注范围内，两层共用同一批标注。" % (100.0 * slot_unknown / slot_total))
    lines.append("")

    lines.append("## 五、附：逐条明细（仅列完全相关/部分相关）")
    lines.append("")
    lines.append("| 题 | 标注rank | 人工判定 | 文件 | 召回rank | 精排rank | 线上rank | 去向 |")
    lines.append("| :--- | ---: | :--- | :--- | ---: | ---: | ---: | :--- |")
    for r in rows:
        if r["人工判定"] not in ("完全相关", "部分相关"):
            continue
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
            r["qid"], r["标注rank"], r["人工判定"], (r["file_path"] or "")[:28],
            r["召回rank"], r["精排rank"], r["线上rank"], r["去向"]))
    lines.append("")
    text = "\n".join(lines)
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(text, encoding="utf-8")
    print("[analyze] 报告输出 %s" % rep)
    print(text[:2000])
    return 0


def main() -> int:
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="L5 线上口径探针（k50 / rerank20 / online10）")
    ap.add_argument("--count", type=int, default=30, help="抽样题数")
    ap.add_argument("--seed", type=int, default=20260910, help="抽样种子（须与 B-24A/B 一致）")
    ap.add_argument("--collection", default="research_reports_v3_full")
    ap.add_argument("--retrieval-k", type=int, default=50, help="召回深度（线上 RETRIEVAL_K）")
    ap.add_argument("--keys", default=str(DEFAULT_KEYS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟）")
    ap.add_argument("--analyze", action="store_true", help="只做阶段 2 归因（读已有 --out）")
    args = ap.parse_args()
    return analyze(args) if args.analyze else run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
