# -*- coding: utf-8 -*-
"""eval/retrieval_cmp.py —— 检索层对比：纯向量 vs 混合检索（向量 + BM25 RRF）

对比口径（引用命中，与 L1 引用核验同源同口径）：
- ground truth 来自 训练结果数据/references_all.json（批量答案引用，带 bid 映射）：
  每题（bid）的引用 = {paper_path, text} 集合。
- 文件级命中：召回 top-K 片段中，payload.file_path 文件名是否覆盖该题引用的研报文件。
- 数字级命中：引用 text 中的数字，是否出现在召回片段正文（comma 归一化，同 L1）。

用法：
  python -m eval retrieval [--collection research_reports_v3] [--k 10] [--limit N] [--match comma] [--out docs/检索对比报告.md]

  [--rerank-top-n N] 可选：对 top-K 候选做 qwen3-rerank 复排，额外统计精排后 top-N 命中

说明：
- 默认对"有引用的题目"（61/80）逐子问题跑两路检索，输出对比报告与 JSON 明细；
- --collection 默认线上集合 research_reports_v3；正式数据全量集合为 research_reports_v3_full。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
GOLDEN_PATH = REPO_ROOT / "database" / "golden" / "v1_2026-08-22.json"
REFS_PATH = REPO_ROOT / "训练结果数据" / "references_all.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "检索对比报告.md"
DEFAULT_JSON = REPO_ROOT / "训练结果数据" / "retrieval_cmp.json"


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _load_golden_items() -> List[Dict[str, Any]]:
    """读取 golden v1 题目（80 题 / 108 子问题）"""
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["items"]


def _load_refs_by_bid() -> Dict[str, List[Dict[str, str]]]:
    """读取参考答案引用，按 bid 聚合"""
    refs = json.loads(REFS_PATH.read_text(encoding="utf-8"))
    by_bid: Dict[str, List[Dict[str, str]]] = {}
    for r in refs:
        by_bid.setdefault(str(r.get("bid", "")), []).append(
            {"paper_path": str(r.get("paper_path", "")), "text": str(r.get("text", ""))}
        )
    return by_bid


class NumberMatcher:
    """数字匹配器：与 pipelines.citation_validator 同口径（raw / comma / loose）"""

    _NUM_RE = re.compile(r"\d+(?:\.\d+)?")

    def __init__(self, mode: str = "comma") -> None:
        self.mode = mode

    def extract(self, text: str) -> List[str]:
        """抽取文本中的数字（整数 / 小数）"""
        return self._NUM_RE.findall(text or "")

    def _norm(self, text: str) -> str:
        if self.mode == "raw":
            return text
        s = text.replace(",", "").replace("\uff0c", "")
        if self.mode == "loose":
            s = re.sub(r"\s+", "", s)
        return s

    def count_hits(self, numbers: List[str], contents: List[str]) -> Tuple[int, List[str]]:
        """统计数字在召回正文中的命中数（comma 归一化）"""
        haystack = self._norm("\n".join(contents))
        hit = 0
        unhit: List[str] = []
        for num in numbers:
            needle = self._norm(num)
            if needle and needle in haystack:
                hit += 1
            else:
                unhit.append(num)
        return hit, unhit


def _candidate_files(results: List[Dict[str, Any]]) -> set:
    """提取召回片段的文件名集合（payload.file_path 可能为完整路径或纯文件名）"""
    out = set()
    for r in results:
        fp = r.get("payload", {}).get("file_path", "")
        if fp:
            out.add(os.path.basename(str(fp)))
    return out


def _candidate_contents(results: List[Dict[str, Any]]) -> List[str]:
    return [str(r.get("payload", {}).get("content", "")) for r in results]


def _rerank_entries(rerank_client: Any, query: str, results: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """对候选列表做 Rerank 复排，返回按精排序的候选条目（保留原 payload 以便统计命中）。"""
    docs = _candidate_contents(results)
    if not docs:
        return []
    reranked = rerank_client.rerank(query=query, documents=docs, top_n=top_n)
    out: List[Dict[str, Any]] = []
    for item in reranked:
        idx = item.get("index")
        if idx is not None and 0 <= idx < len(results):
            out.append({
                "index": idx,
                "score": item.get("relevance_score", 0.0),
                "payload": results[idx].get("payload", {}),
            })
    return out


def _rate(hit: int, total: int) -> Optional[float]:
    return round(hit / total, 4) if total else None


def run(
    collection: str = "research_reports_v3",
    k: int = 10,
    limit: int = 0,
    match_mode: str = "comma",
    rrf_k: int = 0,
    topk_bm25: int = 0,
    vector_floor_ratio: float = -1.0,
    rerank_top_n: int = 0,
    out_md: Optional[Path] = None,
    out_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """执行两路检索对比，返回汇总结果（并写报告文件）"""
    from config.rag_config import RAGConfig
    from core.retrievers import HandwrittenRetriever, HybridRetriever
    from pipelines.rag_pipeline import RAGPipeline

    out_md = out_md or DEFAULT_OUT
    out_json = out_json or DEFAULT_JSON

    config = RAGConfig(QDRANT_COLLECTION_NAME=collection)
    pipeline = RAGPipeline(config)
    print(f"[retrieval_cmp] 集合 {collection}，top-K={k}，数字口径 {match_mode}，构建/加载 BM25 索引…")
    bm25 = pipeline.build_bm25_index()
    vector_retriever = HandwrittenRetriever(
        embedding_client=pipeline.embedding_client,
        qdrant_client=pipeline.qdrant_client,
        top_k=k,
    )
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        top_k=k,
        rrf_k=rrf_k or config.HYBRID_RRF_K,
        topk_vector=config.HYBRID_TOPK_VECTOR,
        topk_bm25=topk_bm25 or config.HYBRID_TOPK_BM25,
        vector_floor_ratio=vector_floor_ratio if vector_floor_ratio >= 0 else config.HYBRID_VECTOR_FLOOR_RATIO,
    )

    rerank_client = None
    if rerank_top_n > 0:
        from chains.rerank import RerankClient
        from core.rerankers import apply_file_diversity, file_keys_from_candidates
        rerank_client = RerankClient(config)
        print(f"[retrieval_cmp] 精排复排：qwen3-rerank top-{rerank_top_n}（含每文件上限对比）")

    matcher = NumberMatcher(match_mode)
    items = _load_golden_items()
    refs_by_bid = _load_refs_by_bid()

    rows: List[Dict[str, Any]] = []
    seen_bids = 0
    t0 = time.time()
    for item in items:
        bid = str(item["编号"])
        refs = refs_by_bid.get(bid)
        if not refs:
            continue
        if limit and seen_bids >= limit:
            break
        seen_bids += 1
        expected_files = {os.path.basename(r["paper_path"]) for r in refs if r["paper_path"]}
        expected_nums = [n for r in refs for n in matcher.extract(r["text"])]
        for sub in item.get("子问题", []):
            if not sub or not sub.strip():
                continue
            vec = vector_retriever.retrieve(sub, top_k=k)
            hyb = hybrid_retriever.retrieve(sub, top_k=k)
            vec_files = _candidate_files(vec)
            hyb_files = _candidate_files(hyb)
            vec_num_hit, vec_unhit = matcher.count_hits(expected_nums, _candidate_contents(vec))
            hyb_num_hit, hyb_unhit = matcher.count_hits(expected_nums, _candidate_contents(hyb))
            rr_caps = {}
            if rerank_client is not None:
                request_n = rerank_top_n * 2
                vec_rr = _rerank_entries(rerank_client, sub, vec, request_n)
                hyb_rr = _rerank_entries(rerank_client, sub, hyb, request_n)
                vec_rr_files = _candidate_files(vec_rr[:rerank_top_n])
                hyb_rr_files = _candidate_files(hyb_rr[:rerank_top_n])
                vec_rr_num_hit, vec_rr_unhit = matcher.count_hits(expected_nums, _candidate_contents(vec_rr[:rerank_top_n]))
                hyb_rr_num_hit, hyb_rr_unhit = matcher.count_hits(expected_nums, _candidate_contents(hyb_rr[:rerank_top_n]))
                vec_file_keys = file_keys_from_candidates(vec)
                hyb_file_keys = file_keys_from_candidates(hyb)
                for cap in (1, 2, 3):
                    v = apply_file_diversity(vec_rr, vec_file_keys, rerank_top_n, cap)
                    h = apply_file_diversity(hyb_rr, hyb_file_keys, rerank_top_n, cap)
                    rr_caps[cap] = {
                        "vf": len(expected_files & _candidate_files(v)),
                        "hf": len(expected_files & _candidate_files(h)),
                        "vn": matcher.count_hits(expected_nums, _candidate_contents(v))[0],
                        "hn": matcher.count_hits(expected_nums, _candidate_contents(h))[0],
                    }
            else:
                vec_rr_files = hyb_rr_files = set()
                vec_rr_num_hit = hyb_rr_num_hit = 0
                vec_rr_unhit = hyb_rr_unhit = []
            rows.append({
                "bid": bid,
                "qtype": str(item.get("问题类型", "")),
                "query": sub,
                "files_total": len(expected_files),
                "vec_files_hit": len(expected_files & vec_files),
                "hyb_files_hit": len(expected_files & hyb_files),
                "vec_file_hit": list(expected_files & vec_files),
                "hyb_file_hit": list(expected_files & hyb_files),
                "num_total": len(expected_nums),
                "vec_num_hit": vec_num_hit,
                "hyb_num_hit": hyb_num_hit,
                "vec_unhit": vec_unhit,
                "hyb_unhit": hyb_unhit,
                "vec_rr_files_hit": len(expected_files & vec_rr_files),
                "hyb_rr_files_hit": len(expected_files & hyb_rr_files),
                "vec_rr_num_hit": vec_rr_num_hit,
                "hyb_rr_num_hit": hyb_rr_num_hit,
                "vec_rr_unhit": vec_rr_unhit,
                "hyb_rr_unhit": hyb_rr_unhit,
                "rr_caps": rr_caps,
                "vec_top_files": sorted(vec_files)[:5],
                "hyb_top_files": sorted(hyb_files)[:5],
            })
        print(f"[retrieval_cmp] {bid} 完成（{seen_bids}/{limit or '全量'} 个有引用题），累计耗时 {time.time() - t0:.0f}s")

    summary = _summarize(rows, rerank_top_n)
    _write_report(summary, rows, out_md, rerank_top_n)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"meta": {"collection": collection, "k": k, "match_mode": match_mode, "rerank_top_n": rerank_top_n}, "summary": summary, "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[retrieval_cmp] 报告：{out_md}")
    print(f"[retrieval_cmp] 明细：{out_json}")
    return summary


def _summarize(rows: List[Dict[str, Any]], rerank_top_n: int = 0) -> Dict[str, Any]:
    """聚合对比指标（rerank_top_n>0 时额外聚合精排后命中）"""
    n = len(rows)
    if not n:
        return {"rows": 0, "bids": 0}

    def rate(hit_key: str, total_key: str) -> Tuple[int, int, Optional[float]]:
        hit = sum(r[hit_key] for r in rows)
        total = sum(r[total_key] for r in rows)
        return hit, total, _rate(hit, total)

    vf_hit, vf_total, vf_rate = rate("vec_files_hit", "files_total")
    hf_hit, hf_total, hf_rate = rate("hyb_files_hit", "files_total")
    vn_hit, vn_total, vn_rate = rate("vec_num_hit", "num_total")
    hn_hit, hn_total, hn_rate = rate("hyb_num_hit", "num_total")

    file_win = {"hybrid_win": 0, "tie": 0, "vector_win": 0}
    num_win = {"hybrid_win": 0, "tie": 0, "vector_win": 0}
    for r in rows:
        vf, hf = r["vec_files_hit"], r["hyb_files_hit"]
        file_win["hybrid_win" if hf > vf else ("vector_win" if vf > hf else "tie")] += 1
        vn, hn = r["vec_num_hit"], r["hyb_num_hit"]
        num_win["hybrid_win" if hn > vn else ("vector_win" if vn > hn else "tie")] += 1

    all_file_hit = sum(1 for r in rows if r["files_total"] > 0 and r["hyb_files_hit"] == r["files_total"])
    all_num_hit = sum(1 for r in rows if r["num_total"] > 0 and r["hyb_num_hit"] == r["num_total"])
    any_file_hit = sum(1 for r in rows if r["hyb_files_hit"] > 0)
    any_num_hit = sum(1 for r in rows if r["hyb_num_hit"] > 0)

    rerank = None
    rerank_caps = None
    if rerank_top_n > 0 and rows and "vec_rr_files_hit" in rows[0]:
        def rr_rate(hit_key: str, total_key: str) -> Tuple[int, int, Optional[float]]:
            hit = sum(r.get(hit_key, 0) for r in rows)
            total = sum(r.get(total_key, 0) for r in rows)
            return hit, total, _rate(hit, total)
        rvf_hit, rvf_total, rvf_rate = rr_rate("vec_rr_files_hit", "files_total")
        rhf_hit, rhf_total, rhf_rate = rr_rate("hyb_rr_files_hit", "files_total")
        rvn_hit, rvn_total, rvn_rate = rr_rate("vec_rr_num_hit", "num_total")
        rhn_hit, rhn_total, rhn_rate = rr_rate("hyb_rr_num_hit", "num_total")
        rerank = {
            "file": {"vec": {"hit": rvf_hit, "total": rvf_total, "rate": rvf_rate},
                     "hybrid": {"hit": rhf_hit, "total": rhf_total, "rate": rhf_rate}},
            "num": {"vec": {"hit": rvn_hit, "total": rvn_total, "rate": rvn_rate},
                    "hybrid": {"hit": rhn_hit, "total": rhn_total, "rate": rhn_rate}},
        }
        if rows[0].get("rr_caps"):
            ft = sum(r["files_total"] for r in rows)
            nt = sum(r["num_total"] for r in rows)
            rerank_caps = []
            for cap in sorted(rows[0]["rr_caps"].keys()):
                vf = sum(r["rr_caps"][cap]["vf"] for r in rows)
                hf = sum(r["rr_caps"][cap]["hf"] for r in rows)
                vn = sum(r["rr_caps"][cap]["vn"] for r in rows)
                hn = sum(r["rr_caps"][cap]["hn"] for r in rows)
                rerank_caps.append({
                    "cap": cap,
                    "file": {"vec": {"hit": vf, "total": ft, "rate": _rate(vf, ft)},
                             "hybrid": {"hit": hf, "total": ft, "rate": _rate(hf, ft)}},
                    "num": {"vec": {"hit": vn, "total": nt, "rate": _rate(vn, nt)},
                            "hybrid": {"hit": hn, "total": nt, "rate": _rate(hn, nt)}},
                })

    bids = len({r["bid"] for r in rows})
    return {
        "rows": n,
        "bids": bids,
        "file": {"vec": {"hit": vf_hit, "total": vf_total, "rate": vf_rate},
                 "hybrid": {"hit": hf_hit, "total": hf_total, "rate": hf_rate},
                 "win": file_win,
                 "all_hit_rows": all_file_hit,
                 "any_hit_rows": any_file_hit},
        "num": {"vec": {"hit": vn_hit, "total": vn_total, "rate": vn_rate},
                "hybrid": {"hit": hn_hit, "total": hn_total, "rate": hn_rate},
                "win": num_win,
                "all_hit_rows": all_num_hit,
                "any_hit_rows": any_num_hit},
        "rerank": rerank,
        "rerank_caps": rerank_caps,
    }


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _write_report(summary: Dict[str, Any], rows: List[Dict[str, Any]], out_md: Path, rerank_top_n: int = 0) -> None:
    """渲染对比报告 markdown（rerank_top_n>0 时追加精排后总览）"""
    lines: List[str] = []
    if not summary.get("rows"):
        lines.append("# 检索层对比报告（纯向量 vs 混合）\n\n（无数据：未找到带引用的题目）\n")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return
    f, num = summary["file"], summary["num"]
    rr = summary.get("rerank")
    retr_cfg = "- 检索配置：top-K 召回后直接比较（未过 Rerank，隔离检索器差异）"
    if rerank_top_n > 0:
        retr_cfg = f"- 检索配置：top-K 召回 + qwen3-rerank 复排 top-{rerank_top_n}（召回层与精排后命中分别统计）"
    lines += [
        "# 检索层对比报告：纯向量 vs 混合检索（引用命中）",
        "",
        f"- 评估范围：{summary['rows']} 个检索测试（{summary['bids']} 个有引用的题目，golden v1 子问题 × 两路检索）",
        retr_cfg,
        "- ground truth：`训练结果数据/references_all.json`（批量答案引用，bid 映射）",
        "- 文件级命中：召回片段文件名覆盖该题引用的研报文件数 / 引用文件总数",
        "- 数字级命中：引用文本中的数字出现在召回片段正文的比例（comma 归一化，同 L1 口径）",
        "",
        "## 一、总览（召回层 top-K）",
        "",
        "| 指标 | 纯向量 | 混合（向量+BM25 RRF） | 变化 |",
        "| --- | --- | --- | --- |",
        f"| 文件级命中（聚合） | {f['vec']['hit']}/{f['vec']['total']} = {_pct(f['vec']['rate'])} | {f['hybrid']['hit']}/{f['hybrid']['total']} = {_pct(f['hybrid']['rate'])} | {_delta(f['vec']['rate'], f['hybrid']['rate'])} |",
        f"| 数字级命中（聚合） | {num['vec']['hit']}/{num['vec']['total']} = {_pct(num['vec']['rate'])} | {num['hybrid']['hit']}/{num['hybrid']['total']} = {_pct(num['hybrid']['rate'])} | {_delta(num['vec']['rate'], num['hybrid']['rate'])} |",
        "",
    ]
    if rr is not None:
        rfile, rnum = rr["file"], rr["num"]
        lines += [
            f"## 二、总览（Rerank 后 top-{rerank_top_n}）",
            "",
            "| 指标 | 纯向量 | 混合（向量+BM25 RRF） | 变化 |",
            "| --- | --- | --- | --- |",
            f"| 文件级命中（聚合） | {rfile['vec']['hit']}/{rfile['vec']['total']} = {_pct(rfile['vec']['rate'])} | {rfile['hybrid']['hit']}/{rfile['hybrid']['total']} = {_pct(rfile['hybrid']['rate'])} | {_delta(rfile['vec']['rate'], rfile['hybrid']['rate'])} |",
            f"| 数字级命中（聚合） | {rnum['vec']['hit']}/{rnum['vec']['total']} = {_pct(rnum['vec']['rate'])} | {rnum['hybrid']['hit']}/{rnum['hybrid']['total']} = {_pct(rnum['hybrid']['rate'])} | {_delta(rnum['vec']['rate'], rnum['hybrid']['rate'])} |",
            "",
        ]
    rerank_caps = summary.get("rerank_caps")
    if rerank_caps:
        lines += [
            f"## {_sec(3)}、Rerank 后 top-{rerank_top_n}：每文件上限对比",
            "",
            "| 变体 | 纯向量 文件级 | 混合 文件级 | 纯向量 数字级 | 混合 数字级 |",
            "| --- | --- | --- | --- | --- |",
            f"| top-{rerank_top_n} 原样 | {_pct(rr['file']['vec']['rate'])} | {_pct(rr['file']['hybrid']['rate'])} | {_pct(rr['num']['vec']['rate'])} | {_pct(rr['num']['hybrid']['rate'])} |",
        ]
        for item in rerank_caps:
            lines.append(
                f"| 每文件≤{item['cap']} | {_pct(item['file']['vec']['rate'])} | {_pct(item['file']['hybrid']['rate'])} | {_pct(item['num']['vec']['rate'])} | {_pct(item['num']['hybrid']['rate'])} |"
            )
        lines.append("")
    sec = 4 if rerank_caps else (3 if rr is not None else 2)
    lines += [
        f"## {_sec(sec)}、胜负统计（按检索测试行）",
        "",
        "| 维度 | 混合 > 向量 | 持平 | 向量 > 混合 |",
        "| --- | --- | --- | --- |",
        f"| 文件级 | {f['win']['hybrid_win']} | {f['win']['tie']} | {f['win']['vector_win']} |",
        f"| 数字级 | {num['win']['hybrid_win']} | {num['win']['tie']} | {num['win']['vector_win']} |",
        "",
        f"- 混合检索“至少命中 1 个引用文件”的测试数：{f['any_hit_rows']}/{summary['rows']}；全命中：{f['all_hit_rows']}/{summary['rows']}",
        f"- 混合检索“至少命中 1 个引用数字”的测试数：{num['any_hit_rows']}/{summary['rows']}；全命中：{num['all_hit_rows']}/{summary['rows']}",
        "",
        f"## {_sec(sec + 1)}、逐题明细（按 bid 聚合）",
        "",
        "| 编号 | 类型 | 测试数 | 文件命中 v/h | 数字命中 v/h |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_bid: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        b = by_bid.setdefault(r["bid"], {"qtype": r["qtype"], "n": 0, "vf": 0, "ft": 0, "hf": 0, "vn": 0, "nt": 0, "hn": 0})
        b["n"] += 1
        b["vf"] += r["vec_files_hit"]; b["ft"] += r["files_total"]
        b["hf"] += r["hyb_files_hit"]
        b["vn"] += r["vec_num_hit"]; b["nt"] += r["num_total"]
        b["hn"] += r["hyb_num_hit"]
    for bid, b in by_bid.items():
        lines.append(
            f"| {bid} | {b['qtype']} | {b['n']} | {b['vf']}/{b['ft']} → {b['hf']}/{b['ft']} | {b['vn']}/{b['nt']} → {b['hn']}/{b['nt']} |"
        )
    lines += [
        "",
        f"> 说明：本报告为检索层对比（{'Rerank 复排后命中另列一节' if rerank_top_n > 0 else 'Rerank 之前'}），用于评估混合检索（BM25+RRF）相对纯向量的引用支撑增益；",
        "> 端到端引用真实性仍以 L1 引用核验（`python -m eval citation`）为准。",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


_CN_NUM = {2: "二", 3: "三", 4: "四", 5: "五"}





def _sec(num: int) -> str:

    """章节序号转中文（超出映射回退为数字）"""

    return _CN_NUM.get(num, str(num))





def _delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{'+' if d >= 0 else ''}{d * 100:.1f}pp"


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口（python -m eval.retrieval_cmp）"""
    _stdout_utf8()
    parser = argparse.ArgumentParser(prog="retrieval_cmp", description="检索层对比：纯向量 vs 混合（引用命中）")
    parser.add_argument("--collection", default="research_reports_v3", help="Qdrant 集合名")
    parser.add_argument("--k", type=int, default=10, help="召回 top-K")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个有引用的题目（冒烟）")
    parser.add_argument("--match", choices=["raw", "comma", "loose"], default="comma", help="数字匹配口径")
    parser.add_argument("--rrf-k", type=int, default=0, help="RRF 常数（0=取配置 HYBRID_RRF_K）")
    parser.add_argument("--topk-bm25", type=int, default=0, help="BM25 路召回量（0=取配置）")
    parser.add_argument("--vector-floor-ratio", type=float, default=-1.0, help="向量路保底比例（-1=取配置，0=纯RRF）")
    parser.add_argument("--rerank-top-n", type=int, default=0, help="对 top-K 候选做 Rerank 复排并统计精排后 top-N 命中（0=不开启）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="markdown 报告路径")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON), help="JSON 明细路径")
    args = parser.parse_args(argv)
    run(
        collection=args.collection,
        k=args.k,
        limit=args.limit,
        match_mode=args.match,
        rrf_k=args.rrf_k,
        topk_bm25=args.topk_bm25,
        vector_floor_ratio=args.vector_floor_ratio,
        rerank_top_n=args.rerank_top_n,
        out_md=Path(args.out),
        out_json=Path(args.json_out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())