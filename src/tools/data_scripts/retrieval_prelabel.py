# -*- coding: utf-8 -*-
"""tools/data_scripts/retrieval_prelabel.py —— B-24A：retrieval 评测集「模型预标注」初筛

做什么：
  1. 从 golden v1 的 108 子问题里按「问题类型分层 + 固定种子」抽 N 题（默认 30）；
  2. 每题用**混合检索**（向量 + BM25 RRF，与线上主链路同口径）取 top-K=10 片段；
  3. 用 LLM（默认 qwen-flash）逐片段预标注「该片段是否包含回答问题所需的信息/数值」；
  4. 产出：JSON（含 `人工修正` 字段，供 B-24B 审核）+ Markdown 待审核清单 + 空白模板。

口径声明（必须随报告一起引用）：
  **这是模型预标注（LLM 初筛），不是 ground truth**；正式指标须等 B-24B 人工审核修正后重算。
  指标计算见 eval/retrieval_metrics.py（人工修正字段优先级高于模型判定，审核后无需改脚本）。

用法：
  python -m tools.data_scripts.retrieval_prelabel --count 30
  python -m tools.data_scripts.retrieval_prelabel --count 5 --collection research_reports_v3_full
  python -m tools.data_scripts.retrieval_prelabel --no-llm      # 只取片段、跳过 LLM（零额度）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_JSON = "训练结果数据/retrieval_prelabel_20260910.json"
OUT_MD = "docs/评估报告/retrieval标注_待审核.md"
OUT_TPL = "docs/评估报告/retrieval标注_模板.md"

JUDGE_PROMPT = """你是检索标注助手。下面给出一个问题，以及检索系统召回的 {n} 个片段（已按排序编号）。

判据：该片段是否包含**回答该问题所需的信息或数值**（不是"主题相近"就算相关；泛泛介绍行业趋势但没有问题所需数据/结论的，判不相关）。

问题：{q}

片段列表：
{chunks}

只输出 JSON 数组，不要任何解释文字，格式：
[{{"idx": 1, "relevant": true, "reason": "含2024年营收具体数值"}}, ...]
要求：每个片段一条，idx 从 1 到 {n}，按顺序给出。"""


def _stdout_utf8() -> None:
    """Windows 控制台统一 UTF-8 输出"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_sub_questions() -> List[Dict[str, Any]]:
    """从 golden v1 展开 108 个子问题（带所属编号与问题类型）"""
    from eval import golden as golden_mod

    golden = golden_mod.load_golden("v1")
    rows: List[Dict[str, Any]] = []
    for it in golden.get("items") or []:
        for i, sub in enumerate(it.get("子问题") or []):
            if sub and str(sub).strip():
                rows.append(
                    {
                        "bid": str(it.get("编号")),
                        "问题类型": it.get("问题类型"),
                        "sub_idx": i + 1,
                        "子问题": str(sub).strip(),
                    }
                )
    return rows


def stratified_sample(rows: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    """按问题类型分层轮转抽样（固定种子 → 可复现）"""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get("问题类型") or "未分类")].append(r)
    queues = {k: sorted(v, key=lambda x: (x["bid"], x["sub_idx"])) for k, v in buckets.items()}
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    keys = sorted(queues)
    while len(picked) < count:
        progressed = False
        for k in keys:
            if len(picked) >= count:
                break
            if queues[k]:
                picked.append(queues[k].pop(rng.randrange(len(queues[k]))))
                progressed = True
        if not progressed:
            break
    return picked


def build_retriever(collection: str, top_k: int):
    """构造与线上同口径的混合检索器（向量 + BM25 RRF）"""
    from config.rag_config import RAGConfig
    from core.retrievers import HandwrittenRetriever, HybridRetriever
    from pipelines.rag_pipeline import RAGPipeline

    config = RAGConfig(QDRANT_COLLECTION_NAME=collection)
    pipeline = RAGPipeline(config)
    bm25 = pipeline.build_bm25_index()
    vector_retriever = HandwrittenRetriever(
        embedding_client=pipeline.embedding_client,
        qdrant_client=pipeline.qdrant_client,
        top_k=top_k,
    )
    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        top_k=top_k,
        rrf_k=config.HYBRID_RRF_K,
        topk_vector=config.HYBRID_TOPK_VECTOR,
        topk_bm25=config.HYBRID_TOPK_BM25,
        vector_floor_ratio=config.HYBRID_VECTOR_FLOOR_RATIO,
    )
    return hybrid


def chunks_of(results: List[Dict[str, Any]], head_chars: int = 400) -> List[Dict[str, Any]]:
    """把检索结果规整为标注用片段结构"""
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(results, start=1):
        payload = r.get("payload") or {}
        text = str(payload.get("content") or "")
        out.append(
            {
                "rank": i,
                "file_path": str(payload.get("file_path") or ""),
                "score": r.get("score"),
                "text_head": text[:head_chars],
                "_text_full_chars": len(text),
            }
        )
    return out


def _parse_judge_json(raw: str, n: int) -> Optional[List[Dict[str, Any]]]:
    """从模型输出里稳健解析 JSON 数组"""
    if not raw:
        return None
    txt = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", txt, re.S)
    if fence:
        txt = fence.group(1).strip()
    start, end = txt.find("["), txt.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(txt[start : end + 1])
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, list):
        return None
    by_idx: Dict[int, Dict[str, Any]] = {}
    for d in data:
        if isinstance(d, dict) and isinstance(d.get("idx"), int):
            by_idx[d["idx"]] = d
    return [by_idx.get(i, {}) for i in range(1, n + 1)]


def judge_chunks(llm: Any, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """LLM 预标注；失败时返回 judge_relevant=None（不阻断整轮）"""
    blocks = []
    for c in chunks:
        blocks.append("[%d] 文件: %s\n正文: %s" % (c["rank"], c["file_path"], c["text_head"]))
    prompt = JUDGE_PROMPT.format(n=len(chunks), q=question, chunks="\n\n".join(blocks))
    for attempt in (1, 2):
        try:
            resp = llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            parsed = _parse_judge_json(raw, len(chunks))
            if parsed:
                vals = []
                for c, p in zip(chunks, parsed):
                    item = dict(c)
                    item["judge_relevant"] = p.get("relevant") if isinstance(p.get("relevant"), bool) else None
                    item["judge_reason"] = str(p.get("reason") or "")
                    item["人工修正"] = None
                    vals.append(item)
                return vals
            print("    [warn] 预标注 JSON 解析失败（第 %d 次），%s" % (attempt, "重试" if attempt == 1 else "放弃"))
        except Exception as exc:  # noqa: BLE001
            print("    [warn] 预标注调用失败（第 %d 次）: %s" % (attempt, exc))
        time.sleep(2)
    out = []
    for c in chunks:
        item = dict(c)
        item["judge_relevant"] = None
        item["judge_reason"] = "judge_error（人工审核时按判据直接标注）"
        item["人工修正"] = None
        out.append(item)
    return out


def write_template(path: Path) -> None:
    """写空白标注模板（B-24B 用）"""
    md = """# retrieval 评测集标注模板（B-24 · 人工审核用）

> 生成方式：本文件为空白模板；实际待审核清单一轮一批由 `tools/data_scripts/retrieval_prelabel.py` 生成到
> `docs/评估报告/retrieval标注_待审核.md`，审核完成后另存为 `retrieval标注_<日期>_人工终稿.md`。

## 1. 口径与判据

- **样本**：golden v1 的 108 子问题中分层抽 30 题（种子固定，可复现）；
- **检索**：混合检索（向量 + BM25 RRF），top-K = 10，与线上主链路同口径；
- **相关性判据（草案，B-24B 定稿）**：
  1. 片段包含回答问题所需的**数值/结论** → 相关；
  2. 仅主题相近、泛泛介绍行业趋势 → 不相关；
  3. 片段给出同一指标但**年份/期间/主体不符** → 不相关（可在备注写明）；
  4. 判据粒度：默认片段级；若需句子级，在备注中标注定位。

## 2. 审核表（逐题填写）

| 字段 | 含义 | 取值 |
| --- | --- | --- |
| 排名 | 检索排序位次（1 = 最相关候选） | 1-10 |
| 来源文件 | payload.file_path 的文件名 | 文本 |
| 模型判定 | B-24A 预标注结果（**非真值**） | 相关 / 不相关 / judge_error |
| 模型理由 | 预标注给出的理由 | 文本 |
| **人工修正** | 审核结论；与模型一致时填「同」，不一致时填「相关」或「不相关」 | 同 / 相关 / 不相关 |
| 备注 | 判据边界、句子级定位、争议点 | 文本 |

## 3. 头部元信息（每次审核填写）

| 项 | 值 |
| --- | --- |
| 审核人 | |
| 审核日期 | |
| 样本范围（题数 / 种子） | |
| 模型改判比例 | |
| 判据补充说明 | |

## 4. 结论（审核完成后填写）

| 指标 | 值 |
| --- | --- |
| 相关片段数 / 标注片段数 | |
| Hit Rate@10 / Precision@10 / MRR | 用 `python eval/retrieval_metrics.py --labels <终稿 JSON>` 计算 |
| 调参建议 | |
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(md)


def write_review_md(path: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """写待审核清单（人工在「人工修正」列填写）"""
    L: List[str] = []
    L.append("# retrieval 标注 · 待审核清单（B-24A 模型预标注）")
    L.append("")
    L.append("> **口径声明**：下表 `模型判定` 与 `模型理由` 均为 **LLM 预标注（初筛）**，不是 ground truth；")
    L.append("> 请逐条在 `人工修正` 列填「同 / 相关 / 不相关」，改判项在 `备注` 说明理由。判据见 `retrieval标注_模板.md`。")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("| --- | --- |")
    for k, v in meta.items():
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## 逐题清单")
    L.append("")
    for r in rows:
        L.append("### %s · 子问题 %d ｜ %s" % (r["bid"], r["sub_idx"], r.get("问题类型") or "-"))
        L.append("")
        L.append("**问题**：%s" % r["子问题"])
        L.append("")
        L.append("| 排名 | 来源文件 | 模型判定 | 模型理由 | 人工修正 | 备注 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for c in r["chunks"]:
            jr = c.get("judge_relevant")
            label = "相关" if jr is True else ("不相关" if jr is False else "judge_error")
            fname = os.path.basename(c.get("file_path") or "") or "-"
            reason = (c.get("judge_reason") or "").replace("|", "\\|")[:60]
            L.append("| %d | %s | %s | %s | | |" % (c["rank"], fname, label, reason))
        L.append("")
        L.append("审核人：　　　　　审核日期：　　　　　题目结论（是否判据需补充）：")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 审核完成后")
    L.append("")
    L.append("1. 把「人工修正」列的填写结果同步回 JSON 的 `人工修正` 字段（`true`/`false`/`null`）；")
    L.append("2. 跑 `python eval/retrieval_metrics.py --labels <终稿 JSON> --json-out 训练结果数据/retrieval_metrics_human.json`；")
    L.append("3. 结果回填 `docs/详细设计方案_智能问数助手系统.md` §6.7.2 与方案 §6.4 检索行。")
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(L) + "\n")


def main() -> int:
    """入口：抽样 → 检索 → 预标注 → 落 JSON/MD → 草稿指标"""
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="B-24A retrieval 评测集模型预标注")
    ap.add_argument("--count", type=int, default=30, help="抽样题数（默认 30）")
    ap.add_argument("--seed", type=int, default=20260910, help="抽样种子（固定 → 可复现）")
    ap.add_argument("--top-k", type=int, default=10, help="每题召回片段数（默认 10）")
    ap.add_argument("--collection", default="research_reports_v3_full", help="Qdrant 集合")
    ap.add_argument("--judge-model", default=os.getenv("PREFILTER_MODEL", "qwen-flash"), help="预标注模型")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM（零额度，只取片段）")
    ap.add_argument("--json-out", default=OUT_JSON, help="预标注 JSON 输出路径")
    args = ap.parse_args()

    rows = stratified_sample(load_sub_questions(), args.count, args.seed)
    print("[prelabel] 子问题池已加载，抽样 %d 题（seed=%s）" % (len(rows), args.seed))

    print("[prelabel] 构造混合检索（集合 %s，top-K=%d）…" % (args.collection, args.top_k))
    retriever = build_retriever(args.collection, args.top_k)

    llm = None
    if not args.no_llm:
        from langchain_openai import ChatOpenAI
        from config.rag_config import get_config

        cfg = get_config()
        llm = ChatOpenAI(
            model=args.judge_model,
            api_key=cfg.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
        )
        print("[prelabel] 预标注模型：%s" % args.judge_model)

    for r in rows:
        t0 = time.time()
        hits = retriever.retrieve(r["子问题"], top_k=args.top_k)
        chunks = chunks_of(hits)
        r["chunks"] = judge_chunks(llm, r["子问题"], chunks) if llm else [
            {**c, "judge_relevant": None, "judge_reason": "未跑 LLM（--no-llm）", "人工修正": None} for c in chunks
        ]
        n_rel = sum(1 for c in r["chunks"] if c.get("judge_relevant") is True)
        print("  [%s-%d] 召回 %d 片段，预标注相关 %d ｜ %.1fs" % (r["bid"], r["sub_idx"], len(chunks), n_rel, time.time() - t0))
    return finish(args, rows)


def finish(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> int:
    """落盘 JSON / MD / 模板，并输出草稿指标"""
    n_rel = sum(1 for r in rows for c in r["chunks"] if c.get("judge_relevant") is True)
    n_chunks = sum(len(r["chunks"]) for r in rows)
    n_err = sum(1 for r in rows for c in r["chunks"] if c.get("judge_relevant") is None)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "golden_version": "v1",
        "seed": args.seed,
        "collection": args.collection,
        "top_k": args.top_k,
        "judge_model": None if args.no_llm else args.judge_model,
        "口径": "模型预标注（LLM 初筛），非 ground truth；人工在 人工修正 字段审核后重算指标",
        "questions": rows,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    meta = {
        "生成时间": payload["generated_at"],
        "golden 版本": "v1（108 子问题）",
        "集合": args.collection,
        "抽样": "%d 题 · seed=%s（分层抽样，可复现）" % (len(rows), args.seed),
        "top-K": args.top_k,
        "预标注模型": "未跑（--no-llm）" if args.no_llm else args.judge_model,
        "预标注片段数": "%d（其中判相关 %d，judge_error %d）" % (n_chunks, n_rel, n_err),
        "JSON": args.json_out,
    }
    write_review_md(Path(OUT_MD), rows, meta)
    write_template(Path(OUT_TPL))

    print("[prelabel] 预标注片段 %d 条（相关 %d，judge_error %d）" % (n_chunks, n_rel, n_err))
    print("[prelabel] JSON：%s" % out)
    print("[prelabel] 待审核清单：%s" % OUT_MD)
    print("[prelabel] 空白模板：%s" % OUT_TPL)

    try:
        from eval.retrieval_metrics import evaluate

        res = evaluate([{**r, "编号": r["bid"]} for r in rows], k_values=(args.top_k,))
        s = res["summary"]
        print("[prelabel] 草稿指标（模型预标注口径，未经人工审核，仅供参考）：")
        print("  Hit Rate@%d=%.4f ｜ Precision@%d=%.4f ｜ MRR=%.4f ｜ 题数 %d（跳过 %d）"
              % (args.top_k, s["Hit Rate@%d" % args.top_k], args.top_k, s["Precision@%d" % args.top_k], s["MRR"], s["题数"], s["跳过题数"]))
    except Exception as exc:  # noqa: BLE001
        print("[prelabel] 草稿指标计算跳过: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())