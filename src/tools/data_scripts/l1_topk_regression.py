"""L1 引用核验回归：量化 TABLE_AGG_TOPK 对管线引用命中率的影响（2026-08-23）

对 golden v1（80 题 / 108 子问题）逐子问题跑管线"引用路径"：
检索 → 软过滤 → 表聚合 → Rerank → 引用构建 → 引用核验过滤（生成阶段 monkeypatch 为空，不产生 LLM 答案）。
收集每条查询返回的 references，再用 L1 核验（文件可溯源 + 数字命中，comma 口径）汇总，
对比 TABLE_AGG_TOPK=20（默认收敛） vs TABLE_AGG_TOPK=0（全量聚合）。

用法:
  python tools/data_scripts/l1_topk_regression.py            # 全量 108 子问题，跑 topk=20 与 topk=0
  python tools/data_scripts/l1_topk_regression.py --limit 6  # 小样本冒烟
  python tools/data_scripts/l1_topk_regression.py --topk 20  # 只跑指定配置

输出:
  训练结果数据/l1_topk{topk}_refs.json            # 逐题管线引用
  训练结果数据/l1_topk{topk}_refs_citation_report.json  # L1 逐条核验明细
  控制台对比汇总（traceable_rate / num_rate + 差值）
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

# 确保以脚本方式运行时能导入仓库根下的包
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from pathlib import Path
from typing import Any, Dict, List

from config.rag_config import RAGConfig
from pipelines.citation_validator import CitationValidator
from pipelines.rag_pipeline import RAGPipeline

GOLDEN = Path("database/golden/v1_2026-08-22.json")
OUT_DIR = Path("训练结果数据")


def load_sub_questions() -> List[Dict[str, str]]:
    """从 golden v1 快照加载全部子问题（108 条）"""
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    subs: List[Dict[str, str]] = []
    for item in data["items"]:
        for q in item["子问题"]:
            subs.append({"编号": item["编号"], "类型": item["问题类型"], "问题": q})
    return subs


def make_pipeline(topk: int) -> RAGPipeline:
    """构造指定聚合收敛档位的管线；生成阶段置空，只保留引用路径"""
    cfg = RAGConfig()
    cfg.TABLE_AGG_TOPK = topk
    pipeline = RAGPipeline(cfg)
    pipeline.generator.generate = lambda **kw: ""  # type: ignore[assignment]
    return pipeline


def collect_references(pipeline: RAGPipeline, question: str) -> List[Dict[str, Any]]:
    """跑单题引用路径，吞掉 Rerank 过程打印，返回管线引用"""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        result = pipeline.query(question, verbose=False)
    finally:
        sys.stdout = old_stdout
    if isinstance(result, dict):
        return result.get("references", []) or []
    return []


def run_config(topk: int, subs: List[Dict[str, str]], limit: int | None = None) -> List[Dict[str, Any]]:
    """对子问题列表跑一遍引用路径，返回逐题结果"""
    scope = subs[:limit] if limit else subs
    print(f"topk={topk}: 开始 {len(scope)} 题（pipeline 初始化含 BM25 加载）", file=sys.stderr, flush=True)
    pipeline = make_pipeline(topk)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, s in enumerate(scope):
        refs = collect_references(pipeline, s["问题"])
        results.append({"编号": s["编号"], "类型": s["类型"], "问题": s["问题"], "references": refs})
        if (i + 1) % 10 == 0 or (i + 1) == len(scope):
            el = time.time() - t0
            print(f"topk={topk}: {i + 1}/{len(scope)} 完成，用时 {el:.0f}s", file=sys.stderr, flush=True)
    return results


def summarize_refs(refs: List[Dict[str, Any]], match_mode: str) -> Dict[str, Any]:
    """对收集到的引用做 L1 汇总（文件可溯源 + 数字命中）"""
    validator = CitationValidator(corpus_root=RAGConfig().CITATION_CORPUS_ROOT, match_mode=match_mode)
    records = validator.check_references(refs)
    summary = validator.summarize(records)
    # 补充：表格占位引用（text="这是一个表格"，无数字，不参与数字命中分母）占比
    placeholder = sum(1 for r in refs if r.get("text", "").strip() == "这是一个表格")
    summary["total_placeholder"] = placeholder
    summary["total_number_bearing"] = len(refs) - placeholder
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1 引用核验回归：TABLE_AGG_TOPK 对比")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    parser.add_argument("--topk", nargs="+", type=int, default=[20, 0], help="要对比的 TABLE_AGG_TOPK 档位")
    args = parser.parse_args(argv)

    subs = load_sub_questions()
    match_mode = RAGConfig().CITATION_MATCH_MODE
    print(f"golden: {len(subs)} 子问题, match_mode={match_mode}", file=sys.stderr, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: Dict[int, Dict[str, Any]] = {}
    for topk in args.topk:
        results = run_config(topk, subs, args.limit or None)
        refs_path = OUT_DIR / f"l1_topk{topk}_refs.json"
        refs_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"topk={topk}: 逐题引用已写 {refs_path}", file=sys.stderr, flush=True)

        all_refs = [r for item in results for r in item["references"]]
        summary = summarize_refs(all_refs, match_mode)
        summaries[topk] = summary
        report_path = OUT_DIR / f"l1_topk{topk}_refs_citation_report.json"
        validator = CitationValidator(corpus_root=RAGConfig().CITATION_CORPUS_ROOT, match_mode=match_mode)
        records = validator.check_references(all_refs)
        report_path.write_text(json.dumps({"records": records, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"topk={topk}: L1 报告已写 {report_path}", file=sys.stderr, flush=True)

    # 对比汇总
    print("\n========== L1 引用核验对比（管线引用路径） ==========")
    print(f"{'档位':<8}{'引用数':<8}{'表格占位':<8}{'可溯源率':<10}{'数字命中率':<10}{'含数字引用子集命中率':<14}")
    prev: Dict[str, Any] | None = None
    for topk in args.topk:
        s = summaries[topk]
        nb_rate = s["num_rate"] if s["num_rate"] is not None else 0.0
        print(f"topk={topk:<5}{s['total']:<8}{s['total_placeholder']:<8}"
              f"{s['traceable_rate'] * 100:<9.1f}%{nb_rate * 100:<9.1f}%{nb_rate * 100:<13.1f}%")
        if prev is not None:
            print(f"  Δ traceable_rate: {(s['traceable_rate'] - prev['traceable_rate']) * 100:+.1f}pp"
                  f"  Δ num_rate: {(s['num_rate'] - prev['num_rate']) * 100 if s['num_rate'] is not None and prev['num_rate'] is not None else 0:+.1f}pp")
        prev = s
    print(f"（对照：官方参考答案引用 references_all.json 基线为 99.3% 可溯源 / 82.8% 数字命中，与管线引用口径不同，仅参考）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
