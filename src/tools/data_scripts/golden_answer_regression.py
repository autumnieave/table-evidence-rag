"""答案质量全量回归（golden v1，108 子问题，真实生成）——2026-08-23

目的：验证 enable_thinking=False（关闭 qwen3.5-plus 思考模式）后答案质量未回退。
对 golden v1 全部子问题跑完整 query()（真实生成，非流式），收集：
1. 基础质量：答案非空率 / 失败率 / 平均耗时 / 平均长度
2. 引用层：L1 文件可溯源率 + 引用文本数字命中率（与 l1_topk_regression 同口径）
3. 端到端答案数字可溯源：从答案提取数字（过滤年份），检查是否出现在该题任一引用源文件中
   （comma 归一化 + 数字内空白折叠 + 单位换算变体归一化；报告同时给出原始口径与归一化口径）
4. 抽检样例：按问题类型取代表性答案供人工浏览

用法:
  python tools/data_scripts/golden_answer_regression.py            # 全量 108 题
  python tools/data_scripts/golden_answer_regression.py --limit 5  # 冒烟
输出:
  训练结果数据/golden_answer_regression.json        # 逐题明细（答案+引用+指标）
  docs/答案质量回归报告.md                          # 汇总报告（含抽检样例）
"""

import argparse
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.rag_config import RAGConfig
from pipelines.citation_validator import CitationValidator
from pipelines.rag_pipeline import RAGPipeline

GOLDEN = Path("database/golden/v1_2026-08-22.json")
OUT_JSON = Path("训练结果数据/golden_answer_regression.json")
OUT_MD = Path("docs/答案质量回归报告.md")
SAMPLE_PER_TYPE = 3  # 每个问题类型最多抽 3 条样例


def load_sub_questions() -> List[Dict[str, str]]:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    subs: List[Dict[str, str]] = []
    for item in data["items"]:
        for q in item["子问题"]:
            subs.append({"编号": item["编号"], "类型": item["问题类型"], "问题": q})
    return subs


def extract_answer_numbers(text: str) -> List[str]:
    """提取答案中的数字并过滤年份（1900-2100 的 4 位整数视为年份）"""
    nums = CitationValidator.extract_numbers(text)
    return [n for n in nums if not (n.isdigit() and len(n) == 4 and 1900 <= int(n) <= 2100)]


class AnswerEvaluator:
    """端到端答案评估：引用 L1 + 答案数字可溯源"""

    def __init__(self, match_mode: str = "comma"):
        self.validator = CitationValidator(
            corpus_root=RAGConfig().CITATION_CORPUS_ROOT, match_mode=match_mode
        )
        self._content_cache: Dict[str, str] = {}

    def _file_content(self, located: Optional[str]) -> str:
        if not located:
            return ""
        if located not in self._content_cache:
            try:
                with open(located, "r", encoding="utf-8", errors="ignore") as fh:
                    self._content_cache[located] = self.validator._normalize_for_match(fh.read())
            except OSError:
                self._content_cache[located] = ""
        return self._content_cache[located]

    def evaluate_answer(self, answer: str, refs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估单题答案：引用 L1 汇总 + 答案数字可溯源"""
        records = self.validator.check_references(refs) if refs else []
        summary = self.validator.summarize(records)
        ans_nums = extract_answer_numbers(answer)
        # 端到端：每个答案数字须出现在任一引用源文件（comma 归一化）
        located_contents = [self._file_content(r.get("located")) for r in records]
        hit_nums = []
        unit_hit_nums = []
        unhit_nums = []
        for n in ans_nums:
            if any(n in c for c in located_contents if c):
                hit_nums.append(n)
                continue
            # 严格口径未命中时，尝试单位换算变体（百万元/万元/千万元 <-> 亿元）
            if any(
                self.validator.number_in_text(n, c, accept_unit_variants=True)[0]
                for c in located_contents if c
            ):
                hit_nums.append(n)
                unit_hit_nums.append(n)
            else:
                unhit_nums.append(n)
        return {
            "ref_total": summary["total"],
            "ref_traceable": summary["traceable"],
            "ref_traceable_rate": summary["traceable_rate"],
            "ref_num_total": summary["num_total"],
            "ref_num_hit": summary["num_hit"],
            "ref_num_rate": summary["num_rate"],
            "ans_num_total": len(ans_nums),
            "ans_num_hit": len(hit_nums),
            "ans_num_rate": round(len(hit_nums) / len(ans_nums), 4) if ans_nums else None,
            "ans_num_rate_raw": (
                round((len(hit_nums) - len(unit_hit_nums)) / len(ans_nums), 4) if ans_nums else None
            ),
            "ans_unit_hit": len(unit_hit_nums),
            "ans_unhit_numbers": unhit_nums[:20],
        }


def run(pipeline: RAGPipeline, evaluator: AnswerEvaluator, subs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, s in enumerate(subs):
        t1 = time.time()
        try:
            answer = pipeline.query(s["问题"], verbose=False)
            if isinstance(answer, str):
                content, refs = answer, []
            else:
                content = answer.get("content", "") or ""
                refs = answer.get("references", []) or []
            ok = bool(content.strip())
            err = ""
        except Exception as e:  # noqa: BLE001
            content, refs, ok, err = "", [], False, f"{type(e).__name__}: {e}"
        eval_ = evaluator.evaluate_answer(content, refs)
        results.append({
            "编号": s["编号"], "类型": s["类型"], "问题": s["问题"],
            "ok": ok, "error": err, "耗时": round(time.time() - t1, 1),
            "答案": content, "引用": refs, **eval_,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(subs):
            el = time.time() - t0
            print(f"{i + 1}/{len(subs)} 完成，用时 {el:.0f}s", file=sys.stderr, flush=True)
    return results


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    ok = sum(1 for r in results if r["ok"])
    ref_trace = sum(r["ref_traceable"] for r in results)
    ref_total = sum(r["ref_total"] for r in results)
    ref_num_hit = sum(r["ref_num_hit"] for r in results)
    ref_num_total = sum(r["ref_num_total"] for r in results)
    ans_hit = sum(r["ans_num_hit"] for r in results)
    ans_unit_hit = sum(r["ans_unit_hit"] for r in results)
    ans_total = sum(r["ans_num_total"] for r in results)
    avg_time = sum(r["耗时"] for r in results) / n if n else 0
    avg_len = sum(len(r["答案"]) for r in results) / n if n else 0
    fail_types = Counter()
    for r in results:
        if not r["ok"]:
            fail_types[r["类型"]] += 1
    return {
        "questions": n, "ok": ok, "ok_rate": round(ok / n, 4),
        "fail_by_type": dict(fail_types),
        "ref_total": ref_total, "ref_traceable": ref_trace,
        "ref_traceable_rate": round(ref_trace / ref_total, 4) if ref_total else None,
        "ref_num_total": ref_num_total, "ref_num_hit": ref_num_hit,
        "ref_num_rate": round(ref_num_hit / ref_num_total, 4) if ref_num_total else None,
        "ans_num_total": ans_total, "ans_num_hit": ans_hit,
        "ans_num_rate": round(ans_hit / ans_total, 4) if ans_total else None,
        "ans_num_rate_raw": round((ans_hit - ans_unit_hit) / ans_total, 4) if ans_total else None,
        "ans_unit_hit": ans_unit_hit,
        "avg_time": round(avg_time, 1), "avg_len": round(avg_len, 1),
    }


def pick_samples(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按问题类型抽样（每种最多 SAMPLE_PER_TYPE 条，取 ok 且答案较完整的）"""
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        if r["ok"] and len(r["答案"]) > 50:
            by_type.setdefault(r["类型"], []).append(r)
    samples = []
    for _, items in sorted(by_type.items()):
        samples.extend(items[:SAMPLE_PER_TYPE])
    return samples


def render_md(summary: Dict[str, Any], results: List[Dict[str, Any]], samples: List[Dict[str, Any]]) -> str:
    def pct(x: Optional[float]) -> str:
        return "—" if x is None else f"{x * 100:.1f}%"

    lines = [
        "# 答案质量回归报告（golden 108 子问题，真实生成）",
        "",
        f"> 2026-08-23 · 全量生成（qwen3.5-plus, enable_thinking=False）· 口径：L1 comma 归一化",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 题目数 | {summary['questions']} |",
        f"| 答案非空率 | {pct(summary['ok_rate'])}（{summary['ok']}/{summary['questions']}） |",
        f"| 平均耗时 / 平均长度 | {summary['avg_time']}s / {summary['avg_len']} 字符 |",
        f"| 引用文件可溯源率（L1） | {pct(summary['ref_traceable_rate'])}（{summary['ref_traceable']}/{summary['ref_total']}） |",
        f"| 引用文本数字命中率（L1） | {pct(summary['ref_num_rate'])}（{summary['ref_num_hit']}/{summary['ref_num_total']}） |",
        f"| **端到端答案数字可溯源率（归一化口径）** | **{pct(summary['ans_num_rate'])}**（{summary['ans_num_hit']}/{summary['ans_num_total']}，含单位换算 {summary['ans_unit_hit']} 个） |",
        f"| 端到端答案数字可溯源率（原始口径） | {pct(summary['ans_num_rate_raw'])}（{summary['ans_num_hit'] - summary['ans_unit_hit']}/{summary['ans_num_total']}） |",
        "",
        "> 端到端口径：答案中的每个数字须出现在该题任一引用源文件中。归一化口径 = comma 逗号归一化",
        "> + 数字内空白折叠（LaTeX 排版）+ 单位换算变体（百万元/万元/千万元 <-> 亿元）；原始口径不含单位换算，仅作对照。",
        "",
    ]
    if summary["fail_by_type"]:
        lines += ["### 失败分布", "", "| 类型 | 失败数 |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in sorted(summary["fail_by_type"].items())]
        lines += [""]
    lines += ["## 抽检样例", ""]
    for s in samples:
        unhit = "，".join(s["ans_unhit_numbers"][:8]) if s["ans_unhit_numbers"] else "无"
        lines += [
            f"### {s['编号']}（{s['类型']}）· {s['耗时']}s",
            f"**问题**：{s['问题']}",
            f"**答案**：{s['答案'][:600]}",
            f"**引用**：{s['ref_total']} 条（可溯源 {s['ref_traceable']}）· 答案未溯源数字：{unhit}",
            "",
        ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="答案质量全量回归")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    args = parser.parse_args(argv)
    subs = load_sub_questions()
    scope = subs[:args.limit] if args.limit else subs
    print(f"golden: {len(scope)} 子问题", file=sys.stderr, flush=True)

    pipeline = RAGPipeline(RAGConfig())
    evaluator = AnswerEvaluator(match_mode=RAGConfig().CITATION_MATCH_MODE)
    results = run(pipeline, evaluator, scope)

    summary = aggregate(results)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "items": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    samples = pick_samples(results) if not args.limit else results
    OUT_MD.write_text(render_md(summary, results, samples), encoding="utf-8")
    print(f"明细已写 {OUT_JSON}", file=sys.stderr, flush=True)
    print(f"报告已写 {OUT_MD}", file=sys.stderr, flush=True)

    print("\n========== 答案质量回归汇总 ==========")
    print(f"题目={summary['questions']} 非空率={pct(summary['ok_rate'])} 平均耗时={summary['avg_time']}s 平均长度={summary['avg_len']}字符")
    print(f"引用可溯源={pct(summary['ref_traceable_rate'])} 引用数字命中={pct(summary['ref_num_rate'])}")
    print(f"端到端答案数字可溯源={pct(summary['ans_num_rate'])}（{summary['ans_num_hit']}/{summary['ans_num_total']}）")
    if summary["fail_by_type"]:
        print(f"失败：{summary['fail_by_type']}")
    return 0


def pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


if __name__ == "__main__":
    sys.exit(main())
