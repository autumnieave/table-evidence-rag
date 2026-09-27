# -*- coding: utf-8 -*-
"""eval/retrieval_metrics.py —— 检索排序质量指标（Hit Rate@K / Precision@K / MRR）

用途（B-24A）：把「检索到底排得准不准」从黑盒（只看生成答案）升级为可量化指标。
- 输入 = 每题的**有序相关性标注**（retrieval top-K 逐条 relevant: true/false，顺序即排序）；
- 输出 = 每题 Hit Rate@K / Precision@K / MRR + 汇总均值。

口径说明（重要）：
- 本模块**只做计算**，不产生标注；标注来源既可以是模型预标注（B-24A 草稿口径），
  也可以是人工终稿（B-24B 正式口径），调用方需在报告里写清是哪一种；
- 相关性判据（建议）：片段中是否包含回答问题所需的信息/数值（而非仅仅主题相近），
  最终以 B-24B 定义为准；
- Hit Rate@K 的分母 = 该题**标注集内**相关的片段总数（若标注集本身只覆盖 top-K，则等价于
  「top-K 内相关片段的加权占比」，此时 Hit Rate@K 与 Precision@K 同源，需在报告中注明）；
  **本指标不等价于标准 Recall@K**——标准 Recall 的分母需全语料 ground truth，本模块不提供。

B-24B 扩展（2026-09-27，四态人工判定 + 新口径）：
- 相关性四态：完全相关 / 部分相关 / 不相关 / 未判定；`relevance_state()` 负责归一
  （优先级：人工判定 > 人工修正 > judge_relevant）；
- 部分相关权重 0.5：Hit Rate 分子、Precision 分子均按 0.5 计；Hit Rate 分母含部分相关；
- Precision 分母只数 `计入Precision != False` 的条数（B-24B 判据 3：同文件第 3 条起不计入）；
- **与 legacy 口径（B-24A 二值模型判定）不可直接对比**——定义不同，CLI 输出的 `口径` 字段已注明。

用法：
  python eval/retrieval_metrics.py --labels 训练结果数据/retrieval_prelabel_20260910.json
  python eval/retrieval_metrics.py --labels xxx.json --k 10 --k20 20 --json-out 训练结果数据/retrieval_metrics.json
  python -m eval.retrieval_metrics --labels xxx.json       # 等价
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _stdout_utf8() -> None:
    """Windows 控制台统一 UTF-8 输出"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def recall_at_k(ranked_relevant: Sequence[bool], k: int = 10, total_relevant: int | None = None) -> float:
    """Hit Rate@K：前 K 条里命中的相关片段数 / 相关片段总数

    total_relevant 缺省时取 len(ranked_relevant)（适用于"标注集只覆盖 top-K"的场景，
    此时等价于「top-K 内相关占比」，报告中须注明分母口径）。
    """
    k = max(0, int(k))
    topk = list(ranked_relevant)[:k]
    hits = sum(1 for x in topk if x)
    denom = total_relevant if total_relevant is not None else len(ranked_relevant)
    if denom <= 0:
        return 0.0
    return hits / denom


def precision_at_k(ranked_relevant: Sequence[bool], k: int = 10) -> float:
    """Precision@K：前 K 条里相关片段数 / K（K 大于实际返回条数时按实际条数算）"""
    k = max(0, int(k))
    topk = list(ranked_relevant)[:k]
    if not topk:
        return 0.0
    return sum(1 for x in topk if x) / len(topk)


def reciprocal_rank(ranked_relevant: Sequence[bool]) -> float:
    """RR：首个相关片段的排名倒数（无相关片段 → 0）"""
    for i, ok in enumerate(ranked_relevant, start=1):
        if ok:
            return 1.0 / i
    return 0.0


# --------------------------------------------------------------------------
# B-24B：四态人工判定 + 新口径（部分相关 0.5；Precision 只数「计入」条数）
# --------------------------------------------------------------------------
JUDGMENT_FULL = "完全相关"
JUDGMENT_PARTIAL = "部分相关"
JUDGMENT_NONE = "不相关"
JUDGMENTS = (JUDGMENT_FULL, JUDGMENT_PARTIAL, JUDGMENT_NONE)

PARTIAL_WEIGHT = 0.5


def _judged_caliber(partial_weight: float = PARTIAL_WEIGHT) -> Dict[str, Any]:
    """B-24B 口径说明（写进 JSON，便于报告与面试口径一致）"""
    return {
        "模式": "judged（四态人工判定，B-24B 定稿 2026-09-27）",
        "相关性四态": "完全相关 / 部分相关 / 不相关 / 未判定（空）",
        "部分相关权重": partial_weight,
        "公式": {
            "Hit Rate@K": "分母 = 标注集内 完全相关+部分相关 的条数；分子 = top-K 内 完全相关×1.0 + 部分相关×%.1f" % partial_weight,
            "Precision@K": "分母 = top-K 内 计入Precision≠False 的条数；分子 = 其中 完全相关×1.0 + 部分相关×%.1f" % partial_weight,
            "MRR": "首个 完全相关或部分相关 片段的排名倒数（无则 0）",
        },
        "判据依据": "判据1（数值/结论 + 主体期间一致）；判据2（部分相关 0.5）；判据3（Precision 同文件最多 2 条，按 rank 序次）",
        "提示": [
            "标注集只覆盖检索 top-10，不是语料全集 → Hit Rate@K 的分母为标注集内相关数，不等价于标准 Recall@K（需全语料 ground truth）",
            "本口径与 B-24A 草稿口径（二值模型判定）不可直接对比：定义不同，数值差异不代表质量变化",
        ],
    }


def relevance_state(chunk: Dict[str, Any]) -> Optional[str]:
    """片段相关性四态：完全相关 / 部分相关 / 不相关 / None（未判定）

    优先级：`人工判定`（四态）> `人工修正`（bool）> `judge_relevant`（bool，B-24A 模型预标注）。
    """
    judged = chunk.get("人工判定")
    if judged in JUDGMENTS:
        return judged
    if chunk.get("人工修正") is True:
        return JUDGMENT_FULL
    if chunk.get("人工修正") is False:
        return JUDGMENT_NONE
    if chunk.get("judge_relevant") is True:
        return JUDGMENT_FULL
    if chunk.get("judge_relevant") is False:
        return JUDGMENT_NONE
    return None


def _credit(state: Optional[str], partial_weight: float = PARTIAL_WEIGHT) -> float:
    """完全相关=1.0；部分相关=partial_weight；其余=0.0"""
    if state == JUDGMENT_FULL:
        return 1.0
    if state == JUDGMENT_PARTIAL:
        return partial_weight
    return 0.0


def evaluate_judged_question(
    chunks: List[Dict[str, Any]],
    k_values: Sequence[int] = (10,),
    partial_weight: float = PARTIAL_WEIGHT,
) -> Dict[str, Any]:
    """单题（B-24B 口径）：四态判定 + 部分相关 0.5 + Precision 只数「计入」条数"""
    states = [relevance_state(c) for c in chunks]
    included = [c.get("计入Precision") is not False for c in chunks]
    full = sum(1 for s in states if s == JUDGMENT_FULL)
    part = sum(1 for s in states if s == JUDGMENT_PARTIAL)
    total_relevant = full + part
    first = next((i for i, s in enumerate(states, 1) if s in (JUDGMENT_FULL, JUDGMENT_PARTIAL)), None)
    out: Dict[str, Any] = {
        "标注片段数": len(states),
        "未判定片段数": sum(1 for s in states if s is None),
        "完全相关片段数": full,
        "部分相关片段数": part,
        "相关片段数": total_relevant,
        "首相关排名": first,
        "MRR": round(1.0 / first, 4) if first else 0.0,
        "计入Precision片段数": sum(1 for x in included if x),
    }
    for k in k_values:
        topk = list(zip(states[:k], included[:k]))
        numerator = sum(_credit(s, partial_weight) for s, _ in topk)
        out["Hit Rate@%d" % k] = round(numerator / total_relevant, 4) if total_relevant else 0.0
        counted = [s for s, keep in topk if keep]
        credit = sum(_credit(s, partial_weight) for s in counted)
        out["Precision@%d" % k] = round(credit / len(counted), 4) if counted else 0.0
    return out


def evaluate_judged(
    rows: List[Dict[str, Any]],
    k_values: Sequence[int] = (10,),
    partial_weight: float = PARTIAL_WEIGHT,
) -> Dict[str, Any]:
    """多题汇总（B-24B 口径）；整题四态全为「未判定」的跳过并计数"""
    per_q: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for r in rows:
        chunks = r.get("chunks") or []
        if not chunks or all(relevance_state(c) is None for c in chunks):
            skipped.append(str(r.get("bid") or r.get("编号")))
            continue
        m = evaluate_judged_question(chunks, k_values, partial_weight)
        m["bid"] = r.get("bid") or r.get("编号")
        m["问题类型"] = r.get("问题类型")
        per_q.append(m)

    def _mean(key: str) -> float:
        vals = [q[key] for q in per_q if q.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary: Dict[str, Any] = {
        "题数": len(per_q),
        "跳过题数": len(skipped),
        "跳过编号": skipped,
        "零相关题数": sum(1 for q in per_q if q["相关片段数"] == 0),
    }
    for k in k_values:
        summary["Hit Rate@%d" % k] = _mean("Hit Rate@%d" % k)
        summary["Precision@%d" % k] = _mean("Precision@%d" % k)
    summary["MRR"] = _mean("MRR")
    return {"summary": summary, "per_question": per_q}


def has_judged_labels(rows: List[Dict[str, Any]]) -> bool:
    """是否含四态人工判定（决定 --mode auto 走哪套口径）"""
    return any(
        c.get("人工判定") in JUDGMENTS
        for r in rows
        for c in (r.get("chunks") or [])
    )


def evaluate_question(chunks: List[Dict[str, Any]], k_values: Sequence[int] = (10,)) -> Dict[str, Any]:
    """对单题的 top-K 片段（有序）计算指标

    chunks: [{"judge_relevant": bool, ...}, ...]，顺序 = 检索排序；
            支持人工修正优先：若 `人工修正` 字段为 True/False 则覆盖模型判定。
    """
    ranked: List[bool] = []
    for c in chunks:
        if c.get("人工修正") is True:
            ranked.append(True)
        elif c.get("人工修正") is False:
            ranked.append(False)
        else:
            ranked.append(bool(c.get("judge_relevant")))
    total_relevant = sum(1 for x in ranked if x)
    out: Dict[str, Any] = {
        "标注片段数": len(ranked),
        "相关片段数": total_relevant,
        "首相关排名": next((i for i, ok in enumerate(ranked, 1) if ok), None),
        "MRR": round(reciprocal_rank(ranked), 4),
    }
    for k in k_values:
        out["Hit Rate@%d" % k] = round(recall_at_k(ranked, k, total_relevant if total_relevant else len(ranked)), 4)
        out["Precision@%d" % k] = round(precision_at_k(ranked, k), 4)
    return out


def evaluate(rows: List[Dict[str, Any]], k_values: Sequence[int] = (10,)) -> Dict[str, Any]:
    """汇总多题指标（返回逐题 + 均值；无标注的题跳过并计数）"""
    per_q: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for r in rows:
        chunks = r.get("chunks") or []
        if not chunks or all(c.get("judge_relevant") is None and c.get("人工修正") is None for c in chunks):
            skipped.append(str(r.get("bid") or r.get("编号")))
            continue
        m = evaluate_question(chunks, k_values)
        m["bid"] = r.get("bid") or r.get("编号")
        m["问题类型"] = r.get("问题类型")
        per_q.append(m)

    def _mean(key: str) -> float:
        vals = [q[key] for q in per_q if q.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "题数": len(per_q),
        "跳过题数": len(skipped),
        "跳过编号": skipped,
        # 标注集内没有任何相关片段的题数：这类题 Hit Rate 的分母退化为 len(标注片段)，按 0 计入，需在报告中注明
        "零相关题数": sum(1 for q in per_q if q["相关片段数"] == 0),
    }
    for k in k_values:
        summary["Hit Rate@%d" % k] = _mean("Hit Rate@%d" % k)
        summary["Precision@%d" % k] = _mean("Precision@%d" % k)
    summary["MRR"] = _mean("MRR")
    return {"summary": summary, "per_question": per_q}


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """读取标注文件（支持 B-24A 预标注 JSON 结构：{questions:[...]} 或直接数组）"""
    data = json.loads(io.open(path, encoding="utf-8").read())
    if isinstance(data, dict):
        return list(data.get("questions") or data.get("rows") or [])
    return list(data)


def main() -> int:
    """CLI 入口"""
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="检索排序质量指标（Hit Rate@K / Precision@K / MRR）")
    ap.add_argument("--labels", required=True, help="标注 JSON（B-24A 预标注或 B-24B 人工终稿）")
    ap.add_argument("--k", type=int, default=10, help="主 K（默认 10）")
    ap.add_argument("--k20", type=int, default=20, help="次 K（默认 20）")
    ap.add_argument("--json-out", default="训练结果数据/retrieval_metrics.json", help="指标 JSON 输出路径")
    ap.add_argument("--mode", choices=("auto", "legacy", "judged"), default="auto",
                    help="口径：auto=含四态人工判定则用 judged，否则 legacy")
    args = ap.parse_args()

    rows = load_rows(Path(args.labels))
    ks = [args.k] if args.k == args.k20 else [args.k, args.k20]
    mode = args.mode
    if mode == "auto":
        mode = "judged" if has_judged_labels(rows) else "legacy"
    if mode == "judged":
        result = evaluate_judged(rows, ks)
        result["口径"] = _judged_caliber()
    else:
        result = evaluate(rows, ks)
        result["口径"] = {
            "模式": "legacy（二值判定，B-24A 草稿口径）",
            "提示": ["该口径下 Hit Rate@K 分母 = top-K 内相关条数，有相关即为 1.0（退化指标），仅作草稿"],
        }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps({"labels": str(args.labels), **result}, ensure_ascii=False, indent=2))

    s = result["summary"]
    print("口径 = %s" % result["口径"]["模式"])
    print("题数 %d（跳过 %d，零相关 %d）" % (s["题数"], s["跳过题数"], s["零相关题数"]))
    for k in ks:
        print("  Hit Rate@%d = %.4f ｜ Precision@%d = %.4f" % (k, s["Hit Rate@%d" % k], k, s["Precision@%d" % k]))
    print("  MRR = %.4f" % s["MRR"])
    for hint in result["口径"].get("提示", []):
        print("  · %s" % hint)
    print("已保存：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())