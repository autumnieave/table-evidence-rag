"""一致性评测套件（B-25A）——同一问题重复生成，量化三层面一致性

AI 应用与确定性程序最大的差别是「同一问题两次回答可能不同」，因此需要专门的
非确定性测试（方案 §6.7.1）。本套件对 golden 子问题集重复生成 n 次，量化：

1. **结构一致性**：答案结构指纹（拒答 / 标题数 / 是否含表格 / 是否含 ECharts / 长度桶）
   与「多数指纹」的一致率；以及 SQL 输出契约通过率（复用 B-17 `utils/output_contracts`）。
2. **数值一致性**：各次生成答案中数值集合两两 IoU 的均值（1.0 = 数值完全一致）。
3. **引用一致性**：各次生成引用集合（paper_path + 片段前 80 字归一化）两两 Jaccard 均值。

用法：
  python -m eval consistency --n 5 --limit 10
  python -m eval consistency --n 5 --limit 10 --reuse 训练结果数据/halluc_audit_20260910/raw_generation.json
  python -m eval consistency --dry-run            # 只抽样，不调用 LLM

产物：
  训练结果数据/consistency_<日期>/consistency_runs.json      # 逐题逐次明细
  训练结果数据/consistency_<日期>/consistency_summary.json   # 基线数字

口径声明：一致性低 ≠ 答案错误（多意图问题允许多种正确路径）；本套件只量化
「稳定性」，正确性由 `eval/llm_judge.py` 与人工（B-25B）判定。
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

GOLDEN_VERSION = "v1"
SQL_SNAPSHOT = _REPO_ROOT / "训练结果数据" / "sql_full_regression_native.json"
DEFAULT_OUT_DIR = _REPO_ROOT / "训练结果数据" / "consistency_20260910"
MAX_REFS_STORED = 10
_MAX_SNIPPET_CHARS = 4000

# 拒答/澄清措辞（与 prompts/fallback.py 的兜底话术同类；判定结构一致性用）
# B-36：统一从 eval/answer_keys.py 引入，避免 judge / consistency / 先验校验三份词表漂移
from eval.answer_keys import REFUSE_MARKERS  # noqa: E402  # noqa: F401

_PUNCT_RE = re.compile(r"[\s：:；;，,、。·“”\"'（）()\[\]【】]")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_HEADING_RE = re.compile(r"^#{1,4}\s", re.M)


# ---------------------------------------------------------------- 采集


def load_golden_items() -> List[Dict[str, Any]]:
    """读取 golden v1 的 items（走版本化清单，避免直接读文件绕过校验）。"""
    from eval import golden as golden_mod

    return list(golden_mod.load_golden(GOLDEN_VERSION).get("items") or [])


def load_sql_flags(snapshot: Path = SQL_SNAPSHOT) -> Dict[str, bool]:
    """从最近一次原生链路回归快照读取「是否有 SQL」，用于分层（缺失时全 False）。"""
    if not Path(snapshot).exists():
        return {}
    records = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    return {
        str(r["编号"]): bool(r.get("有SQL"))
        for r in records
        if isinstance(r, dict) and r.get("编号")
    }


def stratified_sample(
    items: Sequence[Dict[str, Any]],
    sql_flags: Dict[str, bool],
    count: int = 10,
    sub_index: int = 0,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """按「问题类型 × 有/无 SQL」分层抽样。

    Args:
        items: golden items
        sql_flags: 编号 -> 是否有 SQL（来自最近一次原生链路回归快照）
        count: 抽样题数
        sub_index: 每题取第几个子问题
        seed: None = 确定性取每桶首条（与 B-23A 抽样口径一致，可复用其生成产物）；
              传入整数 = 桶内随机取（可复现）
    """

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        subs = item.get("子问题") or []
        if len(subs) <= sub_index:
            continue
        has_sql = bool(sql_flags.get(str(item["编号"]), False))
        key = f"{item['问题类型']}|{'有SQL' if has_sql else '无SQL'}"
        buckets[key].append(
            {
                "编号": str(item["编号"]),
                "问题类型": item["问题类型"],
                "分层键": key,
                "子问题": str(subs[sub_index]),
                "有SQL快照": has_sql,
            }
        )
    rng = random.Random(seed) if seed is not None else None
    queues = {k: sorted(v, key=lambda r: r["编号"]) for k, v in buckets.items()}
    picked: List[Dict[str, Any]] = []
    keys = sorted(queues)
    while len(picked) < count:
        progressed = False
        for key in keys:
            if len(picked) >= count:
                break
            if queues[key]:
                idx = rng.randrange(len(queues[key])) if rng is not None else 0
                picked.append(queues[key].pop(idx))
                progressed = True
        if not progressed:
            break
    return picked[:count]


def load_reuse_runs(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取已有生成产物（如 B-23A 的 raw_generation.json），按编号索引为「第 1 次运行」。

    复用口径必须与本次一致（同为 agent_query 且 QUERY_CACHE_ENABLED=false），
    否则会破坏一致性口径，故在调用处要求显式传入 `--reuse`。
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"未找到复用产物：{path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for rec in payload.get("样本") or []:
        out[str(rec.get("编号"))] = {
            "答案": str(rec.get("答案") or ""),
            "引用": list(rec.get("引用") or []),
            "SQL": str(rec.get("SQL") or ""),
            "耗时": rec.get("耗时"),
            "错误": str(rec.get("错误") or ""),
            "来源": f"reuse:{Path(path).name}",
        }
    return out


def run_generation(
    samples: Sequence[Dict[str, Any]],
    n: int,
    out_dir: Path,
    reuse: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """真实重复生成（agent_query 口径），产物写 out_dir/consistency_runs.json。"""
    from config.rag_config import RAGConfig
    from pipelines.rag_pipeline import RAGPipeline

    config = RAGConfig()
    if config.QUERY_CACHE_ENABLED:
        raise RuntimeError(
            "QUERY_CACHE_ENABLED 必须为 false（重复生成若命中缓存将得到完全一致的假象）；"
            "请设置环境变量 QUERY_CACHE_ENABLED=false"
        )
    pipeline = RAGPipeline(config)

    questions: List[Dict[str, Any]] = []
    t_all = time.time()
    for qi, sample in enumerate(samples, 1):
        runs: List[Dict[str, Any]] = []
        if reuse and sample["编号"] in reuse:
            runs.append(dict(reuse[sample["编号"]]))
            print(f"[{qi}/{len(samples)}] {sample['编号']} 复用第 1 次运行", file=sys.stderr, flush=True)
        while len(runs) < n:
            ridx = len(runs) + 1
            user_id = f"consistency-{sample['编号']}-{ridx}"
            pipeline.reset_conversation(user_id=user_id)
            time.sleep(0.3)
            t0 = time.time()
            answer: Any = None
            error = ""
            try:
                answer = pipeline.agent_query(sample["子问题"], user_id=user_id, verbose=False)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            if isinstance(answer, dict):
                content = str(answer.get("content") or "")
                raw_refs = answer.get("references") or []
            else:
                content = str(answer or "")
                raw_refs = []
            references = [
                {
                    "paper_path": str(r.get("paper_path") or ""),
                    "text": str(r.get("text") or "")[:_MAX_SNIPPET_CHARS],
                }
                for r in raw_refs
                if isinstance(r, dict)
            ][:MAX_REFS_STORED]
            try:
                sql = str(pipeline.get_accumulated_sql(user_id) or "").strip()
            except Exception:  # noqa: BLE001
                sql = ""
            runs.append(
                {
                    "答案": content,
                    "引用": references,
                    "SQL": sql,
                    "耗时": round(time.time() - t0, 1),
                    "错误": error,
                    "来源": "new",
                }
            )
            print(
                f"[{qi}/{len(samples)}] {sample['编号']} run#{ridx} 耗时 {runs[-1]['耗时']}s "
                f"答案 {len(content)} 字 / 引用 {len(references)} 条 / SQL {len(sql)} 字",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.5)
        questions.append({**sample, "runs": runs})

    payload = {
        "日期": "2026-09-10",
        "口径": "agent_query（supervisor-workers 主链路），QUERY_CACHE_ENABLED=false，同题重复 n 次",
        "n": n,
        "题数": len(questions),
        "总耗时秒": round(time.time() - t_all, 1),
        "questions": questions,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "consistency_runs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------- 纯指标（可离线单测）


def extract_number_set(text: str) -> Set[str]:
    """抽取文本中的数值集合（去标点、去重；保留小数与负数）。"""
    from pipelines.citation_validator import CitationValidator

    collapsed = re.sub(r"(?<=\d)\s+(?=[\d.])", "", text or "")
    return set(CitationValidator.extract_numbers(collapsed))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """集合 Jaccard 相似度；两集合均为空时返回 1.0（都无 = 一致）。"""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def mean_pairwise(sets: Sequence[Set[str]]) -> Optional[float]:
    """两两 Jaccard 均值（n<2 时返回 None，不编造数字）。"""
    if len(sets) < 2:
        return None
    total = 0.0
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            total += jaccard(sets[i], sets[j])
            pairs += 1
    return round(total / pairs, 4) if pairs else None


def ref_keys(references: Sequence[Dict[str, Any]], head_chars: int = 80) -> Set[str]:
    """引用集合键：paper_path + 片段前 N 字（归一化标点空白）。"""
    keys: Set[str] = set()
    for ref in references or []:
        path = str(ref.get("paper_path") or "")
        head = str(ref.get("text") or "")[:head_chars]
        keys.add(_PUNCT_RE.sub("", path) + "||" + _PUNCT_RE.sub("", head))
    return keys


def structure_signature(text: str) -> Dict[str, Any]:
    """答案结构指纹（用于结构一致性判定）。"""
    body = text or ""
    length = len(body)
    if length < 200:
        bucket = "短(<200)"
    elif length < 500:
        bucket = "中(200-500)"
    elif length < 1500:
        bucket = "长(500-1500)"
    else:
        bucket = "超长(≥1500)"
    return {
        "拒答": any(m in body for m in REFUSE_MARKERS),
        "标题数": len(_HEADING_RE.findall(body)),
        "有表格": bool(_TABLE_LINE_RE.search(body)),
        "有图表": "echarts" in body.lower(),
        "长度桶": bucket,
    }


def signature_key(signature: Dict[str, Any]) -> str:
    """把结构指纹序列化成可比字符串。"""
    return "|".join(f"{k}={signature[k]}" for k in sorted(signature))


def structure_consistency(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """结构一致性：多数指纹占比 + 拒答判定是否一致（投票）。"""
    signatures = [structure_signature(r.get("答案") or "") for r in runs]
    if not signatures:
        return {"指纹一致率": None, "多数指纹": "", "拒答一致": None, "指纹分布": {}}
    keys = [signature_key(s) for s in signatures]
    counter = Counter(keys)
    modal, modal_count = counter.most_common(1)[0]
    refuse_votes = {bool(s["拒答"]) for s in signatures}
    return {
        "指纹一致率": round(modal_count / len(signatures), 4),
        "多数指纹": modal,
        "拒答一致": len(refuse_votes) == 1,
        "指纹分布": dict(counter),
    }


def sql_contract_rate(runs: Sequence[Dict[str, Any]]) -> Optional[float]:
    """SQL 输出契约通过率（复用 B-17 utils.output_contracts）；无 SQL 时返回 None。"""
    from utils.output_contracts import validate_sql_output

    sql_runs = [r for r in runs if str(r.get("SQL") or "").strip()]
    if not sql_runs:
        return None
    ok = sum(1 for r in sql_runs if validate_sql_output(str(r["SQL"])).ok)
    return round(ok / len(sql_runs), 4)


def aggregate_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """单题一致性汇总（三层面）。"""
    runs = question.get("runs") or []
    number_sets = [extract_number_set(r.get("答案") or "") for r in runs]
    ref_sets = [ref_keys(r.get("引用") or []) for r in runs]
    struct = structure_consistency(runs)
    return {
        "编号": question.get("编号"),
        "问题类型": question.get("问题类型"),
        "分层键": question.get("分层键"),
        "子问题": question.get("子问题"),
        "n": len(runs),
        "数值一致性(IoU均值)": mean_pairwise(number_sets),
        "引用一致性(Jaccard均值)": mean_pairwise(ref_sets),
        "结构一致性(指纹一致率)": struct["指纹一致率"],
        "拒答一致": struct["拒答一致"],
        "多数指纹": struct["多数指纹"],
        "SQL契约通过率": sql_contract_rate(runs),
        "每次数值数": [len(s) for s in number_sets],
        "每次引用数": [len(s) for s in ref_sets],
        "耗时": [r.get("耗时") for r in runs],
        "错误数": sum(1 for r in runs if r.get("错误")),
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """数值均值（跳过 None；无有效值返回 None）。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def aggregate_all(questions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """全量一致性汇总（三层面基线数字）。"""
    rows = [aggregate_question(q) for q in questions]
    number_vals = [r["数值一致性(IoU均值)"] for r in rows]
    ref_vals = [r["引用一致性(Jaccard均值)"] for r in rows]
    struct_vals = [r["结构一致性(指纹一致率)"] for r in rows]
    sql_vals = [r["SQL契约通过率"] for r in rows]
    zero_number_runs = sum(1 for r in rows if all(c == 0 for c in r["每次数值数"]))
    return {
        "题数": len(rows),
        "n": rows[0]["n"] if rows else None,
        "数值一致性(IoU均值)": _mean(number_vals),
        "数值一致性为1.0的题数": sum(1 for v in number_vals if v == 1.0),
        "引用一致性(Jaccard均值)": _mean(ref_vals),
        "引用一致性为1.0的题数": sum(1 for v in ref_vals if v == 1.0),
        "结构一致性(指纹一致率)": _mean(struct_vals),
        "结构指纹全一致的题数": sum(1 for v in struct_vals if v == 1.0),
        "拒答判定不一致的题数": sum(1 for r in rows if r["拒答一致"] is False),
        "SQL契约通过率": _mean(sql_vals),
        "全程无数值的题数": zero_number_runs,
        "总错误次数": sum(r["错误数"] for r in rows),
        "口径声明": "一致性低 ≠ 答案错误；本套件只量化稳定性，正确性由 llm_judge + 人工（B-25B）判定。",
    }


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="一致性评测套件（B-25A）")
    parser.add_argument("--n", type=int, default=5, help="每题重复生成次数（默认 5）")
    parser.add_argument("--limit", type=int, default=10, help="抽样题数（默认 10）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="产物目录")
    parser.add_argument("--reuse", default="", help="复用已有生成产物 JSON（作为第 1 次运行）")
    parser.add_argument("--dry-run", action="store_true", help="只抽样，不调用 LLM")
    parser.add_argument("--seed", type=int, default=None, help="抽样随机种子（默认确定性取每桶首条）")
    args = parser.parse_args(argv)

    items = load_golden_items()
    sql_flags = load_sql_flags()
    samples = stratified_sample(items, sql_flags, count=args.limit, seed=args.seed)
    print(
        f"[consistency] 分层抽样 {len(samples)} 题："
        f"{json.dumps(Counter(s['分层键'] for s in samples), ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )
    if args.dry_run:
        for s in samples:
            print(f"  {s['编号']} {s['分层键']} :: {s['子问题'][:60]}")
        return 0

    out_dir = Path(args.out_dir)
    reuse = load_reuse_runs(Path(args.reuse)) if args.reuse else None
    payload = run_generation(samples, args.n, out_dir, reuse=reuse)
    summary = aggregate_all(payload["questions"])
    summary["总耗时秒"] = payload["总耗时秒"]
    summary = {**summary, "逐题": [aggregate_question(q) for q in payload["questions"]]}
    (out_dir / "consistency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "逐题"}, ensure_ascii=False, indent=2))
    print(f"\n产物：{out_dir / 'consistency_runs.json'}、{out_dir / 'consistency_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())