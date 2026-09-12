"""eval/runner.py —— 统一评估 CLI（python -m eval ...）

子命令：
  golden init/list/verify   golden set 版本化
  sql --suite ...           路由到 tools/data_scripts 回归脚本（全量/Agent/守卫/自测）
  citation --refs ...       L1 引用核验（复用 pipelines.citation_validator）
  report                    聚合最新证据生成 docs/评估报告.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from eval import golden as golden_mod
from eval import metrics as metrics_mod

REPO_ROOT = Path(__file__).resolve().parents[1]

#: SQL 回归套件 → tools/data_scripts 模块名
SQL_SUITES = {
    "full": "tools.data_scripts.sql_full_regression_native",
    "agent": "tools.data_scripts.sql_full_regression_native",
    # guard 套件已下线：SQL 守卫逻辑由 tests/test_sql_guard.py 覆盖，Dify 回归脚本已归档
    "selftest": "tools.data_scripts.sql_validator_selftest",
    "defense": "tools.data_scripts.sql_defense_ablation",
    "backend_cmp": "tools.data_scripts.sql_agent_backend_compare_native",
}


def _stdout_utf8() -> None:
    """Windows 控制台统一 UTF-8 输出"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def cmd_golden_init(args: argparse.Namespace) -> int:
    """golden init：固化题库快照并注册版本"""
    path = golden_mod.init_golden(Path(args.source), args.version, args.tag)
    print(f"golden set 已创建: {path}")
    print(f"版本: {args.version}  tag: {args.tag or '-'}")
    return 0


def cmd_golden_list(args: argparse.Namespace) -> int:
    """golden list：列出全部版本"""
    versions = golden_mod.list_versions()
    if not versions:
        print("（暂无 golden 版本，先运行: python -m eval golden init --source ... --version v1）")
        return 0
    for v in versions:
        c = v["counts"]
        kind = v.get("kind", "sql")
        if kind == "challenge" or "sql_statements" not in c:
            types = v.get("types") or {}
            n_cat = c.get("categories", len(types))
            print(f"{v['version']:<8} {v['created_at']}  {v['tag']}  [{kind} {c.get('questions', '?')} 题/{n_cat} 类]")
        else:
            print(f"{v['version']:<8} {v['created_at']}  {v['tag']}  [{c['questions']}题/{c['sub_questions']}子问题/{c['sql_statements']}句]")
    return 0


def cmd_golden_verify(args: argparse.Namespace) -> int:
    """golden verify：校验快照完整性 + 源文件哈希"""
    result = golden_mod.verify_version(args.version)
    if result["ok"]:
        print(f"golden {args.version} 校验通过（快照完整；源文件哈希一致）")
        print(f"计数: {result['counts']}")
        return 0
    print(f"golden {args.version} 校验失败:")
    for e in result["errors"]:
        print(f"  - {e}")
    return 1


def cmd_sql(args: argparse.Namespace) -> int:
    """sql：路由到 tools/data_scripts 回归脚本（子进程，保留断点续跑）"""
    module = SQL_SUITES.get(args.suite)
    if not module:
        print(f"未知套件: {args.suite}（可选: {', '.join(SQL_SUITES)}）")
        return 2
    cmd = [sys.executable, "-X", "utf8", "-m", module]
    if getattr(args, "limit", None):
        cmd += ["--limit", str(args.limit)]
    if getattr(args, "only", None):
        cmd += ["--only"] + args.only
    if getattr(args, "backend", None):
        cmd += ["--backend", args.backend]
    if getattr(args, "progress_every", 0):
        cmd += ["--progress-every", str(args.progress_every)]
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def cmd_citation(args: argparse.Namespace) -> int:
    """citation：对引用 JSON 执行 L1 核验（文件可溯源 + 数字命中）"""
    refs_path = Path(args.refs)
    if not refs_path.is_file():
        print(f"引用文件不存在: {refs_path}")
        return 2
    from config.rag_config import RAGConfig
    from pipelines.citation_validator import CitationValidator

    references = json.loads(refs_path.read_text(encoding="utf-8"))
    validator = CitationValidator(corpus_root=RAGConfig().CITATION_CORPUS_ROOT, match_mode=args.mode)
    records = validator.check_references(references)
    summary = validator.summarize(records)

    print(f"\n引用核验完成：共 {summary['total']} 条引用")
    print(f"文件可溯源率：{summary['traceable']}/{summary['total']} = {summary['traceable_rate'] * 100:.1f}%"
          f"（exact {summary['exact']} + fuzzy {summary['fuzzy']}，missing {summary['missing']}）")
    if summary["num_total"]:
        print(f"数字命中率：{summary['num_hit']}/{summary['num_total']} = {summary['num_rate'] * 100:.1f}%"
              f"（口径 {args.mode}）")
    out_path = refs_path.with_name(refs_path.stem + "_citation_report.json")
    out_path.write_text(json.dumps({"records": records, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"逐条明细已写入：{out_path}")
    return 0


def cmd_challenge(args: argparse.Namespace) -> int:
    """challenge：对抗挑战集 v2（阶段 A：子集预览；阶段 B：--run 接通真实 Agent 引擎）"""
    from eval import challenge as challenge_mod

    if getattr(args, "rejudge", False):
        from eval import challenge_run
        challenge_run.rejudge_file()
        return 0
    golden = challenge_mod.load_challenge(args.version)
    items = golden["items"]
    if args.categories:
        unknown = [c for c in args.categories if c not in challenge_mod.JUDGE_TYPE]
        if unknown:
            print(f"未知类别: {unknown}")
            return 2
        items = [it for it in items if it["类别"] in args.categories]
    if args.only:
        items = [it for it in items if it["编号"] in args.only]
    if args.limit:
        items = items[: args.limit]
    if args.run:
        if not items:
            print("无可执行条目（检查 --categories/--only/--limit）")
            return 2
        from eval import challenge_run
        return challenge_run.execute_items(items)
    print(f"挑战集 {args.version}（kind=challenge）：快照共 {golden['counts']['questions']} 条 / 五类通过标准：")
    for cat, crit in challenge_mod.category_pass_criteria().items():
        print(f"  - {cat}: {crit}")
    print(f"\n本次子集 {len(items)} 条：")
    for it in items:
        print(f"  {it['编号']} [{it['类别标签']}] {it['问题'][:70]}")
    print("\n[阶段 A 预览] 未执行真实调用；真实执行加 --run（接通 Agent 引擎 + 人工抽审 ≥30%）。")
    return 0


def cmd_retrieval(args: argparse.Namespace) -> int:
    """retrieval：检索层对比（纯向量 vs 混合，引用命中），委托 eval.retrieval_cmp"""
    from eval import retrieval_cmp
    return retrieval_cmp.main([
        "--collection", args.collection,
        "--k", str(args.k),
        "--limit", str(args.limit),
        "--match", args.match,
        "--rrf-k", str(args.rrf_k),
        "--topk-bm25", str(args.topk_bm25),
        "--vector-floor-ratio", str(args.vector_floor_ratio),
        "--rerank-top-n", str(args.rerank_top_n),
        "--out", args.out,
        "--json-out", args.json_out,
    ])


def cmd_report(args: argparse.Namespace) -> int:
    """report：聚合最新证据生成评估报告"""
    versions = golden_mod.list_versions()
    sql_versions = [v for v in versions if v.get("kind", "sql") != "challenge"]
    golden = golden_mod.load_golden(sql_versions[-1]["version"]) if sql_versions else None
    report = metrics_mod.build_report(
        golden=golden,
        sql_full=metrics_mod.sql_metrics(metrics_mod.SQL_FULL_SUMMARY),
        sql_agent=metrics_mod.sql_metrics(metrics_mod.SQL_AGENT_SUMMARY),
        sql_agent_langgraph=metrics_mod.sql_metrics(metrics_mod.SQL_AGENT_LANGGRAPH_SUMMARY),
        sql_native=metrics_mod.sql_metrics(metrics_mod.SQL_NATIVE_SUMMARY),
        citation=metrics_mod.citation_metrics(metrics_mod.CITATION_REPORT),
        badcase=metrics_mod.badcase_count(),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写入：{out}")
    return 0


def cmd_consistency(args: Any) -> int:
    """一致性套件入口（委托 eval/consistency.py）"""
    from eval import consistency as consistency_mod

    argv: List[str] = ["--n", str(args.n), "--limit", str(args.limit)]
    if args.out_dir:
        argv += ["--out-dir", args.out_dir]
    if args.reuse:
        argv += ["--reuse", args.reuse]
    if args.dry_run:
        argv.append("--dry-run")
    return consistency_mod.main(argv)


def cmd_llm_judge(args: Any) -> int:
    """LLM-as-judge 入口（委托 eval/llm_judge.py）"""
    from eval import llm_judge as judge_mod

    argv: List[str] = ["--input", args.input]
    if args.out_dir:
        argv += ["--out-dir", args.out_dir]
    if args.max_runs:
        argv += ["--max-runs", str(args.max_runs)]
    if args.model:
        argv += ["--model", args.model]
    if args.no_judge:
        argv.append("--no-judge")
    if args.no_sql_result:
        argv.append("--no-sql-result")
    if args.answer_key:
        argv += ["--answer-key", args.answer_key]
    if args.no_prior:
        argv.append("--no-prior")
    return judge_mod.main(argv)


def build_parser() -> argparse.ArgumentParser:
    """组装子命令解析器"""
    parser = argparse.ArgumentParser(prog="eval", description="智能问数系统评估闭环（golden 版本化 + SQL/引用回归）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_golden = sub.add_parser("golden", help="golden set 版本化管理")
    gsub = p_golden.add_subparsers(dest="golden_cmd", required=True)
    p_init = gsub.add_parser("init", help="固化题库快照并注册版本")
    p_init.add_argument("--source", required=True, help="源 xlsx 路径（如 训练结果数据/result_3_parallel.xlsx）")
    p_init.add_argument("--version", required=True, help="版本号（如 v1）")
    p_init.add_argument("--tag", default="", help="描述标签（如 80题全量）")
    p_init.set_defaults(func=cmd_golden_init)
    p_list = gsub.add_parser("list", help="列出全部版本")
    p_list.set_defaults(func=cmd_golden_list)
    p_verify = gsub.add_parser("verify", help="校验快照完整性 + 源哈希")
    p_verify.add_argument("--version", required=True)
    p_verify.set_defaults(func=cmd_golden_verify)

    p_sql = sub.add_parser("sql", help="运行 SQL 回归套件（路由到 tools/data_scripts）")
    p_sql.add_argument("--suite", choices=sorted(SQL_SUITES), required=True)
    p_sql.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟）")
    p_sql.add_argument("--only", nargs="+", default=None, help="只跑指定编号（如 B2007 B2041）")
    p_sql.add_argument("--backend", choices=["handwritten", "langgraph"], default=None,
                       help="Agent 规划器后端（透传给 agent 套件；默认取配置 AGENT_PLANNER_BACKEND）")
    p_sql.add_argument("--progress-every", type=int, default=0,
                       help="每 N 题打印一次进度汇总（透传给 agent 套件；0=套件默认）")
    p_sql.set_defaults(func=cmd_sql)

    p_cit = sub.add_parser("citation", help="L1 引用核验")
    p_cit.add_argument("--refs", required=True, help="引用 JSON 路径（元素含 paper_path/text）")
    p_cit.add_argument("--mode", choices=["raw", "comma", "loose"], default="comma", help="数字匹配口径")
    p_cit.set_defaults(func=cmd_citation)

    p_retr = sub.add_parser("retrieval", help="检索层对比：纯向量 vs 混合（引用命中）")
    p_retr.add_argument("--collection", default="research_reports_v3", help="Qdrant 集合名（正式数据全量集合为 research_reports_v3_full）")
    p_retr.add_argument("--k", type=int, default=10, help="召回 top-K（默认 10）")
    p_retr.add_argument("--limit", type=int, default=0, help="只跑前 N 个有引用的题目（冒烟）")
    p_retr.add_argument("--match", choices=["raw", "comma", "loose"], default="comma", help="数字匹配口径")
    p_retr.add_argument("--rrf-k", type=int, default=0, help="RRF 常数（0=取配置）")
    p_retr.add_argument("--topk-bm25", type=int, default=0, help="BM25 路召回量（0=取配置）")
    p_retr.add_argument("--vector-floor-ratio", type=float, default=-1.0, help="向量路保底比例（-1=取配置，0=纯RRF）")

    p_retr.add_argument("--rerank-top-n", type=int, default=0, help="对 top-K 候选做 Rerank 复排并统计精排后 top-N 命中（0=不开启）")
    p_retr.add_argument("--out", default="docs/检索对比报告.md", help="markdown 报告路径")
    p_retr.add_argument("--json-out", default="训练结果数据/retrieval_cmp.json", help="JSON 明细路径")
    p_retr.set_defaults(func=cmd_retrieval)

    p_chal = sub.add_parser("challenge", help="对抗挑战集 v2（提示注入/越界/错别字/绑定/幻觉）")
    p_chal.add_argument("--version", default="v2", help="挑战集版本（默认 v2）")
    p_chal.add_argument("--categories", nargs="+", default=None,
                        help="类别过滤（prompt_injection/out_of_boundary/typo_robustness/binding_entrapment/hallucination_entrapment）")
    p_chal.add_argument("--only", nargs="+", default=None, help="只执行指定编号（如 C2001 C2006）")
    p_chal.add_argument("--limit", type=int, default=0, help="只取前 N 条（预览/冒烟）")
    p_chal.add_argument("--run", action="store_true", help="阶段 B：接通真实 Agent 引擎逐条执行 + 判定 + 抽审报告")
    p_chal.add_argument("--dry-run", action="store_true", help="阶段 A：仅预览子集，不执行真实调用（默认）")
    p_chal.add_argument("--rejudge", action="store_true",
                        help="用最新判定词表重判既有结果 JSON（不调用 LLM）+ 回填人工复核 sidecar")
    p_chal.set_defaults(func=cmd_challenge)

    p_cons = sub.add_parser("consistency", help="一致性套件：同题重复生成，量化结构/数值/引用一致性（B-25A）")
    p_cons.add_argument("--n", type=int, default=5, help="每题重复生成次数（默认 5）")
    p_cons.add_argument("--limit", type=int, default=10, help="抽样题数（默认 10）")
    p_cons.add_argument("--out-dir", default="", help="产物目录（默认 训练结果数据/consistency_<日期>）")
    p_cons.add_argument("--reuse", default="", help="复用已有生成产物 JSON（作为第 1 次运行）")
    p_cons.add_argument("--dry-run", action="store_true", help="只抽样，不调用 LLM")
    p_cons.set_defaults(func=cmd_consistency)

    p_judge = sub.add_parser("llm-judge", help="LLM-as-judge 初判 + 分歧样本清单（B-25A）")
    p_judge.add_argument("--input", required=True, help="一致性套件产物（consistency_runs.json）")
    p_judge.add_argument("--out-dir", default="", help="产物目录（默认 训练结果数据/llm_judge_<日期>）")
    p_judge.add_argument("--max-runs", type=int, default=0, help="每题最多判定几次（0=全部）")
    p_judge.add_argument("--model", default="", help="judge 模型（默认 qwen-flash）")
    p_judge.add_argument("--no-judge", action="store_true", help="不调用 judge 模型（只算规则信号）")
    p_judge.add_argument("--no-sql-result", action="store_true", help="不执行 SQL 取结果预览")
    p_judge.add_argument("--answer-key", default="", help="先验登记表路径（B-36，默认取 eval 内置默认值）")
    p_judge.add_argument("--no-prior", action="store_true", help="不加载先验登记表（B-36）")
    p_judge.set_defaults(func=cmd_llm_judge)

    p_rep = sub.add_parser("report", help="聚合最新证据生成评估报告")
    p_rep.add_argument("--out", default="docs/评估报告.md", help="输出路径（默认 docs/评估报告.md）")
    p_rep.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口"""
    _stdout_utf8()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
