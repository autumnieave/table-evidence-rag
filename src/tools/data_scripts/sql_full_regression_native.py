# -*- coding: utf-8 -*-
"""全量 80 题 SQL 回归（原生链路版）——复刻基线 batch_test.py 的「Agent 多轮累积」口径。

口径说明（与 docs/SQL编译修复前后对比报告.md 第十三节 224/224 完全对齐）：
- 每题同 user_id，逐子问题走 RAGPipeline.agent_query() → Agent 编排循环；
- 财务工具调用走原生 SQL 链路（tools.native_financial，路线 3 替代 Dify）：
  SQL 生成 → 三层防线（静态校验 + MySQL 编译重试）→ 执行 → 分析 → ECharts 图表，
  每次成功生成的 SQL 累积到 conversation_state.sql；
- 子问题异常重试 3 次（指数退避，同基线）；
- 全部子问题完成后取累积 SQL，逐句静态校验 + MySQL 编译终审；
- 空 SQL（Agent 判定无需查询，返回纯分析/研报答案）不计入编译失败。

用法::

    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_full_regression_native            # 全量 80 题
    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_full_regression_native --limit 10 # 冒烟
    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_full_regression_native --only B2001 B2003
    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_full_regression_native --backend handwritten
    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_full_regression_native --progress-every 10

断点续跑：每题完成即追加写 训练结果数据/sql_full_regression_native.jsonl，
重跑时已完成编号自动跳过（--limit/--only 只作用于待跑题目）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# 真实重跑：强制关闭查询缓存，避免命中旧 SQL 结果（config 实例化前生效）
os.environ["QUERY_CACHE_ENABLED"] = "false"

import sqlparse

from tools.sql_validator import compile_check, validate_sql

from config.rag_config import RAGConfig
from pipelines.rag_pipeline import RAGPipeline

GOLDEN_SNAPSHOT = Path("database/golden/v1_2026-08-22.json")
GOLDEN_MANIFEST = Path("database/golden/manifest.json")
OUT_JSONL = Path("训练结果数据/sql_full_regression_native.jsonl")
OUT_JSON = Path("训练结果数据/sql_full_regression_native.json")
OUT_SUMMARY = Path("训练结果数据/sql_full_regression_native_summary.json")

MAX_RETRIES = 3  # 子问题异常重试次数（同基线 batch_test.py）


def _utf8() -> None:
    """Windows 控制台统一 UTF-8 输出。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    """分块计算文件 sha256。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_golden(verify_sha: bool = True) -> List[Dict[str, Any]]:
    """读取 golden 快照题目；可选校验源 xlsx sha256 与 manifest 一致。"""
    if not GOLDEN_SNAPSHOT.is_file():
        raise FileNotFoundError(f"golden 快照不存在: {GOLDEN_SNAPSHOT}")
    data = json.loads(GOLDEN_SNAPSHOT.read_text(encoding="utf-8"))
    if verify_sha:
        manifest = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
        entry = manifest["versions"][0]
        src = Path(entry["source"])
        if not src.is_file():
            print(f"[警告] golden 源文件不存在，跳过哈希校验: {src}", flush=True)
        else:
            actual = sha256_file(src)
            expected = entry["source_sha256"]
            if actual != expected:
                raise RuntimeError(
                    f"golden 源文件哈希不一致: {src}\n  expected={expected}\n  actual  ={actual}\n"
                    "源文件被改动，golden 口径失效。确认后可用 --no-verify-sha 跳过。"
                )
    print(f"golden 加载: {data['version']} tag={data['tag']} counts={data['counts']}", flush=True)
    return data["items"]


def _split_statements(sql_text: str) -> List[str]:
    """把累积 SQL 文本按语句切分（sqlparse，过滤空/注释）。"""
    stmts = [s.strip() for s in sqlparse.split(sql_text or "") if s.strip()]
    return [s for s in stmts if not s.lower().startswith("--")]


def run_question_agent(item: Dict[str, Any], pipeline: RAGPipeline, schema: Optional[Dict], conn: Any) -> Dict[str, Any]:
    """按 batch_test.py 口径执行单题：重置会话 → 逐子问题 agent_query → 取累积 SQL 校验。"""
    bid = item["编号"]
    user_id = f"native-full-{bid}"
    pipeline.reset_conversation(user_id=user_id)
    time.sleep(0.3)

    details: List[Dict[str, Any]] = []
    for qi, q in enumerate(item["子问题"], 1):
        t0 = time.time()
        answer = None
        last_exc = ""
        for attempt in range(MAX_RETRIES):
            try:
                answer = pipeline.agent_query(q, user_id=user_id, verbose=False)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = str(exc)[:150]
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))
        if answer is None:
            content = f"处理失败(重试{MAX_RETRIES}次): {last_exc}"
        elif isinstance(answer, str):
            content = answer
        elif isinstance(answer, dict):
            content = answer.get("content", "")
        else:
            content = str(answer)
        references: List[Dict[str, Any]] = []
        if isinstance(answer, dict):
            for ref in answer.get("references") or []:
                if isinstance(ref, dict) and ref.get("paper_path"):
                    references.append({
                        "paper_path": str(ref.get("paper_path"))[:300],
                        "text": str(ref.get("text") or "")[:300],
                        "paper_image": str(ref.get("paper_image") or "")[:150],
                    })
        details.append({
            "q": q[:150],
            "重试次数": MAX_RETRIES - 1 if answer is None else 0,
            "答案摘要": str(content)[:200],
            "答案引用数": len(references),
            "答案引用": references,
            "耗时": round(time.time() - t0, 1),
        })
        time.sleep(0.5)

    sql = (pipeline.get_accumulated_sql(user_id) or "").strip()
    stmts = _split_statements(sql)
    rec: Dict[str, Any] = {
        "编号": bid,
        "问题类型": item["问题类型"],
        "子问题数": len(details),
        "语句数": len(stmts),
        "通过语句数": 0,
        "有SQL": bool(stmts),
        "SQL": sql[:8000],
        "静态错误": [],
        "编译错误": [],
        "语句明细": [],
        "子问题明细": details,
    }
    for stmt in stmts:
        ok, serrs = validate_sql(stmt, schema)
        cerr = ""
        if ok and conn is not None:
            cerr = compile_check(conn, stmt)
        passed = ok and not cerr
        rec["语句明细"].append({
            "sql": stmt[:300],
            "静态错误": list(serrs) if not ok else [],
            "编译错误": cerr or "",
            "通过": passed,
        })
        if not ok:
            rec["静态错误"].extend(list(serrs)[:4])
        if cerr:
            rec["编译错误"].append(cerr[:200])
        if passed:
            rec["通过语句数"] += 1
    rec["全通过(有SQL且全部语句通过)"] = rec["语句数"] > 0 and rec["通过语句数"] == rec["语句数"]
    rec["错误"] = (rec["编译错误"] + rec["静态错误"])[:6]
    return rec


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总（口径与 sql_compile_summary.md 对齐，字段供 eval/metrics.py 消费）。"""
    total_q = len(results)
    total_sub = sum(r["子问题数"] for r in results)
    total_stmt = sum(r["语句数"] for r in results)
    total_pass = sum(r["通过语句数"] for r in results)
    has_sql_q = sum(1 for r in results if r["有SQL"])
    loose_q = sum(1 for r in results if r["全通过(有SQL且全部语句通过)"])
    no_sql_q = [r["编号"] for r in results if not r["有SQL"]]

    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"题": 0, "有SQL": 0, "全通过": 0, "语句": 0, "通过语句": 0})
    for r in results:
        t = by_type[r["问题类型"]]
        t["题"] += 1
        t["语句"] += r["语句数"]
        t["通过语句"] += r["通过语句数"]
        if r["有SQL"]:
            t["有SQL"] += 1
        if r["全通过(有SQL且全部语句通过)"]:
            t["全通过"] += 1

    failures: List[Dict[str, Any]] = []
    for r in results:
        for d in r["语句明细"]:
            if not d["通过"]:
                failures.append({
                    "编号": r["编号"],
                    "问题类型": r["问题类型"],
                    "静态错误": d["静态错误"],
                    "编译错误": d["编译错误"],
                    "SQL": d["sql"][:300],
                })
    return {
        "题目总数": total_q,
        "子问题总数": total_sub,
        "语句总数": total_stmt,
        "通过语句数": total_pass,
        "语句级通过率": round(total_pass / total_stmt, 4) if total_stmt else 0,
        "有SQL题目": has_sql_q,
        "有SQL且全部语句通过": loose_q,
        "严格全题通过": loose_q,  # 与基线口径一致：空 SQL 视为未通过
        "空SQL题目": no_sql_q,
        "分类型": {k: dict(v) for k, v in by_type.items()},
        "失败语句数": len(failures),
        "失败明细": failures,
    }


def main() -> int:
    """解析参数并执行原生链路 Agent 多轮累积全量回归。"""
    _utf8()
    for _logger_name in ("httpx", "httpcore", "urllib3", "openai", "dashscope"):
        logging.getLogger(_logger_name).setLevel(logging.WARNING)
    logging.getLogger("embeddings.client").setLevel(logging.WARNING)
    logging.getLogger("vectorstore.qdrant_wrapper").setLevel(logging.WARNING)
    logging.getLogger("chains.rerank").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="原生链路 Agent 多轮累积口径全量 80 题 SQL 回归")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟）")
    parser.add_argument("--only", nargs="*", default=None, help="只跑指定编号")
    parser.add_argument("--backend", choices=["handwritten", "langgraph"], default=None,
                        help="Agent 规划器后端覆盖（默认取 .env 的 AGENT_PLANNER_BACKEND，含 multi-agent 开关）")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="每 N 题打印一次进度汇总（默认 10，减少长跑日志噪音；0 表示每题打印完成信息）")
    parser.add_argument("--no-verify-sha", action="store_true", help="跳过 golden 源文件哈希校验（调试用）")
    args = parser.parse_args()

    if args.backend:
        os.environ["AGENT_PLANNER_BACKEND"] = args.backend
        print(f"Agent 后端覆盖: {args.backend}", flush=True)
    else:
        backend = os.environ.get("AGENT_PLANNER_BACKEND", "handwritten")
        multi = os.environ.get("AGENT_LANGGRAPH_MULTI_AGENT", "false")
        print(f"Agent 后端: {backend}（multi-agent={multi}）", flush=True)
    print("查询缓存: 已强制关闭（QUERY_CACHE_ENABLED=false，真实重跑）", flush=True)

    questions = load_golden(verify_sha=not args.no_verify_sha)
    if args.only:
        questions = [q for q in questions if q["编号"] in args.only]
    if args.limit:
        questions = questions[: args.limit]

    done: List[Dict[str, Any]] = []
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    done_ids = {r["编号"] for r in done}
    todo = [q for q in questions if q["编号"] not in done_ids]
    if done_ids:
        print(f"断点续跑：已跳过 {len(done_ids)} 题，本次执行 {len(todo)} 题", flush=True)

    if not todo:
        results = done
    else:
        config = RAGConfig()
        config.ENABLE_MULTI_TURN = True
        pipeline = RAGPipeline(config)
        pipeline.agent_mode_enabled = True
        try:
            from tools.native_financial import _load_schema_conn

            schema, conn = _load_schema_conn(config)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] MySQL schema 加载失败，跳过编译终审: {exc}", flush=True)
            schema, conn = None, None
        try:
            running = {"语句": 0, "通过": 0}
            t_start = time.time()
            for idx, item in enumerate(todo, 1):
                print(f"[{idx}/{len(todo)}] {item['编号']} 开始（{item['问题类型']}，{len(item['子问题'])} 子问题）...", flush=True)
                t0 = time.time()
                rec = run_question_agent(item, pipeline, schema, conn)
                rec["耗时"] = round(time.time() - t0, 1)
                with OUT_JSONL.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.append(rec)
                running["语句"] += rec["语句数"]
                running["通过"] += rec["通过语句数"]
                if args.progress_every and idx % args.progress_every == 0:
                    rate = running["通过"] / running["语句"] * 100 if running["语句"] else 0
                    elapsed = round(time.time() - t_start, 1)
                    print(f"[进度 {idx}/{len(todo)}] 累计语句 {running['通过']}/{running['语句']} = {rate:.1f}%"
                          f"，本段耗时 {elapsed}s", flush=True)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        results = done

    summary = build_summary(results)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 原生链路 Agent 多轮累积全量回归汇总 =====", flush=True)
    print(f"语句级编译通过率: {summary['通过语句数']}/{summary['语句总数']} = {summary['语句级通过率'] * 100:.1f}%")
    print(f"有 SQL 的题目: {summary['有SQL题目']}/{summary['题目总数']}")
    print(f"有 SQL 且全部语句通过: {summary['有SQL且全部语句通过']}/{summary['有SQL题目']}")
    print(f"严格全题（空 SQL 视为未通过）: {summary['严格全题通过']}/{summary['题目总数']}")
    print(f"空 SQL 题目({len(summary['空SQL题目'])}): {summary['空SQL题目']}")
    print("\n分类型（题目数/有SQL/全通过/语句通过率）:")
    for k, v in sorted(summary["分类型"].items()):
        rate = v["通过语句"] / v["语句"] * 100 if v["语句"] else 0
        print(f"  {k}: {v['题']} / {v['有SQL']} / {v['全通过']} / {v['通过语句']}/{v['语句']}={rate:.1f}%")
    print(f"\n失败语句 {summary['失败语句数']} 条：")
    for f in summary["失败明细"][:20]:
        err = (f["编译错误"] or f["静态错误"] or ["未知"])[0]
        print(f"  {f['编号']} -> {err[:120]}")
    if len(summary["失败明细"]) > 20:
        print(f"  ... 其余 {len(summary['失败明细']) - 20} 条见 {OUT_SUMMARY}")
    print(f"\n已保存: {OUT_JSONL} / {OUT_JSON} / {OUT_SUMMARY}")
    return 0 if summary["失败语句数"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
