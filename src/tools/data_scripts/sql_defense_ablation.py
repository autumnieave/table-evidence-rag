# -*- coding: utf-8 -*-
"""SQL 三层防线消融实验（B-04，2026-09-05）——函数级单发口径。

背景：原生 SQL 链路的“三层防线” = 提示词规则（prompts/financial.py SQL_GEN_SYSTEM_PROMPT，
含字段-表归属等规则）/ 静态校验（tools.sql_validator.validate_sql）/
MySQL 编译重试（compile_check + 失败带错误反馈重试）。本脚本按 golden v1 的 108 个子问题，
逐个子问题用与 tools/native_financial._generate_sql 完全相同的消息构造与采样参数重新生成 SQL，
跑四组防线组合（每组独立采样，与 2026-08 单发复跑口径一致）：

- prompt_only（仅提示词）：生成 1 次不加防线，语句直接进入“执行模拟”，MySQL 编译仅测量；
- static（+静态校验）：静态校验作执行闸门，静态错误带反馈重试（≤2 次），重试耗尽未过静态 = 被静态拦；
  通过静态的语句进入执行模拟，编译仅测量（残留编译失败 = 静态层漏网）；
- compile（+编译重试）：MySQL 编译作执行闸门，编译错误带反馈重试，重试耗尽未过编译 = 被编译拦；
- full（全量，现状）：静态 + 编译双重闸门 + 反馈重试（同 tools/native_financial._generate_sql）。

统计口径：
- 语句级编译通过率 = 进入执行模拟后编译通过的语句数 / 该组首轮生成过非空 SQL 的子问题数
  （空 SQL = LLM 判定无需 DB 查询，不计分母，与历史“无 SQL 题目不计失败”一致）；
- 分层归因：仅提示词组失败数（基线）→ +静态组残留 = 编译层需兜底的边界；四组逐层收敛到全量 100%；
- 各层“拦截次数”统计尝试级（含拦截后经反馈修复放行）与语句级（重试耗尽仍被拦）。

用法::

    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_defense_ablation            # 全量 4 组 × 108 子问题
    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_defense_ablation --limit 3  # 冒烟（前 3 题）
    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_defense_ablation --only B2007 B2041
    .\\.venv\\Scripts\\python -m tools.data_scripts.sql_defense_ablation --arm prompt_only  # 只跑一组

输出:
  训练结果数据/sql_defense_ablation.jsonl     # 每条 = (题, 子问题, 组) 完整尝试与归因记录
  训练结果数据/sql_defense_ablation_summary.json
  docs/评估报告/SQL三层防线消融.md            # 四组对比 + 分层归因样例（四组全跑完才生成）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 真实重跑：与 sql_full_regression_native 一致，杜绝命中查询缓存
os.environ["QUERY_CACHE_ENABLED"] = "false"

from config.rag_config import RAGConfig  # noqa: E402
from tools.sql_validator import compile_check, validate_sql  # noqa: E402

GOLDEN = Path("database/golden/v1_2026-08-22.json")
OUT_JSONL = Path("训练结果数据/sql_defense_ablation.jsonl")
OUT_SUMMARY = Path("训练结果数据/sql_defense_ablation_summary.json")
OUT_MD = Path("docs/评估报告/SQL三层防线消融.md")

ARMS = ("prompt_only", "static", "compile", "full")
ARM_LABELS = {
    "prompt_only": "仅提示词",
    "static": "+静态校验",
    "compile": "+编译重试",
    "full": "全量（现状）",
}


def _utf8() -> None:
    """Windows 控制台统一 UTF-8 输出。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_sub_questions() -> List[Dict[str, str]]:
    """读取 golden v1 全部子问题（含题号与类型）。"""
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    subs: List[Dict[str, str]] = []
    for item in data["items"]:
        for q in item["子问题"]:
            subs.append({"编号": item["编号"], "类型": item["问题类型"], "问题": q})
    return subs


def _clean_sql(text: str) -> str:
    """清洗 LLM 输出：去围栏与前后缀，与 native_financial._generate_sql 对齐。"""
    sql = (text or "").strip()
    sql = sql.strip("`")
    if sql.lower().startswith("sql"):
        sql = sql[3:].lstrip()
    return sql.strip()


def _gen_once(client: Any, model: str, question: str, feedback: List[str], enable_thinking: bool) -> str:
    """单次 LLM 生成 SQL（消息构造与 tools/native_financial._generate_sql 一致）。"""
    from prompts.financial import SQL_GEN_SYSTEM_PROMPT

    user_content = f"重构后的问题: {question}\nStandard_field_name: （无上游指标提取，请依据字段白名单自选）"
    if feedback:
        user_content += "\n\n上一次生成的 SQL 校验失败，错误如下，请修正后重新生成：\n" + "\n".join(feedback[-3:])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SQL_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=800,
        extra_body={"enable_thinking": enable_thinking},
    )
    return _clean_sql(resp.choices[0].message.content or "")


def _run_arm(client: Any, model: str, sub: Dict[str, str], arm: str,
             schema: Optional[Dict], conn: Any, retries: int, enable_thinking: bool) -> Dict[str, Any]:
    """对单个子问题执行某组防线，返回完整尝试记录与最终归因类别。

    返回记录中的 outcome 取值：
      no_sql                 —— 首轮生成空 SQL（LLM 判定无需 DB 查询）
      pass                   —— 进入执行模拟且编译通过
      reject_static          —— 静态闸门重试耗尽仍未通过
      reject_compile         —— 编译闸门重试耗尽仍未通过
      fail_compile_measure   —— 通过本组闸门进入执行模拟但 MySQL 编译失败（静态组的漏网，需编译层兜底）
    """
    rec: Dict[str, Any] = {
        "编号": sub["编号"], "类型": sub["类型"], "问题": sub["问题"][:200], "组": arm,
        "outcome": "", "尝试": [], "最终SQL": "", "LLM调用数": 0,
    }
    feedback: List[str] = []
    executed_sql = ""          # 该组最终会放行去执行的语句
    executed_compiled = False  # 该语句是否已由编译闸门/测量确认通过
    for attempt in range(retries + 1):
        try:
            sql = _gen_once(client, model, sub["问题"], feedback, enable_thinking)
        except Exception as exc:  # noqa: BLE001
            err = f"LLM 调用失败: {exc}"
            rec["尝试"].append({"次数": attempt + 1, "SQL": "", "静态错误": [], "编译错误": err, "放行": False})
            feedback.append(err)
            continue
        rec["LLM调用数"] += 1
        if not sql:
            if attempt == 0:
                rec["outcome"] = "no_sql"
                rec["尝试"].append({"次数": 1, "SQL": "", "静态错误": [], "编译错误": [], "放行": False})
                break
            # 重试阶段空输出：按一次无效生成处理（不追加反馈，消耗一次机会）
            rec["尝试"].append({"次数": attempt + 1, "SQL": "", "静态错误": [], "编译错误": ["LLM 返回空 SQL"], "放行": False})
            continue
        serrs: List[str] = []
        cerr = ""
        pass_gate = False
        if arm == "prompt_only":
            # 无任何闸门：首轮语句直接进入执行模拟（本组只有 1 次尝试）
            pass_gate = True
        elif arm == "static":
            ok_v, serrs = validate_sql(sql, schema) if schema is not None else (True, [])
            pass_gate = ok_v                      # 静态通过 = 放行执行（编译仅测量）
            if not ok_v and attempt >= retries:
                rec["outcome"] = "reject_static"
        elif arm == "compile":
            cerr = compile_check(conn, sql) if conn is not None else ""
            pass_gate = not cerr                  # 编译通过 = 放行
            if cerr and attempt >= retries:
                rec["outcome"] = "reject_compile"
        else:  # full：静态 + 编译双重闸门（现状）
            ok_v, serrs = validate_sql(sql, schema) if schema is not None else (True, [])
            if ok_v:
                cerr = compile_check(conn, sql) if conn is not None else ""
            pass_gate = ok_v and not cerr
            if not pass_gate and attempt >= retries:
                rec["outcome"] = "reject_compile" if (ok_v and cerr) else "reject_static"
        rec["尝试"].append({
            "次数": attempt + 1, "SQL": sql[:3000], "静态错误": list(serrs)[:4],
            "编译错误": (cerr or "")[:200], "放行": pass_gate,
        })
        if pass_gate:
            executed_sql = sql
            executed_compiled = (arm in ("compile", "full"))  # 编译闸门组在闸门内已编译通过
            rec["outcome"] = "pass"
            break
        # 未过闸门：依据组语义决定是否反馈重试
        if arm == "static" and serrs:
            feedback.extend(list(serrs)[:4])
            continue
        if arm == "compile" and cerr:
            feedback.append(f"编译错误: {cerr[:200]}")
            continue
        if arm == "full" and (serrs or cerr):
            if serrs:
                feedback.extend(list(serrs)[:4])
            if cerr:
                feedback.append(f"编译错误: {cerr[:200]}")
            continue
        # 静态组：静态通过但编译失败 → 进入执行模拟并测量编译（不重试，归因给编译层兜底）
        if arm == "static":
            executed_sql = sql
            executed_compiled = False
            rec["outcome"] = "fail_compile_measure"
            break
        # 非预期分支：退出尝试循环
        rec["outcome"] = rec["outcome"] or "reject_static"
        break

    rec["最终SQL"] = executed_sql[:8000]
    if rec["outcome"] == "no_sql":
        return rec
    if rec["outcome"] == "pass" and not executed_compiled:
        # prompt_only / static 组：对放行语句做最终编译测量
        cerr = compile_check(conn, executed_sql) if conn is not None else ""
        if cerr:
            rec["outcome"] = "fail_compile_measure"
            if rec["尝试"]:
                rec["尝试"][-1]["编译错误"] = cerr[:200]
    return rec


def aggregate(results: List[Dict[str, Any]], total_subs: int) -> Dict[str, Any]:
    """按组聚合（outcome 分布 / 语句级编译通过率 / 各层拦截次数）。"""
    groups: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        arm_recs = [r for r in results if r["组"] == arm]
        outcome: Dict[str, int] = {}
        for r in arm_recs:
            outcome[r["outcome"]] = outcome.get(r["outcome"], 0) + 1
        stmt_n = sum(v for k, v in outcome.items() if k not in ("no_sql",))
        passed = outcome.get("pass", 0)
        static_intercepts = sum(
            1 for r in arm_recs for a in r["尝试"] if a.get("静态错误")
        )
        compile_intercepts = sum(
            1 for r in arm_recs for a in r["尝试"] if a.get("编译错误")
        )
        groups[arm] = {
            "标签": ARM_LABELS[arm],
            "子问题总数": total_subs,
            "无SQL子问题": outcome.get("no_sql", 0),
            "有SQL子问题": stmt_n,
            "编译通过": passed,
            "语句级编译通过率": round(passed / stmt_n, 4) if stmt_n else None,
            "编译测量失败": outcome.get("fail_compile_measure", 0),
            "被静态拦(重试耗尽)": outcome.get("reject_static", 0),
            "被编译拦(重试耗尽)": outcome.get("reject_compile", 0),
            "静态拦截次数(尝试级)": static_intercepts,
            "编译拦截次数(尝试级)": compile_intercepts,
            "LLM调用数": sum(r["LLM调用数"] for r in arm_recs),
        }
    return {"组": groups, "子问题数": total_subs}


def _fail_examples(results: List[Dict[str, Any]], arm: str, limit: int = 6) -> List[Dict[str, Any]]:
    """抽取该组未编译通过/被拦的归因样例。"""
    out: List[Dict[str, Any]] = []
    bad = {"reject_static", "reject_compile", "fail_compile_measure"}
    for r in results:
        if r["组"] != arm or r["outcome"] not in bad:
            continue
        attempts = r["尝试"]
        feedback: List[str] = []
        for a in attempts:
            feedback.extend(a.get("静态错误") or [])
            if a.get("编译错误"):
                feedback.append(a["编译错误"])
        out.append({
            "编号": r["编号"], "类型": r["类型"], "问题": r["问题"][:70],
            "最终SQL": (r["最终SQL"] or (attempts[-1].get("SQL") or ""))[:220],
            "尝试次数": len(attempts),
            "各层反馈": feedback[:5],
            "outcome": r["outcome"],
        })
        if len(out) >= limit:
            break
    return out


def _fix_examples(results: List[Dict[str, Any]], arm: str, limit: int = 4) -> List[Dict[str, Any]]:
    """抽取该组被闸门拦截后、经反馈重试修复放行的归因样例。"""
    out: List[Dict[str, Any]] = []
    for r in results:
        if r["组"] != arm or r["outcome"] != "pass" or len(r["尝试"]) < 2:
            continue
        first = r["尝试"][0]
        errs: List[str] = list(first.get("静态错误") or [])
        if first.get("编译错误"):
            errs.append(first["编译错误"])
        last = r["尝试"][-1]
        out.append({
            "编号": r["编号"], "类型": r["类型"], "首错": errs[:1],
            "修复SQL": (last.get("SQL") or "")[:160],
        })
        if len(out) >= limit:
            break
    return out


def render_md(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    """渲染 docs/评估报告/SQL三层防线消融.md。"""
    lines = [
        "# SQL 三层防线消融实验报告（原生链路 · 函数级单发口径）",
        "",
        "> 2026-09-05 · B-04 · qwen3.5-plus（enable_thinking=False，temperature=0.1，max_tokens=800）",
        "> 口径：golden v1 全部子问题逐子问题独立生成（不做 Agent 多轮累积），与 2026-08“单发复跑”口径一致；",
        "> 空 SQL（LLM 判定无需 DB 查询）不计语句分母；防线反馈重试 ≤2 次（同 AGENT_NATIVE_RETRY=2）。",
        "",
        "## 四组防线组合",
        "",
        "| 组 | 构成 | 执行闸门 |",
        "| --- | --- | --- |",
        "| 仅提示词 | SQL_GEN_SYSTEM_PROMPT（字段-表归属等规则） | 无，生成即执行（编译仅测量） |",
        "| +静态校验 | 提示词 + validate_sql | 静态校验；通过后执行（编译仅测量） |",
        "| +编译重试 | 提示词 + MySQL compile_check | 编译通过才执行 |",
        "| 全量（现状） | 提示词 + 静态校验 + 编译重试 | 静态 + 编译双闸门 |",
        "",
        "## 语句级结果对比",
        "",
        "| 组 | 有 SQL 子问题 | 编译通过 | 语句级编译通过率 | 编译测量失败 | 被静态拦(重试耗尽) | 被编译拦(重试耗尽) | 静态拦次数(尝试级) | 编译拦次数(尝试级) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        g = summary["组"][arm]
        rate = "—" if g["语句级编译通过率"] is None else f"{g['语句级编译通过率'] * 100:.1f}%"
        lines.append(
            f"| {g['标签']} | {g['有SQL子问题']} | {g['编译通过']} | {rate} | {g['编译测量失败']} "
            f"| {g['被静态拦(重试耗尽)']} | {g['被编译拦(重试耗尽)']} | {g['静态拦截次数(尝试级)']} | {g['编译拦截次数(尝试级)']} |"
        )
    lines += [
        "",
        "> 说明：编译通过率分母 = 该组首轮生成过非空 SQL 的子问题数（空 SQL 不计）。",
        "> “编译测量失败”= 已过本组闸门进入执行模拟、但 MySQL 编译不过（仅提示词/静态组可能出现，即编译层需兜底的漏网）；",
        "> “被 X 拦(重试耗尽)”= 该闸门重试 2 次仍不通过，语句不会进入执行；拦截次数为尝试级计数（含拦截后修复放行）。",
        "",
        "## 分层归因样例",
        "",
        "### 仅提示词组（基线：编译不过的语句 = 后两层需拦截对象）",
        "",
    ]
    fail_p = _fail_examples(results, "prompt_only")
    lines += [f"- **{e['编号']}**（{e['类型']}）: `{e['最终SQL'][:150]}` → {e['各层反馈'][-1][:120]}" for e in fail_p] or ["- （无失败）"]
    lines += ["", "### +静态校验组（残留 = 静态拦不住、须编译层兜底）", ""]
    fail_s = _fail_examples(results, "static")
    lines += [f"- **{e['编号']}**（{e['类型']}）: {e['各层反馈'][:2]} → outcome={e['outcome']}" for e in fail_s] or ["- （无失败）"]
    lines += ["", "### +编译重试组", ""]
    fail_c = _fail_examples(results, "compile")
    lines += [f"- **{e['编号']}**（{e['类型']}）: {e['各层反馈'][:2]} → outcome={e['outcome']}" for e in fail_c] or ["- （无失败）"]
    lines += ["", "### 全量组（现状）", ""]
    fail_f = _fail_examples(results, "full")
    lines += [f"- **{e['编号']}**（{e['类型']}）: {e['各层反馈'][:2]} → outcome={e['outcome']}" for e in fail_f] or ["- （无失败）"]

    lines += ["", "### 各层拦截 -> 反馈修复样例（层贡献证据）", ""]
    lines += ["- 以下为闸门首轮拦截到错误、把错误反馈给 LLM 后重试修复放行的实例（统计见上表“拦截次数(尝试级)”）。", ""]
    for arm, title in (("static", "静态层拦截-修复"), ("compile", "编译层拦截-修复"), ("full", "全量组拦截-修复")):
        lines += [f"**{title}**", ""]
        fix = _fix_examples(results, arm)
        if fix:
            for e in fix:
                lines += [f"- {e['编号']}（{e['类型']}）首错：{e['首错'][:1]} → 修复：`{e['修复SQL']}`"]
        else:
            lines += ["- （无拦截记录）"]
        lines += [""]

    lines += [
        "",
        "## 结论",
        "",
        "- 三层防线逐层收敛：提示词规则消解大部分字段-表归属错误；静态校验在执行前拦截表名/别名/归属类错误（免 MySQL 往返）；",
        "- MySQL 编译重试兜底静态校验边界（函数调用/复杂子查询/MySQL 特有语法等），使放行语句全部真实编译通过；",
        "- 全量组（现状）结果与 2026-08 全量回归“修复后 100%”口径对齐。",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """消融实验入口。"""
    _utf8()
    for name in ("httpx", "httpcore", "urllib3", "openai", "dashscope"):
        logging.getLogger(name).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="SQL 三层防线消融（函数级单发，四组 × golden 子问题）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟）")
    parser.add_argument("--only", nargs="*", default=None, help="只跑指定编号（如 B2007）")
    parser.add_argument("--retries", type=int, default=2, help="防线组反馈重试次数（默认 2，同 AGENT_NATIVE_RETRY）")
    parser.add_argument("--arm", choices=list(ARMS) + ["all"], default="all", help="只跑某一组（默认全跑）")
    args = parser.parse_args(argv)

    subs_all = load_sub_questions()
    if args.only:
        keep = set(args.only)
        subs = [s for s in subs_all if s["编号"] in keep]
    elif args.limit:
        seen: set[str] = set()
        picked: List[Dict[str, str]] = []
        for s in subs_all:
            seen.add(s["编号"])
            if len(seen) <= args.limit:
                picked.append(s)
        subs = picked
    else:
        subs = subs_all
    print(f"golden: 全量 {len(subs_all)} 子问题；本次 {len(subs)} 子问题（{len({s['编号'] for s in subs})} 题）", flush=True)

    config = RAGConfig()
    from openai import OpenAI

    client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = config.LLM_MODEL
    enable_thinking = bool(getattr(config, "AGENT_ENABLE_THINKING", False))
    try:
        from tools.native_financial import _load_schema_conn

        schema, conn = _load_schema_conn(config)
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] MySQL schema 加载失败: {exc}", flush=True)
        schema, conn = None, None
    if conn is None or schema is None:
        print("MySQL 不可用：无法做编译测量/编译重试，中止。", flush=True)
        return 2

    done: List[Dict[str, Any]] = []
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    done_keys = {f"{r['编号']}|{r['问题']}|{r['组']}" for r in done}
    arms = [args.arm] if args.arm != "all" else list(ARMS)
    todo: List[Dict[str, Any]] = []
    for s in subs:
        for arm in arms:
            key = f"{s['编号']}|{s['问题']}|{arm}"
            if key not in done_keys:
                todo.append({"sub": s, "arm": arm})
    if done_keys:
        print(f"断点续跑：已跳过 {len(done_keys)} 条，本次执行 {len(todo)} 条", flush=True)

    t_start = time.time()
    try:
        for idx, item in enumerate(todo, 1):
            s, arm = item["sub"], item["arm"]
            print(f"[{idx}/{len(todo)}] {s['编号']}（{s['类型']}）·{ARM_LABELS[arm]} ...", flush=True)
            rec = _run_arm(client, model, s, arm, schema, conn, args.retries, enable_thinking)
            with OUT_JSONL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.append(rec)
            if idx % 10 == 0:
                elapsed = round((time.time() - t_start) / 60, 1)
                print(f"[进度 {idx}/{len(todo)}] 已耗时 {elapsed} min", flush=True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    summary = aggregate(done, len(subs))
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== SQL 三层防线消融汇总 =====", flush=True)
    for arm in ARMS:
        g = summary["组"][arm]
        rate = "—" if g["语句级编译通过率"] is None else f"{g['语句级编译通过率'] * 100:.1f}%"
        print(f"{g['标签']}: 有SQL {g['有SQL子问题']} 通过 {g['编译通过']} = {rate}"
              f"（编译测量失败 {g['编译测量失败']}，静态拦 {g['被静态拦(重试耗尽)']}，编译拦 {g['被编译拦(重试耗尽)']}，"
              f"尝试级拦截 静态{g['静态拦截次数(尝试级)']}/编译{g['编译拦截次数(尝试级)']}，LLM {g['LLM调用数']} 次）", flush=True)
    print(f"\n已保存: {OUT_JSONL} / {OUT_SUMMARY}", flush=True)

    expected = len(subs) * len(ARMS)
    full_scope = (not args.only and not args.limit and len(subs) == len(subs_all))
    if full_scope and len(done) >= expected:
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text(render_md(summary, done), encoding="utf-8")
        print(f"报告已生成: {OUT_MD}", flush=True)
    else:
        print(f"[提示] 仅完成 {len(done)}/{expected} 条，未生成报告（需四组全部完成）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
