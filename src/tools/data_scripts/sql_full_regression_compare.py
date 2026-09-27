# -*- coding: utf-8 -*-
"""基线 result_3_parallel vs 修复后回归 逐题对比，生成 Markdown 对比章节。

用法::

    # 子问题单发回归（默认）
    python tools/data_scripts/sql_full_regression_compare.py

    # Agent 多轮累积口径回归
    python tools/data_scripts/sql_full_regression_compare.py \
        --regr-jsonl 训练结果数据/sql_agent_regression.jsonl \
        --out-md 训练结果数据/sql_agent_regression_对比.md \
        --title "全量 80 题 Agent 多轮累积口径回归对比"
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

def out(s: str) -> None:
    sys.stdout.write(s + "\n")
    sys.stdout.flush()

BASE_CSV = Path("训练结果数据/sql_compile_report.csv")


def load_baseline() -> Dict[str, Dict]:
    rows = {}
    with BASE_CSV.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["编号"]] = {
                "问题类型": r["问题类型"],
                "SQL语句数": int(r["SQL语句数"]),
                "通过语句数": int(r["通过语句数"]),
                "全部通过": r["全部通过"] == "True",
                "有SQL": r["有SQL"] == "True",
                "首个错误": r["首个错误"],
            }
    return rows


def get_stmt_details(r: Dict) -> List[Dict]:
    """兼容两种回归记录：Agent 级（语句明细）与子问题级（子问题明细内嵌语句明细）。"""
    if "语句明细" in r:
        return r["语句明细"]
    out_list: List[Dict] = []
    for sub in r.get("子问题明细", []):
        out_list.extend(sub.get("语句明细", []))
    return out_list


def load_regression(regr_jsonl: Path) -> Dict[str, Dict]:
    rows = {}
    for line in regr_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["编号"]] = r
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="基线 vs 回归 逐题对比报告")
    parser.add_argument("--regr-jsonl", default="训练结果数据/sql_full_regression.jsonl",
                        help="回归结果 jsonl 路径")
    parser.add_argument("--out-md", default="训练结果数据/sql_full_regression_对比.md",
                        help="对比报告 md 输出路径")
    parser.add_argument("--title", default="全量 80 题修复后回归对比（单发复跑）",
                        help="对比章节标题")
    args = parser.parse_args()
    regr_jsonl = Path(args.regr_jsonl)
    out_md = Path(args.out_md)

    base = load_baseline()
    regr = load_regression(regr_jsonl)
    ids = sorted(base.keys(), key=lambda x: (int(x[1:]), x))
    assert len(regr) == len(ids), f"回归 {len(regr)} 题 != 基线 {len(ids)} 题"

    rows_out: List[List[str]] = []
    status_change: List[str] = []
    sql_change: List[str] = []
    still_fail: List[str] = []
    for bid in ids:
        b, r = base[bid], regr[bid]
        b_ok = b["有SQL"] and b["全部通过"]
        r_ok = r["有SQL"] and r["全通过(有SQL且全部语句通过)"]
        b_stmt, r_stmt = b["SQL语句数"], r["语句数"]
        b_pass, r_pass = b["通过语句数"], r["通过语句数"]
        tag = ""
        if b_ok and not r_ok:
            tag = "回归(修复后未全通过)"
            status_change.append(f"{bid}: 基线通过 -> 回归未全通过")
        elif not b_ok and r_ok:
            tag = "提升(基线未通过 -> 回归通过)"
            status_change.append(f"{bid}: 基线未通过 -> 回归通过")
        if b["有SQL"] and not r["有SQL"]:
            sql_change.append(f"{bid}: 基线有SQL -> 回归空SQL")
        elif not b["有SQL"] and r["有SQL"]:
            sql_change.append(f"{bid}: 基线空SQL -> 回归有SQL")
        err = ""
        if not r_ok and r["有SQL"]:
            for d in get_stmt_details(r):
                if not d["通过"]:
                    err = (d["编译错误"] or d["静态错误"] or ["未知"])[0][:60]
                    break
        rows_out.append([
            bid, b["问题类型"],
            str(b_stmt), str(b_pass), "Y" if b_ok else "N",
            str(r_stmt), str(r_pass), "Y" if r_ok else "N",
            tag, err,
        ])
        if r["有SQL"] and not r["全通过(有SQL且全部语句通过)"]:
            still_fail.append(bid)

    b_all = sum(1 for b in base.values() if b["有SQL"] and b["全部通过"])
    r_all = sum(1 for r in regr.values() if r["有SQL"] and r["全通过(有SQL且全部语句通过)"])
    b_sql = sum(1 for b in base.values() if b["有SQL"])
    r_sql = sum(1 for r in regr.values() if r["有SQL"])
    b_stmt = sum(b["SQL语句数"] for b in base.values())
    b_pass = sum(b["通过语句数"] for b in base.values())
    r_stmt = sum(r["语句数"] for r in regr.values())
    r_pass = sum(r["通过语句数"] for r in regr.values())

    lines = []
    lines.append(f"## {args.title}\n")
    lines.append("| 口径 | 基线 result_3_parallel | 修复后回归 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 语句级编译通过率 | {b_pass}/{b_stmt} = {b_pass / b_stmt * 100:.1f}% | {r_pass}/{r_stmt} = {r_pass / r_stmt * 100:.1f}% |")
    lines.append(f"| 有 SQL 的题目 | {b_sql}/80 | {r_sql}/80 |")
    lines.append(f"| 有 SQL 且全部语句通过 | {b_all}/{b_sql} | {r_all}/{r_sql} |")
    lines.append(f"| 严格全题（空 SQL 视为未通过） | {b_all}/80 | {r_all}/80 |")
    lines.append("")
    lines.append("### 逐题对比\n")
    lines.append("| 编号 | 类型 | 基线语句 | 基线通过 | 基线全通过 | 回归语句 | 回归通过 | 回归全通过 | 状态 | 回归首个错误 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows_out:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if status_change:
        lines.append("### 通过状态变化\n")
        for s in status_change:
            lines.append(f"- {s}")
        lines.append("")
    if sql_change:
        lines.append("### 有 SQL / 空 SQL 状态变化（单发方差或修复影响）\n")
        for s in sql_change:
            lines.append(f"- {s}")
        lines.append("")
    if still_fail:
        lines.append(f"### 回归仍失败的题目（{len(still_fail)}）\n")
        lines.append(f"- " + "、".join(still_fail))
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out(f"\n--- 已生成 {out_md}（{len(ids)} 题）---")


if __name__ == "__main__":
    main()