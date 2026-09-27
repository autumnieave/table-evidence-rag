# -*- coding: utf-8 -*-
"""scripts/ci_golden_subset.py —— CI 第 2 层：golden 子集回归（离线 mock 版）

口径（重要，勿与真实口径混用）：
- 样本：golden v1 的 80 题中，按「问题类型分层 + 固定随机种子」抽取的 N 题（默认 20）；
- 断言：题目参考 SQL 的语句级解析通过率（tools.sql_validator.parse_sql，纯离线、零外部依赖、零 LLM 调用）；
  可选 --schema-from-db 叠加 validate_sql（字段-表归属静态校验，需 MySQL 读 schema，仍不调用 LLM）；
- 产物：题级/语句级通过率 JSON + 控制台摘要；有失败语句时退出码 1（供 CI 门禁）。
- mock 版口径：不含 LLM 生成、不含 MySQL 编译，与 golden 全量 99/99 的真实口径不可直接对比。

用法：
  python scripts/ci_golden_subset.py                             # 默认 20 题、离线解析
  python scripts/ci_golden_subset.py --subset-size 10 --seed 1
  python scripts/ci_golden_subset.py --schema-from-db            # 叠加字段-表归属静态校验
  python scripts/ci_golden_subset.py --json-out 训练结果数据/ci_golden_subset.json
  python scripts/ci_golden_subset.py --only B2001 B2019          # 定向调试
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import golden as golden_mod  # noqa: E402
from tools.sql_validator import parse_sql, validate_sql  # noqa: E402

DEFAULT_OUT = "训练结果数据/ci_golden_subset.json"


def _stdout_utf8() -> None:
    """Windows 控制台统一 UTF-8 输出"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def split_statements(sql: str) -> List[str]:
    """按分号拆分参考 SQL（golden 的 SQL 字段可能含多条语句）"""
    return [s.strip() for s in (sql or "").split(";") if s.strip()]


def stratified_sample(items: List[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    """按问题类型分层轮转抽样（固定种子 → 结果可复现）"""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        buckets[str(it.get("问题类型") or "未分类")].append(it)
    queues = {k: sorted(v, key=lambda x: str(x.get("编号"))) for k, v in buckets.items()}
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    keys = sorted(queues)
    while len(picked) < size:
        progressed = False
        for k in keys:
            if len(picked) >= size:
                break
            if queues[k]:
                picked.append(queues[k].pop(rng.randrange(len(queues[k]))))
                progressed = True
        if not progressed:
            break
    return picked


def load_schema_from_db() -> Optional[Dict[str, Dict[str, str]]]:
    """可选：从 MySQL 读 schema（用于字段-表归属静态校验）；失败返回 None"""
    try:
        from config.rag_config import get_config
        from tools.native_financial import _load_schema_conn

        schema, conn = _load_schema_conn(get_config())
        if conn is not None:
            conn.close()
        return schema
    except Exception as exc:  # noqa: BLE001
        print("[警告] 未能加载 MySQL schema，跳过字段-表归属校验: %s" % exc)
        return None


def run_subset(items: List[Dict[str, Any]], schema: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, Any]:
    """对子集逐题跑参考 SQL 的解析 / 静态校验"""
    rows: List[Dict[str, Any]] = []
    for it in items:
        stmts = split_statements(str(it.get("SQL") or ""))
        detail: List[Dict[str, Any]] = []
        for stmt in stmts:
            parsed = parse_sql(stmt) is not None
            serrs: List[str] = []
            if parsed and schema:
                try:
                    ok, serrs = validate_sql(stmt, schema)
                    if ok:
                        serrs = []
                except Exception as exc:  # noqa: BLE001
                    serrs = ["validate_sql 异常: %s" % exc]
            detail.append(
                {
                    "sql": stmt[:200],
                    "解析通过": parsed,
                    "静态错误": serrs,
                    "通过": parsed and not serrs,
                }
            )
        rows.append(
            {
                "编号": it.get("编号"),
                "问题类型": it.get("问题类型"),
                "语句数": len(stmts),
                "通过语句数": sum(1 for d in detail if d["通过"]),
                "有SQL": bool(stmts),
                "全通过": bool(stmts) and all(d["通过"] for d in detail),
                "语句明细": detail,
            }
        )
    total_stmt = sum(r["语句数"] for r in rows)
    pass_stmt = sum(r["通过语句数"] for r in rows)
    mode = "mock（参考 SQL 解析 + 字段归属静态校验；不含 LLM 生成与 MySQL 编译）" if schema else "mock（参考 SQL 解析；不含 LLM 生成与 MySQL 编译）"
    return {
        "口径": mode,
        "题数": len(rows),
        "语句总数": total_stmt,
        "通过语句数": pass_stmt,
        "语句级通过率": round(pass_stmt / total_stmt, 4) if total_stmt else 0.0,
        "全通过题数": sum(1 for r in rows if r["全通过"]),
        "有SQL题数": sum(1 for r in rows if r["有SQL"]),
        "无SQL题数": sum(1 for r in rows if not r["有SQL"]),
        "rows": rows,
    }


def main() -> int:
    """入口：分层抽样 → 解析/静态校验 → 落 JSON → 退出码"""
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="CI 第 2 层：golden 子集回归（mock 版）")
    ap.add_argument("--golden-version", default="v1", help="golden 版本（默认 v1）")
    ap.add_argument("--subset-size", type=int, default=20, help="子集题数（默认 20）")
    ap.add_argument("--seed", type=int, default=20260910, help="分层抽样随机种子（固定 → 可复现）")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定编号（调试用）")
    ap.add_argument("--schema-from-db", action="store_true", help="叠加字段-表归属静态校验（需 MySQL）")
    ap.add_argument("--json-out", default=DEFAULT_OUT, help="结果 JSON 路径（默认 %s）" % DEFAULT_OUT)
    args = ap.parse_args()

    golden = golden_mod.load_golden(args.golden_version)
    items: List[Dict[str, Any]] = list(golden.get("items") or [])
    print("golden %s: %d 题（tag=%s）" % (args.golden_version, len(items), golden.get("tag")))

    if args.only:
        wanted = {x.strip() for x in args.only}
        picked = [it for it in items if str(it.get("编号")) in wanted]
    else:
        picked = stratified_sample(items, args.subset_size, args.seed)
    print("子集：%d 题（seed=%s）｜编号：%s" % (len(picked), args.seed, ", ".join(str(x.get("编号")) for x in picked)))

    schema = load_schema_from_db() if args.schema_from_db else None
    result = run_subset(picked, schema)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"golden_version": args.golden_version, "seed": args.seed, **result}
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    print("语句级：%d/%d = %.1f%%" % (result["通过语句数"], result["语句总数"], result["语句级通过率"] * 100))
    print("题级：%d/%d 题全通过（有 SQL 题 %d；无 SQL 参考 %d 题不计入）"
          % (result["全通过题数"], result["有SQL题数"], result["有SQL题数"], result["无SQL题数"]))
    for r in result["rows"]:
        if r["有SQL"] and not r["全通过"]:
            bad = [d for d in r["语句明细"] if not d["通过"]]
            print("  [失败] %s 失败语句 %d 条" % (r["编号"], len(bad)))
    print("已保存：%s" % out)
    return 1 if result["通过语句数"] != result["语句总数"] else 0


if __name__ == "__main__":
    raise SystemExit(main())