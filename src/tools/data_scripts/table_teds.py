# -*- coding: utf-8 -*-
"""TEDS 简化版：表格解析质量的**分数**指标（替代「通过 / 不通过」二值判定）。

## 指标定义（明确简化了什么）

把表格看作三层树 **表 → 行 → 格**，与解析结果计算编辑距离后归一：

    TEDS = 1 - EditDist(gold, pred) / max(nodes(gold), nodes(pred))
    nodes(T) = 1 + (1 + len(headers)) + Σ_rows (1 + len(row))

编辑距离用两级序列对齐：行序列对齐（DP）→ 行内单元格序列对齐（DP）。
- 插入 / 删除一行 = ``1 + len(row)``；插入 / 删除一格 = ``1``；
- 替换一格的代价 = ``1 - sim(a, b)``；替换一行的代价 = 行内单元格编辑距离。

``sim`` 的口径（金融表专用）：
1. 完全相同 → ``1.0``；
2. 两边都能解析成「数值 + 单位后缀（%/倍/无）」→ 数值相等且后缀一致得 ``1.0``，否则 ``0.0``
   （所以 ``1,234`` vs ``1234`` 算相同，``18.85%`` vs ``18.85`` 算不同）；
3. 其余 → 字符级 ``difflib.SequenceMatcher`` 相似度；
4. **Gold 该格为空、解析结果是空表头占位名**（``项目`` / ``列N``）视为相同（给空表头补占位名是合理归一）。
   另：``has_header = False`` 的表，**表头行不参与比对**（不存在表头行就不该被扣分）。

**相对标准 TEDS 的简化**：① 只有「表 / 行 / 格」三层，没有 HTML 标签树、没有合并单元格坐标层；
② 单元格相似度用字符级 + 数值归一，不做 token/结构感知；③ 树大小按「格数 + 行数 + 1」计。
因此本分数适合**同一解析器的版本间对比**（相对量），不宣称与论文 TEDS 同值。

## Gold 自检（防「转写错字当标准答案」）

Gold 集是从原文**独立转写**的。为防转写引入错字，脚本对每个非空格子做**原文可定位校验**：
该格文本必须能在原文（去标签 + 折叠空白，或去全部空白）里找到，否则列为可疑。

## 用法

    .venv\\Scripts\\python.exe src/tools/data_scripts/table_teds.py
    .venv\\Scripts\\python.exe src/tools/data_scripts/table_teds.py --verify-only
    .venv\\Scripts\\python.exe src/tools/data_scripts/table_teds.py --gold <gold.json> --out <report.json>
"""

import argparse
import difflib
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data.table_struct import scan_tables  # noqa: E402

_REPO_ROOT = _SRC_ROOT.parent

DEFAULT_CORPUS = _REPO_ROOT / "data_corpus" / "全部数据" / "正式数据" / "附件5：研报数据"
DEFAULT_GOLD = _REPO_ROOT / "训练结果数据" / "gold_set" / "table_gold_set.json"
DEFAULT_OUT = _REPO_ROOT / "训练结果数据" / "gold_set" / "teds_report.json"

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"^\(?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\)?\s*(%|％)?$")
_PLACEHOLDER_RE = re.compile(r"^(?:项目|列\d+)$")

ROW_NODE_COST = 1.0
CELL_INSERT_COST = 1.0


def clean_text(text: str) -> str:
    """单元格文本归一：去零宽字符、折叠空白、去首尾。"""
    s = (text or "").replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return _WS_RE.sub(" ", s).strip()


def parse_number(cell: str) -> Optional[Tuple[float, str]]:
    """把单元格解析为 ``(数值, 单位后缀)``；非纯数值返回 ``None``。

    括号负数（``(1,234.5)``）按负数处理；``%`` / ``％`` 归一为 ``%``。
    """
    text = clean_text(cell).replace(",", "")
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    body = text[1:-1] if negative else text
    body = body.strip()
    suffix = ""
    if body.endswith("%") or body.endswith("％"):
        suffix, body = "%", body[:-1].strip()
    try:
        value = float(body)
    except ValueError:
        return None
    return (-value if negative else value, suffix)


def cell_similarity(gold: str, pred: str) -> float:
    """单元格相似度：数值口径优先，退化到字符级相似度。"""
    left, right = clean_text(gold), clean_text(pred)
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    num_left, num_right = parse_number(left), parse_number(right)
    if num_left is not None or num_right is not None:
        if num_left is None or num_right is None:
            return 0.0
        # 数值用相对容差 1e-9 比较；单位后缀必须一致（18.85% != 18.85）
        if num_left[1] != num_right[1]:
            return 0.0
        base = max(abs(num_left[0]), abs(num_right[0]), 1.0)
        return 1.0 if abs(num_left[0] - num_right[0]) / base <= 1e-9 else 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def pair_similarity(gold: str, pred: str) -> float:
    """Gold 与解析结果的**方向性**相似度。

    除 :func:`cell_similarity` 的口径外，额外承认一条等价：**Gold 该格为空、解析结果是
    空表头占位名**（``项目`` / ``列N``）视为相同——解析器给空表头补占位名是合理归一，
    不应因此扣分。反向（Gold 有文本、解析给占位名）不豁免。
    """
    if not clean_text(gold) and _PLACEHOLDER_RE.match(clean_text(pred)):
        return 1.0
    return cell_similarity(gold, pred)


def sequence_cost(
    gold: Sequence[Any],
    pred: Sequence[Any],
    sub_cost,
    ins_cost,
    del_cost,
) -> float:
    """通用序列编辑距离（DP）：``sub_cost(g, p)`` / ``ins_cost(p)`` / ``del_cost(g)``。"""
    n, m = len(gold), len(pred)
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + ins_cost(pred[j - 1])
    for i in range(1, n + 1):
        cur = [prev[0] + del_cost(gold[i - 1])] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j] + del_cost(gold[i - 1]),
                cur[j - 1] + ins_cost(pred[j - 1]),
                prev[j - 1] + sub_cost(gold[i - 1], pred[j - 1]),
            )
        prev = cur
    return prev[m]


def row_cell_cost(gold_row: Sequence[str], pred_row: Sequence[str]) -> float:
    """一行对一行的单元格级编辑代价（不含行节点代价）。"""
    return sequence_cost(
        list(gold_row),
        list(pred_row),
        sub_cost=lambda g, p: CELL_INSERT_COST - pair_similarity(g, p),
        ins_cost=lambda _p: CELL_INSERT_COST,
        del_cost=lambda _g: CELL_INSERT_COST,
    )


def row_sub_cost(gold_row: Sequence[str], pred_row: Sequence[str]) -> float:
    """替换一整行的代价 = **行内单元格编辑代价**。

    行节点本身在两行被对齐时恒定匹配（代价 0）——否则「完全相同的两行」也会被计 1，
    使满分不可能达到。行的增删仍按 ``行节点 + 格数`` 计（见 :func:`_row_len`）。
    """
    return row_cell_cost(gold_row, pred_row)


def _row_len(row: Sequence[str]) -> float:
    return ROW_NODE_COST + len(row) * CELL_INSERT_COST


def stack_rows(view: Dict[str, Any]) -> List[List[str]]:
    """把 ``{headers, rows, has_header}`` 展平成参与比对的「行序列」。

    ``has_header = False`` 时**不把表头行计入**——此时解析结果的 ``项目 / 列N`` 占位名
    不参与打分（避免用「不存在的表头行」扣分）。
    """
    rows: List[List[str]] = []
    if view.get("has_header", True):
        rows.append(list(view.get("headers") or []))
    rows.extend(list(row) for row in (view.get("rows") or []))
    return rows


def teds(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    """计算 TEDS 简化版分数（``gold`` / ``pred`` 均为 ``{headers, rows}``）。"""
    gold_rows = stack_rows(gold)
    pred_rows = stack_rows(pred)
    if not gold_rows and not pred_rows:
        return 1.0
    cost = sequence_cost(
        gold_rows,
        pred_rows,
        sub_cost=row_sub_cost,
        ins_cost=lambda row: _row_len(row),
        del_cost=lambda row: _row_len(row),
    )
    denom = max(
        1.0 + sum(_row_len(row) for row in gold_rows),
        1.0 + sum(_row_len(row) for row in pred_rows),
    )
    if denom <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - cost / denom))


def raw_plain_text(raw: str, table_format: str = "html") -> str:
    """原文 → 纯文本（去标签 / 折叠空白），用于 Gold 格子可定位校验。"""
    text = raw or ""
    if table_format != "markdown" and ("<tr" in text.lower() or "<table" in text.lower()):
        text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", html.unescape(text)).strip()


def gold_cells_missing_in_raw(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    raw: str,
    table_format: str = "html",
    has_header: bool = True,
) -> List[Dict[str, Any]]:
    """Gold 自检：列出**在原文里定位不到**的格子（转写错字 / 凭空生成）。

    ``has_header = False`` 时**跳过表头行**——此时 Gold 的表头是占位名，不是原文内容。
    """
    plain = raw_plain_text(raw, table_format)
    squashed_plain = _WS_RE.sub("", plain)
    body: List[List[str]] = []
    if has_header:
        body.append(list(headers))
    body.extend(list(r) for r in rows)
    missing: List[Dict[str, Any]] = []
    for index, row in enumerate(body):
        for col, cell in enumerate(row):
            text = clean_text(cell)
            if not text:
                continue
            if text in plain or _WS_RE.sub("", text) in squashed_plain:
                continue
            missing.append({"行": index - 1, "列": col, "文本": text})
    return missing


def load_corpus_tables(corpus: Path) -> Dict[str, Any]:
    """扫描语料，返回 ``table_id -> Table``。"""
    tables: Dict[str, Any] = {}
    for path in sorted(corpus.rglob("*.md")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        _, found = scan_tables(content, source_file=path.name, title=path.stem)
        for table in found:
            tables[table.table_id] = table
    return tables


def table_to_gold_view(table: Any) -> Dict[str, Any]:
    """解析结果 → ``{headers, rows}`` 视图。"""
    return {
        "headers": list(table.headers),
        "rows": [list(r) for r in table.rows],
        "has_header": bool(table.has_header),
    }


def score_gold_set(
    gold_payload: Dict[str, Any], tables: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """对 Gold 集逐表打分并做 Gold 自检。"""
    records: List[Dict[str, Any]] = []
    for entry in gold_payload.get("tables", []):
        table_id = entry["table_id"]
        table = tables.get(table_id)
        gold_view = {
            "headers": entry.get("headers") or [],
            "rows": entry.get("rows") or [],
            "has_header": bool(entry.get("has_header", True)),
        }
        record: Dict[str, Any] = {
            "table_id": table_id,
            "no": entry.get("no"),
            "bucket": entry.get("bucket"),
            "source_file": entry.get("source_file"),
            "gold_rows": len(gold_view["rows"]),
            "gold_cols": len(gold_view["headers"]),
        }
        if table is None:
            record["teds"] = None
            record["状态"] = "语料未命中"
            records.append(record)
            continue
        record["pred_rows"] = table.row_count
        record["pred_cols"] = table.col_count
        record["caption_match"] = clean_text(entry.get("caption", "")) == clean_text(table.caption)
        record["teds"] = round(teds(gold_view, table_to_gold_view(table)), 6)
        missing = gold_cells_missing_in_raw(
            gold_view["headers"],
            gold_view["rows"],
            table.raw,
            table.table_format,
            has_header=bool(entry.get("has_header", True)),
        )
        record["gold_自检_原文中定位不到的格子"] = missing[:20]
        record["gold_自检_通过"] = not missing
        records.append(record)
    return records


def build_report(records: List[Dict[str, Any]], gold_path: Path) -> Dict[str, Any]:
    """汇总报告（含总体 / 分桶平均 TEDS）。"""
    scored = [r for r in records if isinstance(r.get("teds"), (int, float))]
    buckets: Dict[str, List[float]] = {}
    for record in scored:
        buckets.setdefault(str(record.get("bucket")), []).append(float(record["teds"]))
    avg = sum(float(r["teds"]) for r in scored) / len(scored) if scored else 0.0
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gold_set": str(gold_path),
        "统计": {
            "表数": len(records),
            "已打分": len(scored),
            "平均TEDS": round(avg, 6),
            "最低TEDS": round(min((float(r["teds"]) for r in scored), default=0.0), 6),
            "gold自检通过数": sum(1 for r in records if r.get("gold_自检_通过")),
            "分桶平均TEDS": {
                k: round(sum(v) / len(v), 6) for k, v in sorted(buckets.items())
            },
        },
        "records": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：对 Gold 集打分（TEDS 简化版）并落盘报告。"""
    parser = argparse.ArgumentParser(description="TEDS 简化版表格质量打分 + Gold 自检")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Gold 集 json")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="语料根目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="报告输出路径")
    parser.add_argument("--verify-only", action="store_true", help="只做 Gold 自检，不打分")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass

    gold_payload = json.loads(args.gold.read_text(encoding="utf-8"))
    tables = load_corpus_tables(args.corpus)
    records = score_gold_set(gold_payload, tables)
    report = build_report(records, args.gold)
    report["mode"] = "verify-only" if args.verify_only else "score"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = report["统计"]
    print("表数 %d  已打分 %d  gold 自检通过 %d/%d" % (
        stats["表数"], stats["已打分"], stats["gold自检通过数"], stats["表数"]))
    print("平均 TEDS %.6f  最低 %.6f" % (stats["平均TEDS"], stats["最低TEDS"]))
    print("分桶平均: %s" % json.dumps(stats["分桶平均TEDS"], ensure_ascii=False))
    for record in records:
        print("- %-16s %-5s TEDS=%.6f  row %s->%s col %s->%s  gold自检=%s" % (
            str(record.get("bucket")), str(record.get("no")), float(record.get("teds") or 0.0),
            record.get("gold_rows"), record.get("pred_rows"),
            record.get("gold_cols"), record.get("pred_cols"),
            "通过" if record.get("gold_自检_通过") else "有可疑格子"))
    print("报告: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())