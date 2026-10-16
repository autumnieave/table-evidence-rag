# -*- coding: utf-8 -*-
"""全库客观校验：`section_rows` 检测 vs 源 HTML 事实（可复现证据，供 §8.6 引用）。

**为什么要这个脚本**：`_detect_section_rows` 是启发式，光有单测只能证明「我预期的例子对」；
必须拿**源 HTML 的事实**（``<td colspan>`` 合计 + 非空格数）逐行对齐，才有 precision / recall。

两个 GT 口径：
- **GT-strict**（默认口径）：源行**满宽**（``sum(colspan) == 列宽``）且**恰 1 个非空格**；
- **GT-v2**：放宽为「近满宽」（``sum(colspan) >= 列宽 - 1`` 且余下的格全空）——用于量化
  「口径选择对结论的影响」：放宽后会多出大量「单值落在中间列」的伪分节行（实测 recall 掉到 87%）。

用法：

    .venv\\Scripts\\python.exe src/tools/data_scripts/measure_section_rows.py
    .venv\\Scripts\\python.exe src/tools/data_scripts/measure_section_rows.py --out <path.json>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from bs4 import BeautifulSoup  # noqa: E402

from data.table_struct import _clean_cell, parse_html_table_grid, scan_tables  # noqa: E402

_REPO_ROOT = _SRC_ROOT.parent
DEFAULT_CORPUS = _REPO_ROOT / "data_corpus" / "全部数据" / "正式数据" / "附件5：研报数据"
DEFAULT_OUT = _REPO_ROOT / "训练结果数据" / "table_struct_review_20261016" / "section_rows_pr.json"


def source_rows(raw: str) -> Optional[List[Tuple[List[str], List[int]]]]:
    """源 ``<tr>`` → ``[(cells, colspans)]``；无 ``<table>`` 返回 ``None``。"""
    soup = BeautifulSoup(raw, "html.parser")
    table = soup.find("table")
    if table is None:
        return None
    out: List[Tuple[List[str], List[int]]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        texts = [_clean_cell(c.get_text(" ", strip=True)) for c in cells]
        spans: List[int] = []
        for cell in cells:
            try:
                spans.append(max(1, int(cell.get("colspan") or 1)))
            except (TypeError, ValueError):
                spans.append(1)
        out.append((texts, spans))
    return out


def evaluate(corpus: Path, relax: bool) -> Dict[str, Any]:
    """跑一遍全库，返回 TP / FP / FN 与两侧样本。"""
    tp = fp = fn = 0
    gt_total = det_total = tables = skipped = 0
    fp_cases: List[str] = []
    fn_cases: List[str] = []
    for path in sorted(corpus.rglob("*.md")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        _, found = scan_tables(content, source_file=path.name, title=path.stem)
        for table in found:
            if table.is_key_value or not table.raw.strip().startswith("<table"):
                continue
            grid = parse_html_table_grid(table.raw)
            rows = source_rows(table.raw)
            if grid is None or rows is None:
                skipped += 1
                continue
            kept = [(texts, spans) for texts, spans in rows if any(texts)]
            # 行对齐校验：只有对齐得上，坐标才可比
            if len(kept) != grid.consumed_rows + len(grid.rows):
                skipped += 1
                continue
            tables += 1
            width = table.col_count
            gold = set()
            for index, (texts, spans) in enumerate(kept):
                mapped = index - grid.consumed_rows
                if mapped < 0:
                    continue
                nonempty = [t for t in texts if t]
                full = sum(spans) >= (width - 1 if relax else width)
                if len(nonempty) == 1 and full:
                    gold.add(mapped)
            detected = set(table.section_rows)
            gt_total += len(gold)
            det_total += len(detected)
            for mapped in sorted(detected - gold):
                fp += 1
                fp_cases.append("%s | r%d | %s" % (
                    table.table_id, mapped,
                    " | ".join(_clean_cell(c)[:24] for c in table.rows[mapped][:4])))
            for mapped in sorted(gold - detected):
                fn += 1
                fn_cases.append("%s | r%d | %s" % (
                    table.table_id, mapped,
                    " | ".join(_clean_cell(c)[:24] for c in table.rows[mapped][:4])))
            tp += len(gold & detected)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "gt_definition": "近满宽(sum(colspan) >= 列宽-1)" if relax else "满宽(sum(colspan) == 列宽)",
        "tables": tables,
        "skipped": skipped,
        "gt_rows": gt_total,
        "detected": det_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "fp_cases": fp_cases,
        "fn_cases": fn_cases,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="section_rows 检测 vs 源 HTML 事实（全库客观校验）")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass

    report = {
        "corpus": str(args.corpus),
        "note": "GT 由源 HTML 属性推导（colspan 合计 + 非空格数），与解析器的启发式规则相互独立",
        "gt_strict": evaluate(args.corpus, relax=False),
        "gt_v2_near_full": evaluate(args.corpus, relax=True),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("gt_strict", "gt_v2_near_full"):
        item = report[key]
        print("%-16s tables=%d gt=%d TP=%d FP=%d FN=%d P=%.4f R=%.4f F1=%.4f" % (
            key, item["tables"], item["gt_rows"], item["tp"], item["fp"], item["fn"],
            item["precision"], item["recall"], item["f1"]))
    print("out: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
