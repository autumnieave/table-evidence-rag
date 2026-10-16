# -*- coding: utf-8 -*-
"""锅炉板（免责声明 / 分析师声明 / 评级说明 …）语料测量。

**只读**扫描正式语料，回答三个问题，供 §3.2 决定「切 / 不切」：

1. 命中锅炉板标题的文档有多少？
2. 按 :func:`data.boilerplate.strip_trailing_boilerplate` 能切掉多少字符、占比多少？
3. 被**比例护栏拦下**（想切但没切）的文档有多少、都是谁？

用法（仓库根执行）::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe src/tools/data_scripts/measure_boilerplate.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import Any, Dict, List

from data.boilerplate import find_trailing_boilerplate_start, strip_trailing_boilerplate

#: 正式数据 附件5 语料根目录（与 `tools/data_scripts/rebuild_full_index.py` 同源）
CORPUS_ROOT = pathlib.Path("data_corpus/全部数据/正式数据/附件5：研报数据")
OUT = pathlib.Path("训练结果数据/table_struct_smoke/boilerplate_scan_official.json")


def iter_docs() -> List[pathlib.Path]:
    """列出正式语料的全部 Markdown。"""
    if not CORPUS_ROOT.is_dir():
        raise FileNotFoundError(f"正式数据语料不存在: {CORPUS_ROOT}")
    paths: List[pathlib.Path] = []
    for sub in ("个股研报-解析结果-2.0", "行业研报-解析结果-2.0"):
        root = CORPUS_ROOT / sub
        if root.is_dir():
            paths.extend(sorted(root.rglob("*.md")))
    return paths


def main() -> int:
    """扫描并打印/落盘统计。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)
    docs = iter_docs()
    records: List[Dict[str, Any]] = []
    total_chars = 0
    stripped_chars = 0
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="ignore")
        total_chars += len(text)
        start = find_trailing_boilerplate_start(text)
        new_text, removed = strip_trailing_boilerplate(text)
        stripped_chars += removed
        records.append({
            "file": path.name,
            "doc_chars": len(text),
            "has_tail": start is not None,
            "removed_chars": removed,
            "removed_ratio": round(removed / max(1, len(text)), 4),
            "kept_chars": len(new_text),
        })

    hit = [r for r in records if r["removed_chars"] > 0]
    print("文档数:", len(records))
    print("可切文档:", len(hit), "({:.1%})".format(len(hit) / max(1, len(records))))
    print("可切字符:", stripped_chars, "占全文 {:.2%}".format(stripped_chars / max(1, total_chars)))
    if hit:
        ratios = sorted(r["removed_ratio"] for r in hit)
        print("单文档可切占比 中位数 {:.1%} / 最大 {:.1%}".format(ratios[len(ratios) // 2], ratios[-1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "n_docs": len(records),
                "n_strippable": len(hit),
                "total_chars": total_chars,
                "strippable_chars": stripped_chars,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("out:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
