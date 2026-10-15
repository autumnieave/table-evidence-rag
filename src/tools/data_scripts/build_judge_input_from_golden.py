# -*- coding: utf-8 -*-
"""build_judge_input_from_golden.py — 从 golden 答案池生成 judge 输入契约（L3 校准 S1，2026-10-14）

把 `训练结果数据/golden_answer_regression.json` 的 items（108 条答案 / 80 个编号）
摊平成 `eval/llm_judge.py` 的输入契约（与 load_consistency_runs 对齐）：

    {"questions": [{"编号", "问题类型", "子问题", "runs": [{"答案", "引用", "SQL"}]}]}

- 一条答案一个 question + 单 run（不按编号合并，避免丢子问题文本）；
- 引用原样透传（paper_path / text / paper_image / citation）；
- SQL 置空串（本轮 judge 校准走 --no-sql-result 主口径，见方案 §4.1）。

用法：
  python src/tools/data_scripts/build_judge_input_from_golden.py
  python src/tools/data_scripts/build_judge_input_from_golden.py --out 训练结果数据/judge_input_108_20261014.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IN = PROJECT_ROOT / "训练结果数据" / "golden_answer_regression.json"


def _today() -> str:
    return _dt.date.today().strftime("%Y%m%d")


def build(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把答案池 items 摊平为 judge 输入契约（一条答案一个 question + 单 run）。"""
    items = payload.get("items") or []
    questions: List[Dict[str, Any]] = []
    for item in items:
        questions.append(
            {
                "编号": str(item.get("编号") or ""),
                "问题类型": item.get("类型"),
                "子问题": str(item.get("问题") or ""),
                "runs": [
                    {
                        "答案": str(item.get("答案") or ""),
                        "引用": list(item.get("引用") or []),
                        "SQL": "",
                    }
                ],
            }
        )
    return {
        "source": "训练结果数据/golden_answer_regression.json",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "questions": questions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从 golden 答案池生成 judge 输入契约（L3 校准 S1）")
    parser.add_argument("--in", dest="inp", default=str(DEFAULT_IN), help="答案池路径（默认 训练结果数据/golden_answer_regression.json）")
    parser.add_argument("--out", default="", help="输出路径（默认 训练结果数据/judge_input_108_<日期>.json）")
    args = parser.parse_args()

    inp = Path(args.inp)
    out = Path(args.out) if args.out else PROJECT_ROOT / "训练结果数据" / f"judge_input_108_{_today()}.json"

    payload = json.loads(inp.read_text(encoding="utf-8"))
    built = build(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(built, ensure_ascii=False, indent=2), encoding="utf-8")

    qs = built["questions"]
    ids = {q["编号"] for q in qs}
    ref_total = sum(len(q["runs"][0]["引用"]) for q in qs)
    print(f"items in : {len(payload.get('items') or [])}")
    print(f"questions: {len(qs)} ({len(ids)} ids)")
    print(f"refs     : {ref_total}")
    print(f"out      : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())