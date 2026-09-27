"""
tools/data_scripts/export_citations.py
从评测结果 Excel 抽取全部引用并导出 JSON，供 --validate-refs 复跑 L1 引用核验使用。

用法示例：
    python tools/data_scripts/export_citations.py \
        --input 训练结果数据/result_3_parallel.xlsx \
        --output 训练结果数据/references_all.json
    python rag_全流程构建.py --validate-refs 训练结果数据/references_all.json
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _iter_answer_objects(structured_output: str) -> List[Dict[str, Any]]:
    """解析结构化输出字段为问答对象列表。

    Args:
        structured_output: 结构化输出字段原文（JSON 字符串）

    Returns:
        问答对象列表（每项含 Q / A）；解析失败返回空列表
    """
    if not structured_output or not structured_output.strip():
        return []
    try:
        parsed = json.loads(structured_output)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed if isinstance(parsed, list) else []


def _normalize_refs(raw: Any) -> List[Dict[str, str]]:
    """把引用字段归一为 {paper_path, text, paper_image} 列表。

    Args:
        raw: 引用字段原始值

    Returns:
        归一化后的引用列表
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for ref in raw:
        if not isinstance(ref, dict):
            continue
        out.append({
            "paper_path": str(ref.get("paper_path", "") or ""),
            "text": str(ref.get("text", "") or ""),
            "paper_image": str(ref.get("paper_image", "") or ""),
        })
    return out


def extract_references(structured_output: str) -> List[Dict[str, str]]:
    """从单行结构化输出中抽取全部引用。

    支持两种引用位置：
    - A.references 直接为引用列表（当前数据的主要形式）；
    - A.content 为 JSON 字符串且内含 references（兼容嵌套形式）。

    Args:
        structured_output: 结构化输出字段原文

    Returns:
        引用列表，元素为 {paper_path, text, paper_image}
    """
    refs: List[Dict[str, str]] = []
    for item in _iter_answer_objects(structured_output):
        if not isinstance(item, dict):
            continue
        a = item.get("A")
        if not isinstance(a, dict):
            continue
        refs.extend(_normalize_refs(a.get("references")))
        content = a.get("content")
        if isinstance(content, str) and content.strip().startswith(("{", "[")):
            try:
                inner = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(inner, dict):
                refs.extend(_normalize_refs(inner.get("references")))
            elif isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict):
                        refs.extend(_normalize_refs(sub.get("references")))
    return refs


def export_references_from_excel(xlsx_path: str, out_path: str) -> Dict[str, int]:
    """从评测结果 Excel 抽取全部引用并导出 JSON。

    Args:
        xlsx_path: 评测结果 Excel 路径（含 编号 / 问题类型 / 结构化输出 列）
        out_path: 输出引用 JSON 路径（列表，元素含 bid / qtype / paper_path / text / paper_image）

    Returns:
        汇总信息：{questions, question_with_refs, refs}
    """
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    bid_col, qtype_col, so_col = df.columns[0], df.columns[1], df.columns[4]
    rows_out: List[Dict[str, Any]] = []
    refs_total = 0
    question_with_refs = 0
    for i in range(len(df)):
        bid = str(df.iloc[i][bid_col])
        qtype = str(df.iloc[i][qtype_col])
        structured_output = str(df.iloc[i][so_col])
        refs = extract_references(structured_output)
        if refs:
            question_with_refs += 1
        refs_total += len(refs)
        for ref in refs:
            rows_out.append({"bid": bid, "qtype": qtype, **ref})
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows_out, fh, ensure_ascii=False, indent=2)
    return {
        "questions": len(df),
        "question_with_refs": question_with_refs,
        "refs": refs_total,
    }


def main() -> None:
    """命令行入口：--input 评测 Excel，--output 引用 JSON。"""
    parser = argparse.ArgumentParser(description="从评测结果 Excel 抽取引用并导出 JSON")
    parser.add_argument("--input", type=str, required=True, help="评测结果 Excel 路径")
    parser.add_argument("--output", type=str, required=True, help="输出引用 JSON 路径")
    args = parser.parse_args()

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    summary = export_references_from_excel(args.input, args.output)
    print(f"题目数：{summary['questions']}，带引用题目：{summary['question_with_refs']}，引用总数：{summary['refs']}")
    print(f"引用 JSON 已写入：{args.output}")


if __name__ == "__main__":
    main()
