"""LLM-as-judge 自动评估（B-25A）——AI 应用特有的「答案质量自动判定」最小闭环

传统单测只能校验确定性输出；答案正确性过去依赖人工回查（194 个数字人工核对）。
本模块把「数值是否可溯源 / 观点是否有据 / 配对是否错位 / 拒答是否合理」四条判据
写成结构化 prompt，让 judge 模型给出 pass/fail/na 与理由，用于：

1. 对一致性套件（`eval/consistency.py`）的多次生成结果做初判；
2. 与规则信号（数值可溯源率这类确定性指标）交叉，产出「分歧样本清单」，
   人工（B-25B）只需核对分歧样本，不必全量回查。

四判据（与方案 §6.7.3 一致）：
  ① 数值可溯源：答案中关键数值能在引用片段或该题 SQL 结果中找到，且单位/口径一致
  ② 观点有据：结论基于引用内容，而非模型自身记忆或外部常识（不得外推）
  ③ 配对不错位：公司/行业与数值、结论的对应关系正确（不得张冠李戴）
  ④ 拒答合理：库外指标 / 越界年份 / 缺字段时明确拒答或澄清追问 → 不计失败

用法：
  python -m eval llm-judge --input 训练结果数据/consistency_20260910/consistency_runs.json --max-runs 1
  python -m eval llm-judge --input ... --no-sql-result     # 不执行 SQL（无 MySQL 时）

口径声明：judge **未校准**，其判定不得作为对外结论；必须与人工结果对齐（一致率 ≥90%，
见 B-25B）后才可用于回归门禁。

B-36 增补：支持 `--answer-key` 加载**答案先验登记表**（`eval/answer_keys.py`），
用确定性规则识别「应有数据却拒答」。先验来自登记表里**人工审核过的标准 SQL** 的只读复算，
不抄 golden 名单；未登记 / 登记冲突一律按 `unknown` 处理、不产生误拒答判定。
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from eval.answer_keys import (  # noqa: E402  # B-36：先验登记 + 共用词表/只读执行器
    DEFAULT_ANSWER_KEYS,
    REFUSE_MARKERS,  # noqa: F401  # 由本模块再对外暴露，保持旧导入路径可用
    attach_priors,
    execute_readonly_sql,
    key_numbers,
    load_answer_keys,
)

DEFAULT_MODEL = "qwen-flash"
DEFAULT_OUT_DIR = _REPO_ROOT / "训练结果数据" / "llm_judge_20260910"
CHALLENGE_REVIEW = _REPO_ROOT / "训练结果数据" / "challenge_v2_review.json"

MAX_ANSWER_CHARS = 3000
MAX_REF_CHARS = 700
MAX_SQL_CHARS = 2500
MAX_SQL_PREVIEW_CHARS = 2000
MAX_SQL_ROWS = 30

# 拒答/澄清话术与关键数值抽取已下沉到 `eval/answer_keys.py`（judge / consistency / 先验校验
# 共用同一份，避免多份词表漂移）；本模块在文件头 import 后再对外暴露，旧导入路径保持可用。

CRITERIA: List[Dict[str, str]] = [
    {
        "id": "判据1",
        "name": "数值可溯源",
        "desc": "答案中出现的每一个关键数值，都能在该题引用片段或下方 SQL 结果中找到，且单位/口径一致（如‘亿元’对‘亿元’）。",
    },
    {
        "id": "判据2",
        "name": "观点有据",
        "desc": "结论与观点来自引用片段或 SQL 结果的归纳，而非模型自身记忆、外部常识或凭空的行业判断。",
    },
    {
        "id": "判据3",
        "name": "配对不错位",
        "desc": "公司名称/股票代码与数值、结论一一对应正确，不存在张冠李戴或跨公司串行。",
    },
    {
        "id": "判据4",
        "name": "拒答合理",
        "desc": "库外指标、越界年份、缺字段时明确拒答或澄清追问，视为通过（不计失败）。",
    },
]

JUDGE_PROMPT = """你是金融问答系统的质量审核员。请依据四判据审核下面这条回答，严格按 JSON 输出。

【问题】
{question}

【待审核回答】
{answer}

【该题引用片段（最多 10 条）】
{references}

【该题执行的 SQL 及结果（可能为空）】
{sql}

【四判据】
{criterias}

【判定要求】
- 每条判据取以下三值之一：`pass`（通过）/ `fail`（不通过）/ `na`（本题不适用）。
- 判据4：若回答属于"合理拒答或澄清追问"，则判据1/2/3 记 `na`，判据4 记 `pass`。
- 关键数值缺失、检索不到但回答仍给出具体数字，属编造 → 判据1 `fail`。
- **单位换算口径**：同一数值经单位换算后一致即视为可溯源（如 3,285,671.65 元 ≈ 328.57 万元 ≈ 0.033 亿元），不得因单位写法不同判 `fail`。
- **SQL 结果可能只展示部分行**（预览首行会标注总行数）：答案中的数值若不在展示范围内，**不得判定为编造**，应记 `na` 并在理由中说明"证据不足，需完整结果核对"。
- 派生指标（同比、均值、占比、排名）只要其计算口径与所给数据一致，即可判 `pass`。
- 只依据上面给出的引用与 SQL 结果判断；信息不足时判 `na`，不要猜测。

【输出格式（只输出 JSON，不要多余文字）】
{{"判据1": "pass|fail|na", "判据2": "pass|fail|na", "判据3": "pass|fail|na", "判据4": "pass|fail|na", "总判定": "pass|fail", "理由": "一句话说明关键依据，并指出失败项"}}
"""


# ---------------------------------------------------------------- prompt 组装 / 解析（可离线单测）


def build_judge_prompt(
    question: str,
    answer: str,
    references: Sequence[Dict[str, Any]] = (),
    sql_text: str = "",
    sql_preview: str = "",
    max_refs: int = 10,
) -> str:
    """组装 judge prompt。

    Args:
        question: 用户问题
        answer: 待审核回答
        references: 引用列表（paper_path / text）
        sql_text: 该题执行的 SQL 原文
        sql_preview: SQL 结果预览（前若干行；为空表示未提供结果）
        max_refs: 最多纳入的引用条数

    Returns:
        完整 prompt 字符串
    """
    ref_blocks: List[str] = []
    for i, ref in enumerate(list(references or [])[:max_refs], 1):
        path = str(ref.get("paper_path") or "")
        text = str(ref.get("text") or "")[:MAX_REF_CHARS]
        ref_blocks.append(f"[{i}] 文件：{path}\n片段：{text}")
    refs_text = "\n\n".join(ref_blocks) if ref_blocks else "（本题无研报引用）"

    if sql_text.strip():
        sql_body = f"SQL：\n{sql_text[:MAX_SQL_CHARS]}"
        if sql_preview.strip():
            sql_body += f"\n\n查询结果（前 {MAX_SQL_ROWS} 行）：\n{sql_preview[:MAX_SQL_PREVIEW_CHARS]}"
        else:
            sql_body += "\n\n（查询结果预览未提供：仅依据引用片段判断数值可溯源性，证据不足时判 na）"
    else:
        sql_body = "（本题未执行 SQL）"

    criterias = "\n".join(f"- {c['id']} {c['name']}：{c['desc']}" for c in CRITERIA)
    return JUDGE_PROMPT.format(
        question=(question or "").strip(),
        answer=(answer or "")[:MAX_ANSWER_CHARS],
        references=refs_text,
        sql=sql_body,
        criterias=criterias,
    )


_TRISTATE = {
    "pass": "pass", "passes": "pass", "通过": "pass", "正确": "pass", "yes": "pass", "true": "pass",
    "fail": "fail", "failed": "fail", "不通过": "fail", "失败": "fail", "no": "fail", "false": "fail",
    "na": "na", "n/a": "na", "不适用": "na", "无法判定": "na", "unknown": "na", "": "na",
    "pending": "na", "待定": "na",
}


def normalize_tristate(value: Any) -> str:
    """把模型输出的判定值规范化为 pass / fail / na。"""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    text = str(value or "").strip().lower()
    return _TRISTATE.get(text, "na")


def parse_judge_response(raw: str) -> Dict[str, Any]:
    """稳健解析 judge 输出（容忍代码块与多余文字）。

    Args:
        raw: 模型原始输出

    Returns:
        {"判据1".."判据4", "总判定", "理由"}；解析失败时 总判定 = "judge_error"
    """
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    data: Optional[Dict[str, Any]] = None
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = None
    if data is None:
        return {
            "判据1": "na", "判据2": "na", "判据3": "na", "判据4": "na",
            "总判定": "judge_error", "理由": "judge 输出无法解析为 JSON",
        }

    out: Dict[str, Any] = {c["id"]: normalize_tristate(data.get(c["id"])) for c in CRITERIA}
    verdict = normalize_tristate(data.get("总判定"))
    if verdict == "na":  # 模型未给总判定 → 由分项推导
        if any(out[c["id"]] == "fail" for c in CRITERIA):
            verdict = "fail"
        elif all(out[c["id"]] in ("pass", "na") for c in CRITERIA):
            verdict = "pass"
        else:
            verdict = "na"
    out["总判定"] = verdict
    out["理由"] = str(data.get("理由") or "").strip()[:500]
    return out


def rule_signal(answer: str, references: Sequence[Dict[str, Any]], sql_text: str) -> str:
    """规则信号（确定性口径，与 judge 交叉用于找分歧）。

    规则：
      - 空答案 → fail
      - 含拒答措辞 → pass（合理拒答，判据4 口径）
      - 无关键数值 → uncertain
      - 关键数值全部能在引用片段中找到 → pass
      - 命中率 ≥50% → uncertain；<50% 时：该题有 SQL → uncertain（需查 SQL 结果，规则不判），无 SQL → fail
    """
    body = answer or ""
    if not body.strip():
        return "fail"
    if any(m in body for m in REFUSE_MARKERS):
        return "pass"
    numbers = key_numbers(body)
    if not numbers:
        return "uncertain"
    ref_text = "".join(
        re.sub(r"\s+", "", str(r.get("text") or "")).replace(",", "").replace("，", "")
        for r in references or []
    )
    hit = sum(1 for n in numbers if n in ref_text)
    ratio = hit / len(numbers)
    if ratio == 1.0:
        return "pass"
    if ratio >= 0.5:
        return "uncertain"
    if sql_text.strip():
        return "uncertain"
    return "fail"


def find_disagreements(
    rows: Sequence[Dict[str, Any]], review_map: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Dict[str, str]]:
    """找出 judge 与规则/历史人工结论不一致的样本。

    Args:
        rows: judge 结果行（编号/子问题/judge判定/规则信号）
        review_map: 历史人工复核结论（如 challenge v2 sidecar），可为 None

    Returns:
        分歧样本列表，含「judge 判定」「待人工核对」两列（后者留空待填）
    """
    review_map = review_map or {}
    out: List[Dict[str, str]] = []
    for row in rows:
        judge = str(row.get("judge判定") or "")
        rule = str(row.get("规则信号") or "")
        human = ""
        reviewed = review_map.get(str(row.get("编号")))
        if isinstance(reviewed, dict):
            human = str(reviewed.get("结论") or "")
        types: List[str] = []
        if judge == "judge_error":
            types.append("judge 解析失败")
        if judge == "pass" and rule == "fail":
            types.append("judge 通过但规则判失败（疑似漏判）")
        if judge == "fail" and rule == "pass":
            types.append("judge 失败但规则判通过（疑似误判）")
        if judge == "fail" and rule == "uncertain":
            types.append("judge 失败而规则不确定（证据不足）")
        if human == "通过" and judge == "fail":
            types.append("与历史人工结论不一致（人工通过 / judge 失败）")
        if human == "不通过" and judge == "pass":
            types.append("与历史人工结论不一致（人工不通过 / judge 通过）")
        # B-36：先验判定的「误拒答（应有数据）」单列一类，与"合理拒答"区分
        prior_flag = str(row.get("误拒答判定") or "")
        if prior_flag:
            types.append(prior_flag)
        if not types:
            continue
        out.append(
            {
                "编号": str(row.get("编号") or ""),
                "子问题": str(row.get("子问题") or "")[:120],
                "第几次": str(row.get("第几次") or ""),
                "judge判定": judge,
                "规则信号": rule,
                "历史人工结论": human or "—",
                "分歧类型": "；".join(types),
                "judge理由": str(row.get("理由") or "")[:200],
                "待人工核对": "",
                "先验": str(row.get("先验") or "unknown"),
                "先验依据": str(row.get("先验依据") or "")
                + (f"；{row.get('先验理由')}" if row.get("先验理由") else ""),
                "本次SQL行数": str(row.get("本次SQL行数") if row.get("本次SQL行数") is not None else "—"),
                "误拒答判定": prior_flag,
                "先验数值命中率": str(
                    row.get("先验数值命中率") if row.get("先验数值命中率") is not None else "—"
                ),
            }
        )
    return out


DISAGREEMENT_COLUMNS = [
    "编号", "子问题", "第几次", "judge判定", "规则信号", "历史人工结论",
    "分歧类型", "judge理由", "待人工核对",
    # B-36：先验列（确定性规则，与 judge 判定并列进入人工核对）
    "先验", "先验依据", "本次SQL行数", "误拒答判定", "先验数值命中率",
]


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    """写 CSV（utf-8-sig，Excel/WPS 直接打开不乱码）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------- 运行（需 LLM / 可选 MySQL）


def format_sql_preview(
    rows: Optional[List[Dict[str, Any]]], err: str = "", max_rows: int = MAX_SQL_ROWS
) -> str:
    """把只读查询结果渲染成 judge 可读的预览文本。

    Args:
        rows: 结果行（None 表示执行失败/被拦截）
        err: 错误说明（失败时给出）
        max_rows: 最多展示行数

    Returns:
        预览文本；执行失败时返回空串
    """
    if rows is None:
        return ""
    if not rows:
        return "（查询返回 0 行）"
    shown = min(max_rows, len(rows))
    lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in rows[:max_rows]]
    header = f"（结果集共 {len(rows)} 行，仅展示前 {shown} 行；未展示行的数值不得判为编造）"
    return header + "\n" + "\n".join(lines)


def collect_sql_preview(sql_text: str, max_rows: int = MAX_SQL_ROWS) -> Tuple[str, str]:
    """执行该题 SQL 并返回结果预览（只读 SELECT；失败返回 ("", 错误说明)）。

    Returns:
        (预览文本, 错误说明)；错误非空时预览为空
    """
    if not (sql_text or "").strip():
        return "", ""
    rows, err = execute_readonly_sql(sql_text)
    if rows is None:
        return "", err
    return format_sql_preview(rows, err, max_rows), ""


def load_consistency_runs(path: Path) -> List[Dict[str, Any]]:
    """把一致性套件产物摊平成（题 × 次）记录列表。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for question in payload.get("questions") or []:
        for idx, run in enumerate(question.get("runs") or [], 1):
            out.append(
                {
                    "编号": str(question.get("编号")),
                    "问题类型": question.get("问题类型"),
                    "子问题": str(question.get("子问题") or ""),
                    "第几次": idx,
                    "答案": str(run.get("答案") or ""),
                    "引用": list(run.get("引用") or []),
                    "SQL": str(run.get("SQL") or ""),
                }
            )
    return out


def judge_rows(
    rows: Sequence[Dict[str, Any]],
    llm: Any,
    with_sql_result: bool = True,
    sleep_seconds: float = 0.4,
    answer_keys: Optional[Dict[str, Any]] = None,
    prior_executor: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """逐条判定（judge 失败不阻断整轮，标 judge_error）。

    Args:
        rows: 一致性套件明细行
        llm: judge 模型（None = 不调用模型，只算规则信号）
        with_sql_result: 是否只读执行该题 SQL 取结果预览与行数
        sleep_seconds: 每条之间的限速间隔
        answer_keys: 先验登记表（B-36）；为空时先验列一律 unknown
        prior_executor: 先验标准 SQL 的只读执行器（默认取真实库；单测可注入 stub）
    """
    out: List[Dict[str, Any]] = []
    sql_cache: Dict[str, Tuple[str, str]] = {}
    row_count_cache: Dict[str, Optional[int]] = {}
    for i, row in enumerate(rows, 1):
        sql_text = str(row.get("SQL") or "")
        preview, sql_err = ("", "")
        sql_row_count: Optional[int] = None
        if with_sql_result and sql_text:
            if sql_text not in sql_cache:
                # B-36：一次执行同时拿「预览」与「行数」，供 judge 与先验校验共用
                sql_rows, exec_err = execute_readonly_sql(sql_text)
                sql_cache[sql_text] = (format_sql_preview(sql_rows, exec_err), exec_err)
                row_count_cache[sql_text] = None if sql_rows is None else len(sql_rows)
            preview, sql_err = sql_cache[sql_text]
            sql_row_count = row_count_cache[sql_text]
        prompt = build_judge_prompt(
            row.get("子问题") or "", row.get("答案") or "", row.get("引用") or [], sql_text, preview
        )
        verdict: Dict[str, Any] = {}
        if llm is None:
            verdict = {**{c["id"]: "na" for c in CRITERIA}, "总判定": "judge_error",
                       "理由": "未提供 judge 模型（--no-judge）"}
        else:
            for attempt in (1, 2):
                try:
                    resp = llm.invoke(prompt)
                    raw = resp.content if hasattr(resp, "content") else str(resp)
                    verdict = parse_judge_response(raw)
                    if verdict["总判定"] != "judge_error":
                        break
                except Exception as exc:  # noqa: BLE001
                    verdict = {**{c["id"]: "na" for c in CRITERIA}, "总判定": "judge_error",
                               "理由": f"judge 调用失败: {type(exc).__name__}: {str(exc)[:120]}"}
                time.sleep(2)
        out.append(
            {
                "编号": row["编号"],
                "问题类型": row.get("问题类型"),
                "子问题": row.get("子问题"),
                "第几次": row.get("第几次"),
                "judge判定": verdict["总判定"],
                "判据1": verdict["判据1"],
                "判据2": verdict["判据2"],
                "判据3": verdict["判据3"],
                "判据4": verdict["判据4"],
                "理由": verdict["理由"],
                "规则信号": rule_signal(row.get("答案") or "", row.get("引用") or [], sql_text),
                "引用数": len(row.get("引用") or []),
                "SQL结果": "已执行" if preview else ("未执行" if not sql_text else f"未获取({sql_err})"),
                "本次SQL行数": sql_row_count,
                "答案字数": len(row.get("答案") or ""),
            }
        )
        print(
            f"[judge {i}/{len(rows)}] {row['编号']}#{row.get('第几次')} → {out[-1]['judge判定']} "
            f"(规则 {out[-1]['规则信号']})",
            file=sys.stderr,
            flush=True,
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)
    # B-36：统一附加先验列（未登记也显式标 unknown，便于核对覆盖度）。
    # 注意：输出行只保留「答案字数」以控制产物体积，而误拒答判定需要答案原文，
    # 故先临时挂上原文完成判定，再摘掉（否则 has_data 题会被全量误判为误拒答）。
    for row_out, row_src in zip(out, rows):
        row_out["答案"] = str(row_src.get("答案") or "")
    enriched = attach_priors(out, answer_keys or {}, prior_executor)
    for row_out in enriched:
        row_out.pop("答案", None)
    return enriched


def summarize(rows: Sequence[Dict[str, Any]], disagreements: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总 judge 初判结果（含未校准声明）。"""
    total = len(rows)
    verdicts: Dict[str, int] = {}
    for row in rows:
        verdicts[str(row["judge判定"])] = verdicts.get(str(row["judge判定"]), 0) + 1
    per_criterion: Dict[str, Dict[str, int]] = {}
    for key in ("判据1", "判据2", "判据3", "判据4"):
        counter: Dict[str, int] = {}
        for row in rows:
            counter[str(row.get(key))] = counter.get(str(row.get(key)), 0) + 1
        per_criterion[key] = counter
    rule_agree = sum(1 for r in rows if r["judge判定"] == r["规则信号"])
    comparable = sum(1 for r in rows if r["规则信号"] != "uncertain" and r["judge判定"] != "judge_error")
    prior_counter: Dict[str, int] = {}
    for row in rows:
        prior_key = str(row.get("先验") or "unknown")
        prior_counter[prior_key] = prior_counter.get(prior_key, 0) + 1
    misrefusal_count = sum(1 for r in rows if str(r.get("误拒答判定") or ""))
    return {
        "判定条数": total,
        "judge判定分布": verdicts,
        "各判据分布": per_criterion,
        "与规则信号可比的条数": comparable,
        "与规则信号一致的条数": rule_agree,
        "与规则信号一致率": round(rule_agree / comparable, 4) if comparable else None,
        "分歧样本数": len(disagreements),
        "先验分布": prior_counter,
        "误拒答（应有数据）条数": misrefusal_count,
        "先验口径声明": (
            "先验为确定性规则（不依赖 judge）：登记表 database/answer_keys/v1.json，"
            "has_data 项由人工审核过的标准 SQL 只读复算自校验；未登记 / 登记冲突一律 unknown，"
            "不产生误拒答判定。误拒答是否成立最终由人工核对（B-36 验收 / B-25B）。"
        ),
        "口径声明": "judge 未校准，不得作为对外结论；一致性阈值与启用边界由 B-25B 人工决策。",
    }


def render_md(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]],
              disagreements: Sequence[Dict[str, Any]]) -> str:
    """渲染 judge 初判报告（含分歧清单）。"""
    lines = [
        "# LLM-as-judge 初判结果（B-25A）",
        "",
        "> 口径：judge 模型初判，**未经人工校准**；只用于定位需要人工核对的样本，不得对外当结论。",
        "",
        "## 汇总",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 判定条数 | {summary['判定条数']} |",
        f"| judge 判定分布 | {json.dumps(summary['judge判定分布'], ensure_ascii=False)} |",
        f"| 与规则信号一致率 | {summary['与规则信号一致率']}（可比 {summary['与规则信号可比的条数']} 条） |",
        f"| 分歧样本数 | {summary['分歧样本数']} |",
        f"| 先验分布 | {json.dumps(summary.get('先验分布', {}), ensure_ascii=False)} |",
        f"| 误拒答（应有数据）条数 | {summary.get('误拒答（应有数据）条数', 0)} |",
        "",
        "## 各判据分布",
        "",
        "| 判据 | pass | fail | na |",
        "| --- | --- | --- | --- |",
    ]
    for key, counter in summary["各判据分布"].items():
        lines.append(
            f"| {key} | {counter.get('pass', 0)} | {counter.get('fail', 0)} | {counter.get('na', 0)} |"
        )
    lines += ["", "## 分歧样本清单（judge 判定 / 待人工核对）", ""]
    if not disagreements:
        lines.append("（无分歧样本）")
    else:
        lines.append(
            "| 编号 | 第几次 | judge判定 | 规则信号 | 先验 | 误拒答判定 | 分歧类型 | judge理由 | 待人工核对 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for d in disagreements:
            lines.append(
                f"| {d['编号']} | {d['第几次']} | {d['judge判定']} | {d['规则信号']} | "
                f"{d.get('先验', '—')} | {d.get('误拒答判定', '') or '—'} | {d['分歧类型']} | "
                f"{d['judge理由'].replace('|', '/')} | {d['待人工核对']} |"
            )
    lines += ["", "## 逐条判定明细", "",
              "| 编号 | 第几次 | judge判定 | 判据1 | 判据2 | 判据3 | 判据4 | 规则信号 | 引用数 | SQL结果 | 先验 | 本次SQL行数 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['编号']} | {row['第几次']} | {row['judge判定']} | {row['判据1']} | {row['判据2']} | "
            f"{row['判据3']} | {row['判据4']} | {row['规则信号']} | {row['引用数']} | {row['SQL结果']} | "
            f"{row.get('先验', 'unknown')} | {row.get('本次SQL行数') if row.get('本次SQL行数') is not None else '—'} |"
        )
    lines += ["", "## 先验与误拒答（B-36）", "",
              f"- 先验口径：{summary.get('先验口径声明', '')}",
              f"- 误拒答（应有数据）：**{summary.get('误拒答（应有数据）条数', 0)}** 条"
              "（确定性规则，先验成立 + 本次回答未给数据）；",
              "- 判读顺序：先看「误拒答判定」列 → 再核对「先验依据」（标准 SQL 复算行数），"
              "prior_conflict / prior_error 表示登记项本身待复核、**不代表答案有问题**；",
              "- 附加提示：「先验数值命中率」只作参考（答案含派生指标时天然偏低），不参与判定。", ""]
    lines += ["", "## 待人工处理（B-25B）", "",
              "1. 只核对上表「分歧样本清单」（目标 ≤20 条），逐条填「待人工核对」；",
              "2. 决定 judge 是否可用于门禁与一致率阈值（建议 ≥90%）；",
              "3. 校准后写明适用边界（哪些判据可自动、哪些仍需人工）。", ""]
    return "\n".join(lines)


def load_review_map(path: Path = CHALLENGE_REVIEW) -> Dict[str, Dict[str, Any]]:
    """读取历史人工复核 sidecar（编号 -> 结论），缺失时返回空表。"""
    if not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    reviews = payload.get("reviews")
    return reviews if isinstance(reviews, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="LLM-as-judge 自动评估（B-25A）")
    parser.add_argument("--input", required=True, help="一致性套件产物（consistency_runs.json）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="产物目录")
    parser.add_argument("--max-runs", type=int, default=0, help="每题最多判定几次（0=全部）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="judge 模型（默认 qwen-flash）")
    parser.add_argument("--no-judge", action="store_true", help="不调用 judge 模型（只算规则信号）")
    parser.add_argument("--no-sql-result", action="store_true", help="不执行 SQL 取结果预览")
    parser.add_argument("--review", default=str(CHALLENGE_REVIEW), help="历史人工复核 sidecar 路径")
    parser.add_argument(
        "--answer-key", default=str(DEFAULT_ANSWER_KEYS),
        help="答案先验登记表路径（B-36；默认 database/answer_keys/v1.json）",
    )
    parser.add_argument("--no-prior", action="store_true", help="不加载先验登记表（B-36，先验列全 unknown）")
    args = parser.parse_args(argv)

    rows = load_consistency_runs(Path(args.input))
    if args.max_runs > 0:
        rows = [r for r in rows if r["第几次"] <= args.max_runs]
    if not rows:
        print("无待判定记录", file=sys.stderr)
        return 1
    print(f"[judge] 待判定 {len(rows)} 条（{len({r['编号'] for r in rows})} 题）", file=sys.stderr, flush=True)

    llm = None
    if not args.no_judge:
        from langchain_openai import ChatOpenAI
        from config.rag_config import get_config

        cfg = get_config()
        llm = ChatOpenAI(
            model=args.model,
            api_key=cfg.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
        )
        print(f"[judge] 模型：{args.model}", file=sys.stderr, flush=True)

    keys = {} if args.no_prior else load_answer_keys(Path(args.answer_key))
    if keys:
        print(f"[prior] 先验登记 {len(keys)} 题：{'、'.join(sorted(keys))}", file=sys.stderr, flush=True)
    else:
        why = "--no-prior" if args.no_prior else f"未找到登记表 {args.answer_key}"
        print(f"[prior] 先验未启用（{why}），先验列全部为 unknown", file=sys.stderr, flush=True)

    judged = judge_rows(rows, llm, with_sql_result=not args.no_sql_result, answer_keys=keys)
    disagreements = find_disagreements(judged, load_review_map(Path(args.review)))
    summary = summarize(judged, disagreements)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_results.json").write_text(
        json.dumps({"汇总": summary, "明细": judged}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "分歧样本清单.csv", DISAGREEMENT_COLUMNS, disagreements)
    (out_dir / "judge_报告.md").write_text(render_md(summary, judged, disagreements), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n产物：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())