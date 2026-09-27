"""答案先验登记与「误拒答（应有数据）」判定（B-36）。

背景（B-25A 实测）：B2053 同题 n=5 中 4 次误拒答
（「抱歉，未查询到「…」的相关数据。…（SQL 已执行成功，但未返回任何数据）」），
而 `eval/llm_judge.py` 的四判据与规则信号**均判 pass** —— 因为两者都拿不到
「该题本应有数据」的先验信息。

本模块把先验做成**可复算的登记表**，而不是抄一份「正确答案名单」：

- 每题登记三态 `expect`：`has_data` / `no_data` / `unknown`；
- `has_data` 的题必须给出**人工审核过的标准 SQL**，评测时只读执行该 SQL，
  行数 ≥1 先验才算成立 —— 先验自校验：标准 SQL 复算 0 行 → 登记冲突，
  降级为 unknown 并提示复核（避免错误登记把正确回答判 fail）；
- `no_data` 覆盖两类情形：库内确实无对应数据（如开放性问题），以及**拒答本身合理**
  （如 B2036 触发 B-30 指标一致性守卫）—— 二者都不应被判为误拒答；
- 判定是**确定性规则**，不调用 LLM：先验成立 + 本次回答未给数据 → `误拒答（应有数据）`；
- 本模块同时持有 judge / 一致性套件共用的 `REFUSE_MARKERS` 与 `key_numbers`，
  由 `eval/llm_judge.py` 导入后再对外暴露，避免两份词表漂移。

口径：本模块产出的是**信号**（与 judge 判定并列进入分歧清单），不单独作为结论；
误拒答是否成立最终由人工核对（B-36 验收 / B-25B）。
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

DEFAULT_ANSWER_KEYS = _REPO_ROOT / "database" / "answer_keys" / "v1.json"

EXPECT_HAS_DATA = "has_data"
EXPECT_NO_DATA = "no_data"
EXPECT_UNKNOWN = "unknown"
EXPECT_VALUES = (EXPECT_HAS_DATA, EXPECT_NO_DATA, EXPECT_UNKNOWN)

STATUS_HAS_DATA = "has_data"
STATUS_NO_DATA = "no_data"
STATUS_UNKNOWN = "unknown"
STATUS_CONFLICT = "prior_conflict"
STATUS_ERROR = "prior_error"

MISREFUSAL_TYPE = "误拒答（应有数据）"

# 拒答 / 澄清话术（judge 与一致性套件共用；B-25A 起逐步补齐）
REFUSE_MARKERS = (
    "未包含", "未披露", "未显示", "不包含", "没有该字段", "无法提供", "无法回答",
    "请补充", "请明确", "不在本次查询范围", "查询结果中不包含", "未找到",
    # B-25A 补：误拒答/空结果话术（B2053 连续 4 次命中，原文见 consistency_20260910/）
    "未查询到", "尚未收录", "暂未收录", "换个已覆盖范围", "未返回任何数据",
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def key_numbers(text: str) -> set:
    """抽取「需核对的关键数值」：过滤年份与单个数字（季度/序号等噪声）。

    Args:
        text: 答案文本

    Returns:
        关键数值集合（保留小数）
    """
    collapsed = re.sub(r"(?<=\d)\s+(?=[\d.])", "", text or "")
    out = set()
    for n in _NUMBER_RE.findall(collapsed):
        if len(n) == 4 and n.isdigit() and 1900 <= int(n) <= 2100:
            continue  # 年份：核对价值低且引用片段常不写年份
        if n.isdigit() and len(n) <= 1:
            continue  # 单个数字：多为 top10 / Q3 / 第3条 之类噪声
        out.add(n)
    return out


def looks_like_refusal(answer: str) -> bool:
    """答案是否命中拒答/澄清话术。"""
    body = answer or ""
    return any(m in body for m in REFUSE_MARKERS)


def answer_gives_data(answer: str, question: str = "") -> bool:
    """答案是否给出了可核对的数据（含关键数值）。

    两条口径：

    - **扣除问题自带数值**：拒答模板会把原问题复述一遍
      （如「未查询到「…比值在 0.8 到 1.2 之间的公司…」的相关数据」），
      若不扣除，问题里的 0.8 / 1.2 会被误当成「已给数据」→ 假阳性；
    - **不以拒答词表直接判死**：「无法确认…为正，实际净流出 2298.86 万元」
      这类回答含「无法」字样但确实给了数据（B2069），故只按数值判定。

    Args:
        answer: 待判定回答
        question: 该题问题文本（用于扣除问题自带数值），可空

    Returns:
        True 表示答案给出了问题之外的关键数值
    """
    return bool(key_numbers(answer) - key_numbers(question))


@dataclass(frozen=True)
class AnswerKey:
    """单题先验登记项。

    Attributes:
        code: 题目编号（如 B2053）
        expect: 三态之一（has_data / no_data / unknown）
        standard_sql: 标准 SQL（expect=has_data 时必填，须人工审核）
        note: 备注（题目语义 / 登记理由）
        review: 审核状态（approved = 已生效）
        review_evidence: 审核证据（如「只读复算 14 行」）
        waiver_reason: no_data 的豁免理由（区分「库内无数据」与「拒答合理」）
    """

    code: str
    expect: str = EXPECT_UNKNOWN
    standard_sql: str = ""
    note: str = ""
    review: str = "draft"
    review_evidence: str = ""
    waiver_reason: str = ""

    @property
    def enabled(self) -> bool:
        """是否已审核通过（draft 不产生先验，避免未审核登记直接参与判定）。"""
        return str(self.review).strip().lower() == "approved"


def load_answer_keys(path: Optional[Path] = None) -> Dict[str, AnswerKey]:
    """加载先验登记表（缺失 / 损坏时返回空表，不抛异常）。

    Args:
        path: 登记表路径，默认 ``database/answer_keys/v1.json``

    Returns:
        编号 -> AnswerKey（仅包含已审核通过的项）
    """
    target = Path(path) if path else DEFAULT_ANSWER_KEYS
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out: Dict[str, AnswerKey] = {}
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("编号") or "").strip()
        if not code:
            continue
        expect = str(item.get("expect") or EXPECT_UNKNOWN).strip()
        if expect not in EXPECT_VALUES:
            expect = EXPECT_UNKNOWN
        key = AnswerKey(
            code=code,
            expect=expect,
            standard_sql=str(item.get("标准SQL") or ""),
            note=str(item.get("备注") or ""),
            review=str(item.get("审核") or "draft"),
            review_evidence=str(item.get("审核证据") or ""),
            waiver_reason=str(item.get("豁免理由") or ""),
        )
        if key.enabled:
            out[code] = key
    return out


def execute_readonly_sql(sql: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """只读执行一段 SQL（judge/先验/预览共用）。

    安全口径与线上一致：仅允许 SELECT / WITH / 括号开头；15s 语句级超时；
    任何异常都退化为 (None, 说明) 而不抛出。

    Args:
        sql: SQL 文本（可含多条语句）

    Returns:
        (结果行列表 或 None, 错误说明)；错误为空表示执行成功
    """
    if not (sql or "").strip():
        return [], ""
    statements = [s.strip() for s in re.split(r";\s*", sql) if s.strip()]
    risky = [s for s in statements if not s.lower().lstrip().startswith(("select", "with", "("))]
    if risky:
        return None, "存在非 SELECT 语句，跳过执行（只读口径）"
    try:
        from config.rag_config import get_config
        from tools.native_financial import _execute_sql, _load_schema_conn

        schema, conn = _load_schema_conn(get_config())
        if conn is None:
            return None, "MySQL 不可用"
        try:  # 与线上一致：15s 语句级超时（MySQL 5.7+，只影响 SELECT）
            conn.cursor().execute("SET SESSION max_execution_time=15000")
        except Exception:  # noqa: BLE001
            pass
        return _execute_sql(conn, sql), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"


def resolve_prior(
    key: AnswerKey, executor: Optional[Callable[[str], Tuple[Optional[List[Dict[str, Any]]], str]]] = None
) -> Dict[str, Any]:
    """把登记项解析为本次判定可用的先验状态（has_data 会只读复算自校验）。

    Args:
        key: 登记项
        executor: 只读执行器（默认 :func:`execute_readonly_sql`，单测可注入 stub）

    Returns:
        含 status / rows / reason / rows_data 的字典；
        status ∈ {has_data, no_data, unknown, prior_conflict, prior_error}，
        其中 prior_conflict / prior_error 一律按 unknown 处理（不产生误拒答判定）。
    """
    run = executor or execute_readonly_sql
    base: Dict[str, Any] = {"code": key.code, "expect": key.expect, "standard_sql": key.standard_sql}
    if key.expect == EXPECT_NO_DATA:
        return {
            **base,
            "status": STATUS_NO_DATA,
            "rows": None,
            "rows_data": [],
            "reason": key.waiver_reason or "该题不应有数据 / 拒答属合理",
        }
    if key.expect == EXPECT_UNKNOWN:
        return {**base, "status": STATUS_UNKNOWN, "rows": None, "rows_data": [], "reason": "未登记先验"}
    if not key.standard_sql.strip():
        return {
            **base,
            "status": STATUS_CONFLICT,
            "rows": None,
            "rows_data": [],
            "reason": "登记为 has_data 但缺标准 SQL，先验不启用（请补人工审核过的标准 SQL）",
        }
    rows, err = run(key.standard_sql)
    if rows is None:
        return {
            **base,
            "status": STATUS_ERROR,
            "rows": None,
            "rows_data": [],
            "reason": f"标准 SQL 执行失败：{err[:80]}",
        }
    if len(rows) < 1:
        return {
            **base,
            "status": STATUS_CONFLICT,
            "rows": 0,
            "rows_data": [],
            "reason": "标准 SQL 复算 0 行 → 先验不成立，请复核登记项（可能是标准 SQL 口径错）",
        }
    return {
        **base,
        "status": STATUS_HAS_DATA,
        "rows": len(rows),
        "rows_data": rows,
        "reason": f"标准 SQL 只读复算 {len(rows)} 行",
    }


def judge_misrefusal(prior_status: str, answer: str, question: str = "") -> Tuple[str, str]:
    """判定「应有数据却拒答」（确定性规则，不调用 LLM）。

    Args:
        prior_status: :func:`resolve_prior` 的 status
        answer: 待判定回答
        question: 该题问题文本（用于扣除问题自带数值），可空

    Returns:
        (分歧类型, 理由)；不成立时返回 ("", "")
    """
    if prior_status != STATUS_HAS_DATA:
        return "", ""
    if answer_gives_data(answer, question):
        return "", ""
    why = "命中拒答/澄清话术且未给出数值" if looks_like_refusal(answer) else "未给出可核对的关键数值"
    return MISREFUSAL_TYPE, f"先验显示该题应有数据，但本次回答{why}"


def prior_number_hit_rate(
    answer: str, rows: Optional[List[Dict[str, Any]]], question: str = ""
) -> Optional[float]:
    """答案关键数值在「标准 SQL 复算结果」中的命中率（C 附加提示，不参与判定）。

    Args:
        answer: 待判定回答
        rows: 标准 SQL 复算结果行
        question: 该题问题文本（扣除问题自带数值），可空

    Returns:
        命中率 0~1；无法计算（无答案数值 / 无结果行）时返回 None
    """
    numbers = key_numbers(answer) - key_numbers(question)
    if not numbers or not rows:
        return None
    blob = "|".join(str(v) for row in rows for v in row.values())
    hit = sum(1 for n in numbers if n in blob)
    return round(hit / len(numbers), 4)


PRIOR_COLUMNS = ("先验", "先验依据", "本次SQL行数", "误拒答判定", "先验数值命中率")


def attach_priors(
    rows: List[Dict[str, Any]],
    keys: Optional[Dict[str, AnswerKey]] = None,
    executor: Optional[Callable[[str], Tuple[Optional[List[Dict[str, Any]]], str]]] = None,
) -> List[Dict[str, Any]]:
    """给判定明细行附加先验列（编号级先验只复算一次）。

    Args:
        rows: judge 明细行（需含「编号」「答案」）
        keys: 先验登记表（默认空表 = 全部 unknown）
        executor: 只读执行器（单测可注入 stub）

    Returns:
        新行列表（不修改入参），额外含 :data:`PRIOR_COLUMNS` 五列
    """
    keys = keys or {}
    resolved_cache: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        code = str(row.get("编号") or "")
        if code not in resolved_cache:
            key = keys.get(code)
            resolved_cache[code] = (
                resolve_prior(key, executor)
                if key is not None
                else {"status": STATUS_UNKNOWN, "rows": None, "rows_data": [], "reason": "未登记先验"}
            )
        resolved = resolved_cache[code]
        answer = str(row.get("答案") or "")
        question = str(row.get("子问题") or "")
        flag, reason = judge_misrefusal(resolved["status"], answer, question)
        rate = None
        if resolved["status"] == STATUS_HAS_DATA:
            rate = prior_number_hit_rate(answer, resolved.get("rows_data"), question)
        sql_rows = row.get("本次SQL行数")
        out.append(
            {
                **row,
                "先验": resolved["status"],
                "先验依据": resolved["reason"],
                "本次SQL行数": sql_rows if sql_rows is not None else "—",
                "误拒答判定": flag,
                "先验数值命中率": "—" if rate is None else rate,
                "先验理由": reason,
            }
        )
    return out