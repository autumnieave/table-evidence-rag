# -*- coding: utf-8 -*-
"""tools/native_financial.py —— 原生财务查询链路（路线 3 阶段 1）

替代 Dify 工作流的「SQL 生成 → MySQL 执行 → 分析生成」闭环，与 call_financial_chatflow 同接口
（返回 JSON 字符串 {"content", "image", "sql", "chart_json"}），供 AgentPlanner / LangGraphMultiAgentPlanner 共用。

链路：
1. sql_gen：LLM 生成 SQL（prompts/financial.py 的 SQL_GEN_SYSTEM_PROMPT，字段白名单 + 11 条规则）
2. 三层防线：静态校验 validate_sql + MySQL 编译 compile_check，失败带错误重试（AGENT_NATIVE_RETRY）
3. mysql_exec：pymysql 执行，结果转 list[dict]
4. analysis_gen：LLM 基于查询结果生成分析文本（ANALYSIS_SYSTEM_PROMPT，模式一/二）

图表：阶段 2 由 chart_gen 生成 ECharts JSON（chart_json），前端直接渲染交互图表，
不再依赖 selenium 截图；image 保持空列表（兼容旧前端）。
缓存：按 (user_id, question, FINANCIAL_PROMPT_VERSION, backend) 走 RAGPipeline.query_cache。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from prompts.financial import (
    ANALYSIS_SYSTEM_PROMPT,
    CHART_GEN_SYSTEM_PROMPT,
    FINANCIAL_PROMPT_VERSION,
    METRIC_STANDARDIZATION_SYSTEM_PROMPT,
    SQL_GEN_SYSTEM_PROMPT,
)
from utils.output_contracts import (
    get_stats as _contract_stats,
    validate_metric_plan as _contract_metric,
    validate_sql_output as _contract_sql,
)
from prompts.fallback import (
    build_human_risk_advice,
    build_refuse_data_not_found,
    build_refuse_metric_out_of_scope,
    build_refuse_system_unavailable,
    log_fallback as _log_fallback,
)

logger = logging.getLogger(__name__)

#: 分析文本中命中即视为「投资建议类高风险表述」的关键词（B-18 人工兜底）
_ADVICE_RISK_KEYWORDS: Tuple[str, ...] = (
    "建议买入", "建议卖出", "买入评级", "卖出评级", "强烈推荐",
    "推荐买入", "加仓", "减仓", "目标价",
)


#: SQL 生成失败错误中命中即视为「库内无对应字段/指标」的信号（B-31 → 走库外指标话术）
_FIELD_ERROR_MARKERS: Tuple[str, ...] = (
    "无可用字段", "字段不存在", "Unknown column", "白名单", "字段-表归属", "字段不属于",
)


class _GenStats:
    """SQL 生成环节的最近一次调用记录（B-39 对照实验用，只读观测、不影响主链路）。

    记录「尝试次数 / 首次是否通过 / few-shot 模式 / 注入示例数 / 是否成功」，
    便于实验侧统计「首次生成成功率」而无需额外调用 LLM（每问每题只跑 1 次）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: Dict[str, Any] = {}

    def record(
        self,
        *,
        attempts: int,
        first_ok: bool,
        mode: str,
        injected: int,
        ok: bool,
        reason: str = "",
        sql: str = "",
    ) -> None:
        """记录一次 _generate_sql 调用结果（sql=本次最后一次 LLM 输出，便于结构核验）。"""
        with self._lock:
            self._last = {
                "attempts": attempts,
                "first_ok": bool(first_ok),
                "mode": mode,
                "injected": int(injected),
                "ok": bool(ok),
                "reason": reason,
                "sql": (sql or "")[:8000],
            }

    def last(self) -> Dict[str, Any]:
        """返回最近一次调用记录（无记录时为空 dict）。"""
        with self._lock:
            return dict(self._last)


_gen_stats_instance = _GenStats()


def _gen_stats() -> _GenStats:
    """模块级 SQL 生成记录器（实验/回归侧读取）。"""
    return _gen_stats_instance


def _resolve_few_shot_mode(rag: Any) -> str:
    """解析当前 few-shot 模式（B-39）：AGENT_FEWSHOT_MODE 优先，留空回退 AGENT_DYNAMIC_FEWSHOT。"""
    from utils.few_shot_retriever import resolve_mode

    cfg = getattr(rag, "config", None)
    mode = getattr(cfg, "AGENT_FEWSHOT_MODE", "") if cfg is not None else ""
    enabled = bool(getattr(cfg, "AGENT_DYNAMIC_FEWSHOT", False)) if cfg is not None else False
    return resolve_mode(enabled=enabled, mode=mode or None)



def unsupported_metrics_of(metric_plan: Optional[Dict[str, Any]]) -> List[str]:
    """取指标标准化标注的「库外指标」列表（B-31）；无标注返回空列表。

    Args:
        metric_plan: _standardize_metrics 的输出（可为 None）

    Returns:
        用户索要但库内无对应字段的指标名列表
    """
    if not isinstance(metric_plan, dict):
        return []
    raw = metric_plan.get("unsupported_metrics")
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]


#: 已知库外指标词表（B-30 兜底）：问题里出现且模型未标注 unsupported_metrics 时，
#: 仍按库外口径拒答——防「模型把库外指标硬映射成近似字段」（如问每股公积金却选 net_asset_per_share）。
#: 词表为策划式清单（对应 prompts/financial.py 中的库外指标示例），已比对 golden 80 题 + 挑战集无其他命中。
_OUT_OF_SCOPE_QUESTION_TERMS: Tuple[str, ...] = (
    "股价", "总市值", "市值", "成交量", "成交额", "换手率", "市盈率", "市净率", "市销率", "股息率",
    "每股公积金", "公积金", "gmv", "门店数量", "门店数", "客单价", "复购率",
)


def unmarked_out_of_scope_terms(question: str, metric_plan: Optional[Dict[str, Any]]) -> List[str]:
    """问题问到已知库外指标、但指标标准化未标注 unsupported_metrics 的词（B-30 兜底）。

    已标注的库外指标由 unsupported_metrics 出口处理，这里只补「模型漏标/硬映射」的情况，
    避免同一问题被两条出口重复处置。

    Args:
        question: 重构后的财务问题文本
        metric_plan: _standardize_metrics 的输出（None 时不判，保持旧兜底行为）

    Returns:
        命中且未标注的库外指标词列表（最多 5 个）
    """
    if not isinstance(metric_plan, dict):
        return []
    marked_text = " ".join(m.lower() for m in unsupported_metrics_of(metric_plan))
    text = str(question or "").lower()
    hits: List[str] = []
    # 长词优先，短词若已被更长的命中词包含则跳过（「每股公积金」不再额外报「公积金」）
    for term in sorted(_OUT_OF_SCOPE_QUESTION_TERMS, key=len, reverse=True):
        low = term.lower()
        if low in text and low not in marked_text and not any(low in hit.lower() for hit in hits):
            hits.append(term)
    return hits[:5]


def _sql_failure_reply(errors: List[str]) -> Tuple[str, str]:
    """SQL 生成失败 → 用户可见话术（技术明细只进日志/事件通道，B-31）。

    字段/白名单类失败按「库内未收录该指标」回复（与 unsupported_metrics 同一出口），
    其余失败给不含技术堆栈的短因由。

    Args:
        errors: _generate_sql 返回的错误列表

    Returns:
        (content, template_id)
    """
    joined = "；".join(errors or [])
    if any(m in joined for m in _FIELD_ERROR_MARKERS):
        return build_refuse_metric_out_of_scope(None)
    if "全角标点" in joined or "契约" in joined or "格式" in joined:
        return build_refuse_system_unavailable("查询语句未通过库内格式校验（已自动重试）")
    return build_refuse_system_unavailable("查询语句未通过库内校验（已自动重试）")


#: 标签/维度列（非指标列）：不参与「指标-问题一致性」校验（B-30）
_TAG_COLUMNS = frozenset({"stock_code", "stock_abbr", "report_year", "report_period"})

#: 「指标-问题一致性」审计信号的标记（写入日志/事件通道，不参与拦截）
_METRIC_MISMATCH_MARKER = "指标-问题一致性校验未通过"


def _allowed_metric_fields(metric_plan: Dict[str, Any]) -> set:
    """指标计划允许出现在 SELECT 的指标列（B-30）。

    Args:
        metric_plan: 规整后的指标计划 JSON

    Returns:
        允许的字段名集合 = standard_fields + order_by 分子分母 + threshold 字段
    """
    allowed = {str(x).strip() for x in (metric_plan.get("standard_fields") or []) if str(x).strip()}
    calculation = metric_plan.get("calculation")
    if isinstance(calculation, dict):
        order_by = calculation.get("order_by")
        if isinstance(order_by, str):
            for part in re.split(r"[/+\-*()\s]+", order_by):
                if part.strip():
                    allowed.add(part.strip())
    filter_terms = metric_plan.get("filter_terms")
    if isinstance(filter_terms, dict):
        threshold = filter_terms.get("threshold")
        if isinstance(threshold, dict) and isinstance(threshold.get("field"), str):
            allowed.add(threshold["field"].strip())
    return allowed


def _selected_metric_fields(sql: str) -> Optional[List[str]]:
    """取单条 SQL 的 SELECT 指标列（去标签列）；出现函数/表达式/``*`` 等无法判定项时返回 None。

    Args:
        sql: 单条 SELECT 语句

    Returns:
        指标列名列表；无法判定返回 None（调用方跳过校验，避免误伤）
    """
    if not sql:
        return None
    try:
        from tools.sql_validator import parse_sql

        structure = parse_sql(sql)
    except Exception:  # noqa: BLE001
        return None
    if not structure:
        return None
    cols: List[str] = []
    for _alias, col in structure.get("select_cols") or []:
        if not col:
            return None
        name = str(col).strip().strip("`")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return None  # 星号 / 带函数或引号的表达式 → 无法判定，跳过
        if name not in _TAG_COLUMNS:
            cols.append(name)
    return cols


def metric_field_consistency_error(sql: str, metric_plan: Optional[Dict[str, Any]]) -> Optional[str]:
    """指标-问题一致性校验（B-30）：SQL 取用的指标列必须来自指标计划，禁止近似字段替代。

    背景：C2016 问「每股公积金」，SQL 实际取 net_asset_per_share（每股净资产）作答。
    B-31 已在指标标准化契约上标注库外指标；本函数是生成侧的确定性**审计信号**，零 LLM 成本，
    只比较 SELECT 里的纯列名——函数 / 表达式 / ``*`` 一律跳过。

    **定位：审计，不拦截**。80 题全量回归实测：94 次判定中 6 次命中，逐条核验均为「计划欠规范」
    而非真实指标替换（SQL 合法补用 ``*_yoy_growth``、``other_income``、口径校验原料字段等），
    硬门禁会把这些题从「有数」退化为「仅研报回答」——所以只记录、不重试、不拒答；
    真实库外指标由 ② 出口（unsupported_metrics / 库外词表）负责拒答。

    Args:
        sql: 生成待校验的 SQL（可多条，分号分隔）
        metric_plan: _standardize_metrics 的输出；为 None 或未标注标准字段时不校验

    Returns:
        不一致的说明（写入日志与事件通道 kind=metric_consistency 供审计）；一致或无法判定返回 None
    """
    if not isinstance(metric_plan, dict):
        return None
    allowed = _allowed_metric_fields(metric_plan)
    if not allowed:
        return None
    extras: set = set()
    decided = False
    for stmt in [x.strip() for x in str(sql or "").split(";") if x.strip()]:
        selected = _selected_metric_fields(stmt)
        if selected is None:
            continue
        decided = True
        extras |= {c for c in selected if c not in allowed}
    if not decided or not extras:
        return None
    return (
        f"{_METRIC_MISMATCH_MARKER}（B-30）：SQL 取用了指标计划（standard_fields）之外的字段 "
        + "、".join(sorted(extras))
        + "；请只用 standard_fields 内字段，严禁用近似字段替代所问指标。"
    )


def _advice_risk_type(text: str) -> Optional[str]:
    """检测文本是否含投资建议类表述；命中返回风险类型，否则 None。"""
    if not text:
        return None
    for keyword in _ADVICE_RISK_KEYWORDS:
        if keyword in text:
            return "投资建议"
    return None


def _fmt_rows(rows: List[Dict[str, Any]]) -> str:
    """查询结果转紧凑 JSON 文本（值转 str，供 LLM 阅读）。"""
    return json.dumps(rows, ensure_ascii=False, default=str)[:12000]


_RATIO_FIELDS = ("asset_liability_ratio", "gross_profit_margin", "net_profit_margin", "roe")


def _company_rows(rows: List[Dict[str, Any]]) -> int:
    """统计结果中含公司标识（stock_code/stock_abbr）的行数。"""
    seen = set()
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        code = r.get("stock_code")
        abbr = r.get("stock_abbr")
        key = code if code not in (None, "") else abbr
        if key not in (None, ""):
            seen.add(str(key))
    return len(seen)


def _mean_column_for_question(question: str) -> Optional[str]:
    """按问题关键词返回应做均值聚合的比率字段名；无匹配返回 None。"""
    lowered = (question or "").lower()
    rules = (
        ("asset_liability_ratio", ("资产负债率", "负债率", "负债总额/资产总额", "总负债/总资产")),
        ("gross_profit_margin", ("毛利率",)),
        ("net_profit_margin", ("净利率",)),
        ("roe", ("roe", "净资产收益率")),
    )
    for field, keywords in rules:
        if any(kw in lowered for kw in keywords):
            return field
    return None


def _wants_aggregate_mean(question: str) -> bool:
    """判断问题是否隐含“对多家公司求平均/行业均值/口径校验”意图。"""
    q = question or ""
    if any(kw in q for kw in ("行业均值", "行业平均", "均值", "平均", "所有公司", "全部公司")):
        return True
    return re.search(r"总负债\s*/\s*总资产|负债总额\s*/\s*资产总额|资产总额\s*/\s*负债总额", q) is not None


def _industry_mean_hint(question: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    """多公司明细 + 均值意图时，生成“行业均值”口径提示（供分析 prompt 的【计算结果】）。

    仅对比率类字段生效（百分比口径统一，便于后续数值自洽核验）。
    明细行可能不含公司标签（如 SELECT 未输出 stock_abbr）：此时只要结果无时间维度列、
    行数 >= 2，即视为多公司明细参与求均值。
    """
    if not _wants_aggregate_mean(question):
        return None
    field = _mean_column_for_question(question)
    if field not in _RATIO_FIELDS:
        return None
    values: List[float] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        raw = r.get(field)
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    labeled = _company_rows(rows)
    has_time_cols = any(
        isinstance(r, dict) and (r.get("report_year") not in (None, "") or r.get("report_period") not in (None, ""))
        for r in rows or []
    )
    if labeled == 1:
        # 只有一家公司：多行可能是该公司多期/多口径，不应称“行业均值”
        return None
    if labeled == 0 and has_time_cols:
        # 无公司标签且带时间列：可能是单公司多期数据，不可当作多公司样本
        return None
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return f"行业均值（按全部 {len(values)} 家公司的 {field} 简单算术平均）≈ {mean:.2f}%（比率字段原值即百分比）"


def _mean_consistency_error(
    question: str, rows: List[Dict[str, Any]], analysis: str, hint: Optional[str]
) -> Optional[str]:
    """分析文本与多公司均值口径的自洽性核验；无问题返回 None，有问题返回中文纠错说明。"""
    text = analysis or ""
    if not question or not text:
        return None
    # B2036-Q2：禁止“平均负债总额 / 平均资产总额相除 = 比率均值”这类无依据等价句
    formula_issue = _mean_ratio_equivalence_issue(text)
    if formula_issue:
        return formula_issue
    if "行业均值" not in text and "行业平均" not in text:
        # 问题明确要均值但文本完全未给均值结论时，交由提示词约束处理
        return None
    if _company_rows(rows) == 1 and _wants_aggregate_mean(question):
        return "查询结果仅含 1 家公司行，不得称“行业均值/行业平均”，应改为该公司单值表述。"
    if not hint:
        return None
    matched = re.search(r"≈\s*(\d{1,3}(?:\.\d+)?)", hint)
    if not matched:
        return None
    expected = float(matched.group(1))
    tokens = re.findall(r"(?<![\w.])-?\d{1,3}(?:\.\d+)?(?![\w.])", text)
    if any(abs(float(t) - expected) <= 0.06 for t in tokens):
        return None
    return (
        f"行业均值应等于全部公司行的简单算术平均 {expected:.2f}%（已由【计算结果】给出口径与样本数），"
        "当前文本中的均值数值与它不一致或缺失（常见错误：拿首行单公司的总负债/总资产推导行业均值），请按系统规则重写。"
    )


def _mean_ratio_equivalence_issue(text: str) -> Optional[str]:
    """检测“平均负债/平均资产相除后与比率均值一致”等错误等价句（B2036-Q2 修复）。

    比率均值（算数口径：各公司负债率先算再平均）与加权口径（总负债合计/总资产合计）
    概念不同、数值未必相等；只有样本各公司资产相等时才巧合一致，不能作为验证口径。
    """
    t = (text or "").replace(" ", "").replace("，", ",").replace("。", ".")
    patterns = (
        r"平均负债总额?.{0,24}平均资产总额?.{0,24}(相除|两者相除|直接相除|相除后).{0,16}(一致|相同|相等|等于|逻辑闭环|验证通过)",
        r"平均负债总额?.{0,40}(相除|除以).{0,20}平均资产总额?.{0,40}(一致|相同|相等|等于|逻辑闭环|验证通过)",
    )
    if any(re.search(p, t) for p in patterns):
        return (
            "文本把“平均负债总额/平均资产总额相除”与“各公司资产负债率的算术平均”混为一谈："
            "两者口径不同（比率均值=算数平均；平均负债/平均资产=加权口径），数值未必相等，"
            "严禁用平均负债/平均资产相除来“验证”行业均值。请改写为：直接对全部公司行的负债率取算术平均并注明样本数，"
            "若要说明加权口径，须用真实合计值计算并如实指出两口径差异。"
        )
    return None


def _load_schema_conn(config: Any) -> Tuple[Optional[Dict], Any]:
    """复用 agents.planner 的 MySQL schema/连接缓存；失败返回 (None, None)。"""
    try:
        from agents.planner import _load_schema

        return _load_schema(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("原生财务查询：schema 加载失败: %s", exc)
        return None, None


def _execute_sql(conn: Any, sql: str) -> List[Dict[str, Any]]:
    """pymysql 执行 SQL，返回 list[dict]（列名 -> 值）。

    支持一条请求内以分号分隔的多条 SELECT：逐条执行并合并结果，避免整串执行触发
    pymysql 的 packet 不同步错误（B2040 重跑曾出现 "Packet sequence number wrong"）。
    """
    import sqlparse

    stmts = [s.strip() for s in sqlparse.split(sql or "") if s.strip()]
    if not stmts:
        return []
    cur = conn.cursor()
    try:
        rows: List[Dict[str, Any]] = []
        for statement in stmts:
            cur.execute(statement)
            columns = [d[0] for d in (cur.description or [])]
            if not columns:
                continue
            rows.extend(dict(zip(columns, row)) for row in cur.fetchall())
        return rows
    finally:
        cur.close()


def _company_time_key(row: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """生成“公司 + 时间”分组键；公司缺省或时间列残缺时返回按公司分组键，两者皆无可返回 None。"""
    code = row.get("stock_code")
    abbr = row.get("stock_abbr")
    company = code if code not in (None, "") else abbr
    if company in (None, ""):
        return None
    year = row.get("report_year")
    period = row.get("report_period")
    if year not in (None, "") and period not in (None, ""):
        return ("t", str(company), str(year), str(period))
    if year in (None, "") and period in (None, ""):
        return ("c", str(company))
    # 时间列只给了一半：视为无时间标签，但保守起见不与带完整时间的行混并
    return None


def _merge_company_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把多条语句的结果按“公司（+时间）”对齐合并，字段取并集（B2040 前五名单指标错位修复）。

    适用场景：一条语句返回“公司+营收（名单）”，另一条返回“公司+负债率+毛利率（指标）”，
    两条结果行序不一致导致分析侧公司-数值错位。此处按公司键确定性合并：
    - 同一公司/同一期的多条行记录字段取并集；
    - 同一字段出现多个不同非空值时（多为缺失时间标签的多期数据）跳过合并，避免数据丢失；
    - 无公司键的行（如 AVG 聚合行）原样保留。
    """
    if not rows:
        return []
    groups: "Dict[Tuple[Any, ...], List[Dict[str, Any]]]" = {}
    order: List[Tuple[Any, ...]] = []
    for row in rows:
        key = _company_time_key(row)
        if key is None:
            order.append(("single", id(row)))
            groups[("single", id(row))] = [row]
        elif key not in groups:
            groups[key] = [row]
            order.append(key)
        else:
            groups[key].append(row)
    merged: List[Dict[str, Any]] = []
    for key in order:
        items = groups[key]
        if len(items) == 1:
            merged.append(items[0])
            continue
        combined: Dict[str, Any] = {}
        conflict = False
        for item in items:
            for k, v in item.items():
                if k in combined and combined[k] not in (None, "") and v not in (None, "") and combined[k] != v:
                    conflict = True
                    break
                if combined.get(k) in (None, "") and v not in (None, ""):
                    combined[k] = v
            if conflict:
                break
        if conflict:
            merged.extend(items)
            continue
        merged.append(combined)
    return merged


_METRIC_STANDARDIZE_MAX_TOKENS = 400


def _parse_metric_json_text(text: str) -> Optional[Dict[str, Any]]:
    """容错解析指标标准化 LLM 输出的 JSON（Markdown 围栏/前缀解释/尾部逗号兼容）。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    candidates: List[str] = [cleaned]
    start = cleaned.find("{")
    if start > 0:
        candidates.append(cleaned[start:])
    for candidate in candidates:
        for fixed in (candidate, _remove_trailing_commas(candidate)):
            try:
                obj = json.loads(fixed)
            except (TypeError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _normalize_metric_plan(plan: Any) -> Optional[Dict[str, Any]]:
    """规整/校验指标标准化 JSON；缺关键键或类型错误返回 None（由调用方走旧路径兜底）。"""
    if not isinstance(plan, dict):
        return None
    standard_fields = plan.get("standard_fields")
    time_grain = plan.get("time_grain")
    calculation = plan.get("calculation")
    # B-31：standard_fields 显式为 null/[] 且标注了 unsupported_metrics = 库内无对应字段
    # （合规信号，非解析失败）→ 返回空字段计划 + 库外指标名，由上层走库外口径话术，禁止自由生成 SQL
    if standard_fields is None or standard_fields == []:
        scope_metrics = unsupported_metrics_of(plan)
        if not scope_metrics:
            return None
        time_grain = plan.get("time_grain")
        if not isinstance(time_grain, dict):
            time_grain = {"mode": "none"}
        calculation = plan.get("calculation")
        if not isinstance(calculation, dict):
            calculation = {"kind": "raw"}
        filter_terms = plan.get("filter_terms")
        if not isinstance(filter_terms, dict):
            filter_terms = {}
        return {
            "standard_fields": [],
            "time_grain": time_grain,
            "calculation": calculation,
            "filter_terms": filter_terms,
            "unsupported_metrics": scope_metrics,
        }
    if not isinstance(standard_fields, list) or not all(
        isinstance(x, str) and x.strip() for x in standard_fields
    ):
        return None
    if not isinstance(time_grain, dict) or not isinstance(time_grain.get("mode"), str) or not time_grain.get("mode"):
        return None
    if not isinstance(calculation, dict) or not isinstance(calculation.get("kind"), str) or not calculation.get("kind"):
        return None
    seen: List[str] = []
    for field in standard_fields:
        if field not in seen:
            seen.append(field)
    filter_terms = plan.get("filter_terms")
    if not isinstance(filter_terms, dict):
        filter_terms = {}
    return {
        "standard_fields": seen,
        "time_grain": time_grain,
        "calculation": calculation,
        "filter_terms": filter_terms,
    }


def _metric_plan_to_text(metric_plan: Optional[Dict[str, Any]]) -> str:
    """指标标准化结果转紧凑文本（供 SQL 生成 user content；失败兜底走旧自选文案）。"""
    if not metric_plan:
        return "（无上游指标提取，请依据字段白名单自选）"
    return json.dumps(metric_plan, ensure_ascii=False)[:4000]


def _standardize_metrics(rag: Any, question: str) -> Optional[Dict[str, Any]]:
    """指标标准化小调用：问题 → JSON（standard_fields/time_grain/calculation/filter_terms，B-12 第 1 步）。

    max_tokens 收紧到 _METRIC_STANDARDIZE_MAX_TOKENS 控制成本；
    JSON 解析/规整失败或调用异常返回 None，由 _generate_sql 回退旧自选路径，避免单点故障阻断查询。

    Args:
        rag: RAGPipeline 实例（config / llm_generator）
        question: 重构后的财务问题文本

    Returns:
        规整后的指标计划 JSON；不可用返回 None
    """
    config = getattr(rag, "config", None)
    if config is not None and not getattr(config, "AGENT_METRIC_STANDARDIZE", True):
        return None
    model = getattr(config, "LLM_MODEL", "qwen3.5-plus") if config is not None else "qwen3.5-plus"
    try:
        resp = rag.llm_generator.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": METRIC_STANDARDIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"重构后的问题：{question}"},
            ],
            temperature=0.1,
            max_tokens=_METRIC_STANDARDIZE_MAX_TOKENS,
            # 小 JSON 调用统一关思考，避免思考 token 挤占 max_tokens
            extra_body={"enable_thinking": False},
        )
        plan = _normalize_metric_plan(_parse_metric_json_text(resp.choices[0].message.content or ""))
        if plan is None:
            _contract_stats().record("metric_plan", False, ["标准化输出 JSON 非法/规整失败（B-17）"])
            logger.warning("原生财务查询：指标标准化 JSON 非法，回退自选路径: %s", str(question)[:80])
            return None
        _contract_stats().record("metric_plan", True, [])
        logger.info("原生财务查询：指标标准化成功: %s", json.dumps(plan, ensure_ascii=False)[:600])
        return plan
    except Exception as exc:  # noqa: BLE001
        _contract_stats().record("metric_plan", False, [f"标准化调用异常: {str(exc)[:120]}（B-17）"])
        logger.warning("原生财务查询：指标标准化调用失败，回退自选路径: %s", exc)
        return None


def _generate_sql(
    rag: Any, question: str, schema: Dict, conn: Any, retries: int, metric_plan: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str]]:
    """LLM 生成 SQL + 三层防线校验；失败带错误重试。返回 (sql, 错误列表)。

    B-12 两步分解：默认先 _standardize_metrics 得到 JSON（standard_fields/time_grain/calculation/filter_terms），
    SQL 生成只做“映射+拼装”；metric_plan 为 None（关闭/失败）时回退旧“依据字段白名单自选”文案。
    """
    if metric_plan is None:
        metric_plan = _standardize_metrics(rag, question)
    scope_metrics = unsupported_metrics_of(metric_plan)
    if scope_metrics:
        # B-31：库内无对应字段 → 不再自由生成 SQL（避免生成非法 SQL 后回落技术性报错）
        _gen_stats().record(attempts=0, first_ok=False, mode="none", injected=0, ok=False, reason="out_of_scope")
        return "", ["库外指标（库内无对应字段）: " + "、".join(scope_metrics)]
    plan_text = _metric_plan_to_text(metric_plan)
    fs_mode = _resolve_few_shot_mode(rag)
    sql_gen_system = SQL_GEN_SYSTEM_PROMPT
    fs_examples: List[Dict[str, Any]] = []
    if fs_mode != "none":
        from utils.few_shot_retriever import build_sql_gen_system

        cfg = getattr(rag, "config", None)
        sql_gen_system, fs_examples = build_sql_gen_system(
            question,
            metric_plan,
            mode=fs_mode,
            static_k=int(getattr(cfg, "AGENT_FEWSHOT_STATIC_K", 2) or 2),
            static_strategy=str(getattr(cfg, "AGENT_FEWSHOT_STATIC_STRATEGY", "per_type") or "per_type"),
        )
        if fs_examples:
            logger.info("原生财务查询：few-shot(%s) 注入 %d 条示例（B-16/B-39）", fs_mode, len(fs_examples))
    errors: List[str] = []
    last_sql = ""
    for attempt in range(retries + 1):
        user_content = f"重构后的问题: {question}\n指标标准化结果(JSON): {plan_text}"
        if errors:
            user_content += "\n\n上一次生成的 SQL 校验失败，错误如下，请修正后重新生成：\n" + "\n".join(errors[-3:])
        try:
            resp = rag.llm_generator.client.chat.completions.create(
                model=rag.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": sql_gen_system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=800,
                extra_body={"enable_thinking": getattr(rag.config, "AGENT_ENABLE_THINKING", False)},
            )
            sql = (resp.choices[0].message.content or "").strip()
            sql = sql.strip("`")
            if sql.lower().startswith("sql"):
                sql = sql[3:].lstrip()
            last_sql = sql
        except Exception as exc:  # noqa: BLE001
            errors.append(f"LLM 调用失败: {exc}")
            continue
        # 契约闸门（B-17）：危险关键字/非 SELECT/全角标点等格式问题直接拒绝并带错重试
        sql_res = _contract_sql(sql)
        if not sql_res.ok:
            errors.append("格式契约拒绝: " + "；".join(sql_res.errors[:3]))
            _contract_stats().record("sql_output", False, sql_res.errors)
            continue
        _contract_stats().record("sql_output", True, [])
        # 三层防线：静态校验 + MySQL 编译终审
        try:
            from tools.sql_validator import compile_check, validate_sql

            ok, serrs = validate_sql(sql, schema)
            if ok:
                # B-30：指标-问题一致性（纯规则、零 LLM 成本，先于 MySQL 编译执行）
                cons_err = metric_field_consistency_error(sql, metric_plan)
                _contract_stats().record(
                    "metric_consistency", not cons_err, [cons_err] if cons_err else []
                )
                if cons_err:
                    # 审计信号：只记录（B-30 全量回归实测硬拦截会把合法补用字段的题误伤）
                    logger.warning("原生财务查询：指标计划与 SQL 的字段差异（审计）: %s", cons_err)
            cerr = ""
            if ok and conn is not None:
                cerr = compile_check(conn, sql)
            if ok and not cerr:
                _gen_stats().record(
                    attempts=attempt + 1,
                    first_ok=(attempt == 0),
                    mode=fs_mode,
                    injected=len(fs_examples),
                    ok=True,
                    sql=sql,
                )
                return sql, []
            errors.extend(list(serrs)[:4] if not ok else [])
            if cerr:
                errors.append(f"编译错误: {cerr[:200]}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"校验异常: {exc}")
            _gen_stats().record(
                attempts=attempt + 1, first_ok=False, mode=fs_mode,
                injected=len(fs_examples), ok=False, reason="validate_error", sql=last_sql,
            )
            return "", errors
    _gen_stats().record(
        attempts=retries + 1, first_ok=False, mode=fs_mode,
        injected=len(fs_examples), ok=False, reason="retry_exhausted", sql=last_sql,
    )
    return "", errors


def _generate_analysis(rag: Any, question: str, rows: List[Dict[str, Any]]) -> str:
    """LLM 基于查询结果生成分析文本（模式一/二）。"""
    hint = _industry_mean_hint(question, rows)
    calc_result = hint or "（无）"

    def _user_content(extra: str = "") -> str:
        """组装用户消息；extra 用于自洽性核验失败后的纠错重试。"""
        return (
            f"重构问题：{question}\n"
            f"查询结果：{_fmt_rows(rows)}\n"
            f"计算结果：{calc_result}\n"
            "注意：查询结果为数据库原始值（如 net_profit 单位为元、net_profit_10k_yuan 为万元、"
            "total_operating_revenue 为万元）。请按数值量级换算为合适的单位（万元/亿元）后再表述，"
            "全文单位必须统一，严禁直接复读原始大数字而忽略单位。"
            + (f"\n{extra}" if extra else "")
        )

    def _call(user_content: str) -> str:
        """调用一次分析 LLM，异常返回兜底文案。"""
        try:
            resp = rag.llm_generator.client.chat.completions.create(
                model=rag.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=500,
                extra_body={"enable_thinking": getattr(rag.config, "AGENT_ENABLE_THINKING", False)},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("原生财务查询：分析生成失败: %s", exc)
            return "查询完成，但分析文本生成失败。"

    last_error = ""
    analysis = ""
    for _attempt in (1, 2):
        extra = ""
        if last_error:
            extra = (
                "上一轮自洽性核验未通过：" + last_error +
                "请严格按系统规则重写：行业均值必须为全部公司行指标的平均值（AVG）并注明样本数与口径，"
                "严禁用首行单公司数据冒充行业均值；涉及“是否符合负债总额/资产总额”时需区分算数平均与加权口径，"
                "不得用平均负债总额/平均资产总额相除来冒充验证结果。"
            )
        analysis = _call(_user_content(extra))
        issue = _mean_consistency_error(question, rows, analysis, hint)
        if not issue:
            return analysis
        last_error = issue
        logger.warning("原生财务查询：分析自洽性核验未通过，重试一次: %s", issue)
    return analysis


def _extract_balanced_json(text: str) -> Optional[str]:
    """从文本中截取首个括号深度归零的完整 JSON 对象子串（正确处理字符串内括号）。"""
    start = (text or "").find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _remove_trailing_commas(text: str) -> str:
    """去除对象/数组结尾的多余逗号（LLM 常见输出；仅在校验失败后的容错路径使用）。"""
    try:
        return re.sub(r",\s*([}\]])", r"\1", text)
    except Exception:  # noqa: BLE001
        return text


def _parse_chart_json_text(text: str) -> Optional[Dict[str, Any]]:
    """容错解析图表 JSON：去 Markdown 围栏 → 提取首个完整 JSON 对象 → 去尾部逗号。"""
    if not (text or "").strip():
        return None
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    candidates: List[str] = []
    if cleaned:
        candidates.append(cleaned)
    start = cleaned.find("{")
    if start > 0:
        candidates.append(cleaned[start:])
    balanced = _extract_balanced_json(cleaned)
    if balanced:
        candidates.append(balanced)
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for fixed in (candidate, _remove_trailing_commas(candidate)):
            try:
                obj = json.loads(fixed)
            except (TypeError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _dump_chart_failure(question: str, text: str, finish_reason: Optional[str]) -> None:
    """把图表 JSON 解析失败的原文追加到运行时文件（供 B-11 归集；失败静默）。"""
    try:
        from pathlib import Path

        target = Path(__file__).resolve().parents[1] / "训练结果数据" / "chart_parse_failures.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "question": str(question)[:200],
            "len": len(text or ""),
            "finish_reason": finish_reason,
            "text": text or "",
        }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _to_num(value: Any) -> Any:
    """尽力把图表数据点转为数值（LLM 偶尔输出字符串数字）。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _sanitize_chart(option: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验并收敛 ECharts option：必须含非空 series[].data；超限截断；数值归一。"""
    if not isinstance(option, dict):
        return None
    series = option.get("series")
    if not isinstance(series, list) or not series:
        return None
    for item in series:
        if not isinstance(item, dict) or "data" not in item:
            return None
        data = item.get("data")
        if not isinstance(data, list) or not data:
            return None
        item["data"] = [_to_num(v) for v in data[:200]]
    return option


def _generate_chart(rag: Any, question: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """LLM 生成 ECharts 配置；无需图表或生成/校验失败返回 None（不影响文字答案）。"""
    if not rows:
        return None
    user_content = f"用户问题：{question}\n查询结果：{_fmt_rows(rows)}"
    try:
        resp = rag.llm_generator.client.chat.completions.create(
            model=rag.config.LLM_MODEL,
            messages=[
                {"role": "system", "content": CHART_GEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=3000,
            extra_body={"enable_thinking": getattr(rag.config, "AGENT_ENABLE_THINKING", False)},
        )
        text = (resp.choices[0].message.content or "").strip()
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("原生财务查询：图表生成失败: %s", exc)
        return None
    obj = _parse_chart_json_text(text)
    if obj is None:
        snippet = (text or "").replace("\r", " ").replace("\n", " ")[:160]
        tail = (text or "").replace("\r", " ").replace("\n", " ")[-100:]
        _dump_chart_failure(question, text, finish_reason)
        logger.warning(
            "原生财务查询：图表 JSON 解析失败，跳过图表（len=%d finish=%s 首片段: %s ... 尾片段: %s）",
            len(text or ""), finish_reason, snippet, tail,
        )
        return None
    if not isinstance(obj, dict) or obj.get("need_chart") is False:
        return None
    option = obj.get("chart") if isinstance(obj.get("chart"), dict) else obj
    return _sanitize_chart(option)


def native_financial_query(rag: Any, user_query: str, user_id: str = "default") -> str:
    """原生财务查询入口（与 call_financial_chatflow 同接口，返回 JSON 字符串）。

    Args:
        rag: RAGPipeline 实例（llm_generator.client / config / query_cache）
        user_query: 用户自然语言财务问题
        user_id: 会话用户标识

    Returns:
        JSON 字符串：{"content": ..., "image": [], "sql": ..., "chart_json": {...}|null}
    """
    config = getattr(rag, "config", None)
    cache = getattr(rag, "query_cache", None)
    cache_key = None
    if cache is not None:
        try:
            from utils.query_cache import make_cache_key

            cache_key = make_cache_key("fin-native", user_id, user_query, FINANCIAL_PROMPT_VERSION)
            hit = cache.get(cache_key)
            if hit is not None:
                try:
                    from agents.planner import _append_sql

                    _append_sql(rag, hit.get("sql") or "")
                except Exception:  # noqa: BLE001
                    pass
                return json.dumps(
                    {
                        "content": hit.get("content", ""),
                        "image": [],
                        "sql": hit.get("sql", ""),
                        "chart_json": hit.get("chart_json") or None,
                    },
                    ensure_ascii=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("原生财务查询缓存读取失败（按未命中处理）: %s", exc)
            cache_key = None
    try:
        retries = int(getattr(config, "AGENT_NATIVE_RETRY", 2)) if config else 2
        schema, conn = _load_schema_conn(config)
        if conn is None or schema is None:
            content, _tid = build_refuse_system_unavailable("MySQL schema/连接加载失败")
            _log_fallback(_tid, detail="mysql_schema_load_failed")
            return json.dumps({"content": content, "image": []}, ensure_ascii=False)
        # 每个查询使用独立短连接：并行子 Agent/多 tool_call 共享同一 pymysql 连接会触发
        # "Packet sequence number wrong" 等协议错乱（B2040 并行重跑曾复现），此处隔离执行连接。
        run_conn = None
        try:
            import pymysql

            run_conn = pymysql.connect(
                host=config.MYSQL_HOST,
                port=int(getattr(config, "MYSQL_PORT", 3306) or 3306),
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                charset="utf8mb4",
                connect_timeout=5,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("原生财务查询：独立 MySQL 连接建立失败: %s", exc)
            content, _tid = build_refuse_system_unavailable(f"MySQL 连接失败：{exc}")
            _log_fallback(_tid, detail="mysql_connect_failed")
            return json.dumps({"content": content, "image": []}, ensure_ascii=False)
        try:
            # B-28 错别字归一化：精简映射（资产负债绿→资产负债率 等）+ 库内公司简称编辑距离纠错
            # （云白要→云南白药 等），先还原再走指标标准化/SQL 生成，避免按错字查库返回空直接拒答
            from tools.typo_normalizer import load_company_abbrs, normalize_question_typos  # noqa: PLC0415

            canonical_names = load_company_abbrs(run_conn)
            query_text, typo_fixes = normalize_question_typos(user_query, canonical_names)
            if typo_fixes:
                logger.info("B-28 错字归一化: %r -> %r（%s）", user_query, query_text, typo_fixes)
            # B-31：先看指标标准化是否标注「库外指标」——库内无对应字段时直接走库外口径话术，
            # 不再自由生成 SQL（原路径会生成非法 SQL 后回落技术性报错，用户看到的是格式契约细节）
            metric_plan = _standardize_metrics(rag, query_text)
            scope_metrics = unsupported_metrics_of(metric_plan)
            if scope_metrics:
                content, _tid = build_refuse_metric_out_of_scope(scope_metrics)
                _log_fallback(_tid, detail="metric_out_of_scope: " + "、".join(scope_metrics))
                logger.info("原生财务查询：库外指标拒答（%s）: %s", "、".join(scope_metrics), str(query_text)[:80])
                return json.dumps(
                    {"content": content, "image": [], "sql": "", "chart_json": None}, ensure_ascii=False
                )
            # B-30：模型漏标库外指标（把库外指标硬映射成近似字段）时的确定性兜底
            stray_terms = unmarked_out_of_scope_terms(query_text, metric_plan)
            if stray_terms:
                content, _tid = build_refuse_metric_out_of_scope(stray_terms)
                _log_fallback(_tid, detail="metric_out_of_scope_unmarked: " + "、".join(stray_terms))
                logger.info("原生财务查询：库外指标漏标兜底拒答（%s）: %s", "、".join(stray_terms), str(query_text)[:80])
                return json.dumps(
                    {"content": content, "image": [], "sql": "", "chart_json": None}, ensure_ascii=False
                )
            sql, errors = _generate_sql(rag, query_text, schema, run_conn, retries, metric_plan=metric_plan)
            if not sql:
                detail = "；".join(errors[:4]) if errors else "未知原因"
                content, _tid = _sql_failure_reply(errors)
                logger.warning("原生财务查询：SQL 生成经 %d 次校验未通过：%s", retries, detail)
                _log_fallback(_tid, detail="sql_gen_failed")
                return json.dumps(
                    {"content": content, "image": [], "sql": "", "chart_json": None}, ensure_ascii=False
                )
            rows = _merge_company_rows(_execute_sql(run_conn, sql))
            if not rows:
                content, _tid = build_refuse_data_not_found(
                    subject=query_text, detail="（SQL 已执行成功，但未返回任何数据）"
                )
                _log_fallback(_tid, detail="sql_ok_rows_empty")
                # B-36：空结果分支在 return 前也要累积 SQL —— 否则「执行成功但 0 行」的
                # 故障现场不会进入 conversation_state.sql，评测侧（judge / 先验校验）
                # 拿到的记录里 SQL 为空，既无法复核也无法归因（B2053 4/5 误拒答即此盲区）
                try:
                    from agents.planner import _append_sql

                    _append_sql(rag, sql)
                except Exception:  # noqa: BLE001
                    pass
                return json.dumps(
                    {"content": content, "image": [], "sql": sql, "chart_json": None},
                    ensure_ascii=False,
                )
            # 分析/图表只依赖查询结果 rows，互不依赖：并行生成，省一次串行 LLM 延迟
            try:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=2) as _pool:
                    analysis_f = _pool.submit(_generate_analysis, rag, query_text, rows)
                    chart_f = _pool.submit(_generate_chart, rag, query_text, rows)
                    analysis = analysis_f.result()
                    chart = chart_f.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("分析/图表并行失败，回退串行: %s", exc)
                analysis = _generate_analysis(rag, query_text, rows)
                chart = _generate_chart(rag, query_text, rows)
            risk_type = _advice_risk_type(analysis)
            if risk_type:
                human_note, _tid = build_human_risk_advice(risk_type=risk_type)
                analysis = f"{analysis}\n\n{human_note}"
                _log_fallback(_tid, detail="analysis_contained_advice_keyword")
            try:
                from agents.planner import _append_sql

                _append_sql(rag, sql)
            except Exception:  # noqa: BLE001
                pass
            result = {"content": analysis, "image": [], "sql": sql, "chart_json": chart}
            if cache is not None and cache_key is not None:
                try:
                    cache.set(cache_key, {"content": analysis, "sql": sql, "chart_json": chart})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("原生财务查询缓存写入失败（忽略）: %s", exc)
            return json.dumps(result, ensure_ascii=False)
        finally:
            if run_conn is not None:
                try:
                    run_conn.close()
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("原生财务查询异常")
        content, _tid = build_refuse_system_unavailable(f"查询失败：{exc}")
        _log_fallback(_tid, detail="unexpected_exception")
        return json.dumps({"content": content, "image": []}, ensure_ascii=False)
