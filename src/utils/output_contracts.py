# -*- coding: utf-8 -*-
"""B-17 输出契约层：supervisor 拆解 / aggregator 汇总 / metric 标准化 / SQL 产物
四类模型输出统一做契约校验 + 格式错误事件落盘（jsonl）。

与设计方案 §3.5.5（输出格式约束）和 §6.7.1（结构一致性度量）对齐：
- 校验失败 = 一条格式错误事件，调用侧据此「拒绝透传」并走既有重试/兜底；
- 事件落 ``训练结果数据/output_contract_stats.jsonl``（运行时数据目录，不入库），
  可按 kind 汇总格式错误率，供评测套件复用（B-21 分层门禁 / n=5 结构一致性）。
纯逻辑模块：零外部依赖（不调 LLM/MySQL/Qdrant），可离线单测。

用法::

    from utils.output_contracts import (
        validate_supervisor_output, validate_aggregate_result,
        validate_metric_plan, validate_sql_output, get_stats,
    )
    res = validate_metric_plan(raw_json)
    if not res.ok:
        get_stats().record("metric_plan", False, res.errors)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: 默认事件落盘路径（运行时数据目录，已被 .gitignore 忽略）
DEFAULT_STATS_PATH = Path(__file__).resolve().parents[2] / "训练结果数据" / "output_contract_stats.jsonl"
#: 可用 env 关闭事件落盘（校验逻辑不受影响）
_STATS_ENV = "OUTPUT_CONTRACT_STATS"

#: supervisor 子 Agent 白名单（与 agents/langgraph_multi_agent._parse_tasks 一致）
ALLOWED_AGENTS: tuple = ("financial", "research")
#: SQL 生成产物允许的语句首关键字
_SQL_ALLOWED_HEADS = ("select", "with", "(")
#: SQL 危险/变更关键字（静态粗筛，详细校验仍走 tools/sql_validator + sql_guard 三层防线）
_SQL_BANNED_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "grant", "revoke", "set", "load", "replace into",
    "into outfile", "information_schema",
)
_SQL_MAX_STATEMENTS = 20
_SQL_MAX_LENGTH = 8000
#: SQL 全角标点（MySQL 不识别，B2011 真实 badcase）
_FULLWIDTH_RE = re.compile(r"[，；：（）、【】]")


@dataclass
class ContractResult:
    """契约校验结果。

    Attributes:
        kind: 产物类型（supervisor_tasks / aggregate_result / metric_plan / sql_output）。
        ok: 是否通过契约。
        errors: 违规描述列表（空 = 通过）。
        normalized: 规整后的对象（supervisor 返回任务列表；其余为 None 或原对象）。
    """

    kind: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    normalized: Any = None


class ContractStats:
    """格式错误事件统计器：线程安全计数 + 追加写 jsonl。

    Args:
        path: 事件落盘路径（默认 DEFAULT_STATS_PATH；单测传 tmp_path）。
        enabled: 是否落盘（默认读 env OUTPUT_CONTRACT_STATS，默认开）。
    """

    def __init__(self, path: Optional[Path] = None, enabled: Optional[bool] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_STATS_PATH
        if enabled is None:
            enabled = os.getenv(_STATS_ENV, "true").lower() != "false"
        self.enabled = enabled
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, int]] = {}
        self._fallback: Dict[str, int] = {}

    def record(self, kind: str, ok: bool, errors: Optional[List[str]] = None) -> None:
        """记录一次校验事件（ok=False 即一条格式错误事件）。"""
        errs = [str(e) for e in (errors or [])][:3]
        with self._lock:
            c = self._counters.setdefault(kind, {"calls": 0, "ok": 0, "fail": 0})
            c["calls"] += 1
            if ok:
                c["ok"] += 1
            else:
                c["fail"] += 1
            if self.enabled:
                self._append_event(
                    {"kind": kind, "ok": bool(ok), "errors": errs}
                )

    def _append_event(self, payload: Dict[str, Any]) -> None:
        """线程内追加写一条事件到 jsonl（落盘失败不影响校验主流程）。"""
        event = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:  # 落盘失败不影响校验主流程
            logger.warning("输出契约统计落盘失败（%s）: %s", self.path, exc)

    def record_fallback(self, template_id: str, category: str, detail: str = "") -> None:
        """记录一次兜底话术命中（B-18）：按 template_id 单调计数，事件落盘 kind=fallback。

        Args:
            template_id: 命中模板 ID（如 refuse.data_not_found）。
            category: 兜底类别（refuse / suggest / human）。
            detail: 命中分支/原因说明（用于分支分布统计）。
        """
        with self._lock:
            self._fallback[template_id] = self._fallback.get(template_id, 0) + 1
            if self.enabled:
                self._append_event(
                    {"kind": "fallback", "category": category, "template_id": template_id, "detail": detail}
                )

    def fallback_summary(self) -> Dict[str, int]:
        """汇总各 template_id 的命中次数（兜底分支分布）。"""
        with self._lock:
            return dict(sorted(self._fallback.items()))

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """汇总各 kind 的调用/通过/失败数，便于打印格式错误率。"""
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for kind, c in self._counters.items():
                rate = round(c["fail"] / c["calls"], 4) if c["calls"] else None
                out[kind] = {**c, "format_error_rate": rate}
            return out


_default_stats = ContractStats()


def get_stats() -> ContractStats:
    """模块级默认统计器（运行时调用侧共用）。"""
    return _default_stats


def validate_supervisor_output(value: Any) -> ContractResult:
    """supervisor 拆解输出契约：tasks 子项必须 agent 白名单 + 非空 query；
    无任务时必须带非空 direct_answer（澄清/直答为合法输出）。"""
    kind = "supervisor_tasks"
    errors: List[str] = []
    tasks: List[Dict[str, str]] = []
    direct: Optional[str] = None
    if isinstance(value, dict):
        direct = value.get("direct_answer")
        raw_tasks = value.get("tasks") or []
    elif isinstance(value, list):
        raw_tasks = value
    else:
        return ContractResult(kind, False, ["输出不是 JSON 对象或数组"], value)
    for t in raw_tasks:
        if not isinstance(t, dict):
            errors.append("任务项不是对象")
            continue
        agent = t.get("agent")
        query = t.get("query")
        if agent not in ALLOWED_AGENTS:
            errors.append(f"任务 agent 不在白名单: {agent!r}")
            continue
        if not isinstance(query, str) or not query.strip():
            errors.append("任务 query 缺失或为空")
            continue
        tasks.append({"agent": str(agent), "query": query.strip()})
    if not tasks:
        if direct is None or (isinstance(direct, str) and not direct.strip()):
            errors.append("未拆出可执行任务且无 direct_answer")
    if errors:
        return ContractResult(kind, False, errors, tasks)
    return ContractResult(kind, True, [], tasks)


def validate_aggregate_result(value: Any) -> ContractResult:
    """aggregator 汇总输出契约：必须是对象且含非空 content 字符串；
    image / references / chart_json 若存在则必须为合法类型。"""
    kind = "aggregate_result"
    if not isinstance(value, dict):
        return ContractResult(kind, False, ["汇总结果不是 JSON 对象"], value)
    content = value.get("content")
    if not isinstance(content, str) or not content.strip():
        return ContractResult(kind, False, ["content 缺失或为空"], value)
    for key, exp in (("image", list), ("references", list)):
        v = value.get(key)
        if v is not None and not isinstance(v, exp):
            return ContractResult(kind, False, [f"{key} 应为 {exp.__name__}"], value)
    chart = value.get("chart_json")
    if chart is not None and not isinstance(chart, (dict, list)):
        return ContractResult(kind, False, ["chart_json 应为对象或 null"], value)
    return ContractResult(kind, True, [], value)


def validate_metric_plan(value: Any) -> ContractResult:
    """指标标准化输出契约（与 tools/native_financial._normalize_metric_plan 同口径）：
    standard_fields 非空字符串列表；time_grain.mode、calculation.kind 非空字符串；
    filter_terms 若存在须为对象。"""
    kind = "metric_plan"
    if not isinstance(value, dict):
        return ContractResult(kind, False, ["指标标准化输出不是 JSON 对象"], value)
    fields = value.get("standard_fields")
    if not isinstance(fields, list) or not fields or not all(isinstance(x, str) and x.strip() for x in fields):
        return ContractResult(kind, False, ["standard_fields 应为非空字符串列表"], value)
    time_grain = value.get("time_grain")
    if not isinstance(time_grain, dict) or not isinstance(time_grain.get("mode"), str) or not time_grain.get("mode"):
        return ContractResult(kind, False, ["time_grain.mode 缺失或为空"], value)
    calculation = value.get("calculation")
    if not isinstance(calculation, dict) or not isinstance(calculation.get("kind"), str) or not calculation.get("kind"):
        return ContractResult(kind, False, ["calculation.kind 缺失或为空"], value)
    filter_terms = value.get("filter_terms")
    if filter_terms is not None and not isinstance(filter_terms, dict):
        return ContractResult(kind, False, ["filter_terms 应为对象"], value)
    return ContractResult(kind, True, [], value)


def validate_sql_output(sql: str) -> ContractResult:
    """SQL 生成产物契约（生成后、正式三层防线前的快速闸门）：
    去代码围栏后按语句切分，语句首关键字须 SELECT/WITH，扫描危险关键字与全角标点，
    语句数与长度设上限。非法即拒绝透传，交由既有重试带错误重生成。"""
    kind = "sql_output"
    errors: List[str] = []
    text = (sql or "").strip().strip("`")
    if text.lower().startswith("sql"):
        text = text[3:].lstrip().strip("`")
    if not text:
        return ContractResult(kind, False, ["SQL 为空"], text)
    if len(text) > _SQL_MAX_LENGTH:
        errors.append(f"SQL 超长（{len(text)} > {_SQL_MAX_LENGTH}）")
    lowered = text.lower()
    for banned in _SQL_BANNED_KEYWORDS:
        if re.search(rf"\b{re.escape(banned)}\b", lowered):
            errors.append(f"包含危险/变更关键字: {banned}")
            break
    if _FULLWIDTH_RE.search(text):
        errors.append("包含全角标点（MySQL 不识别）")
    try:
        import sqlparse

        statements = [s.strip() for s in sqlparse.split(text) if s.strip()]
    except Exception:  # noqa: BLE001 - sqlparse 异常按格式错误处理
        statements = [text]
        errors.append("SQL 无法切分语句")
    if not statements:
        errors.append("未解析出任何 SQL 语句")
    if len(statements) > _SQL_MAX_STATEMENTS:
        errors.append(f"语句数超限（{len(statements)} > {_SQL_MAX_STATEMENTS}）")
    for stmt in statements:
        head = re.sub(r"^[`\s(]+", "", stmt.lstrip(), flags=re.IGNORECASE).split(None, 1)[0].lower() if stmt.split() else ""
        if head not in _SQL_ALLOWED_HEADS:
            errors.append(f"语句首关键字非法: {head or '(空)'}")
            break
    if errors:
        return ContractResult(kind, False, errors[:5], text)
    return ContractResult(kind, True, [], text)
