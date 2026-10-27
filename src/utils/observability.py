"""结构化请求追踪与 JSONL 日志。

设计目标：

- 每次请求一个 ``RequestTrace``，通过 ``contextvars`` 在检索/生成线程间传递；
- 阶段耗时、token、模型名、成本状态一次落盘；
- 单价未配置时明确写 ``unit_price_not_registered``，不编造金额；
- 线程安全追加到 ``训练结果数据/logs/YYYYMMDD.jsonl``。

该模块只依赖标准库，便于离线单测。
"""

from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_CURRENT_TRACE: ContextVar[Optional["RequestTrace"]] = ContextVar(
    "current_request_trace", default=None
)
_WRITE_LOCK = threading.Lock()


def _now_iso() -> str:
    """返回本地时区 ISO 时间（毫秒精度）。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass
class RequestTrace:
    """单请求可观测数据。字段缺失时保留 ``None``，不填 0 冒充实测。"""

    trace_id: str
    path: str = ""
    method: str = ""
    mode: Optional[str] = None
    status: str = "running"
    status_code: Optional[int] = None
    started_at: str = field(default_factory=_now_iso)
    started_perf: float = field(default_factory=time.perf_counter)
    stages_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_name: Optional[str] = None
    estimated_cost: Optional[float] = None
    cost_status: str = "unit_price_not_registered"
    usage_status: str = "usage_unavailable"
    cache_hit: bool = False
    error: Optional[str] = None
    total_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def record_stage(self, name: str, elapsed_ms: Optional[float]) -> None:
        """记录阶段耗时；``None`` 表示该阶段未执行或不可测。"""
        self.stages_ms[name] = None if elapsed_ms is None else round(float(elapsed_ms), 3)

    def record_usage(
        self,
        *,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        input_price_per_1k: Optional[float] = None,
        output_price_per_1k: Optional[float] = None,
    ) -> None:
        """记录一次 LLM 调用的 token 与成本；单价缺失时明确标记未登记。"""
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        self.prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
        self.completion_tokens = int(completion_tokens) if completion_tokens is not None else None
        self.total_tokens = int(total_tokens) if total_tokens is not None else None
        if model_name:
            self.model_name = model_name

        if self.prompt_tokens is None and self.completion_tokens is None:
            self.usage_status = "usage_unavailable"
            return

        self.usage_status = "recorded"
        if input_price_per_1k is None or output_price_per_1k is None:
            self.estimated_cost = None
            self.cost_status = "unit_price_not_registered"
            return

        prompt = float(self.prompt_tokens or 0)
        completion = float(self.completion_tokens or 0)
        self.estimated_cost = round(
            prompt / 1000.0 * float(input_price_per_1k)
            + completion / 1000.0 * float(output_price_per_1k),
            8,
        )
        self.cost_status = "estimated"

    def mark_usage_unavailable(self, reason: str = "usage_unavailable") -> None:
        """标记 token 用量不可得（例如流式响应未返回 usage）。"""
        self.usage_status = reason

    def finalize(
        self,
        *,
        total_ms: Optional[float] = None,
        status: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> "RequestTrace":
        """结束请求；默认用 perf_counter 计算请求总耗时。"""
        if total_ms is None:
            total_ms = (time.perf_counter() - self.started_perf) * 1000.0
        self.total_ms = round(float(total_ms), 3)
        if status is not None:
            self.status = status
        if status_code is not None:
            self.status_code = int(status_code)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """输出固定字段的日志记录，便于聚合脚本消费。"""
        return {
            "ts": self.started_at,
            "trace_id": self.trace_id,
            "path": self.path,
            "method": self.method,
            "mode": self.mode,
            "status": self.status,
            "status_code": self.status_code,
            "retrieval_ms": self.stages_ms.get("retrieval_ms"),
            "rerank_ms": self.stages_ms.get("rerank_ms"),
            "generate_ms": self.stages_ms.get("generate_ms"),
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "estimated_cost": self.estimated_cost,
            "cost_status": self.cost_status,
            "usage_status": self.usage_status,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "stages_ms": dict(self.stages_ms),
            **self.extra,
        }


def get_trace() -> Optional[RequestTrace]:
    """返回当前上下文的请求 trace；不在请求上下文时返回 ``None``。"""
    return _CURRENT_TRACE.get()


def set_trace(trace: Optional[RequestTrace]) -> Token:
    """设置当前上下文 trace，返回用于 reset 的 token。"""
    return _CURRENT_TRACE.set(trace)


def reset_trace(token: Token) -> None:
    """恢复上一个 contextvar 值。"""
    _CURRENT_TRACE.reset(token)


def log_path(log_dir: str, when: Optional[datetime] = None) -> Path:
    """返回 ``<log_dir>/YYYYMMDD.jsonl``。"""
    day = (when or datetime.now()).strftime("%Y%m%d")
    return Path(log_dir) / f"{day}.jsonl"


def append_trace(trace: RequestTrace, *, log_dir: str, enabled: bool = True) -> Optional[Path]:
    """线程安全追加一条 JSONL；``enabled=False`` 时只返回目标路径不写盘。"""
    path = log_path(log_dir)
    if not enabled:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(trace.as_dict(), ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    return path


def finalize_and_log(
    trace: RequestTrace,
    *,
    log_dir: str,
    enabled: bool = True,
    total_ms: Optional[float] = None,
    status: Optional[str] = None,
    status_code: Optional[int] = None,
) -> Optional[Path]:
    """结束 trace 并落盘，返回日志路径。"""
    trace.finalize(total_ms=total_ms, status=status, status_code=status_code)
    return append_trace(trace, log_dir=log_dir, enabled=enabled)
