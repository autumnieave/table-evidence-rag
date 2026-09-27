import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

from filters.query_filters import QueryFilters


class ClarifyStatus(Enum):
    READY = "ready"               # 信息充分，可直接检索
    NEED_CLARIFY = "need_clarify" # 需要澄清


@dataclass
class ConversationState:
    """多轮会话状态（可序列化，支持按 user_id 持久化）。

    字段说明：
    - filters: 已积累的过滤条件
    - history: 对话历史 [{"role": "user/assistant", "content": ...}]
    - status: READY / NEED_CLARIFY
    - last_active: 最后活跃时间戳（超时判定用）
    - rounds: 每轮 Agent 结构化输出
    - sql: 最近一次工具调用产生的 SQL（Agent 多轮累积）
    """

    filters: QueryFilters = field(default_factory=QueryFilters)
    missing_fields: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    clarify_question: Optional[str] = None
    status: ClarifyStatus = ClarifyStatus.READY
    last_active: float = field(default_factory=time.time)
    pending_question: Optional[str] = None  # 待补全的原始问题（已累积的上下文）
    user_id: str = "default"                # 会话标识
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    sql: str = ""                           # 最近一次工具调用产生的 SQL

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 存储的字典（filters/status 展开为原生类型）"""
        return {
            "filters": asdict(self.filters) if self.filters else {},
            "missing_fields": list(self.missing_fields),
            "history": list(self.history),
            "clarify_question": self.clarify_question,
            "status": self.status.value,
            "last_active": self.last_active,
            "pending_question": self.pending_question,
            "user_id": self.user_id,
            "rounds": list(self.rounds),
            "sql": self.sql,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """从 to_dict() 输出还原会话状态（缺失字段取默认值）"""
        if not isinstance(data, dict):
            return cls()
        filters = QueryFilters(**data.get("filters") or {})
        try:
            status = ClarifyStatus(data.get("status", ClarifyStatus.READY.value))
        except ValueError:
            status = ClarifyStatus.READY
        return cls(
            filters=filters,
            missing_fields=list(data.get("missing_fields") or []),
            history=list(data.get("history") or []),
            clarify_question=data.get("clarify_question"),
            status=status,
            last_active=float(data.get("last_active") or time.time()),
            pending_question=data.get("pending_question"),
            user_id=str(data.get("user_id") or "default"),
            rounds=list(data.get("rounds") or []),
            sql=str(data.get("sql") or ""),
        )
