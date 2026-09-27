"""app/schemas.py —— API 请求/响应模型"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """对话请求：mode 为 rag（默认）或 agent"""

    question: str
    user_id: Optional[str] = "default"
    mode: Optional[str] = "rag"  # "rag" 或 "agent"


class ChatResponse(BaseModel):
    """对话响应（兼容字符串返回）"""

    content: str
    image: List[str] = []
    references: List[dict] = []
    chart_json: Optional[Dict] = None


class ClarifyRequest(BaseModel):
    """多轮澄清请求"""

    input: str
    user_id: Optional[str] = "default"
