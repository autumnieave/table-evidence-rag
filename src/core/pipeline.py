"""core/pipeline.py - 装配中枢（2026-08-22 入口收敛 #4）

所有入口（FastAPI app.api / CLI scripts.interactive / 回归脚本）统一经
get_pipeline() / get_config() 装配，避免各入口各自组装配置与依赖。

- get_config(): 全局配置单例（复用 config.rag_config 单例，含 validate）
- get_pipeline(): RAGPipeline 懒加载单例（线程安全）
"""

import threading
from typing import Optional

from config.rag_config import RAGConfig, get_config as _get_config
from pipelines.rag_pipeline import RAGPipeline

__all__ = ["RAGConfig", "RAGPipeline", "get_config", "get_pipeline"]

_lock = threading.Lock()
_pipeline: Optional[RAGPipeline] = None


def get_config() -> RAGConfig:
    """获取全局配置单例（复用 config.rag_config 单例）"""
    return _get_config()


def get_pipeline() -> RAGPipeline:
    """懒加载全局 RAGPipeline 单例（线程安全，重复调用返回同一实例）"""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = RAGPipeline(get_config())
    return _pipeline
