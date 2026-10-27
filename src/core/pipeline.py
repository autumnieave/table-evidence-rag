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

__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "get_config",
    "get_pipeline",
    "get_multi_stage_runtime",
]

_lock = threading.Lock()
_pipeline: Optional[RAGPipeline] = None
_multi_stage_runtime = None


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


def get_multi_stage_runtime():
    """懒加载多阶段检索运行时单例（表级 BM25 常驻 + 词表自动加载）。"""
    global _multi_stage_runtime
    if _multi_stage_runtime is None:
        with _lock:
            if _multi_stage_runtime is None:
                from core.multi_stage_runtime import build_multi_stage_runtime

                _multi_stage_runtime = build_multi_stage_runtime(get_config())
    return _multi_stage_runtime
