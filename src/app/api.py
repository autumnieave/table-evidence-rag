"""app/api.py —— FastAPI Web 入口（收敛自根目录 app.py）

启动: uvicorn app.api:app --reload --port 8000
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import ChatRequest, ChatResponse, ClarifyRequest
from utils.api_security import create_rate_limiter, extract_api_key, parse_api_keys
from utils.query_cache import make_cache_key
from utils.sse_stream import (
    SSEResumeStore,
    format_sse,
    parse_last_event_id,
    question_fingerprint,
)
from utils.observability import RequestTrace, finalize_and_log, reset_trace, set_trace
from core.pipeline import get_config, get_pipeline

logger = logging.getLogger(__name__)

# 初始化配置和 Pipeline（全局单例，避免重复加载）
config = get_config()
pipeline = get_pipeline()

# 启动预热：加载 BM25 索引与引用核验语料索引，避免首个请求冷启动
try:
    pipeline.build_bm25_index()
    print("[startup] BM25 索引预热完成")
except Exception as e:  # noqa: BLE001
    logger.warning(
        "bm25_unavailable_fallback scope=hybrid_chunk_bm25 collection=%s error=%s",
        config.QDRANT_COLLECTION_NAME,
        e,
    )
    print(f"[startup] BM25 索引预热失败（可忽略，将按需构建）: {e}")
if getattr(config, "MULTI_STAGE_RETRIEVAL_ENABLED", False):
    try:
        from core.pipeline import get_multi_stage_runtime

        runtime = get_multi_stage_runtime()
        info = (
            runtime.warmup()
            if getattr(config, "MULTI_STAGE_WARMUP_ENABLED", True)
            else runtime.as_dict()
        )
        print(f"[startup] 多阶段检索预热完成: {info}")
    except Exception as e:  # noqa: BLE001
        logger.warning("multi_stage_warmup_failed error=%s", e)
        print(f"[startup] 多阶段检索预热失败: {e}")
try:
    from pipelines.citation_validator import CitationValidator
    pipeline.citation_validator = CitationValidator(
        corpus_root=config.CITATION_CORPUS_ROOT,
        match_mode=config.CITATION_MATCH_MODE,
    )
    pipeline.citation_validator.build_index()
    print("[startup] 引用核验语料索引预热完成")
except Exception as e:  # noqa: BLE001
    print(f"[startup] 引用核验索引预热失败（可忽略）: {e}")

app = FastAPI(title="智能问数系统 API")
# 1. 挂载 result 目录（让图片可以通过 URL 访问）
app.mount("/result", StaticFiles(directory="result"), name="result")

# 允许跨域（前端开发时需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_origin_regex="http://localhost:\\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _observability_log_dir() -> str:
    """日志目录从配置读取；测试 stub 配置缺字段时用默认值。"""
    return getattr(config, "OBSERVABILITY_LOG_DIR", "训练结果数据/logs")


def _observability_enabled() -> bool:
    """日志开关默认开；测试可显式关闭。"""
    return bool(getattr(config, "OBSERVABILITY_LOG_ENABLED", True))


_RATE_LIMITER: Any = None


def _api_keys() -> set[str]:
    """读取允许的 API key；测试 stub 配置缺字段时返回空集。"""
    return parse_api_keys(getattr(config, "API_KEYS", "") or "")


def _api_auth_enabled() -> bool:
    """显式开关优先；未设置时仅当 API_KEYS 非空才启用鉴权。"""
    explicit = getattr(config, "API_AUTH_ENABLED", None)
    if explicit is not None:
        return bool(explicit)
    return bool(_api_keys())


def _api_exempt_paths() -> tuple[str, ...]:
    raw = getattr(config, "API_AUTH_EXEMPT_PATHS", "/health,/docs,/redoc,/openapi.json,/result")
    paths = {part.strip() for part in str(raw or "").split(",") if part.strip()}
    paths.update({"/health", "/docs", "/redoc", "/openapi.json", "/result"})
    return tuple(paths)


def _is_api_exempt(path: str) -> bool:
    for prefix in _api_exempt_paths():
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def _get_rate_limiter() -> Any:
    """按配置的每分钟限额懒加载；限额/后端变化时重建（redis 不可用自动降级 memory）。"""
    global _RATE_LIMITER
    limit = int(getattr(config, "API_RATE_LIMIT_PER_MINUTE", 10) or 10)
    backend = str(getattr(config, "API_RATE_LIMIT_BACKEND", "redis")).strip().lower()
    redis_url = str(getattr(config, "API_RATE_LIMIT_REDIS_URL", "") or "")
    signature = (max(1, limit), backend, redis_url)
    if _RATE_LIMITER is None or getattr(_RATE_LIMITER, "_signature", None) != signature:
        _RATE_LIMITER = create_rate_limiter(
            limit=limit,
            backend=backend,
            redis_url=redis_url,
            prefix="rl:api_key",
        )
        _RATE_LIMITER._signature = signature
    return _RATE_LIMITER


_USER_RATE_LIMITER: Any = None


def _get_user_rate_limiter() -> Any:
    """按 user_id 的每分钟限额懒加载（与 API key 限流相互独立）。"""
    global _USER_RATE_LIMITER
    limit = int(getattr(config, "USER_RATE_LIMIT_PER_MINUTE", 20) or 20)
    backend = str(getattr(config, "API_RATE_LIMIT_BACKEND", "redis")).strip().lower()
    redis_url = str(getattr(config, "API_RATE_LIMIT_REDIS_URL", "") or "")
    signature = (max(1, limit), backend, redis_url)
    if _USER_RATE_LIMITER is None or getattr(_USER_RATE_LIMITER, "_signature", None) != signature:
        _USER_RATE_LIMITER = create_rate_limiter(
            limit=limit,
            backend=backend,
            redis_url=redis_url,
            prefix="rl:user",
        )
        _USER_RATE_LIMITER._signature = signature
    return _USER_RATE_LIMITER


async def _enforce_user_rate_limit(user_id: Optional[str]) -> None:
    """按 user_id 做 20 req/min 限流；超限抛 429 并带 Retry-After。"""
    if not bool(getattr(config, "USER_RATE_LIMIT_ENABLED", True)):
        return
    limiter = _get_user_rate_limiter()
    allowed, retry_after = await asyncio.to_thread(
        limiter.allow, str(user_id or "default")
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(1, int(round(retry_after))))},
        )


def reset_api_rate_limiter() -> None:
    """清空 API key 与 user_id 限流窗口（测试/运维用）。"""
    global _RATE_LIMITER, _USER_RATE_LIMITER
    if _RATE_LIMITER is not None:
        _RATE_LIMITER.reset()
    if _USER_RATE_LIMITER is not None:
        _USER_RATE_LIMITER.reset()


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """API key 鉴权 + 每 key 10 req/min 滑动窗口限流。"""
    if not _api_auth_enabled() or _is_api_exempt(request.url.path):
        return await call_next(request)

    api_key = extract_api_key(request.headers)
    if not api_key:
        return JSONResponse(status_code=401, content={"detail": "missing api key"})
    if api_key not in _api_keys():
        return JSONResponse(status_code=403, content={"detail": "invalid api key"})

    allowed, retry_after = _get_rate_limiter().allow(api_key)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded", "retry_after_seconds": retry_after},
            headers={"Retry-After": str(max(1, int(round(retry_after))))},
        )
    request.state.api_key = api_key
    return await call_next(request)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """为每个请求生成 trace，并在响应完成时写一条 JSONL。

    - 普通响应：``call_next`` 返回后立即落盘；
    - 流式响应：包装 ``body_iterator``，在 SSE 真正结束后落盘，避免把
      “首字节返回时间”误当成端到端耗时；
    - 异常：记录 error 后继续抛出，不吞业务错误。
    """
    trace = RequestTrace(
        trace_id=uuid.uuid4().hex,
        path=request.url.path,
        method=request.method,
    )
    token = set_trace(trace)
    request.state.trace = trace
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        trace.error = str(exc)
        finalize_and_log(
            trace,
            log_dir=_observability_log_dir(),
            enabled=_observability_enabled(),
            total_ms=(time.perf_counter() - started) * 1000.0,
            status="error",
            status_code=500,
        )
        try:
            reset_trace(token)
        except Exception:  # noqa: BLE001
            set_trace(None)
        raise

    trace.status_code = response.status_code
    trace.status = "ok" if response.status_code < 500 else "error"

    if isinstance(response, StreamingResponse):
        original_iterator = response.body_iterator

        async def _logged_iterator():
            try:
                async for chunk in original_iterator:
                    yield chunk
            finally:
                finalize_and_log(
                    trace,
                    log_dir=_observability_log_dir(),
                    enabled=_observability_enabled(),
                    total_ms=(time.perf_counter() - started) * 1000.0,
                    status=trace.status,
                    status_code=trace.status_code,
                )
                # 流式 body 在独立 task 中执行，不能 reset middleware 的 token；
                # 直接清空当前 task 的 contextvar 即可。
                set_trace(None)

        response.body_iterator = _logged_iterator()
    else:
        finalize_and_log(
            trace,
            log_dir=_observability_log_dir(),
            enabled=_observability_enabled(),
            total_ms=(time.perf_counter() - started) * 1000.0,
            status=trace.status,
            status_code=trace.status_code,
        )
        try:
            reset_trace(token)
        except Exception:  # noqa: BLE001
            set_trace(None)

    return response


def _get_query_cache():
    """返回启用的查询缓存实例（QUERY_CACHE_ENABLED=false 或初始化失败时为 None）。"""
    if not getattr(config, "QUERY_CACHE_ENABLED", True):
        return None
    return getattr(pipeline, "query_cache", None)


_semantic_cache: Any = None


def _get_semantic_cache():
    """返回启用的语义缓存实例（SEMANTIC_CACHE_ENABLED=false 时为 None）。

    懒加载：避免默认关闭时也初始化 Redis / Embedding 依赖。
    """
    global _semantic_cache
    if not getattr(config, "SEMANTIC_CACHE_ENABLED", False):
        return None
    if _semantic_cache is None:
        from utils.semantic_cache import SemanticCache

        _semantic_cache = SemanticCache(
            embedding_client=pipeline.embedding_client,
            redis_url=getattr(config, "SEMANTIC_CACHE_REDIS_URL", "redis://localhost:6379/1"),
            threshold=float(getattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.92)),
            ttl=int(getattr(config, "SEMANTIC_CACHE_TTL", 86400)),
            max_entries=int(getattr(config, "SEMANTIC_CACHE_MAX_ENTRIES", 1000)),
            version=str(getattr(config, "SEMANTIC_CACHE_VERSION", "v1")),
        )
    return _semantic_cache


@app.get("/cache/stats")
async def cache_stats():
    """返回精确缓存与语义缓存的运行统计（语义缓存未启用时 semantic=None）。"""
    exact = _get_query_cache()
    semantic = _get_semantic_cache()
    return {
        "exact": exact.stats() if exact is not None and hasattr(exact, "stats") else None,
        "semantic": semantic.stats() if semantic is not None else None,
    }


def _with_default_route_type(result: Any, mode: str) -> Any:
    """RAG 同步接口补默认路由；agent 保持真实路由或 None。"""
    if not isinstance(result, dict):
        result = {"content": str(result or ""), "image": [], "references": []}
    if mode == "rag" and not result.get("route_type"):
        result = dict(result)
        result["route_type"] = "rag"
    return result


def _normalize_images(images: List[str]) -> List[str]:
    """
    将后端图片路径统一为可访问的相对 URL：
    - 本地文件（如 result/xxx.jpg 或绝对路径）→ /result/文件名
    - http(s) 完整 URL 原样保留（如外部直链）
    """
    normalized: List[str] = []
    for img in images or []:
        if not img:
            continue
        if str(img).startswith(("http://", "https://")):
            normalized.append(str(img))
        else:
            filename = Path(str(img)).name
            if filename:
                normalized.append(f"/result/{filename}")
    return normalized


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request, response: Response):
    trace = getattr(http_request.state, "trace", None)
    if trace is not None:
        response.headers["X-Trace-Id"] = trace.trace_id
        trace.mode = request.mode or "rag"
        trace.extra["question_chars"] = len(request.question or "")
        # 显式注入当前上下文：不依赖 ASGI 中间件的 contextvar 传播，
        # 否则 pipeline.query 里的 get_trace() 可能拿不到 trace，阶段耗时全为 null。
        set_trace(trace)
    await _enforce_user_rate_limit(request.user_id)
    cache = _get_query_cache()
    cache_key = None
    if cache is not None:
        cache_key = make_cache_key("chat", request.mode, request.user_id, request.question)
        hit = cache.get(cache_key)
        if hit is not None:
            if trace is not None:
                trace.cache_hit = True
            if isinstance(hit, str):
                hit = {"content": hit, "image": [], "references": []}
            if isinstance(hit, dict) and hit.get("image"):
                hit["image"] = _normalize_images(hit["image"])
            response.headers["X-Cache-Status"] = "exact_hit"
            return _with_default_route_type(hit, request.mode)

    semantic = _get_semantic_cache()
    semantic_ns = f"{(request.mode or 'rag')}:{request.user_id}"
    if semantic is not None:
        semantic_hit = await asyncio.to_thread(semantic.get, request.question, semantic_ns)
        if semantic_hit is not None:
            if trace is not None:
                trace.cache_hit = True
                trace.extra["cache_layer"] = "semantic"
                trace.extra["semantic_similarity"] = semantic.last_similarity
                trace.extra["semantic_lookup_ms"] = semantic.last_lookup_ms
            if isinstance(semantic_hit.get("image"), list):
                semantic_hit["image"] = _normalize_images(semantic_hit["image"])
            response.headers["X-Cache-Status"] = "semantic_hit"
            response.headers["X-Semantic-Similarity"] = str(semantic.last_similarity)
            return _with_default_route_type(semantic_hit, request.mode)

    try:
        if request.mode == "agent":
            result = pipeline.agent_query(request.question, user_id=request.user_id)
        else:
            result = pipeline.query(request.question, verbose=False)

        # 统一图片路径为可访问的相对 URL（/result/文件名；http(s) 直链原样保留）
        if isinstance(result, dict) and result.get("image"):
            result["image"] = _normalize_images(result["image"])

        # 兼容字符串返回
        if isinstance(result, str):
            result = {"content": result, "image": [], "references": []}
        result = _with_default_route_type(result, request.mode)
        if cache is not None and cache_key is not None:
            cache.set(cache_key, result)
        if semantic is not None:
            await asyncio.to_thread(semantic.set, request.question, result, semantic_ns)
        response.headers["X-Cache-Status"] = "miss"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class _LiveStream:
    """一条正在生成的流；后台任务独立于 HTTP 连接，断线后可继续写事件。"""

    def __init__(self, stream_id: str, user_id: str, question: str, mode: str) -> None:
        self.stream_id = stream_id
        self.user_id = user_id
        self.question = question
        self.mode = mode
        self.events: List[Dict[str, Any]] = []
        self.queue: Optional[asyncio.Queue] = asyncio.Queue()
        self.done = False
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.trace: Any = None


class _ReplayStream:
    """从 Redis/内存回放的流；没有活动任务，只能读取已落盘事件。"""

    def __init__(
        self,
        stream_id: str,
        user_id: str,
        question: str,
        mode: str,
        events: List[Dict[str, Any]],
        done: bool,
        error: Optional[str] = None,
    ) -> None:
        self.stream_id = stream_id
        self.user_id = user_id
        self.question = question
        self.mode = mode
        self.events = list(events or [])
        self.queue = None
        self.done = bool(done)
        self.error = error
        self.task = None
        self.trace = None


_SSE_STORE: Any = None
_LIVE_STREAMS: Dict[str, _LiveStream] = {}


def _get_sse_store() -> SSEResumeStore:
    """懒加载续传存储；Redis 不可用时自动降级进程内内存。"""
    global _SSE_STORE
    backend = str(getattr(config, "SSE_RESUME_BACKEND", "redis") or "memory").strip().lower()
    redis_url = str(getattr(config, "SSE_RESUME_REDIS_URL", "") or "")
    ttl_seconds = int(getattr(config, "SSE_RESUME_TTL_SECONDS", 1800) or 1800)
    signature = (backend, redis_url, ttl_seconds)
    if _SSE_STORE is not None and getattr(_SSE_STORE, "_signature", None) == signature:
        return _SSE_STORE
    if _SSE_STORE is not None:
        try:
            _SSE_STORE.reset()
        except Exception:  # noqa: BLE001
            pass
    redis_client = None
    if backend == "redis" and redis_url:
        try:
            import redis  # noqa: PLC0415

            redis_client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            redis_client.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning("sse_resume_redis_unavailable fallback=memory error=%s", exc)
            redis_client = None
    _SSE_STORE = SSEResumeStore(
        redis_client=redis_client,
        ttl_seconds=ttl_seconds,
    )
    _SSE_STORE._signature = signature
    return _SSE_STORE


def reset_sse_resume_store() -> None:
    """清空续传存储与活动任务（测试/运维用）。"""
    global _SSE_STORE
    for session in list(_LIVE_STREAMS.values()):
        task = getattr(session, "task", None)
        if task is not None and not task.done():
            task.cancel()
    _LIVE_STREAMS.clear()
    if _SSE_STORE is not None:
        try:
            _SSE_STORE.reset()
        except Exception:  # noqa: BLE001
            pass
    _SSE_STORE = None


_RETRYABLE_ERROR_MARKERS = (
    "timeout", "timed out", "超时", "rate limit", "too many requests", "429",
    "502", "503", "504", "connection reset", "connection aborted",
    "connection refused", "temporarily unavailable", "网络", "network",
    "redis", "pool exhausted",
)
_NON_RETRYABLE_ERROR_MARKERS = (
    "invalid", "validation", "bad request", "unauthorized", "forbidden",
    "not found", "unsupported", "参数", "校验失败", "拒答", "拒绝",
    "权限", "schema", "400", "401", "403", "404",
)


def _is_retryable_error(value: Any) -> bool:
    """网络超时/限流可重试；参数错误/业务拒答/权限问题不可重试。"""
    if isinstance(value, BaseException):
        text = f"{type(value).__name__}: {value}"
    else:
        text = str(value or "")
    lowered = text.lower()
    if any(marker in lowered for marker in _NON_RETRYABLE_ERROR_MARKERS):
        return False
    return any(marker in lowered for marker in _RETRYABLE_ERROR_MARKERS)


def _resolve_route_type(result: Any, mode: str, stages: List[str]) -> str:
    """优先使用结果里的 route_type；否则按 stage 推断。"""
    if isinstance(result, dict):
        value = str(result.get("route_type") or "").strip().lower()
        if value in ("financial", "research", "mixed"):
            return value
    seen = set(stages or [])
    if "query_financial" in seen and "search_reports" in seen:
        return "mixed"
    if "query_financial" in seen:
        return "financial"
    return "research"


def _meta_event(
    result: Any,
    mode: str,
    stages: List[str],
    cache_hit: bool,
    cache_layer: Optional[str],
    semantic_similarity: Optional[float] = None,
) -> Dict[str, Any]:
    """构造 meta 事件；缓存/非缓存路径共用同一字段集。"""
    if not isinstance(result, dict):
        result = {"content": str(result or ""), "image": [], "references": []}
    images = result.get("image") or []
    if not isinstance(images, list):
        images = [images]
    event: Dict[str, Any] = {
        "type": "meta",
        "image": _normalize_images(images),
        "references": result.get("references") or [],
        "chart_json": result.get("chart_json"),
        "cache_hit": bool(cache_hit),
        "cache_layer": cache_layer,
        "route_type": _resolve_route_type(result, mode, stages),
    }
    if semantic_similarity is not None:
        event["semantic_similarity"] = semantic_similarity
    return event


def _emit_event(session: Any, event: Dict[str, Any], store: SSEResumeStore) -> Optional[Dict[str, Any]]:
    """写 store + 入队；store 返回 None 时说明流已被新请求替换。"""
    stored = store.append(session.user_id, session.stream_id, event)
    if stored is None:
        return None
    session.events.append(stored)
    if session.queue is not None:
        session.queue.put_nowait(stored)
    return stored


async def _iter_stream(session: Any, after_id: int = 0):
    """先回放已有事件，再消费活动队列；保证不重复、不丢 done/error。"""
    sent = max(0, int(after_id or 0))
    for event in list(session.events):
        event_id = int(event.get("id") or 0)
        if event_id <= sent:
            continue
        yield format_sse(event)
        sent = event_id
        if event.get("type") in ("done", "error"):
            return
    if session.queue is None:
        if not session.done:
            yield format_sse({
                "id": sent + 1,
                "type": "error",
                "message": "流式生成已中断，请重试",
                "retryable": True,
            })
            yield format_sse({"id": sent + 2, "type": "done"})
        return
    while True:
        event = await session.queue.get()
        if event is None:
            break
        event_id = int(event.get("id") or 0)
        if event_id <= sent:
            continue
        sent = event_id
        yield format_sse(event)
        if event.get("type") in ("done", "error"):
            break


def _build_cached_session(
    store: SSEResumeStore,
    request: ChatRequest,
    result: Any,
    cache_layer: str,
    semantic_similarity: Optional[float] = None,
) -> _ReplayStream:
    """把精确/语义缓存命中回放成一条可续传的流。"""
    stream_id = store.start(request.user_id, request.question, request.mode)
    session = _ReplayStream(
        stream_id, request.user_id, request.question, request.mode, [], True
    )

    def add(event: Dict[str, Any]) -> None:
        stored = store.append(request.user_id, stream_id, event)
        if stored is not None:
            session.events.append(stored)

    add(_meta_event(result, request.mode, [], True, cache_layer, semantic_similarity))
    content = str(result.get("content") or "") if isinstance(result, dict) else str(result or "")
    for i in range(0, len(content), 4):
        add({"type": "content", "text": content[i:i + 4]})
    add({"type": "done"})
    store.mark_done(request.user_id, stream_id)
    return session


def _find_resumable(
    store: SSEResumeStore,
    request: ChatRequest,
    last_event_id: Optional[int],
) -> Optional[Any]:
    """按 user_id + question 指纹匹配断点；不匹配视为新请求。"""
    if last_event_id is None:
        return None
    meta = store.get_meta(request.user_id)
    if not meta:
        return None
    if meta.get("mode") != request.mode:
        return None
    if meta.get("fingerprint") != question_fingerprint(request.question, request.mode):
        return None
    if last_event_id > int(meta.get("last_id") or 0):
        return None
    stream_id = str(meta.get("stream_id") or "")
    if not stream_id:
        return None
    live = _LIVE_STREAMS.get(stream_id)
    if live is not None:
        return live
    events = store.load_events(request.user_id, last_event_id)
    return _ReplayStream(
        stream_id,
        request.user_id,
        request.question,
        request.mode,
        events,
        bool(meta.get("done")),
        meta.get("error"),
    )


async def _run_generation(
    session: _LiveStream,
    store: SSEResumeStore,
    cache: Any,
    cache_key: Optional[str],
    semantic: Any,
    semantic_ns: str,
) -> None:
    """后台生成：独立于 HTTP 连接，事件实时写 store，断线后可续传。"""
    loop = asyncio.get_running_loop()
    stages: List[str] = []

    def emit(event: Dict[str, Any]) -> None:
        _emit_event(session, event, store)

    def on_stage(stage: str) -> None:
        stages.append(str(stage))
        loop.call_soon_threadsafe(emit, {"type": "stage", "stage": str(stage)})

    def on_chunk(text: str) -> None:
        loop.call_soon_threadsafe(emit, {"type": "content", "text": str(text)})

    def run_sync() -> Any:
        streamed_parts: List[str] = []
        if session.mode == "agent":
            def agent_on_chunk(text: str) -> None:
                streamed_parts.append(str(text))
                on_chunk(str(text))

            result = pipeline.agent_query(
                session.question,
                user_id=session.user_id,
                on_stage=on_stage,
                on_chunk=agent_on_chunk,
            )
        else:
            def rag_on_chunk(text: str) -> None:
                streamed_parts.append(str(text))
                on_chunk(str(text))

            result = pipeline.query(
                session.question, verbose=False, stream_callback=rag_on_chunk
            )
        return result, "".join(streamed_parts)

    try:
        result, streamed_text = await asyncio.to_thread(run_sync)
        # 让 worker 线程通过 call_soon_threadsafe 排入的 stage/content 先落库，
        # 再发 meta/done；否则可能出现 done 先到、后续 content 被丢弃的竞态。
        await asyncio.sleep(0)
        if isinstance(result, str):
            result = {"content": result, "image": [], "references": []}
        if not isinstance(result, dict):
            result = {"content": str(result or ""), "image": [], "references": []}
        if result.get("image"):
            result["image"] = _normalize_images(result["image"])
        if cache is not None and cache_key is not None:
            try:
                cache.set(cache_key, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sse_exact_cache_set_failed error=%s", exc)
        if semantic is not None:
            try:
                await asyncio.to_thread(semantic.set, session.question, result, semantic_ns)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sse_semantic_cache_set_failed error=%s", exc)

        content = str(result.get("content") or "")
        emit(_meta_event(result, session.mode, stages, False, None))
        if streamed_text and content.strip() == streamed_text.strip():
            emit({"type": "done"})
        else:
            if streamed_text:
                emit({"type": "final"})
            for i in range(0, len(content), 4):
                emit({"type": "content", "text": content[i:i + 4]})
                await asyncio.sleep(0.008)
            emit({"type": "done"})
        store.mark_done(session.user_id, session.stream_id)
        if session.trace is not None:
            session.trace.status = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.exception("sse_generation_failed stream_id=%s", session.stream_id)
        emit({
            "type": "error",
            "message": str(exc),
            "retryable": _is_retryable_error(exc),
        })
        session.error = str(exc)
        store.mark_done(session.user_id, session.stream_id, error=str(exc))
        if session.trace is not None:
            session.trace.status = "error"
            session.trace.error = str(exc)
    finally:
        session.done = True
        if session.queue is not None:
            session.queue.put_nowait(None)
        _LIVE_STREAMS.pop(session.stream_id, None)


@app.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def chat_stream(request: ChatRequest, http_request: Request):
    """流式接口：SSE 事件（stage / content / meta / final / done / error）。

    - 每条事件带自增 ``id``；
    - 支持 ``Last-Event-ID``：同一 user_id + 同一 question 从断点回放；
    - 后台生成与 HTTP 连接解耦，断线后仍会继续写入续传存储。
    """
    trace = getattr(http_request.state, "trace", None)
    if trace is not None:
        set_trace(trace)
        trace.mode = request.mode or "rag"
        trace.extra["question_chars"] = len(request.question or "")
    await _enforce_user_rate_limit(request.user_id)

    store = _get_sse_store()
    last_event_id = parse_last_event_id(http_request.headers.get("Last-Event-ID"))
    resume_enabled = bool(getattr(config, "SSE_RESUME_ENABLED", True))
    session = _find_resumable(store, request, last_event_id) if resume_enabled else None
    resumed = session is not None

    if session is None:
        cache = _get_query_cache()
        cache_key = None
        cached_result = None
        cache_layer = None
        semantic_similarity = None
        if cache is not None:
            cache_key = make_cache_key("chat", request.mode, request.user_id, request.question)
            hit = cache.get(cache_key)
            if hit is not None:
                cached_result = hit
                cache_layer = "exact"

        semantic = None
        semantic_ns = f"{request.mode}:{request.user_id}"
        if cached_result is None:
            semantic = _get_semantic_cache()
            if semantic is not None:
                hit = await asyncio.to_thread(semantic.get, request.question, semantic_ns)
                if hit is not None:
                    cached_result = hit
                    cache_layer = "semantic"
                    semantic_similarity = getattr(semantic, "last_similarity", None)

        if cached_result is not None:
            if trace is not None:
                trace.cache_hit = True
                trace.extra["cache_layer"] = cache_layer
                if semantic_similarity is not None:
                    trace.extra["semantic_similarity"] = semantic_similarity
            session = _build_cached_session(
                store, request, cached_result, str(cache_layer), semantic_similarity
            )
        else:
            if trace is not None:
                trace.cache_hit = False
            stream_id = store.start(request.user_id, request.question, request.mode)
            session = _LiveStream(
                stream_id, request.user_id, request.question, request.mode
            )
            session.trace = trace
            _LIVE_STREAMS[stream_id] = session
            session.task = asyncio.create_task(
                _run_generation(session, store, cache, cache_key, semantic, semantic_ns)
            )

    after_id = last_event_id if resumed else 0

    async def generate():
        async for frame in _iter_stream(session, after_id):
            yield frame

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/clarify")
async def chat_clarify(request: ClarifyRequest, http_request: Request):
    """多轮澄清对话：自动提取过滤条件，缺字段时反问，补齐后执行检索生成答案"""
    await _enforce_user_rate_limit(request.user_id)
    trace = getattr(http_request.state, "trace", None)
    if trace is not None:
        trace.mode = "clarify"
        trace.extra["question_chars"] = len(request.input or "")
        set_trace(trace)
    try:
        reply, done = pipeline.conversational_query(request.input, user_id=request.user_id)
        return {
            "content": reply,
            "image": [],
            "references": [],
            "clarify_done": done,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health(http_request: Request):
    trace = getattr(http_request.state, "trace", None)
    if trace is not None:
        trace.mode = "health"
    return {"status": "ok", "collection": config.QDRANT_COLLECTION_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
