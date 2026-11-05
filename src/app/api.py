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


@app.post("/chat")
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
            if isinstance(hit, dict) and hit.get("image"):
                hit["image"] = _normalize_images(hit["image"])
            response.headers["X-Cache-Status"] = "exact_hit"
            return hit

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
            return semantic_hit

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
        if cache is not None and cache_key is not None:
            cache.set(cache_key, result)
        if semantic is not None:
            await asyncio.to_thread(semantic.set, request.question, result, semantic_ns)
        response.headers["X-Cache-Status"] = "miss"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """
    流式接口：SSE 推送事件（stage / content / meta / done / error）。
    - rag：真实逐 token（检索/精排完成后立即输出生成内容）
    - agent：阶段事件（stage）即时反馈 + 完成后逐字输出，避免长任务等待期无反馈
    """

    await _enforce_user_rate_limit(request.user_id)

    async def generate():
        trace = getattr(http_request.state, "trace", None)
        if trace is not None:
            set_trace(trace)
            trace.mode = request.mode or "rag"
            trace.extra["question_chars"] = len(request.question or "")
        # 路线 1 缓存：命中直接回放结果（content/meta/done），跳过真实检索与生成
        cache = _get_query_cache()
        cache_key = None
        if cache is not None:
            cache_key = make_cache_key("chat", request.mode, request.user_id, request.question)
            hit = cache.get(cache_key)
            if hit is not None:
                if trace is not None:
                    trace.cache_hit = True
                content = hit.get("content", "") if isinstance(hit, dict) else str(hit or "")
                image = hit.get("image", []) if isinstance(hit, dict) else []
                references = hit.get("references", []) if isinstance(hit, dict) else []
                chart_json = hit.get("chart_json") if isinstance(hit, dict) else None
                meta = json.dumps(
                    {"type": "meta", "image": _normalize_images(image), "references": references,
                     "chart_json": chart_json, "cache_hit": True, "cache_layer": "exact"},
                    ensure_ascii=False,
                )
                yield f"data: {meta}\n\n"
                for i in range(0, len(content), 4):
                    piece = content[i:i + 4]
                    data = json.dumps({"type": "content", "text": piece}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.008)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        # 语义缓存：与精确缓存同一回放协议（meta → content → done）
        semantic = _get_semantic_cache()
        semantic_ns = f"{(request.mode or 'rag')}:{request.user_id}"
        if semantic is not None:
            semantic_hit = await asyncio.to_thread(
                semantic.get, request.question, semantic_ns
            )
            if semantic_hit is not None:
                if trace is not None:
                    trace.cache_hit = True
                    trace.extra["cache_layer"] = "semantic"
                    trace.extra["semantic_similarity"] = semantic.last_similarity
                    trace.extra["semantic_lookup_ms"] = semantic.last_lookup_ms
                content = semantic_hit.get("content", "") if isinstance(semantic_hit, dict) else str(semantic_hit or "")
                image = semantic_hit.get("image", []) if isinstance(semantic_hit, dict) else []
                references = semantic_hit.get("references", []) if isinstance(semantic_hit, dict) else []
                chart_json = semantic_hit.get("chart_json") if isinstance(semantic_hit, dict) else None
                meta = json.dumps(
                    {"type": "meta", "image": _normalize_images(image), "references": references,
                     "chart_json": chart_json, "cache_hit": True, "cache_layer": "semantic",
                     "semantic_similarity": getattr(semantic, "last_similarity", None)},
                    ensure_ascii=False,
                )
                yield f"data: {meta}\n\n"
                for i in range(0, len(content), 4):
                    piece = content[i:i + 4]
                    data = json.dumps({"type": "content", "text": piece}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.008)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        queue: asyncio.Queue = asyncio.Queue()
        box: Dict[str, Any] = {}

        def on_chunk(text: str) -> None:
            queue.put_nowait(("content", text))

        def on_stage(stage: str) -> None:
            queue.put_nowait(("stage", stage))

        def run_query() -> None:
            try:
                streamed_parts: List[str] = []
                if request.mode == "agent":
                    def agent_on_chunk(text: str) -> None:
                        streamed_parts.append(text)
                        on_chunk(text)

                    result = pipeline.agent_query(
                        request.question,
                        user_id=request.user_id,
                        on_stage=on_stage,
                        on_chunk=agent_on_chunk,
                    )
                else:
                    def rag_on_chunk(text: str) -> None:
                        streamed_parts.append(text)
                        on_chunk(text)

                    result = pipeline.query(request.question, verbose=False, stream_callback=rag_on_chunk)
                box["streamed_text"] = "".join(streamed_parts)
                if cache is not None and cache_key is not None and isinstance(result, dict):
                    cache.set(cache_key, result)
                if semantic is not None and isinstance(result, dict):
                    semantic.set(request.question, result, semantic_ns)
                box["result"] = result
                queue.put_nowait(("finish", None))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                queue.put_nowait(("error", str(e)))

        task = asyncio.create_task(asyncio.to_thread(run_query))
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "stage":
                    data = json.dumps({"type": "stage", "stage": payload}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                elif kind == "content":
                    data = json.dumps({"type": "content", "text": payload}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                elif kind == "finish":
                    result = box.get("result") or {}
                    content = result.get("content", "") if isinstance(result, dict) else str(result or "")
                    image = result.get("image", []) if isinstance(result, dict) else []
                    references = result.get("references", []) if isinstance(result, dict) else []
                    chart_json = result.get("chart_json") if isinstance(result, dict) else None
                    meta = json.dumps(
                        {"type": "meta", "image": _normalize_images(image), "references": references,
                         "chart_json": chart_json, "cache_hit": False, "cache_layer": None},
                        ensure_ascii=False,
                    )
                    yield f"data: {meta}\n\n"
                    streamed_text = box.get("streamed_text") or ""
                    # 已真流式输出且与最终内容一致（rag 模式 / 单任务直出）：跳过重发，避免内容重复
                    if streamed_text and content.strip() == streamed_text.strip():
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break
                    # 流式输出过但最终内容不同（混合题：研报草稿 → 汇总终稿）：先发 final 让前端重置再重发终稿
                    if streamed_text:
                        yield f"data: {json.dumps({'type': 'final'})}\n\n"
                    # agent 全量结果一次性返回：按小块逐字输出，保留打字机效果
                    for i in range(0, len(content), 4):
                        piece = content[i:i + 4]
                        data = json.dumps({"type": "content", "text": piece}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                        await asyncio.sleep(0.008)
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                elif kind == "error":
                    if trace is not None:
                        trace.status = "error"
                        trace.error = str(payload)
                    data = json.dumps({"type": "error", "message": payload}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    break
        finally:
            task.cancel()

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
