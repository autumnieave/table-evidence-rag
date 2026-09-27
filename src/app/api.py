"""app/api.py —— FastAPI Web 入口（收敛自根目录 app.py）

启动: uvicorn app.api:app --reload --port 8000
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import ChatRequest, ChatResponse, ClarifyRequest
from utils.query_cache import make_cache_key
from core.pipeline import get_config, get_pipeline

# 初始化配置和 Pipeline（全局单例，避免重复加载）
config = get_config()
pipeline = get_pipeline()

# 启动预热：加载 BM25 索引与引用核验语料索引，避免首个请求冷启动
try:
    pipeline.build_bm25_index()
    print("[startup] BM25 索引预热完成")
except Exception as e:  # noqa: BLE001
    print(f"[startup] BM25 索引预热失败（可忽略，将按需构建）: {e}")
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


def _get_query_cache():
    """返回启用的查询缓存实例（QUERY_CACHE_ENABLED=false 或初始化失败时为 None）。"""
    if not getattr(config, "QUERY_CACHE_ENABLED", True):
        return None
    return getattr(pipeline, "query_cache", None)

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
async def chat(request: ChatRequest):
    cache = _get_query_cache()
    cache_key = None
    if cache is not None:
        cache_key = make_cache_key("chat", request.mode, request.user_id, request.question)
        hit = cache.get(cache_key)
        if hit is not None:
            if isinstance(hit, dict) and hit.get("image"):
                hit["image"] = _normalize_images(hit["image"])
            return hit
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
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式接口：SSE 推送事件（stage / content / meta / done / error）。
    - rag：真实逐 token（检索/精排完成后立即输出生成内容）
    - agent：阶段事件（stage）即时反馈 + 完成后逐字输出，避免长任务等待期无反馈
    """

    async def generate():
        # 路线 1 缓存：命中直接回放结果（content/meta/done），跳过真实检索与生成
        cache = _get_query_cache()
        cache_key = None
        if cache is not None:
            cache_key = make_cache_key("chat", request.mode, request.user_id, request.question)
            hit = cache.get(cache_key)
            if hit is not None:
                content = hit.get("content", "") if isinstance(hit, dict) else str(hit or "")
                image = hit.get("image", []) if isinstance(hit, dict) else []
                references = hit.get("references", []) if isinstance(hit, dict) else []
                chart_json = hit.get("chart_json") if isinstance(hit, dict) else None
                meta = json.dumps(
                    {"type": "meta", "image": _normalize_images(image), "references": references, "chart_json": chart_json},
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
                        {"type": "meta", "image": _normalize_images(image), "references": references, "chart_json": chart_json},
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
                    data = json.dumps({"type": "error", "message": payload}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    break
        finally:
            task.cancel()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/clarify")
async def chat_clarify(request: ClarifyRequest):
    """多轮澄清对话：自动提取过滤条件，缺字段时反问，补齐后执行检索生成答案"""
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
async def health():
    return {"status": "ok", "collection": config.QDRANT_COLLECTION_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
