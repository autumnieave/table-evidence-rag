# TECH_NOTES.md — 技术笔记

## 一、开发中解决的关键问题

### 1. DashScope Embedding SDK 版本不兼容（`input.contents` 报错）
- **现象**：`dashscope.TextEmbedding.call(input=texts)` 报 `Value error, contents is neither str nor list of str.: input.contents`；后改 HTTP API 直接传数组又报 `The input parameter requires json format`。
- **根因**：SDK 与 text-embedding-v2 接口不匹配；且 HTTP API 的 `input` 必须是对象。
- **解决**：`embeddings/client.py` 用 `requests.post` 直连，payload 为 `{"model": ..., "input": {"texts": texts}, "parameters": {"text_type": text_type}}`（`input.texts` 数组，`text_type` 用 document/query 区分）。
- **经验**：改第三方 API 调用前先 curl/单测确认 payload 格式，不要盲信 SDK。

### 2. LCEL 链 retriever 收到 dict 导致 embed_query 报错
- **现象**：`chain on` 提问时 embedding 报 `input.contents`，而单独调用 adapter 正常。
- **根因**：`{"context": retriever | format_docs, "question": ...}` 把整个输入 dict 传给 retriever，`embed_query(dict)` 序列化异常。
- **解决**：`chains/rag_chain.py` 改为 `RunnablePassthrough.assign(context=lambda i: format_docs(retriever.invoke(i["question"])))`，只传 question 字符串。
- **经验**：LCEL 中 retriever 的输入必须显式提取字段，避免 dict 透传。

### 3. QdrantVectorStore 维度验证触发 embed_documents 报错
- **现象**：`from_existing_collection` 初始化时调用 `embed_documents(["dummy_text"])` 校验维度，触发 DashScope 报错。
- **解决**：`config/langchain_config.py` 新增 `get_vector_store_direct(client, collection_name, embedding)`，直接用 `QdrantVectorStore(client=..., ...)` 构造，并传 `validate_collection_config=False`（该默认值会走 `_validate_collection_for_dense` 调 embed_documents）。
- **经验**：langchain-qdrant 构造参数 `validate_embeddings`/`validate_collection_config` 默认 True，绕过校验需显式关闭。

### 4. qwen3.5-plus 推理模型空回答
- **现象**：LCEL 链返回 `''`，`finish_reason=length`。
- **根因**：推理模型的思考内容消耗全部 `max_tokens`（2048），`content` 为空。
- **解决**：`get_chat_model()` 增加 `extra_body={"enable_thinking": False}`；备选方案为调大 `max_tokens`（8192 也可行）。
- **经验**：DashScope 兼容模式的推理模型需显式关思考或加足 token 预算。

### 5. Windows GBK 控制台 emoji 崩溃
- **现象**：交互模式打印 `✅/⚠️` 报 `'gbk' codec can't encode character '✅'`。
- **解决**：`scripts/interactive.py::main()` 对 stdout/stderr `reconfigure(encoding="utf-8", errors="replace")`。
- **经验**：Windows 下中文项目统一在入口设置 UTF-8 输出。

### 6. Windows PowerShell 管道中文编码损坏（排查"空回答/乱码"的陷阱）
- **现象**：通过 `@'...'@ | python -` 把中文脚本喂给 Python 时，中文全部变成 `?`，导致测试脚本中问题文本损坏、模型误以为输入是乱码而"返回空"或答非所问。
- **解决**：管道前设置 `$OutputEncoding = [System.Text.Encoding]::UTF8`；或把测试脚本写成 UTF-8 文件再 `python xxx.py` 运行。
- **经验**：排查"模型空回答"时，先确认输入文本是否在传输/落盘时被编码破坏，再怀疑模型本身。

### 7. 项目 git 化：大文件与密钥防护
- **背景**：项目总计约 7.3GB（<原始数据> 4.3GB、<数据目录> 1.3GB、qdrant_storage 946MB、.venv 353MB），且 `.env` 含 `DASHSCOPE_API_KEY`。
- **解决**：`git init` 前先创建 `.gitignore`（排除 `.env`、`.venv/`、`qdrant_storage/`、`<数据目录>/`、`<数据根目录>/`、`node_modules/`、`archive/` 等）；首次提交仅 92 个文件、0.47MB。
- **经验**：含密钥或大数据目录的项目，git 化第一步永远是 `.gitignore` 再 `git init`；提交前用 `git status --short` 复查。

## 二、性能数据

> 代码当前未内置耗时统计，以下为配置规模与参考量级，实测方法见下方可复用片段。

| 环节 | 规模/参考值 | 说明 |
| --- | --- | --- |
| 索引数据量 | 正式数据附件5 全量 473 篇（个股 160 / 行业 313） | 分块 57,178 块入库 Qdrant（`research_reports_v3_full`） |
| 向量维度 | 1536（text-embedding-v2，COSINE） | `RAGConfig.VECTOR_DIMENSION` |
| 向量召回 K | 50（`RETRIEVAL_K`） | 三条链路共用 |
| Rerank 精排 TopN | 10（`RERANK_TOP_N`） | qwen3-rerank，仅手写/LangChain 检索器链路 |
| 文本分块 | CHUNK_SIZE 1024 / overlap 100 | `data/splitter` 与 `config` 统一 100（2026-08-24 对比实验确认，见 `docs/overlap对比实验.md`） |
| Embedding 批处理 | batch_size=10，指数退避重试 3 次 | `embeddings/client.py` |
| Agent 工具调用上限 | 10 轮 | `agents/planner.py` |

**实测方法**：执行下方测速片段，可分别得到检索/精排/生成/索引耗时；建议在 README 或 CI 中沉淀基准值。

## 三、可复用代码片段

### 1. 分阶段测速（检索/精排/生成）

```python
import time
from config.rag_config import RAGConfig
from pipelines.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(RAGConfig())
question = "请介绍一下贵州茅台近期的业绩情况"

t0 = time.perf_counter()
vec = pipeline.embedding_client.generate_embeddings([question], text_type="query")[0]
t1 = time.perf_counter()
docs = pipeline.qdrant_client.search_similar(vec, top_k=pipeline.config.RETRIEVAL_K)
t2 = time.perf_counter()
ranked = pipeline.rerank_client.rerank(question, [d["text"] for d in docs])
t3 = time.perf_counter()
print(f"embed={t1-t0:.2f}s search={t2-t1:.2f}s rerank={t3-t2:.2f}s")
```

### 2. 直连 DashScope Embedding HTTP API（正确 payload）

```python
import requests

def embed_texts(texts, api_key, model="text-embedding-v2", text_type="document"):
    resp = requests.post(
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": {"texts": texts}, "parameters": {"text_type": text_type}},
        timeout=30,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["output"]["embeddings"]]
```

### 3. 关闭思考模式的 ChatOpenAI（避免空回答）

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=cfg.DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    max_tokens=2048,
    extra_body={"enable_thinking": False},  # 关键：关闭推理思考
)
```

### 4. 跳过维度验证的 QdrantVectorStore 构造

```python
from langchain_qdrant import QdrantVectorStore

vs = QdrantVectorStore(
    client=qdrant_client,          # qdrant_client.QdrantClient 实例
    collection_name="research_reports_v3_full",
    embedding=embedding_adapter,   # 实现 Embeddings 接口
    content_payload_key="content",
    validate_collection_config=False,  # 跳过 embed_documents 维度校验
)
```

### 5. 表格聚合判定（研报表格续页处理）

```python
def _should_aggregate_table(query: str) -> bool:
    """根据查询关键词判断是否需要将表格行合并为完整表格"""
    keywords = ["趋势", "变化", "对比", "历年", "过去", "逐年",
                "增长趋势", "下降趋势", "近三年", "近五年"]
    query_lower = query.lower()
    return True  # 当前实现为强制聚合（保留关键词逻辑便于后续收紧）
```
（对应 `utils/helpers.py` 实际实现；`_aggregate_parent_table` 通过 `parent_id` 拉取整表。）
