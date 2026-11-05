# 复杂表格 Agentic RAG：解析 → 检索 → Agent 证据治理 → 评测

> **30 秒速览**
> - **业务场景**：财务研报与财报的复杂表格 RAG，核心解决“表格被切碎”与“数字幻觉无法溯源”的痛点。
> - **硬核指标**：端到端 **P50 从 22.3s 降至 0.26s（提速 85.9 倍）**；表级 **Hit@10 达 87.5%**（MRR 提升至 0.7326）；Claim 支持率 **98.36%（120/122）**。
> - **架构亮点**：自研 Table Grid 解析、Schema-Aware 前置硬过滤、LangGraph 混合工具调度、Semantic Cache 降本增效、Redis 分布式基建。

本项目以结构保留解析、行级事实、混合检索、Agent 证据选择与 Claim 级评测构成闭环。不仅跑通了算法链路，还完成了服务化交付与高并发降本观测。

**技术栈**：Python · FastAPI (SSE) · LangGraph · Qdrant · Redis · Semantic Cache · Cross-Encoder · pytest (1086 单测)

[![CI](https://github.com/autumnieave/table-evidence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/autumnieave/table-evidence-rag/actions/workflows/ci.yml) [![tests](https://img.shields.io/badge/1086%20tests-passing-brightgreen)]() [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## 核心指标与工程收益

- **解析与表征**：受控 Gold Set TEDS 1.0（解决跨页/合并单元格），碎片 Chunk 减少 99.4%，建立 Cell-Level Anchors 支撑数字级溯源。
- **检索与重排**：针对跨期/跨公司噪声，引入 Schema-Aware 前置硬过滤与 Cross-Encoder 两阶段精排，表级 Hit@10 从 25% 提升至 87.5%，MRR@10 从 0.1292 提升至 0.7326。
- **降本与高并发**：引入 Semantic Cache 拦截重复 Query，免去昂贵的 LLM 生成 Token，端到端 P50 从 22.3s 降至 0.26s（提速 85.9 倍）；基于 Redis+Lua 实现分布式滑动窗口限流。
- **工程闭环**：基于 Docker Compose 一键编排交付；打通 SSE 流式输出（首字延迟从 38.3s 降至 0.674s）；配套 1086 条离线单测，CI 自动执行。

## 架构图

```mermaid
flowchart LR
    subgraph D[数据管道层]
        D1["研报 Markdown / 财报 PDF"] --> D2["MinerU 解析<br/>Table Grid 结构化"]
        D2 --> D3["结构化 Chunk + 元数据<br/>行级事实 + Cell-Level Anchors"]
    end

    subgraph R[检索层]
        R1["Query 理解 + 意图路由"] --> R2["Dense + Sparse + RRF 混合召回"]
        R2 --> R3["Schema-Aware 前置硬过滤"]
        R3 --> R4["qwen3-rerank（线上）<br/>Cross-Encoder 两阶段精排"]
    end

    subgraph A[Agent 与工具调度层]
        A1["LangGraph ReAct 状态机"] --> A2["研报 RAG Tool"]
        A1 --> A3["SQL Tool (多层安全拦截与柔性审计)"]
        A2 --> A4["证据动作：重写 / 换表 / 结构化拒答"]
        A3 --> A4
    end

    subgraph I[基建与可观测层]
        I1["Trace 链路追踪"] --> I2["Semantic Cache (Redis 向量检索)"]
        I2 --> I3["Redis+Lua 分布式限流"]
        I3 --> I4["SSE 流式输出 / FastAPI"]
    end

    D --> R --> A --> I
```

## 核心能力深度拆解

1. **数据解析与结构化**：
   - 自研 Table Grid 解析做跨页拼接与 Merged-Cell 语义恢复。
   - 引入 Parent Document 回拉机制，行级命中后向上拉取整表上下文，杜绝“表头丢失”。
2. **混合检索与过滤精排**：
   - 基线 Dense+Sparse+RRF 受噪声干扰严重，引入 Schema 实体与年份的强制过滤，大幅收敛候选池。
   - 线上用 qwen3-rerank 对 Top-50 结果重排，压住长尾干扰（消融实验基于 BGE-Reranker，见边界声明）。
3. **Agent 证据治理与状态路由**：
   - 基于 LangGraph 构建状态机，当数值口径（指标/期间/单位）不匹配时触发重写/拒答。
   - 引入 Intent Routing，将精确查数分发至带安全护栏的 SQL Tool，复杂推理走 RAG。
   - 引入 Targeted Retry（P50 388ms）替代全局盲搜。
4. **LLM 评测与 Faithfulness 纠偏**：
   - 搭建不合成单一分数的 RAG 四层评测矩阵（引用/答案/对抗/检索）。
   - 引入 Atomic Claim 分解与规则级证据定位，通过数值单位容差消除比对假阴性。严格核算 Claim 支持率达 98.36%（120/122）。
5. **高并发基建与降本观测**：
   - **Semantic Cache**：利用 Embedding 相似度在 Redis 中匹配同义 Query，命中后直接返回，免去 LLM 生成，速度提升 85.9 倍。
   - **状态外置**：将 LangGraph Checkpoint 由 SQLite 迁移至 Redis，支持多副本无状态扩展与断线重连。

## 快速开始

### 最短验证（零依赖离线单测）

```bash
pip install -r requirements.txt
python -m pytest tests/ -q    # 1086 个离线单测，无需 API Key / 数据库
```

### 完整服务运行

#### 1. 环境准备

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
DASHSCOPE_API_KEY=sk-xxx        # 阿里云百炼（必填）
MYSQL_HOST=127.0.0.1            # 可选：仅 Agent 的 SQL Tool 使用
```

#### 3. 一键编排启动（推荐）

项目提供了 `docker-compose.yml`，可一键启动 Qdrant + Redis + FastAPI 后端：

```bash
docker compose up -d --build
# 访问 API 文档：http://localhost:18000/docs
```

*(注：首次启动请确保在 `cli.py` 或 API 中触发过一次 `--build` 索引构建)*

## 测试与评估

```bash
# 运行全部单测 (1086 passed)
python -m pytest tests/ -q

# 运行评测闭环 (需本地 golden 数据支持)
python -m eval citation           # RAG-L1 引用核验
python -m eval challenge --run    # RAG-L3 对抗挑战集
```

## API 契约与 SSE 流式返回

后端对外只暴露两个问答入口：`POST /chat`（非流式）与 `POST /chat/stream`（SSE）。两者共用同一份请求模型，错误语义、意图路由与降级行为保持一致。

### 1. 请求模型

`ChatRequest`（Pydantic，`mode` 为严格枚举）：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `question` | `string` | 是 | - | 用户问题原文 |
| `user_id` | `string` | 否 | `"default"` | 会话、限流、SSE 断点续传的隔离键 |
| `mode` | `Literal["rag", "agent"]` | 否 | `"rag"` | `rag`=研报 RAG；`agent`=LangGraph 多工具编排。其他值或 `null` 返回 422 |

`POST /chat` 的响应由 `ChatResponse` 约束：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `content` | `string` | 最终答案（SQL 不可用时为结构化拒答话术） |
| `image` | `string[]` | 图表/图片地址 |
| `references` | `object[]` | 引用条目，含 `chunk_type`、`markdown`、`chunk_index`、`citation` 等 |
| `chart_json` | `object?` | ECharts 配置（仅财务查询可能返回） |
| `route_type` | `"financial" \| "research" \| "mixed"?` | 意图路由结果 |

### 2. SSE 事件类型与时序

每个 SSE 帧的格式固定为：

```text
id: <自增整数>
data: {"type": "...", ...}

```

`id` 只出现在 SSE 的 `id:` 行，不重复写入 JSON 载荷；客户端据此实现断点续传。

| 事件 | 载荷字段 | 含义 |
| --- | --- | --- |
| `stage` | `stage` | Agent 阶段进度（如 `query_financial`、`search_reports`、`generate`） |
| `content` | `text` | 增量文本片段；客户端按到达顺序拼接 |
| `meta` | `references`、`image`、`chart_json`、`cache_hit`、`cache_layer`、`route_type`、`semantic_similarity?` | 引用、缓存与路由元信息 |
| `final` | 无 | Agent 草稿流结束，随后会推送最终答案的 `content` |
| `done` | 无 | 本次流正常结束 |
| `error` | `message`、`retryable` | 生成失败；`retryable=true` 表示可重试（超时/限流/网络），`false` 表示参数或业务错误 |

时序（按实际实现）：

- **RAG 未命中缓存**：`content...` → `meta` → `done`（大模型生成完毕后，核验器才会下发引用 `meta`）
- **精确/语义缓存命中**：`meta` → `content...` → `done`（无生成过程，直接返回缓存结果）
- **Agent 多工具**：`stage...` → `content...`（子 Agent 草稿）→ `stage(generate)` → `meta` → `final` → `content...`（终稿）→ `done`
- **失败**：`...` → `error(retryable=...)`，流随后关闭

### 3. 真实报文示例

`meta` 事件（表格引用保留结构化 Markdown 与块位置）：

```text
id: 42
data: {"type":"meta","image":[],"references":[{"paper_path":"data_corpus/全部数据/正式数据/附件5：研报数据/个股研报/片仔癀/片仔癀2023年年度报告.md","text":"肝病用药毛利率 67.9%，上年同期 81.8%","chunk_type":"table","markdown":"| 项目 | 2023 | 2022 |\n|---|---|---|\n| 营业收入(亿元) | 100.4 | 88.2 |\n| 肝病用药毛利率 | 67.9% | 81.8% |","markdown_truncated":false,"chunk_index":17,"chunk_total":834,"text_total":74,"text_truncated":false,"citation":{"status":"fuzzy","located":"data_corpus/.../片仔癀2023年年度报告.md","nums":10,"num_hit":10,"num_ratio":1.0,"unhit":[]}}],"chart_json":null,"cache_hit":false,"cache_layer":null,"route_type":"research"}

```

`content` / `done` 事件：

```text
id: 43
data: {"type":"content","text":"片仔癀2023年肝病用药毛利率为"}

id: 44
data: {"type":"done"}

```

`error` 事件：

```text
id: 45
data: {"type":"error","message":"upstream timeout after 30s","retryable":true}

```

### 4. 错误处理与降级机制

| 场景 | 系统行为 | 客户端可见 |
| --- | --- | --- |
| MySQL schema/连接加载失败 | `native_financial` 捕获后返回 `build_refuse_system_unavailable` 结构化拒答，不抛 500 | HTTP 200 + 可读拒答话术；流式 `meta` 按实际阶段标记 `route_type="financial"` |
| MySQL 建连超时 | 单次连接 `connect_timeout=5s`，失败即降级为结构化拒答 | HTTP 200 + 拒答话术 |
| SQL 执行过长 | 会话级 `MAX_EXECUTION_TIME=15000ms`，校验器返回错误并由财务 Agent 转为拒答 | HTTP 200 + 拒答话术 |
| 多 Agent 单个子任务异常 | 并行子任务各自捕获异常，其余子任务继续；全部为空时返回“抱歉，未能完成查询。” | HTTP 200 + 兜底文案 |
| Agent 汇总 JSON 非法 | 契约闸门拒绝透传原始 JSON，替换为可读兜底并记录格式事件 | HTTP 200 + 兜底文案 |
| 网络超时 / 上游 429 / 5xx | SSE 推送 `error`，`retryable=true` | 可安全重试 |
| 参数缺失 / `mode` 非枚举 | HTTP 422（FastAPI 校验） | 不可重试，需修正请求 |
| 业务拒答（口径不匹配/无数据） | 正常返回结构化拒答文案，不作为异常抛出 | HTTP 200 + 拒答话术；若以 `error` 帧透传则 `retryable=false` |
| 用户级限流（默认 20 req/min） | 返回 429，并带 `Retry-After` 秒数 | 按 `Retry-After` 退避 |

### 5. SSE 断点续传

- 每条事件由服务端分配自增 `id`；同一 `user_id` 只保留最新一条流。
- 重连时携带请求头 `Last-Event-ID: <id>`，服务端按 `user_id + question + mode` 指纹匹配。
- 指纹匹配且 `id` 未越界：从 `id + 1` 继续推送。生成与 HTTP 连接解耦，断线期间事件仍会写入存储。
- 指纹不匹配、无记录或 `id` 越界：视为新请求，从头生成（旧流记录被覆盖）。
- 存储优先 Redis（`sse:resume:{user_id}:meta` + `sse:resume:events:{stream_id}`），TTL 默认 1800 秒；Redis 不可用时自动降级进程内内存，接口行为不变。
- 进程重启后若只剩历史事件，服务端会先回放已缓存事件，再发送 `retryable=true` 的中断错误，避免客户端静默卡死。

相关环境变量：

```bash
SSE_RESUME_ENABLED=true
SSE_RESUME_BACKEND=redis          # redis | memory
SSE_RESUME_REDIS_URL=redis://localhost:6379/3
SSE_RESUME_TTL_SECONDS=1800
```

## 边界与 Trade-off 声明

1. **SQL Tool 定位**：SQL 是 Agent 的受控 Tool，主要用于精确查数。我们通过多层安全拦截与柔性审计确保安全，但不代表 SQL 返回结果 100% 业务正确。
2. **精排模型选择**：消融实验基于本地 BGE-Reranker，受限于单机内存；线上为了低延迟切换为云端 `qwen3-rerank` API，以保障吞吐与延迟。
3. **评测集性质**：24 题/22 题盲测集用于消融方向验证与指标纠偏，属于小样本，不作为统计学意义上的绝对泛化能力证明。
4. **缓存降级边界**：语义缓存采用内存 O(n) 余弦相似度匹配，在 <1000 条缓存时性能极佳（约 226ms 内），若大规模上生产将替换为 RediSearch 或专用向量索引。Checkpoint 强依赖 Redis Stack。
5. **MySQL 降级边界**：MySQL 缺失或凭据错误时，系统会自动拦截并返回结构化拒答（HTTP 200），不影响 RAG 主流程；SQL Tool 的失败不会升级为 500。

## License

MIT License，见 [LICENSE](LICENSE)。
