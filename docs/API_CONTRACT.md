# API 契约与 SSE 流式返回

本文档是 README 的接口细则，覆盖请求/响应模型、SSE 事件时序、真实报文、错误降级与断点续传。

后端对外只暴露两个问答入口：`POST /chat`（非流式）与 `POST /chat/stream`（SSE）。两者共用同一份请求模型，错误语义、意图路由与降级行为保持一致。

## 1. 请求模型

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

## 2. SSE 事件类型与时序

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

## 3. 真实报文示例

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

## 4. 错误处理与降级机制

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

## 5. SSE 断点续传

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
