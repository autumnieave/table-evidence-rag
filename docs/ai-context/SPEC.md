# SPEC.md — 功能规格（AI 可读版）

## 1. 三种链路模式

三种模式共用配置 `RAGConfig` 与 Qdrant 集合（`QDRANT_COLLECTION_NAME`，线上为正式数据全量语料 `research_reports_v3_full`），在 `RAGPipeline.query()` 中按开关分支执行。

| 模式 | 开关 | 输入 | 处理流程 | 输出 |
| --- | --- | --- | --- | --- |
| 手写链路 | `chain off` + `langchain off`（默认） | `question: str` | `EmbeddingClient` 生成查询向量 → `QdrantClientWrapper.search_similar`（K=50）→ `RerankClient` 精排（TopN=10）→ `LLMGenerator` 生成 | `dict{content: str, image: list, references: list[dict]}` |
| LangChain 检索器 | `langchain on` | `question: str` | `get_vectorstore()`（QdrantVectorStore）→ `similarity_search_with_score(question, k=50)` → 转旧检索器兼容格式 → 精排 → 生成 | 同上 |
| LCEL 完整链路 | `chain on` | `question: str` | `LangChainRAGChain.invoke`：retriever（复用 LangChain 检索器）→ `format_docs` → `ChatPromptTemplate` → `ChatOpenAI`（enable_thinking=False）→ `StrOutputParser` | `dict{content: str, image: [], references: []}` |

关键点：
- LCEL 链路跳过手写 Embedding/精排/生成逻辑，直接由链内 retriever 检索。
- `query()` 返回 dict，交互层按 `content` 打印、按 `references` 输出引用；调用方需兼容 `dict` 与 `str` 两种返回（历史版本返回过 str）。
- 手写与 LangChain 检索器共用 `_parse_filters_with_llm`（软过滤，不硬过滤）。
- `tools/data_scripts/` 中的脚本（pdf处理+校验入库/重抽取/batch_test/list_files）为独立数据处理工具，**不参与在线问答链路**，勿在主流程中 import。

## 2. Agent 工具清单（`tools/tools_registry.py` + `agents/planner.py`）

`AgentPlanner.execute(user_query, history, user_id, verbose)` 使用 Function Calling，最多 **10 轮** 工具调用，最终输出 JSON：`{content, image[], references[]}`；超轮数时返回兜底 JSON。

| 工具名 | 适用场景 | 输入参数 | 输出格式 |
| --- | --- | --- | --- |
| `search_reports` | 研报知识库检索：归因分析、公司评价、政策解读 | `{query: str}`（含主体、时间、问题焦点） | `rag.query()` 的 dict 序列化 JSON 字符串（含 content/image/references） |
| `query_financial_and_visualize` | 结构化财务数据查询 + 图表生成（可查业绩指标/资产负债表/现金流量表/利润表字段） | `{query: str}` | 问题残缺→返回反问文本；完整→生成 SQL 查询 + ECharts 图表，返回 `{content, image[]}` JSON 字符串 |

注意：
- 财务字段白名单定义在 `query_financial_and_visualize` 的 description 中（约 60+ 字段，含 `eps`、`roe`、`net_profit_yoy_growth` 等）。
- 工具底层走 `tools.native_financial.native_financial_query`（原生 SQL：生成→MySQL 执行→分析→ECharts）；异常时 `call_financial_chatflow` 返回 `{"content": "查询失败: ...", "image": []}`，不抛异常。
- Agent 模式入口：交互命令 `agent on` 直接启用，`RAGPipeline.agent_query()` 维护会话状态并调用 `agent_planner.execute()`。
- **规划器后端**：`AGENT_PLANNER_BACKEND`（代码默认 `handwritten` 自研 `AgentPlanner`；`langgraph` 为 LangGraph 后端，同 prompt/同 tools/同输出契约）；交互命令 `planner handwritten|langgraph` 切换，`status` 显示当前后端。**当前运行环境通过 `.env` 切换为 `langgraph` + `AGENT_LANGGRAPH_MULTI_AGENT=true`，即 supervisor-workers 多 Agent 为主链路，`handwritten` 为回退路径。** 对照口径见 `docs/评估报告/LangGraph对照.md`。
- **SQL 守卫**：SQL 生成经 `tools/sql_guard.py` 静态校验（表名/别名/字段归属/子查询表名 + 全角标点）+ MySQL 编译终审，失败把错误与字段建议拼回问题重问；配置 `AGENT_SQL_VALIDATE`（默认开）/ `AGENT_NATIVE_RETRY`（默认 2 次）/ `MYSQL_*`。
- **Agent 思考模式**：`AGENT_ENABLE_THINKING`（默认 `false`，qwen3.5-plus 推理模型必须关闭，避免耗尽 max_tokens）。

## 3. 交互命令（简要版）

| 命令 | 作用 |
| --- | --- |
| 直接输入问题 | 普通 RAG 检索问答 |
| `agent on` / `agent off` | 开启/关闭 Agent 多步推理 |
| `multi-turn on` / `multi-turn off` | 开启/关闭多轮澄清对话（缺失字段自动追问） |
| `langchain on` / `langchain off` | 切换 LangChain 向量检索器 |
| `chain on` / `chain off` | 切换 LCEL 完整链路 / 手写链路 |
| `hybrid on` / `hybrid off` | 切换混合检索（向量 + BM25 RRF，默认开） |
| `planner handwritten` / `planner langgraph` | 切换 Agent 规划器后端（自研 / LangGraph 实验） |
| `status` | 查看各模式开关状态 |
| `rebuild` | 强制重建索引（需确认） |
| `addstock` / `addindustry` | 增量插入个股 / 行业研报 |
| `new` | 开启新话题（重置 ConversationState） |
| `quit` / `exit` / `q` | 退出 |

### 命令行参数（简要）

| 命令 | 作用 |
| --- | --- |
| `python rag_全流程构建.py --build` | 仅构建索引 |
| `python rag_全流程构建.py --rebuild` | 强制重建索引（清空现有数据） |
| `python rag_全流程构建.py --query "问题"` | 单次查询（非交互） |
| `python rag_全流程构建.py --add-stock` | 增量插入个股研报 |
| `python rag_全流程构建.py --add-industry` | 增量插入行业研报 |

## 4. 其他功能规格要点

- **多轮澄清**：`conversational_query(user_input)` 用 LLM 解析过滤条件→检查 `stock_name` 等必填字段→缺失时返回澄清问题（`ClarifyStatus.NEED_CLARIFY`），补齐后再检索。
- **增量索引**：`add_new_stock_reports()` / `add_new_industry_reports()` 按目录读取新文档去重插入；`build_index(force_rebuild)` 全量构建/重建。
- **表格聚合**：分块阶段表格行带 `is_table_row` + `parent_id`；检索命中表格行时 `_aggregate_parent_table` 拉全父表；非表格行批量摘要（`_generate_summaries`）。
- **Web API**：`app/api.py`（`uvicorn app.api:app`）提供 `POST /chat`、`POST /chat/stream`（SSE：meta/content/done/error）、`GET /health`、静态 `/result`。
