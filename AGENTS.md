# AGENTS.md — AI 开发指南

本文件为 AI 编码代理提供项目速览与开发约束，修改代码前请先阅读。

## 项目一句话

RAG 金融研报智能问数系统：基于检索增强生成（RAG）的上市公司研报问答系统，支持手写 / LangChain 检索器 / LCEL 三种链路，Agent 多步推理、多轮澄清对话与增量索引更新。

## 模块结构速览（10 个核心模块）

| 模块 | 职责（一句话） |
| --- | --- |
| `config/` | 全局配置：`RAGConfig`（模型/路径/检索/引用校验参数，唯一配置源，含组件工厂与 `get_config()` 单例）；`EmbeddingClientAdapter` 已并入本模块（原兼容层 `langchain_config.py` 已删除） |
| `prompts/` | 唯一 Prompt 目录：`rag.py`（RAG 问答，手写/LCEL 同源）、`pipeline.py`（字段提取/摘要/图片检测）、`agent.py`（Agent system prompt）、`financial.py`（SQL 生成）/ `multi_agent.py`（supervisor-workers）、`fallback.py`（兜底话术）、`examples/`（few-shot 示例库）、`registry.json`（版本注册表，**唯一事实源**）；新增 Prompt 一律放此，修改后同步 registry 版本 |
| `core/` | 链路收敛接口：`interfaces.py`（`IRetriever/IReranker/IGenerator` 三协议）、`retrievers.py`（`HandwrittenRetriever` 自研基线=代码默认 / `HybridRetriever` 混合检索（`HYBRID_ENABLED` **默认 true**，即**当前默认使用**）/ `LangChainRetriever` 对照实验）、`rerankers.py` + `generators.py`（适配层） |
| `eval/` | 评估闭环：`golden.py`（golden set 版本化，`database/golden/`，本地资产不入库）、`runner.py`（`python -m eval` 统一入口：golden / sql / citation / report / **llm-judge / challenge / consistency**）、`metrics.py`（报告聚合）、`answer_keys.py`（答案先验三态登记）、`llm_judge.py`（LLM-as-judge）、`consistency.py`（同题一致率）、`retrieval_metrics.py`（Recall@K / MRR）、`challenge.py`（对抗挑战集 v2） |
| `data/` | 研报 Markdown 加载、Excel 元数据匹配、文本分块与 HTML/Markdown 表格抽取 |
| `embeddings/` | `EmbeddingClient`：通过 DashScope HTTP API 生成向量（text-embedding-v2，1536 维） |
| `vectorstore/` | `QdrantClientWrapper`：Qdrant 集合读写、检索、清空封装 |
| `chains/` | `LangChainRAGChain`（LCEL 完整链路）、`RerankClient`（qwen3-rerank 精排）；Prompt 已移至 `prompts/` |
| `llm/` | `LLMGenerator`：答案生成与流式输出（qwen3.5-plus） |
| `agents/` | `langgraph_multi_agent.py`（**当前主链路**：supervisor-workers 多 Agent，规划→财务/研报子 Agent 并行→汇总）、`langgraph_planner.py`（StateGraph 单 Agent 对照）、`planner.py`（`AgentPlanner`：Function Calling 多步推理，**回退路径**）；运行链路由 `AGENT_PLANNER_BACKEND` + `AGENT_LANGGRAPH_MULTI_AGENT` 决定（代码默认 handwritten，当前 `.env` 为 langgraph 多 Agent） |
| `memory/` | `ConversationState` / `ClarifyStatus`（多轮对话状态、澄清标记）+ `store.py`（记忆持久化：SQLite 默认 / Redis 可选，按 `user_id` 存取，TTL 过期） |
| `filters/` | `QueryFilters`：查询条件软过滤（匹配得分排序，不做硬过滤） |
| `pipelines/` | `RAGPipeline`：全流程编排（build_index / query / agent_query / conversational_query / 增量插入） |
| `scripts/` | 交互式问答入口：`interactive_mode` / `main`（含 CLI 参数解析） |
| `app/` | FastAPI 入口包：`api.py`（路由/SSE/静态挂载）、`schemas.py`（请求响应模型）；`uvicorn app.api:app` |
| `tools/` | Agent 工具注册表（`tools_registry.py`）、原生财务查询（`native_financial.py`：SQL 生成→MySQL 执行→分析→ECharts）、SQL 校验守卫（`sql_guard.py`）；`tools/typo_normalizer.py`（错别字归一化）；`tools/data_scripts/` 存放数据处理脚本（pdf处理+校验入库/重抽取/batch_test/list_files） |
| `utils/` | 通用工具：表格聚合、摘要生成、引用构建（`helpers.py`） |

### 补充登记（2026-09-10 核对：模块速览遗漏项）

| 路径 / 开关 | 说明 |
| --- | --- |
| `eval/answer_keys.py` | 答案先验登记（`has_data` / `no_data` / `unknown`）+ 只读复算，供 judge 判「应有数据却拒答」（B-36） |
| `eval/llm_judge.py` | LLM-as-judge 评测（`python -m eval llm-judge`，支持 `--answer-key` / `--no-prior` / `--rejudge`） |
| `eval/consistency.py` | 同题多次生成一致率（结构指纹 / 数值 IoU / 引用 Jaccard） |
| `eval/retrieval_metrics.py` | 检索质量指标 Recall@K / Precision@K / MRR（B-24A） |
| `eval/challenge.py` | 对抗挑战集 v2（18 条 / 5 类，`python -m eval challenge --run` / `--rejudge`） |
| `prompts/registry.json` | Prompt 版本注册表（唯一事实源；防漂移单测 `tests/test_prompt_registry.py`） |
| `prompts/fallback.py` | 兜底话术模板（refuse / suggest / human 三层，B-18） |
| `prompts/examples/` | few-shot 示例库（B-15 示例库 + B-16 三态开关：none/static/dynamic） |
| `agents/langgraph_planner.py` | LangGraph StateGraph 单 Agent 编排（对照实验） |
| `agents/langgraph_multi_agent.py` | **当前主链路**：supervisor-workers 多 Agent 协作 |
| `tools/typo_normalizer.py` | 错别字归一化（对抗挑战集 B-28 修复） |
| `utils/output_contracts.py` | 输出契约校验（B-17，四类校验 + 事件落盘） |
| `utils/query_cache.py` | SQLite 查询缓存（并行/缓存路线 1，2026-08-30） |
| `.github/workflows/ci-layered.yml` | 分层 CI（L1 必过 / L2 默认关 / L3 nightly 占位，B-21A） |
| `tests/test_api_contracts.py` | API 契约 + SSE 事件序列测试（B-19） |
| `AGENT_PLANNER_BACKEND` / `AGENT_LANGGRAPH_MULTI_AGENT` | 决定运行链路的两个 env 开关（代码默认 handwritten；当前 `.env` 为 langgraph + multi-agent=true） |

## 目录结构（2026-08 整理后）

- 源码包：`app/ core/ eval/ config/ data/ embeddings/ vectorstore/ chains/ llm/ agents/ memory/ filters/ pipelines/ scripts/ tools/ utils/`
- 运行核心：`rag_全流程构建.py`（CLI 启动器，委托 `scripts.interactive`）、`app/`（FastAPI 包，`uvicorn app.api:app` 启动）
- `notebooks/`：数据分析 Notebook（pdf解析 等）
- `database/`：SQL 建表脚本、数据 CSV
- `docs/`：项目文档与资料（论文/、问题记录/、评估报告/、课程作业/）
- `archive/`：历史归档（旧版程序、临时脚本、备份文件）——**不要向其中添加新代码**
- 运行时数据（不入 git）：`<数据目录>/ qdrant_storage/ result/ 汇总结果/ <结果数据>/ <数据根目录>/`

## 编码规范

- **Python 版本**：3.11（`.venv` 已配置）；Windows 下统一用 `.\.venv\Scripts\python` 执行。
- **类型注解**：所有公开函数/方法必须带类型注解（`-> List[Dict[str, Any]]` 等）。
- **docstring**：使用中文、`"""三引号"""` 风格，写清参数与返回值（参照现有模块）。
- **模块划分**：新增功能放入对应职责模块，禁止在主入口 `rag_全流程构建.py` 写业务逻辑。
- **链路接口**：新链路/新组件必须实现 `core/interfaces.py` 的三协议（`IRetriever/IReranker/IGenerator`），`pipelines/rag_pipeline.py` 的 `query()`/`conversational_query()` 统一走接口，禁止直连具体类。
- **评估入口**：回归/核验统一走 `python -m eval`（golden 版本化 + sql 套件 + citation + report），禁止另起散装评估脚本；新增评估套件放入 `eval/` 或登记到 `eval/runner.py` 的 `SQL_SUITES`。
- **会话记忆**：会话状态存取统一走 `RAGPipeline._load_conversation/_save_conversation/reset_conversation`（按 `user_id` 持久化），禁止直接改 `conversation_state` 绕过存储；新状态字段必须支持 `to_dict/from_dict` 序列化。
- **单元测试**：新增纯逻辑模块（校验器/分块/解析器/存储层等）必须配 `tests/` 用例，测试不得依赖外部服务（Qdrant/MySQL/LLM）；改代码后本地跑 `python -m pytest tests/ -q`，CI 在 `.github/workflows/ci.yml` 自动执行。
- **配置**：新增可调参数优先加在 `config/rag_config.py`（支持环境变量覆盖），不要在函数内硬编码。
- **编码**：所有源文件保持 UTF-8，不要引入 BOM。
- **脚本位置**：新增数据处理脚本放 `tools/data_scripts/`，禁止放根目录；临时调试脚本用 `_` 前缀（会被 `.gitignore` 忽略，不入库）。
- **Prompt 管理**：所有 Prompt 统一放 `prompts/` 包（RAG 问答 / 字段提取 / 摘要 / 图片检测 / Agent），禁止在业务代码内联新增 Prompt；修改模板后更新 `prompts/__init__.py` 的 `PROMPT_VERSION`。

## 修改代码时的注意事项

- **改 `pipelines/` 需同步检查 `scripts/interactive.py` 与 `app/api.py`**：它们直接调用 `RAGPipeline` 的方法与属性（`query`、`agent_query`、`build_index`、`agent_mode_enabled` 等），改名/删参会导致入口崩溃。
- **改 `config/rag_config.py` 的 `get_chat_model()`**：`qwen3.5-plus` 是推理模型，必须保留 `extra_body={"enable_thinking": False}`，否则思考过程会耗尽 `max_tokens` 导致空回答。
- **分块口径已统一为 overlap=100**（2026-08-24 对比实验确认最优：`docs/评估报告/overlap对比实验.md`）：`data/splitter.py` 默认与 `config/rag_config.py` 的 `CHUNK_OVERLAP` 一致（100），pipeline 显式传 config 值。改分块参数必须同步两处默认值并重建索引，否则新旧语料混用。
- **改 Embedding 调用**：DashScope v2 的 HTTP API payload 必须是 `{"input": {"texts": [...]}}`（`input` 是对象不是数组），参数用 `text_type`；不要改回 `dashscope.TextEmbedding.call`（SDK 版本不兼容）。
- **改 `chains/rag_chain.py`**：retriever 必须只接收 question 字符串（通过 `RunnablePassthrough.assign` 提取），不能把整个输入 dict 传给 retriever，否则 `embed_query` 会收到 dict 报错。
- **改 `pipelines/rag_pipeline.py` 的 `get_vectorstore()`**：必须走 `get_vector_store_direct`（带 `validate_collection_config=False`），不能改回 `from_existing_collection`（会触发 `embed_documents(["dummy_text"])` 维度验证报错）。
- **交互模式中的 `self`**：`interactive_mode(pipeline)` 是普通函数不是方法，内部只能用 `pipeline.xxx`，禁止出现 `self`。
- **Windows 控制台**：`main()` 已对 stdout/stderr 做 UTF-8 reconfigure，新加的 `print` 不要依赖 GBK 可编码字符。
- **Git 自查**：项目已 git 化（`master`，首次提交 `02ce98a`）。改完代码运行 `git status --short` + `git diff` 自查；确认 `.env` 与大数据目录从未被 `git add`（`.gitignore` 已防护）。

## 常用命令（Windows / PowerShell）

```bash
# 启动交互式问答
python rag_全流程构建.py

# 构建 / 强制重建索引
python rag_全流程构建.py --build
python rag_全流程构建.py --rebuild

# 单次查询（非交互）
python rag_全流程构建.py --query "贵州茅台近期业绩如何"

# 增量插入个股 / 行业研报
python rag_全流程构建.py --add-stock
python rag_全流程构建.py --add-industry

# 语法检查（改完文件必跑）
python -m py_compile scripts/interactive.py pipelines/rag_pipeline.py

# 单元测试（零外部依赖，可离线跑）
python -m pytest tests/ -q

# 导入冒烟测试
python -c "from pipelines.rag_pipeline import RAGPipeline; print('OK')"
python -c "from scripts.interactive import main; print('OK')"

# Web 后端
uvicorn app.api:app --reload --port 8000

# Git 自查（改完代码必看）
git status --short        # 查看改动概览
git diff                  # 查看未暂存改动
git diff --cached         # 查看已暂存改动
git log --oneline         # 查看提交历史
git commit -am "中文简述改动"   # 提交前先确认无 .env/大文件
```

## 查询缓存与并行（路线 1，2026-08-30）

- `utils/query_cache.py`：SQLite 查询缓存（`SQLiteQueryCache` + `make_cache_key`），线程安全（每线程独立连接），表 `query_cache`。
- 并行：`AgentPlanner.execute`（同一轮多 tool_call）与 `LangGraphMultiAgentPlanner._run_subtasks`（financial/research 子任务）默认并行，`AGENT_PARALLEL_TOOLS=false` 可关。
- 缓存接入点：`agents/planner.py::call_financial_chatflow`（原生财务查询，key 含 `FINANCIAL_PROMPT_VERSION`）；`app/api.py` 的 `/chat` 与 `/chat/stream`（按 mode+user_id+question）。
- 证据链口径：做「修复后真实重跑」类回归必须设 `QUERY_CACHE_ENABLED=false` 或清空缓存库（`python -c "from utils.query_cache import SQLiteQueryCache; SQLiteQueryCache('database/query_cache.db').clear()"`）；修改 SQL/分析/图表提示词后 bump `FINANCIAL_PROMPT_VERSION` 或 `QUERY_CACHE_VERSION`（env），避免命中过期结果。
- 配置：`QUERY_CACHE_ENABLED` / `QUERY_CACHE_DB`（默认 `database/query_cache.db`，已 gitignore）/ `QUERY_CACHE_TTL`（默认 86400s）/ `QUERY_CACHE_VERSION`。详见 `docs/评估报告/性能优化_并行缓存.md`。

## 数据依赖

- Qdrant 需在 `localhost:6333` 运行（`docker compose up -d qdrant`），集合名 `research_reports_v3_full`。
- 环境变量在 `.env`：`DASHSCOPE_API_KEY`（必填）、`MYSQL_*`（财务查询，Agent 用）。
- Agent 财务查询走原生 SQL 链路（MySQL + LLM 三层防线），不依赖 Dify；MySQL 不可用时返回友好错误 JSON。
- 混合检索（#8 实验）：`hybrid on` 首次使用会滚动 Qdrant 全量点构建 BM25 索引，缓存于 `database/bm25_index.*.pkl`（按 集合名+点数 自动重建，已 gitignore）；纯 Python 实现，无外部依赖。
- 部署：后端镜像入口为 `uvicorn app.api:app`（`Dockerfile` 已修复）；环境变量模板 `.env.example`，部署说明见 `docs/DEPLOYMENT.md`。
