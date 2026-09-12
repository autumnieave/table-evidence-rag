# CONTEXT.md — 项目背景与上下文

## 一句话介绍

面向上市公司研报的 RAG 智能问答系统（个人技术实践项目），支持多链路问答、Agent 推理、多轮澄清与增量索引。

## 项目背景

- 项目主题："上市公司财报'智能问数'助手"，数据来自 `<数据目录>/研报数据/`（个股研报解析结果 96 个 .md、行业研报解析结果 68 个 .md、元数据 Excel）。
- 原始版本为单体 `rag_全流程构建.py`，后按模块拆分（config/data/embeddings/vectorstore/chains/llm/agents/memory/filters/pipelines/scripts/tools/utils），主文件仅保留导入与入口。
- 财务结构化数据在 MySQL（`financial_database`，7 张表：dim_stock_info、core_performance_indicators_sheet、balance_sheet、income_sheet、cash_flow_sheet、dify、log_data_validation），由原生 SQL 链路查询（`tools/native_financial.py`）；建表脚本见 `database/schema.sql`（数据由 `tools/data_scripts/pdf处理+校验入库.py` 从公开财报抽取入库，原始数据不入库）。
- 2026-08 完成目录整理：Notebook → `notebooks/`，SQL/YML/CSV → `database/`，文档资料 → `docs/`，数据处理脚本 → `tools/data_scripts/`，旧版程序与临时文件 → `archive/`。
- 2026-08 初始化 git 仓库（`master`，首次提交 `02ce98a`）；`.gitignore` 排除 `.env` 与约 7.3GB 大数据目录（<原始数据>/<数据目录>/qdrant_storage/.venv 等），仅提交源码+文档+配置。

## 核心技术决策及原因

### 1. 为什么用"软过滤"而非"硬过滤"
- 研报检索场景查询条件多样（公司/时间/评级/券商/行业等），硬过滤（Qdrant Filter）会漏掉未完全匹配的记录，且字段映射复杂。
- 方案：向量召回 K=50 后，用 LLM 解析过滤条件为 `QueryFilters`，通过 `compute_match_score` 对结果软打分排序，再做精排。`to_qdrant_filter` 已架空（返回 None），确保不进行硬过滤。

### 2. 为什么做"表格聚合"
- 研报中的表格常跨页或标注"表格续页"，分块后同一张表被拆成多个片段，检索命中子片段时上下文割裂。
- 方案：分块时标记 `is_table_row` + `parent_id`；检索命中任意表格行时，`_aggregate_parent_table` 拉取整个父表内容；非表格行批量生成摘要（batch=20），降低上下文占用并保留关键信息。

### 3. 为什么保留三种链路
- 手写链路便于调试与对照、LangChain 检索器可对比检索质量、LCEL 链路为生产级编排，三者在同一 `query()` 内按开关切换，方便 A/B 对比。

### 4. 为什么用 DashScope HTTP API 而非 SDK
- 当前 `dashscope` SDK 与 text-embedding-v2 接口不兼容（`TextEmbedding.call` 报 `input.contents` 错误），改为 requests 直连 HTTP API，payload `{"input": {"texts": [...]}}`，稳定可控。

### 5. 为什么关闭 qwen3.5-plus 思考模式
- `qwen3.5-plus` 是推理模型，思考过程会消耗 `max_tokens`（2048），长上下文时 `content` 被截断为空（`finish_reason=length`）。在 `get_chat_model()` 加 `extra_body={"enable_thinking": False}` 后稳定输出。

### 6. 为什么分块 overlap 定为 100
- 早期存在双默认值：`data/splitter.py` 默认 150、`config/rag_config.py` 默认 100，pipeline 以 config 为准（线上实际 100），简历叙事却写"150 最佳"，无实验证据。
- 2026-08-24 做对比实验定案：① 离线统计（全量 473 篇，overlap∈{50,100,150,200,250}）发现 `RecursiveCharacterTextSplitter` 实际重叠仅为请求值约一半（150 请求 → 实际 5.8%），表格行 49,917 块不受影响；② 检索命中对比（golden 引用子集 103 篇，100/150 双集合，K=50+Rerank）显示 **100 文件级命中全面领先、数字级持平**。
- 结论：overlap=100 最优，`splitter.py` 默认已统一为 100，全量重建 57,178 点验证无退化。实验报告 `docs/overlap对比实验.md`。

## 已知问题

- **检索质量依赖数据**：`research_reports_v3_full` 集合若混入非研报数据，检索结果相关性下降（此前出现过客服类文档混入）。
- **Agent 财务查询依赖 MySQL**：原生 SQL 链路（生成→执行→分析→ECharts）需 MySQL 可用；MySQL 不可用时返回友好错误提示，RAG 检索不受影响。
- **前端为 React 19**：`qa-frontend/` 基于 React 19 + Vite + Tailwind（README 已同步修正）。
- **LCEL 链路错误处理**：`chains/rag_chain.py` 的 `format_docs` 带大量 DEBUG print，生产可清理；链异常以文本拼接返回，缺少结构化错误。
- ~~**配置分散**：`RAGConfig` 与 `LangChainConfig` 双份配置~~ → 已解决（2026-08-22 合并到 `RAGConfig` 唯一来源，`langchain_config.py` 降为兼容层待删）。

## 后续优化方向

1. 统一配置：以 `RAGConfig` 为唯一来源，`LangChainConfig` 复用同一实例，消除双份配置。
2. 补充性能计时：在 `query()`/`build_index()` 内埋点（`time.perf_counter`），输出检索/精排/生成分阶段耗时。
3. 清理 DEBUG 输出：移除 `format_docs`、`query()` 中的 `[DEBUG]`/`[QUERY DEBUG]` print，改为 logging。
4. 向量库数据治理：增加集合内容校验（仅研报类文档入库），修复混入数据。
5. 引入评估集：用项目需求集"问题汇总.xlsx"构建 QA 评测集，量化三链路效果差异。
6. 补齐 Web 端 Agent 模式入口（前端框架描述已与代码统一，README 已修正为 React 19）。
7. 收尾进展（2026-08-22）：阶段 3（Dify 职责收敛）已完成 —— `dify_tool.py`/`langchain_tools.py` 迁入 `tools/`，全部引用同步更新，消除根目录互相顶层 import；阶段 4（RAGConfig/LangChainConfig 双份配置合并）亦已完成，`langchain_config.py` 降为兼容层。
8. 收尾进展（2026-08-23）：Agent 编排后端对照（#9 实验）—— 新增 `agents/langgraph_planner.py`（LangGraph StateGraph 版，与自研 `AgentPlanner` 同 prompt/同 tools/同输出契约），默认仍用自研（`AGENT_PLANNER_BACKEND=handwritten`），LangGraph 标实验；对照口径见 `docs/LangGraph对照.md`。
9. 收尾进展（2026-08-24）：SQL 守卫 + Agent 关思考 + LangGraph 全量回归 —— 新增 `tools/dify_guard.py`（静态校验 + 全角标点检查 + MySQL 编译 + 失败带错误提示重问，`AGENT_DIFY_RETRY=1`），挂接 `agents/planner.py::call_dify_chatflow` 两版共用；Agent 循环统一关思考（`AGENT_ENABLE_THINKING=false` 默认）；LangGraph 后端同口径 80 题回归：91.3%（116/127）→ 守卫 v1 97.2%（104/107）→ 守卫 v2（yoy 字段白名单提示，Dify prompt 规则 5 写入 Dify 提示词配置文件（规则见 `docs/问题记录/提示词.txt`））**108/108 = 100.0%**，与手工基线 224/224 同口径持平；`docs/评估报告.md` 由 `python -m eval report` 自动聚合 LangGraph 指标小节。
10. 收尾进展（2026-08-24）：overlap 分块参数统一 —— 双默认值（splitter 150 vs config 100）经对比实验（5 档离线统计 + 100/150 双集合检索命中对比）确认 **100 最优**；`splitter.py` 默认改为 100，`rebuild_full_index.py` 新增 `--chunk-overlap/--chunk-size` 参数，全量重建 `research_reports_v3_full`（57,178 点，33.6 分钟）验证指标无退化；简历/面试口径同步为"overlap 对比实验确认 100"。报告 `docs/overlap对比实验.md`。
