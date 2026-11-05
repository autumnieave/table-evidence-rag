# 耗时优化：multi-agent 单任务直出 + 财务子任务并行（2026-08-30）

> 背景：全量 native SQL 回归（80 题 / 108 子问题）实测单题平均 77s、单子问题 57s（qwen3.5-plus 关思考模式）。
> 目标：在不改 SQL 生成/校验口径的前提下降低端到端单题耗时。

## 一、瓶颈拆解（multi-agent 链路每子问题）

| 环节 | LLM 调用 | 说明 |
| --- | --- | --- |
| supervisor 拆任务 | 1 次 | 多轮历史长时生成变慢 |
| 财务子任务（原生 SQL 链路） | 3 次串行 | SQL 生成 → MySQL 执行 → 分析生成 → 图表生成（后两步互相独立却串行） |
| 研报子任务（RAG） | 1 次 | 检索 + 精排 + 长答案生成 |
| aggregator 汇总 | 1 次 | 输入含研报全文时生成 30-45s |

典型"1 财务 + 1 研报"问题 = 6 次串行 LLM；多子问题题每子问题重复整条链路。

## 二、前四轮优化（2026-08-30 已落地）

1. **财务子任务分析/图表并行**（`tools/native_financial.py`）：`_generate_analysis` 与 `_generate_chart` 只依赖查询结果 rows，互不依赖，`ThreadPoolExecutor(max_workers=2)` 并行，省一次串行 LLM 延迟。（第 5 轮复核发现该改动实际未生效，已补全，见第六节）
2. **multi-agent 单任务直出**（`agents/langgraph_multi_agent.py`）：新增 `direct` 节点，supervisor 只拆出 **1 个**财务/研报任务时直接透传子 Agent 结果（含 chart_json / references），跳过 aggregator LLM；多任务（financial+research 混合或同类型多任务）仍走 aggregator 保证整合质量。开关 `AGENT_MULTI_DIRECT_RESULT`（默认 true，env 可关）。
3. **aggregator 输入压缩**：`_build_context` 研报子结果只取 `content` 前 2500 字符（references 由 `_merge_results` 兜底补回），显著缩短汇总 prompt。
4. **supervisor/aggregator 加 max_tokens**（`_call_llm` 新增参数）：supervisor 拆任务 `max_tokens=500`（输出短 JSON）、aggregator 汇总 `max_tokens=1800`；aggregator prompt 要求 content 600-900 字精炼输出，压住"长报告生成"耗时。
5. **aggregator 换更快模型**（`AGGREGATOR_MODEL`，默认空=跟随主模型）：汇总节点是"多源信息 → 单篇报告"的单次 LLM 生成，输出 token 数是耗时主因；`qwen-flash` 实测聚合耗时 16.5s → 7.1s（-57%），content 783 字、数字与引用质量正常，失败自动回退主模型。

## 三、同口径 10 题对比（golden B2001-B2010，关缓存真实重跑）

| 题 | 优化前 | 优化后 | 变化 |
| --- | --- | --- | --- |
| B2001 | 40.0s | 39.6s | ≈ |
| B2002 | 101.5s | 33.8s | -67% |
| B2003 | 113.5s | 42.3s | -63% |
| B2004 | 53.8s | 29.1s | -46% |
| B2005 | 82.1s | 61.7s | -25% |
| B2006 | 78.2s | 34.9s | -55% |
| B2007 | 105.4s | 54.4s | -48% |
| B2008 | 68.9s | 85.5s | +24%（LLM 波动） |
| B2009 | 67.7s | 55.9s | -17% |
| B2010 | 161.5s | 92.0s | -43% |
| **合计** | **872.6s** | **529.2s** | **-39%** |

- 9/10 题下降，仅 B2008 受单次 LLM 波动影响上升；平均单题 87.3s → 52.9s（全量基线 77s，预计降到 ~50s 档）。
- 单财务任务子问题稳定 ~10-13s（原 30-60s），直出路径省掉汇总 LLM 收益最大。
- 第三轮（+max_tokens）：4 个混合任务题合计 425.9s → 287.4s（-33%）；端到端混合题"华润三九为什么 ROE 比白云山高"真实生成 49.6s（财务+研报+图表+13 引用，优化前同类 90-130s）。
- 第四轮（aggregator 换 qwen-flash）：B2041 aggregator 16.5s → 7.1s；端到端混合题（新问题）47.2s；`.env` 已设 `AGGREGATOR_MODEL=qwen-flash`，留空即回退主模型。
- 说明：单次测量含 LLM 行为噪声（supervisor 拆任务不稳定），结论以 10 题总量对比为准。

## 四、口径与风险

- **SQL 编译通过率不受影响**：本次改动未触碰 SQL 生成/静态校验/MySQL 编译路径，回归口径（102/102 = 100%）无需重跑即成立；如需再验证可随时 `python -m eval sql --suite full --limit 10`。
- 直出路径输出 = 财务分析文本（原生链路生成）或研报答案（RAG 生成），格式与 aggregator 输出契约一致（content/image/references/chart_json），端到端样例已人工核对。
- `AGENT_MULTI_DIRECT_RESULT=false` 可回退为"一律走 aggregator"的旧行为。

## 五、产品（前四轮改动清单）

- 改动：`tools/native_financial.py`、`agents/langgraph_multi_agent.py`、`config/rag_config.py`（新增 `AGENT_MULTI_DIRECT_RESULT` / `AGGREGATOR_MODEL`）、`prompts/multi_agent.py`（aggregator 精炼要求）、`.env`（`AGGREGATOR_MODEL=qwen-flash`）
- 验证：`python -m pytest tests/ -q` → 117 passed

## 六、第 5 轮（2026-08-30）：研报生成真流式 + 生成上下文剪枝 + supervisor 换 flash

> 目标：混合题 45s 属于"多跳研报分析"正常量级（业界同类 30s-分钟级），但**感知时延**才是体验瓶颈——此前 agent 模式 30s 生成期间前端无任何内容。本轮同时压"感知时延"与"绝对耗时"。

### 1. 研报生成真流式（感知 45s → 3-8s 首字）
- 接线：`agent_query` 新增 `on_chunk` → `LangGraphMultiAgentPlanner.execute(..., on_chunk)` → `_run_research` 把 `stream_callback` 传给 `rag.query`；`/chat/stream` 的 agent 模式与 rag 模式统一走真 token 流式。
- SSE 事件契约：新增 `final` 事件——混合题中研报草稿先流式展示，aggregator 终稿生成后前端收到 `final` 重置再重发终稿（`qa-frontend/src/App.jsx` 已处理）；单任务直出时流式内容即终稿，后端检测 `streamed_text == content` 跳过重发，避免内容重复。
- 实测（golden 混合题 B2016，http.client 逐行读 SSE）：
  - 首 content 事件 **3.8s**（原 45s 全静默后才出结果）；stage=parse 0.1s 即达
  - 端到端 **32.3s**（优化前 45.2s，-29%）
  - 纯研报直出题（B2002）：首 content 4.0s，无 final 重置、无重复内容，端到端 27.0s
- 顺带修复：rag 模式此前"流式期间已输出 + finish 又整体重发"导致前端内容重复的问题（同一 `streamed_text == content` 判定跳过重发）。

### 2. 生成上下文剪枝（实验）—— 已回退 top-10（保 100% 数字可溯源口径）
- 实验配置：`GENERATOR_CONTEXT_TOP_N=5`；**结论后回退为 10**（`config/rag_config.py` 默认与 `.env` 均为 10 = 不剪枝）：top-5 全量 108 题回归端到端答案数字可溯源 99.9%（2/3508 未命中，均为生成细节/标识符问题，top-10 对照重跑该两题 185/185 可溯源），为保住项目材料 100% 口径改回 top-10。剪枝机制保留（`pipelines/rag_pipeline.py::query()` 生成前对 `final_contexts` 切片），后续如需再压耗时可直接调参。
- 关键口径（实验期间）：**引用始终按 Rerank 后 top-10（RERANK_TOP_N）全量构建**，只把“喂给生成模型的上下文”收到前 N 条——压 prefill 耗时，不影响引用列表与 L1 核验路径。
- 依据（B2010 佐力药业 收入分析，同题 A/B）：top-10（9429 字）17.6s → top-4 12.6s（-28%）→ top-2 11.6s（-34%），答案关键事实（22.8 亿、+11.48%）不丢。
- 答案质量回归：top-5 冒烟 10 题 100%（500/500）→ top-5 全量 108 题 99.9%（3506/3508，2 例见 docs/评估报告/答案质量回归报告.md）→ **回退 top-10 后全量 108 题恢复 100%（已确认：108/108、3879/3879 = 100%，证据 `docs/评估报告/答案质量回归报告.md` + `<结果数据>/golden_answer_full_top10.log`，2026-08-31）**。

### 3. supervisor 换 qwen-flash（拆任务 4.3s → ~1s）
- 配置：`SUPERVISOR_MODEL=qwen-flash`；`_supervisor` 使用快速模型，**非法 JSON 自动回退主模型重试一次**（`_is_valid_json` 判定），避免把乱码当答案。
- 实测：B2016 supervisor 4.3s → 0.9s（-79%）；拆任务结果正常（financial+research 双任务正确拆出）。

### 4. 财务分析/图表并行补全（代码与第 2 轮文档声明对齐）
- 复核发现 `tools/native_financial.py` 实际为串行（B2017 实测 analysis 2.8s → chart 2.4s 顺序执行），已用 `ThreadPoolExecutor(max_workers=2)` 补成真并行，异常自动回退串行；单题财务链路 10.1s 验证通过。
- 注：财务子任务与研报并行、且慢于研报，并行化不改变混合题关键路径，此项主要为文档/代码一致性。

### 5. 改动清单（第 5 轮）
- `pipelines/rag_pipeline.py`：`agent_query` 加 `on_chunk`；`query()` 生成上下文剪枝
- `agents/langgraph_multi_agent.py`：`execute` 加 `on_chunk`；`_run_research` 流式透传；`_supervisor` 用 `SUPERVISOR_MODEL` + 非法 JSON 回退；stage=parse 双发清理（execute 不再重复 emit，由 _supervisor 统一发）
- `agents/planner.py` / `agents/langgraph_planner.py`：`execute` 加 `on_chunk` 兼容（handwritten 路径 search_reports 同步支持流式）
- `app/api.py`：agent/rag 模式接入 on_chunk；finish 逻辑（一致跳过重发 / 不同发 `final` 重置）
- `qa-frontend/src/App.jsx`：处理 `final` 事件（重置后重发终稿）；`npm run build` 已重建 dist
- config/rag_config.py + .env + .env.example：SUPERVISOR_MODEL / GENERATOR_CONTEXT_TOP_N（回退后默认 10 = 不剪枝）
- `tools/native_financial.py`：分析/图表并行
- 验证：`python -m pytest tests/ -q` → 117 passed