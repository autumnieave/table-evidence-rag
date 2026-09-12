# LangGraph 版 Agent 对照（自研循环 vs StateGraph）

> **历史文档标注（2026-08-30）**：本文记录 Dify 迁移下线前的实验与回归口径，属历史证据；当前财务查询走原生 SQL 链路（`tools/native_financial.py`），运行时已无 Dify 依赖。

> 2026-08-23 · 实验：#9 Agent 编排后端对照。原则：**同 prompt、同 tools、同工具执行、同输出契约，只换编排层**。


> **时效标注**：该描述限于 2026-08-23 对照期；**2026-08-30 起 `.env` 已切 `AGENT_PLANNER_BACKEND=langgraph` + `AGENT_LANGGRAPH_MULTI_AGENT=true`（supervisor-workers 多 Agent 主链路）**，见 `MultiAgent对照.md`。

项目主链路 Agent 为自研 `AgentPlanner`（while 循环 + Function Calling）。为对齐市场对 **LangChain/LangGraph 框架熟练度**的要求，新增 `LangGraphPlanner`（StateGraph 显式状态机）做对照验证，回答"用 LangGraph 重写 Agent 会怎样、值不值"。


| 维度 | 自研 `AgentPlanner` | `LangGraphPlanner`（实验） |
| --- | --- | --- |
| 编排方式 | `while` 循环（最多 10 轮） | `StateGraph`：`call_model → tools → finalize` 状态机 |
| 状态管理 | 隐式 `messages` 列表 | 显式 `AgentState`（`messages/rounds/user_id/result`） |
| 路由 | 代码内 `if msg.tool_calls` 判断 | 条件边 `route_after_model` / `route_after_tools` |
| 轮次上限 | `for _ in range(10)` | 条件边 `rounds < MAX_ROUNDS` |
| 超时兜底 | 循环结束返回固定提示 | `finalize` 节点统一兜底（超时/空/非 JSON） |
| 工具执行 | 方法内 `if/elif` 分发 | `_run_tool` 分发；Dify 调用**共用** `call_dify_chatflow` |
| 可观测性 | `print` 日志 | 图结构可查（`get_graph()`）、状态显式、可插 checkpoint |
| 依赖 | 无 | `langgraph>=1.0.0` |
| 状态输出 | 字典 | 字典（`execute()` 返回 `output["result"]`） |


- 新增：`agents/langgraph_planner.py`（实验）、`config/rag_config.py::AGENT_PLANNER_BACKEND`（env `AGENT_PLANNER_BACKEND`，默认 `handwritten`）
- `RAGPipeline._get_agent_planner()` 按配置懒加载；`agent_query()` 统一走该方法，入口签名不变
- 交互切换：`planner langgraph` / `planner handwritten`（`status` 显示当前后端）
- 复现对照：`python tools/data_scripts/agent_planner_compare.py --stub --backend both`


场景：stub 检索工具（Qdrant 未运行）+ Dify 未启动（财务工具连接超时后降级返回失败），同一问题"结合研报分析马应龙的成本控制优势"。

| 后端 | 耗时 | 工具调用 | 输出契约 |
| --- | --- | --- | --- |
| 自研 `AgentPlanner` | 44.7s | 3 次 `search_reports` | `{content, image:0, references:3}` |
| `LangGraphPlanner` | 119.9s | 2 次检索 + 3 次财务查询（Dify 连接超时） | `{content, image:0, references:2}` |

**结论（口径必须写清）**：
1. 两版均完成多步工具调用，返回结构一致的答案，**输出契约一致**。
2. 耗时差异主要来自**单次轨迹随机性**（LangGraph 本轮多触发了 3 次 Dify 调用，每次连接超时叠加延迟），**不能归因于框架**；需固定工具序列、多轮采样才有统计意义。
3. LangGraph 图结构：`nodes = [__start__, call_model, tools, finalize, __end__]`，路由与轮次由条件边声明式表达。


`tests/test_langgraph_planner.py` 6 例（fake OpenAI client + stub RAG，不依赖外部服务）：
图构建 / 单轮工具调用 / 直接 JSON 解析 / 非法 JSON 兜底 / 超时兜底 / 历史截断 12 条。
`python -m pytest tests/ -q` → **77 passed**。


**口径**：与 golden v1 同源（80 题 / 108 子问题 / 291 句基线），Agent 多轮累积口径
（逐题 `agent_query` → `conversation_state.sql` 累积 → 静态校验 + MySQL 编译）。
命令：`python -X utf8 -m tools.data_scripts.sql_agent_regression --backend langgraph --progress-every 10`
（Dify / Qdrant / MySQL 全程在线）。

| 指标 | 自研 handwritten（08-18 基线） | LangGraph 首跑（无守卫） | LangGraph + SQL 守卫（终值） |
| --- | ---: | ---: | ---: |
| 语句级编译通过率 | 224/224 = **100.0%** | 116/127 = 91.3% | **108/108 = 100.0%** |
| 有 SQL 题目 | 61/80 | 58/80 | 58/80 |
| 有 SQL 且全部语句通过 | 61/80 | 53/80 | **58/58** |
| 空 SQL 题目 | 19 | 22 | 22 |
| 单题耗时中位数 | 130.9s | 91.6s | — |

**三轮迭代（同一 80 题 / LangGraph 后端）**：
1. **首跑（无守卫）**：116/127 = 91.3%，11 条失败语句全部来自 Dify 工作流 SQL 输出质量
   （B2011 全角逗号；B2047 未定义别名 / 非 SELECT 文本；B2060/B2073/B2076 编造 `*_yoy_growth` 字段）；
2. **守卫 v1**（`tools/dify_guard.py`：静态校验 + MySQL 编译 → 失败带错误提示重问一次）：
   104/107 = 97.2%——B2011/B2047/B2076 修复；
3. **守卫 v2**（错误提示追加"可用字段参考 + yoy 字段白名单"）：108/108 = **100.0%**——
   B2060/B2073 的字段幻觉被白名单提示纠正。

**守卫机制（本轮新增，两版后端共用）**：
- `tools/dify_guard.py`：`sql_errors()`（静态校验 + 全角标点启发式 + MySQL 编译终审）+
  `call_dify_with_guard()`（校验失败 → 把错误与字段建议拼回问题重问，`AGENT_DIFY_RETRY` 默认 1 次）；
- 挂接点：`agents/planner.py::call_dify_chatflow`（自研 / LangGraph 同源）；
- 配置：`RAGConfig.AGENT_SQL_VALIDATE`（默认开）/ `AGENT_DIFY_RETRY`（默认 1）/ `MYSQL_*`
  （schema + 编译终审）；单元测试 `tests/test_dify_guard.py` 8 例（离线）。

**源头修复（prompt 规则）**：Dify 工作流 SQL 生成节点【再次强调】新增规则 5——yoy/qoq
字段白名单（4 张表真实 yoy 字段逐一列出；费用科目无 yoy 字段、禁止编造任何变体），已同步
写入 Dify 提示词配置文件（规则见 `docs/问题记录/提示词.txt`）；
**需重新导入该配置文件到 Dify 后生效**（当前 100% 由守卫兜底达成，不依赖重新导入）。

**轨迹差异（与手工基线逐题对比）**：LangGraph 在 **17 题**上未产生 SQL（其中 B2015/B2022
手工为 1/1 通过）、在 **9 题**上新增了通过的 SQL（B2003/B2012/B2017/B2018/B2023/B2027/B2029/
B2043/B2058，手工为空 SQL）。两版都产生 SQL 的 48 题上：handwritten 187/187 = 100%；
LangGraph 终值本轮 58/58 有 SQL 题全部通过（含该 48 题）。

**切换评估结论**：
1. **功能等价**：同 prompt / 同 tools / 同输出契约，全量 80 题跑通，无框架级故障；
2. 首跑 91.3% 的失败全部是 Dify 生成 SQL 质量问题，**不能归因于 LangGraph**；接入守卫后
   语句级 108/108 = **100.0%**，与手工基线同口径持平；
3. 本对照同时叠加了 thinking ON→OFF 与不同 Dify 状态，属"端态对照"而非受控实验；
   轨迹方差大（17 丢 / 9 得），单次跑批不足以判定框架优劣；
4. **决策：生产默认仍保留自研 handwritten**（回归基线、零依赖）；LangGraph 作为实验后端保留。
   （**时效标注**：该结论限于 2026-08-23 对照期；**2026-08-30 起 `.env` 已切 `AGENT_PLANNER_BACKEND=langgraph` + `AGENT_LANGGRAPH_MULTI_AGENT=true`（supervisor-workers 多 Agent 主链路）**，见 `MultiAgent对照.md`。）
   若日后切换，前置条件已具备（守卫 + 字段白名单）；建议再做同日同 thinking 的多轮采样对照。

**关思考提速（本次一并落地）**：Agent 循环（handwritten + LangGraph）统一走
`RAGConfig.AGENT_ENABLE_THINKING`（默认 `false`，env `AGENT_ENABLE_THINKING=true` 可开）；
LangGraph 关思考全量跑批实测中位耗时 91.6s（对比 handwritten 130.9s，含 thinking ON/OFF 与
不同日期混淆，仅作参考）。



LangGraph 版 Agent 接入 LangGraph **checkpointer**，按 `thread_id=user_id` 持久化对话状态，
形成"LangGraph 自带记忆"的完整示例（对照：主链路自研版记忆走 `memory/store.py` + `RAGPipeline`）。

**设计与行为**
- 编译：`graph.compile(checkpointer=...)`，后端可选 `sqlite`（默认，落盘 `database/langgraph_checkpoints.sqlite`）/ `memory` / `none`；
- 读取：每次 `execute()` 先 `graph.get_state({\"configurable\": {\"thread_id\": user_id}})` 读回该用户上次会话消息，作为本轮上下文；超过 `CONVERSATION_TIMEOUT_SECONDS` 视为新话题；
- 写入：本轮结束 `graph.update_state(..., {"last_active": time.time()})` 记录活跃时间，图执行过程自动落 checkpoint；
- 隔离：不同 `user_id` 对应不同 thread，互不串话；sqlite 落盘后**重启进程可恢复**（有离线单测覆盖跨实例恢复）。

**配置**（`config/rag_config.py`）
- `AGENT_LANGGRAPH_CHECKPOINT`（默认 `true`）/ `AGENT_LANGGRAPH_CHECKPOINT_BACKEND`（`sqlite|memory|none`）
- `AGENT_LANGGRAPH_CHECKPOINT_PATH` / `AGENT_LANGGRAPH_MAX_HISTORY`（默认 40 条，保留首条 system，避免截断后 tool 引用悬空）
- 依赖：`langgraph-checkpoint-sqlite>=3.1.1`（已加入 `requirements.txt`）

**口径提醒**（**时效标注**：该结论限于 2026-08-23 对照期；**2026-08-30 起 `.env` 已切 `AGENT_PLANNER_BACKEND=langgraph` + `AGENT_LANGGRAPH_MULTI_AGENT=true`（supervisor-workers 多 Agent 主链路）**，见 `MultiAgent对照.md`。）：checkpoint 是 LangGraph 后端的能力，切换 `planner langgraph` 后生效；
自研 `handwritten`（现为**回退路径**）的记忆由 `memory/store.py`（SQLite/Redis 按 user_id）负责。
面试可讲："LangGraph 版用 thread_id + checkpointer 实现了会话状态落盘与恢复（SQLite，重启不丢），
验证了框架级持久化能力，为后续迁移 LangGraph 生态铺路。"


- **LangGraph 优点**：显式状态机、声明式条件边、可 checkpoint/人审/可视化、生态标准、便于多人协作与复用现成模式。
- **自研优点**：零依赖、完全可控、易调试（状态就是一个列表）、少一层抽象、已通过 224/224 Agent 回归基线。
- **本项目决策**（**时效标注**：该结论限于 2026-08-23 对照期；**2026-08-30 起 `.env` 已切 `AGENT_PLANNER_BACKEND=langgraph` + `AGENT_LANGGRAPH_MULTI_AGENT=true`（supervisor-workers 多 Agent 主链路）**，见 `MultiAgent对照.md`。）：当时默认保留自研 `handwritten`（回归基线、零依赖）；**现状已切换为 LangGraph supervisor-workers 多 Agent 主链路**，`handwritten` 降为回退路径。
- **面试口径建议**（已按现状更新）："演进路径是自研 Function Calling → LangGraph StateGraph 对照版 → **supervisor-workers 多 Agent 主链路**（`.env` 默认，同 prompt/同工具/同输出契约，经 80 题同口径对照后落地），自研版保留一键回退。LangGraph 的优势在显式状态、条件路由与 checkpointer，自研的优势在零依赖与完全可控。"
- **简历表述建议**：可写"自研 Agent 编排（Function Calling 多步推理）并实现 LangGraph StateGraph 对照版"，避免写成"基于 LangChain Agent"。


- ~~Agent 循环统一关闭思考模式~~ → 已完成（`AGENT_ENABLE_THINKING=false` 默认，两版同源）。
- ~~全量 Agent 回归（224 题口径）基线跑在自研后端；若切换 LangGraph 需先跑同口径回归~~ → 已完成，见"全量回归对照"一节；结论暂不切换。
- ~~Dify SQL 质量加固（校验器拒收重试 / 字段白名单）~~ → 已完成：`tools/dify_guard.py` 守卫
  挂接 Agent 工具循环，语句级 91.3% → 100.0%（108/108）；prompt 规则 5 已写入 YAML/提示词文档，
  待重新导入 Dify 生效。
- 真实生产对比需 Qdrant + Dify 就绪后执行：`python tools/data_scripts/agent_planner_compare.py --backend both`。
