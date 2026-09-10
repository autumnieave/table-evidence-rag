# LangGraph 多 Agent 协作对照（supervisor-workers）

> **历史文档标注（2026-08-30）**：本文记录 Dify 迁移下线前的多 Agent 对照口径，属历史证据；当前财务查询走原生 SQL 链路（`tools/native_financial.py`），运行时已无 Dify 依赖。

> 2026-08-26 · 实验：#10 在 LangGraph 实验后端上扩展多 Agent 协作。
> 2026-08-30 · **前端默认后端已切换**（.env：AGENT_PLANNER_BACKEND=langgraph + AGENT_LANGGRAPH_MULTI_AGENT=true），回退：设 handwritten。
> 原则：自研单 Agent（224/224 回归基线）仍是回归基准；LangGraph 版从"等价对照"升级为多 Agent 协作 + 生产默认编排。


- **切换**：前端默认后端 = LangGraph multi-agent（`.env` 配置，可一键回退 handwritten）。
- **端到端**：归因题 89.8s，stage 序列 `parse → query_financial → search_reports → generate`，返回图表 + 20 条引用。
- **SQL 证据链**：multi-agent 子集 8 题（6 财务向），财务向题 SQL 语句级通过率 **100%**（7/7 复核，SQL 仍由 Dify + 三层防线生成，multi-agent 只换编排层）；B2002/B2004 空 SQL 属意图模糊/开放性问题（supervisor 正确判定非财务查询）。
- **延迟代价**：multi-agent 每题 109-155s（supervisor+aggregator 额外 LLM 轮次 + 每题倾向拆 research），比自研 Agent 纯财务题（~23s）明显更慢。**结论：multi-agent 适合归因/复合分析题（结果更丰富，带研报引用）；纯数据查询场景建议回退 handwritten 或按问题类型路由。**
- 明细：`result/lg_multiagent_sql_regression.json`。


回答"分析 X 公司 2023 营收结构并给出研报观点"这类**复合问题**：财务查数 + 研报找观点 + 汇总成报告。
单 Agent 也能做（工具循环逐步调用），但多 Agent 的核心是**角色分工 + 显式状态 + 结果汇合**，
这正是 StateGraph 的设计初衷，因此放在 LangGraph 实验后端验证。


**supervisor-workers（主管-工人）模式**

```text
START → supervisor（拆任务）─┬─ tasks 非空 → tools（财务/研报子 Agent）→ aggregator（汇总）→ END
                              └─ tasks 空   → finalize（直接回答）→ END
```

| 节点 | 职责 | 对应能力 |
| --- | --- | --- |
| supervisor | LLM 把用户问题拆成子任务（financial / research） | 任务规划 |
| tools | 按任务类型调用财务查询 / 研报检索 | 子 Agent 执行 |
| aggregator | 把子结果整合成最终 `{content, image, references}` | 汇总报告 |
| finalize | 未拆出任务时直接回答（兜底） | 闲聊/无效输入 |

**状态**（`MultiAgentState`，checkpoint 持久化，thread_id=user_id 隔离）
- `tasks`：supervisor 拆出的任务列表 `[{agent: financial|research, query}]`
- `subtask_results`：子 Agent 产出 `{financial: [...], research: [...]}`
- 其余沿用：`messages / user_query / rounds / user_id / result / last_active`

**复用**
- 财务子 Agent → `agents/planner.py::call_dify_chatflow`（Dify 查询 + SQL 守卫，与单 Agent 同源）
- 研报子 Agent → `rag.query()`（RAG 检索 + 引用）
- checkpoint / `on_stage` 回调 / `AGENT_ENABLE_THINKING`：与 `LangGraphPlanner` 同一契约（前端阶段提示自动生效）

**新增**
- `agents/langgraph_multi_agent.py`：`LangGraphMultiAgentPlanner`（`execute` 契约与单 Agent 一致）
- `prompts/multi_agent.py`：supervisor / aggregator 两段 Prompt（遵守 Prompt 统一收口规范）
- 工具可注入（`financial_tool` / `research_tool`），便于离线单测与替换后端


- 配置：`AGENT_LANGGRAPH_MULTI_AGENT`（默认 `false`，env `AGENT_LANGGRAPH_MULTI_AGENT=true` 可开），需配合 `AGENT_PLANNER_BACKEND=langgraph`
- 交互式：`planner multi-agent` 切换 / `planner langgraph` 切回单 Agent / `planner handwritten` 切回自研；`status` 显示当前后端
- `RAGPipeline._get_agent_planner()` 按配置惰性加载，`agent_query()` 入口不变


`tests/test_langgraph_multi_agent.py` 6 例（fake OpenAI 客户端 + stub 工具，不依赖外部服务）：
图构建 / 拆 2 任务并汇总（图片引用合并）/ 无任务 direct_answer / supervisor 非法 JSON 兜底 /
aggregator 缺引用时代码兜底合并（按 paper_path 去重）/ 默认关闭思考模式。
`python -m pytest tests/ -q` → **95 passed**（含此前 89 例，无回归）。


- **不是**把生产切到多 Agent：`AGENT_LANGGRAPH_MULTI_AGENT` 默认关闭，自研主链路与 LangGraph 单 Agent 行为不变。
- 面试口径："自研单 Agent（双工具编排，224/224 回归基线）保生产稳定；LangGraph 版在 StateGraph 上扩展了
  supervisor-workers 多 Agent 协作——规划拆任务、财务/研报子 Agent 并行取数、汇总成报告，验证了复杂任务的
  编排能力；多 Agent 的核心是显式状态和路由，所以放在图框架上做，而不是手写循环。"
- 待办：真实链路（Qdrant + Dify 在线）跑 2-3 个复合问题样例记录到本文件；若要做"并行取数"可改用 LangGraph
  `Send` API（当前为串行执行，优先确定性）。


- 单 Agent 对照：`docs/评估报告/LangGraph对照.md`（StateGraph 版 + checkpoint 持久化）
- 自研**回退路径**：`agents/planner.py`（`AgentPlanner`，Function Calling 循环）。**时效标注**：该表述限于 2026-08-23 对照期；**2026-08-30 起 `.env` 已切 `AGENT_PLANNER_BACKEND=langgraph` + `AGENT_LANGGRAPH_MULTI_AGENT=true`（supervisor-workers 多 Agent 主链路）**。
