# 阶段5：Agent 迭代证据定位与 claim faithfulness
> 口径更新（2026-10-18）：44.05% 是 108 题混合「引用卡片 + L1 上下文」口径，不再作为 RAG 能力指标；A 类 12 题最新 90.62% 见 阶段1-5数字盘点更新_20261018.md。


日期：2026-10-18

## 一、结论

本轮完成三件事：

1. **Router 轻量版**：supervisor 输出 `route_type`（`financial` / `research` / `mixed`），**不改 LangGraph 图结构**，不新增 Mixed 独立 Skill 节点，不做路由评测。
2. **Agent 迭代证据定位**：`audit_evidence` 检查 table schema / value mismatch，输出 `retry_table` / `route_to_sql` / `reject_with_reason` 三种动作（完全匹配时返回 `accept`）；`retrieve_with_iteration` 用扩大召回深度做 1–2 轮重试。
3. **原子化 claim faithfulness**：新增 `src/core/claim_faithfulness.py`，逐 claim 检查是否被整表全文 / SQL 结果支持，**独立于 `row_hits`**；默认只核数字 claim，观点 claim 需注入 LLM/embedding。

最重要的实测结论：

> **`accept`（schema/value 匹配）不等于 answer faithful。** B2005、B2015 的检索审计均为 `accept`，但当前整表全文口径下的 claim 支持率分别只有 16.7%、22.2%。检索命中只能证明“表里有这个指标/期间”，不能证明“答案里的每个数字都能在表里找到”。

两个口径不能混用：

1. **44.05%** = 108 条答案对**自身引用材料**（引用卡片 + L1 数字命中上下文）的题均支持率，离线可复现，但不是全文口径；
2. **28.47%** = 4 条严重题对**当前多阶段检索整表全文 + SQL 行**的题均支持率，是 live 口径。

两者材料来源、样本量、题目集合都不同，只能分别陈述，不能相加、平均或互相替代。

## 二、Router 轻量版

实现位置：`src/agents/langgraph_multi_agent.py`

- `MultiAgentState` 新增 `route_type`；
- 新增 `_classify_route_type(question, tasks)`：
  - 已派发 financial + research → `mixed`；
  - 只派 financial → `financial`；
  - 只派 research → `research`；
  - 无任务时按关键词兜底；纯闲聊归入 `research`（非财务问答桶）；
- `_looks_like_financial_query` 保持 bool 兼容，既有 B-26 兜底逻辑不变；
- supervisor 优先解析 JSON 里的 `route_type`，但**有任务时以实际派发为准**；
- `execute()` 结果额外带 `route_type`，`content/image/references` 契约不变。

版本：`MULTI_AGENT_PROMPT_VERSION=2026-10-18-v4`，`PROMPT_VERSION=2026-10-18-v8`，registry 已同步。

边界：

- 本轮没有做路由准确率评测，不能写成“已完成完整 Router”；
- 没有 Mixed 独立 Skill 节点；
- `route_type` 只用于观测/后续分流，不改变本轮工具执行路径。

## 三、Agent 迭代证据定位

实现位置：`src/core/evidence_iteration.py`、`src/core/multi_stage.py`

### 3.1 决策表

| 条件 | 动作 |
|---|---|
| 无候选表，未超重试轮数 | `retry_table` |
| 无候选表，已超重试轮数 | `reject_with_reason` |
| 问题无可确定指标/期间 | `accept`（不做硬判） |
| 候选表无查询指标 schema，未超轮数 | `retry_table` |
| 候选表无查询指标 schema，已超轮数 | `route_to_sql` |
| 表里有指标 schema，但无匹配期间/值 | `route_to_sql`（季度题典型） |
| 至少一张表同时满足指标与期间 | `accept` |

重试时召回深度：

```text
coarse_k = min(max(coarse_k * 2, coarse_k + 50), 400)
row_k    = min(max(row_k * 2, row_k + 50), 400)
```

`MultiStageTrace` 新增 `iterations` / `final_action` / `final_mismatch`；每条结果 payload 写 `evidence_decision`。

### 3.2 本轮修复的表级 metadata 丢失

单测发现：整表回取来自切片 payload，切片上没有 `metric_names` / `report_period`，导致 schema 审计误判为“表里没有指标”。已修复：`_as_result` 在字段缺失时从粗召回命中继承表级 metadata（`metric_names` / `metric_name` / `report_period` / `anchor_cells`）。

### 3.3 4 条严重题实测

集合 `research_reports_v6_rowfacts`，`retrieve_with_iteration(max_rounds=2)`：

| 题号 | 决策 | 迭代轮数 | 关键路径 |
|---|---|---:|---|
| B2065|1 | `route_to_sql` | 3 | schema_mismatch → 扩到 200 → 400，仍无指标 schema |
| B2069|1 | `route_to_sql` | 1 | value_mismatch（季度实际值缺失） |
| B2015|1 | `accept` | 1 | 指标与期间均命中 |
| B2005|1 | `accept` | 1 | 指标与期间均命中 |

边界：`route_to_sql` 只是**决策**，本轮评测没有执行 SQL；真正转 SQL/财报库是后续链路动作。

### 3.4 最终检索单元确认（B2015/B2065/B2069）

实测脚本：`eval/retrieval_unit_check.py`；产物：`训练结果数据/stage5_measure/retrieval_unit_check.json`。

| 题号 | 返回单元 | 整表数 | `row_hits` | 决策 | 迭代轮数 |
|---|---|---:|---:|---|---:|
| B2015|1 | 整表 | 10 | 0 | `accept` | 1 |
| B2065|1 | 整表 | 10 | 0 | `route_to_sql`（`schema_mismatch`） | 3（100→200→400） |
| B2069|1 | 整表 | 10 | 0 | `route_to_sql`（`value_mismatch`） | 1 |

结论：

- **当前最终检索单元是整表**。三题都返回 10 张候选表的 `payload.content`，不是“整表+行证据”。
- `row_hits` 是可选排序证据：B2015 的 `row_candidates=15`、B2065 的 `row_candidates=400`，但最终 `row_hits` 均为 0；不能把 `row_hits` 当成必然返回物。
- `route_to_sql` 不是拒答：B2065/B2069 仍返回 10 张候选整表，同时在 `evidence_decision` 写 `route_to_sql`；真正转 SQL 是后续链路动作。
- `reject_with_reason` 才返回空结果 + 理由（无候选且重试耗尽时）。
- 因此，L5 工作单必须按整表 `content` 重建，不能依赖 `row_hits`；`row_hits` 只能作为辅助定位线索。

## 四、原子化 claim faithfulness

实现位置：`src/core/claim_faithfulness.py`、`eval/faithfulness_eval.py`

### 4.1 口径

- 材料来自多阶段返回的整表 `payload.content`（可选追加 `sql_rows_to_material` 的 SQL 结果行）；
- **不读 `row_hits`**，因此与行级精定位解耦；
- 数字 claim：所有数字都在材料中命中才算 `supported`；
- 观点 claim：需注入 `llm` 抽取 + `embed` 定位，高分语义证据标为 `semantic_similarity`，仍属线索，不能当 NLI 结论；
- 输出每条 claim 的 `supported` / `missing_numbers` / `source_tables` / `evidence`，并给出 `support_rate`。

### 4.2 本轮修正的两个数字抽取 bug

1. **千分位被拆开**：`3,369` 原先被抽成 `3` + `369`。已修 `_NUMBER_RE`，支持千分位，匹配前仍去逗号，L1 比对口径不变。
2. **序号与年份被折成小数**：`### 1. 2025 年...` 原先被折成 `1.2025`。已在 `_collapse_number_spaces` 保护“序号. 年份”结构。

faithfulness 层另有等长遮罩：股票代码（`600436.SH`）、`Top10`、`第N名` 不参与数字 claim 核验，遮罩不改变字符偏移，展示仍回原文。

### 4.3 实测结果

**A. 108 条 agent 答案 vs 自身引用材料（可离线复现）**

- 材料：引用卡片文本 + L1 数字命中上下文（来自源文件，但**不是 full_text 全文**）；
- 结果：1,439 条数字 claim，623 条 supported；**题均 support rate = 44.05%**（按题等权），按 claim 汇总为 623/1,439 = 43.29%；
- 325 条 claim 完全无证据；1,318 个数字未在材料中命中；
- 108 题中只有 9 题支持率 ≥ 0.8。

**B. 4 条严重题 vs 当前多阶段整表全文 + SQL 行**

| 题号 | 决策 | claims | supported | support rate |
|---|---|---:|---:|---:|
| B2065|1 | `route_to_sql` | 4 | 3 | 75.0% |
| B2069|1 | `route_to_sql` | 3 | 0 | 0.0% |
| B2015|1 | `accept` | 9 | 2 | 22.2% |
| B2005|1 | `accept` | 6 | 1 | 16.7% |
| **合计** |  | **22** | **6** | 池化 27.27% / 题均 28.47% |

### 4.4 如何解读

- 44.05% 是**数字句支持率的下界**，不是“事实准确率”：材料是引用卡片 + 命中上下文，不是 full_text 全文；claim 抽取把“带数字的句子”都当 claim，包含少量描述性枚举；
- B2069 没有匹配到 SQL 行（`sql_rows=0`），数值只在研报表格/文本里，当前检索未覆盖；
- B2005 的 2023A–2027E 利润序列主要来自预测表，当前整表 top-10 未覆盖全部预测列；这属于**证据覆盖问题**，不是“模型编造”已证；
- 最重要的边界仍是：**`accept` 不能替代 faithfulness 门禁**。B2005/B2015 都是 `accept`，但支持率只有 16.7%/22.2%。

## 五、验证

- `python -m py_compile`：通过；
- `python -m pytest tests/ -q`：**928 passed**；
- 新增/修改测试：`test_evidence_iteration.py`、`test_claim_faithfulness.py`、`test_multi_stage.py`、`test_langgraph_multi_agent.py`、`test_citation_validator.py`；
- 实测产物：`训练结果数据/stage5_measure/faithfulness_refs.json`、`faithfulness_live.json`、`retrieval_unit_check.json`。

## 六、本轮明确未做

- 没有做 Router 路由准确率评测；
- 没有 Mixed 独立 Skill 节点；
- 没有重跑 27/108 题全量 faithfulness（只跑 108 条引用材料口径 + 4 条严重题 live 口径）；
- 观点 claim 默认不抽取、不做 NLI；
- `route_to_sql` 没有在本轮实际执行 SQL；
- faithfulness 还不是发布门禁：需要 full_text 工作单 + 人工抽检校准后，才能作为硬指标。

## 七、下一步建议

1. 用 full_text 工作单重跑 claim-level 人工抽检，校准 44.05% 这个下界；
2. 给 live 评测补齐每题 SQL 行（当前只匹配到 3/4 题），否则 SQL 路答案会被系统性低估；
3. 增加“非断言枚举句”过滤，避免把“数值列（如：0.72/0.39…）”算进支持率；
4. 把 `accept + faithfulness < 阈值` 定义为新的联合门禁，而不是只看检索审计；
5. 观点 claim 需要单独的 NLI/人工口径，不能复用数字规则。
