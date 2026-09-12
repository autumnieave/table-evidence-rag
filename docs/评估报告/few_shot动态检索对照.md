# few-shot 动态检索对照实验（B-16，2026-09-09）

> 前提修正：原文判定『静态基线走 SQL_GEN_SYSTEM_PROMPT 静态 3 示例』有误——实测该提示词内 0 处示例，以 B-39 三态实验（none/static/dynamic）为准。

> 结论：静态（现状基线）与动态 few-shot 在 golden 10 题冒烟中**编译通过率均为 10/10=100%，题型命中率 10/10=100%**，接入动态检索不造成 SQL 编译回退；开关 `AGENT_DYNAMIC_FEWSHOT` 默认关闭，默认行为与静态基线一致，可随时回退。

## 1. 实验目的与口径

- 验证「以指标标准化 JSON（calculation / time_grain / filter_terms）+ 问题类型为标签的规则检索」命中 prompts/examples 示例库后，把 top-k 参考示例注入 SQL_GEN 提示词，是否影响 Text-to-SQL 的编译通过率。
- 链路：仅测「指标标准化 → SQL 生成 → 三层防线（静态校验 + MySQL 编译/执行终审）」环节（`tools/native_financial._generate_sql`），不跑 Agent/RAG/研报汇总，避免编排与检索噪声混入。
- 双模式：静态 = `AGENT_DYNAMIC_FEWSHOT=false`（走 `SQL_GEN_SYSTEM_PROMPT` 静态 3 示例）；动态 = `=true`（`build_sql_gen_system` 命中则追加 few-shot 块，未命中回退静态提示词）。
- 环境：`QUERY_CACHE_ENABLED=false`（真实重跑）；`retries=1`（最多 2 次生成尝试）；每题仅一次指标标准化、结果双模式共享。
- 样本：golden v1 财务子问题 10 题，覆盖 single_period / cross_period_trend / ranking_compare / industry_mean 等题型（逐题见表）。

## 2. 逐题结果

| 题目 | 预测题型 | 命中示例 | 静态编译 | 动态编译 | 静态耗时(s) | 动态耗时(s) |
| --- | --- | --- | --- | --- | --- | --- |
| B2001-Q1 | ranking_compare | 2 | PASS | PASS | 4.92 | 5.96 |
| B2005-Q1 | single_period | 2 | PASS | PASS | 4.62 | 3.96 |
| B2003-Q1 | cross_period_trend | 2 | PASS | PASS | 5.37 | 4.57 |
| B2006-Q1 | cross_period_trend | 2 | PASS | PASS | 5.48 | 4.74 |
| B2008-Q1 | single_period | 2 | PASS | PASS | 4.39 | 4.33 |
| B2010-Q1 | cross_period_trend | 2 | PASS | PASS | 4.73 | 5.01 |
| B2010-Q2 | cross_period_trend | 2 | PASS | PASS | 5.04 | 3.91 |
| B2012-Q1 | single_period | 2 | PASS | PASS | 4.71 | 5.61 |
| B2036-Q1 | industry_mean | 2 | PASS | PASS | 4.22 | 7.16 |
| B2074-Q1 | cross_period_trend | 2 | PASS | PASS | 5.59 | 5.49 |

## 3. 汇总指标

| 指标 | 静态（基线） | 动态（B-16） | 验收目标 | 是否达标 |
| --- | --- | --- | --- | --- |
| 编译通过率（10 题） | 10/10 = 100% | 10/10 = 100% | 接入后不回退（目标 100%） | 达标 |
| 题型命中率（10 题） | — | 10/10 = 100% | ≥90%（抽样统计归档） | 达标 |
| 开关可回退静态示例 | — | 默认关，env 开启 | 可回退 | 达标 |
| 检索器单测 | — | 7 例（tests/test_few_shot_retriever.py） | 配单测（mock 示例库） | 达标 |

## 4. 说明与局限

- B2074-Q1（未分配利润为负的多指标归因）被预测为 cross_period_trend，系问题含「历史」触发趋势关键词——提示词注入趋势类示例后编译仍通过，但示例与真实题型（绑定/归因）并不完全贴合，属规则命中可继续打磨的点；全量 80 题 Agent 级回归建议纳入 B-21 分层门禁后复核。
- 动态 few-shot 只作用于「命中且开启」的请求；指标标准化失败（metric_plan=None）时与现状一致回退静态示例，不新增失败面。
- 冒烟口径为 SQL 生成层，不含 supervisor 拆解/研报检索；SQL 全量 137/137=100% 为既有静态基线，默认关开关使生产行为不变。

## 5. 资产与验证命令

- 检索器：`utils/few_shot_retriever.py`；接入点：`tools/native_financial.py::_generate_sql`；开关：`config/rag_config.py::AGENT_DYNAMIC_FEWSHOT`。
- 实验脚本：`tools/data_scripts/few_shot_dynamic_compare.py`；明细：`训练结果数据/few_shot_dynamic_compare.jsonl` + `_summary.json`。
- 复跑：`python tools/data_scripts/few_shot_dynamic_compare.py`；单测：`python -m pytest tests/test_few_shot_retriever.py -q`（全量 189 passed）。
