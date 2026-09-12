# SQL 编译失败 Badcase 修复前后对比报告

> **历史文档标注（2026-08-30）**：本文记录 Dify 迁移下线前的实验与回归口径，属历史证据；当前财务查询走原生 SQL 链路（`tools/native_financial.py`），运行时已无 Dify 依赖。

- **项目**：上市公司"智能问数"助手（RAG + Agent）
- **数据来源**：`<结果数据>/result_3_parallel.xlsx`（80 题 6 类典型查询批量回归）
- **校验方式**：逐句 SQL 在 MySQL `financial_database` 上真实执行（`MAX_EXECUTION_TIME=15s`）
- **报告日期**：2026-08-17


批量回归共生成 **291 句 SQL**，其中 **282 句编译通过（96.9%）**，**9 句失败**。失败集中在多表 JOIN 场景下的三类问题：字段表归属错误、表别名未定义、同名字段歧义。本报告记录这 9 条 badcase 的修复前后对比。


| 编号 | 问题类型 | 分类 | 修复前错误 | 旧 SQL 片段 |
|---|---|---|---|---|
| B2007 | 归因分析 | B | (1054, "Unknown column 't2.net_profit_10k_yuan' in 'field list'") | SELECT t1.stock_abbr, t1.report_year, t1.total_operating_revenue, t2.net_profit_10k_yuan, t3.asset_total_asset… |
| B2041 | 多意图 | A | (1054, "Unknown column 'net_profit' in 'field list'") | SELECT stock_abbr, report_year, report_period, net_profit_excl_non_recurring, net_profit FROM core_performance… |
| B2049 | 多意图 | A | (1054, "Unknown column 't1.net_profit' in 'field list'") | SELECT t1.stock_code, t1.stock_abbr, t1.report_year, t1.report_period, t2.operating_expense_selling_expenses, … |
| B2052 | 多意图 | A | (1054, "Unknown column 't2.operating_expense_rnd_expenses' in 'field list'") | SELECT t1.stock_code, t1.stock_abbr, t1.operating_expense_rnd_expenses, t1.operating_expense_selling_expenses,… |
| B2053 | 多意图 | B | (1054, "Unknown column 't1.stock_code' in 'field list'") | SELECT t1.stock_code, t1.stock_abbr, t1.report_year, t1.report_period, t3.operating_cf_net_amount, t4.net_prof… |
| B2063 | 归因分析 | A | (1054, "Unknown column 'operating_expense_cost_of_sales' in 'field list'") | SELECT stock_abbr, report_year, report_period, gross_profit_margin, total_operating_revenue, operating_expense… |
| B2068 | 归因分析 | A | (1054, "Unknown column 'net_profit' in 'field list'") | SELECT net_profit_excl_non_recurring, net_profit FROM core_performance_indicators_sheet WHERE stock_abbr LIKE … |
| B2074 | 归因分析 | B | (1054, "Unknown column 't1.report_year' in 'field list'") | SELECT t1.report_year, t2.equity_unappropriated_profit, t4.net_profit FROM balance_sheet t2 JOIN income_sheet … |
| B2075 | 归因分析 | C | (1052, "Column 'total_operating_revenue' in field list is ambiguous") | SELECT stock_code, stock_abbr, report_year, report_period, net_profit, total_operating_revenue, net_profit_exc… |

> 完整失败 SQL 与逐条分析见 `docs/问题记录/badcase_台账.md`。


| 分类 | 条数 | 根因 | 旧版提示词覆盖情况 |
| --- | --- | --- | --- |
| A 字段表归属错误 | 5 | 同名字段（如 `net_profit`）被挂到不存在的表 | 仅有"严禁臆造字段"，未强制字段-表归属 |
| B 别名/表未定义 | 3 | `SELECT` 引用了 `FROM` 中未定义的表别名 | 仅约束表名，未约束别名 |
| C 同名字段歧义 | 1 | JOIN 后同名字段未加表前缀 | 无规则 |


| 轮次 | 新增内容 | 落点 |
| --- | --- | --- |
| 第 1 轮 | 核心禁令 6「字段-表归属严格校验」、7「多表别名强制规则」、8「同名字段歧义规则」 | `docs/问题记录/提示词.txt` + `提示词配置文件（本地保留，不入库）` |
| 第 2 轮 | 规则 6 补充「字段→表速查」：`net_profit`→income_sheet、`net_profit_10k_yuan`→core 表等 | 同上 |
| 第 3 轮 | 规则 6 增加 JOIN 正例/反例（few-shot） | 同上 |

规则要点：

1. **字段-表归属**：生成 SQL 前必须核对每个字段所属表，严禁跨表乱用；附「字段→表速查」与 JOIN 正例/反例。
2. **多表别名强制**：JOIN 时 SELECT 每个字段必须带表别名前缀；FROM 中每个别名必须有定义。
3. **同名字段歧义**：JOIN/USING 后同名字段必须显式加表前缀。


| 编号 | 问题类型 | 分类 | 修复后新 SQL 片段 | 回归 |
|---|---|---|---|---|
| B2007 | 归因分析 | B | SELECT t1.stock_code, t1.stock_abbr, t1.total_operating_revenue, t1.net_profit_10k_yuan, t2.asset_total_assets… | ✓ 通过 |
| B2041 | 多意图 | A | SELECT t1.net_profit_excl_non_recurring, t2.net_profit FROM core_performance_indicators_sheet t1 JOIN income_s… | ✓ 通过 |
| B2049 | 多意图 | A | SELECT t1.stock_code, t1.stock_abbr, t2.operating_expense_selling_expenses, t2.total_operating_revenue, t1.net… | ✓ 通过 |
| B2052 | 多意图 | A | SELECT t1.stock_code, t1.stock_abbr, t2.operating_expense_rnd_expenses, t2.operating_expense_selling_expenses … | ✓ 通过 |
| B2053 | 多意图 | B | SELECT t1.operating_cf_net_amount, t2.net_profit, t3.asset_liability_ratio, t3.asset_accounts_receivable, t3.a… | ✓ 通过 |
| B2063 | 归因分析 | A | SELECT t1.gross_profit_margin, t2.total_operating_revenue, t2.operating_expense_cost_of_sales FROM core_perfor… | ✓ 通过 |
| B2068 | 归因分析 | A | SELECT t1.net_profit_excl_non_recurring, t2.net_profit FROM core_performance_indicators_sheet t1 JOIN income_s… | ✓ 通过 |
| B2074 | 归因分析 | B | SELECT t1.equity_unappropriated_profit, t2.net_profit FROM balance_sheet t1 JOIN income_sheet t2 ON t1.stock_c… | ✓ 通过 |
| B2075 | 归因分析 | C | SELECT t1.net_profit, t1.total_operating_revenue, t2.net_profit_excl_non_recurring, t1.other_income, t1.total_… | ✓ 通过 |

**迭代曲线**：

| 轮次 | 规则状态 | 通过 |
| --- | --- | --- |
| 基线 | 旧版 prompt | 0/9 |
| 第 1 轮 | 加规则 6/7/8 | 6/9 |
| 第 2 轮 | 补字段→表速查 | 8/9 |
| 第 3 轮 | 加正例/反例 | **9/9**（B2068 稳定 3/3） |


- 语句级编译通过率：**96.9%（282/291）→ 修复样本 9/9 全过**（9 条 badcase 全部闭环）。
- 端到端复验（Dify 全链路）：复验前 6/9 通过 → 补充修复后 **9/9 编号级通过、语句级 9/9 SQL 编译执行通过**。
- 失败根因从"模型臆断字段/别名"转为可约束规则，后续可纳入 SQL 生成回归指标（目标语句级 ≥99%）。


**阶段一：Prompt 级回归（先行验证）**
- 提取 Dify 工作流 SQL 生成节点的最终 system prompt，以 qwen3.5-plus（temperature 0.7，与工作流节点一致）重新生成 SQL，并在 MySQL 上逐句执行校验。
- 输入口径：`Standard_field_name` 取原失败 SQL 的 SELECT 字段，`上游逻辑判断结果` 置 `is_consistent: true`。
- 结果：3 轮迭代 0/9 → 9/9。

**阶段二：真端到端复验（Dify 全链路，2026-08-17）**
- 环境：Dify v1.13.0（Docker，localhost:5001），工作流 `提示词配置文件` 已导入并发布，API Key `app-xxxx`。
- 链路：用户问题 → Dify「问题重构 → 语义转化 → SQL 生成」→ 提取 SQL → MySQL 逐句编译。
- 复验前：6/9 通过（B2049/B2053 语义节点反问、B2075 JOIN 不存在表）；补充修复后 **9/9 全部通过**（含 B2049 稳定性连跑 4/4）。
- 边界：本复验口径为"生成 SQL 能否在 MySQL 真实编译执行"；答案语义正确性不在本次范围（B2007 子问题 2/3 被正确判为"与数据查询无关"）。

- `docs/问题记录/badcase_台账.md` —— 9 条 badcase 逐条明细与状态跟踪
- `docs/问题记录/提示词.txt` —— SQL 生成 Prompt（含新规则 6/7/8）
- `提示词配置文件（本地保留，不入库）` —— Dify 工作流导出（SQL 生成节点已同步）
- `<结果数据>/sql_compile_report.csv`、`regression_final.json` 等 —— 校验明细数据（不入 git）
- `<结果数据>/e2e_diag.json` / `e2e_diag_temp01.json` / `e2e_diag_final.json` —— 端到端复验前后明细（不入 git）
- `tools/sql_validator.py`、`tools/data_scripts/sql_validator_selftest.py` / `sql_guard_regression.py` / `sql_guard_repair_demo.py` —— 工程化拦截（SQL 校验器 + 守卫回归）
- `<结果数据>/sql_guard_results.json`、`sql_guard_repair_demo.json` —— 守卫回归与修复演示明细（不入 git）


在 SQL 生成后、执行前增加静态校验 + 失败自动重试，把"模型输出"纳入可控质量闸门：

- `tools/sql_validator.py`：表名白名单 / 别名定义 / 字段-表归属 / 裸字段歧义 四类静态校验 + MySQL 编译终审。
- 自测：原始 9 条失败 SQL **拦截 9/9**、修复后 9 条通过 SQL **零误报**。
- 守卫回归：9 题 11 个子问题首轮全部通过；B2049 连跑 4/4。
- 重试修复（故障注入）：B2049/B2053/B2075 三类典型错误注入后，错误反馈追问均产出通过校验的修正 SQL。
- 意义：SQL 编译通过率从"靠提示词约束"升级为"提示词 + 执行前校验 + 失败自动修复"三层保障，可直接纳入批量回归指标。


将证据从「9 条 badcase 样例」扩展到「全量 80 题」：对 `result_3_parallel.xlsx` 全部 108 个子问题，
走 Dify 真实链路逐子问题重新生成 SQL，再用校验器静态校验 + MySQL 真实编译，产出修复后全量指标。


- **生成方式**：子问题级单发（每个子问题独立新会话、只问一次、不做错误追问重试），与 9 题端到端复验
  （`e2e_diag_final.json`）口径一致；基线 `result_3_parallel.xlsx` 为 Agent 多轮工具调用 + SQL 累积生成。
  因此**回归语句量（53 句）与基线（291 句）不可直接对比**，可对比的是"生成语句的编译通过率"这一口径。
- **空 SQL**：意图模糊/开放性问题/分析类子问题不生成 SQL 属预期行为，不计入编译失败；"有 SQL 题目数"
  下降主要来自单发模式下分析类子问题不再累积 SQL，而非编译失败。
- **API 异常**：0 题，无网络/服务侧失败混入。


| 口径 | 基线 result_3_parallel | 修复后全量回归（单发复跑） |
| --- | --- | --- |
| 语句级编译通过率 | 282/291 = 96.9% | 52/52 = 100.0% |
| 有 SQL 的题目 | 70/80 | 52/80 |
| 有 SQL 且全部语句通过 | 61/70 | 52/52 |
| 严格全题（空 SQL 视为未通过） | 61/80 | 52/80 |
| 空 SQL 题目 | 10 | 28（其中 10 题与基线一致，18 题为单发未生成，含 B2035 本轮未生成 SQL） |


9 条 badcase 所在题目在本次全量回归中**全部通过**（基线 0/9 未通过 → 回归 9/9 通过）：

- B2007、B2041、B2049、B2052、B2053、B2063、B2068、B2074、B2075：每题均生成 SQL 且编译通过。
- 其中 B2007 的子问题 2/3（归因分析、数据来源可靠性）被正确判为"与数据查询无关"（空 SQL），符合预期。


| 编号 | 问题类型 | 根因分类 | 错误 | 修复方向 |
| --- | --- | --- | --- | --- |
| B2035 | 融合查询 | D. 虚构表 + MySQL 方言 | (1235) MySQL 不支持 `LIMIT & IN/ALL/ANY/SOME subquery`；且 JOIN 了库中不存在的 `research_reports` 表（研报数据在 Qdrant/RAG，不在 MySQL） | 研报侧"行业龙头"应从 RAG 检索取，不混入 SQL；提示词补"严禁虚构表（含子查询内）" |
| B2051 | 多意图 | A. 字段表归属（漏网） | `asset_liability_ratio` 属于 `balance_sheet`，被挂到 `core_performance_indicators_sheet`（t1） | 该字段已在字段白名单（balance_sheet 节），属单发漏网；已由静态校验器拦截，可补入"字段→表速查"强提示 |

说明：B2035 的虚构表位于派生表（子查询）内部，原静态校验器跳过子查询内表名校验，由 MySQL 编译兜底；已扩展子查询内表名检查（见第十二节，实现见 `tools/sql_validator.py`）。


- 修复规则在全量尺度上成立：9/9 badcase 全部闭环；第三轮补强（规则 9）后，B2045/B2051/B2063 复跑全部通过，B2035 复跑未生成 SQL（空 SQL，非编译失败），单发口径语句级 52/52 = 100%。
- 语句级通过率：第三轮补强前单发口径 51/53 = 96.2%（B2035/B2051 两处失败）→ 补强后 52/52 = 100.0%；
- 单发与基线（96.9%）因"单发 vs Agent 多轮累积"口径不同不可直接对比，但"生成语句的编译通过率"口径下双 100%。
- 全量回归的工程价值：新抓出 2 条坏例（1 条虚构表 + 方言、1 条字段归属漏网），形成"badcase → 规则 → 全量回归 → 新 badcase"的持续闭环。


- `tools/data_scripts/sql_full_regression.py` —— 全量 80 题回归脚本（断点续跑、单发、静态+编译双重校验）
- `tools/data_scripts/sql_full_regression_compare.py` —— 基线 vs 回归逐题对比报告生成
- `<结果数据>/sql_full_regression.jsonl` / `sql_full_regression.json` / `sql_full_regression_summary.json` —— 回归明细与汇总（不入 git）
- `<结果数据>/sql_full_regression_对比.md` —— 逐题对比（不入 git）


上一节为"子问题单发"口径（与 9 题端到端复验一致）；本节复刻基线 `result_3_parallel.xlsx` 的
**Agent 多轮累积**生成方式（`tools/data_scripts/batch_test.py`：每题同 user_id，逐子问题走
`RAGPipeline.agent_query` → AgentPlanner Function Calling 循环，每次 `query_financial_and_visualize`
工具调用经 Dify 生成 SQL 并累积到 `conversation_state.sql`），与基线 291 句口径直接可比。


- **同口径**：Agent 工具循环、Dify SQL 生成、SQL 累积逻辑与基线 `batch_test.py` 完全一致；
  子问题异常重试 3 次（指数退避）同基线。
- **运行期补丁**（仅回归进程内生效，未改仓库代码）：DashScope 对 qwen3.5-plus 推理模式长上下文
  生成延迟超过默认 60s 超时，导致研报叙事生成（`search_reports` 辅助链路，不产生 SQL）反复超时重试；
  回归进程内临时调整为 `enable_thinking=False` + 300s 超时（AGENTS.md 明确建议该模型禁用 thinking）。
  SQL 生成链路不受影响。
- **空 SQL**：Agent 判定无需查询时返回纯分析/研报答案，不计入编译失败。


| 口径 | 基线 result_3_parallel | Agent 多轮累积回归 |
| --- | --- | --- |
| 语句级编译通过率 | 282/291 = 96.9% | **224/224 = 100.0%** |
| 有 SQL 的题目 | 70/80 | 61/80 |
| 有 SQL 且全部语句通过 | 61/70 | 61/61 |
| 严格全题（空 SQL 视为未通过） | 61/80 | 61/80 |
| 空 SQL 题目 | 10 | 19 |

- 语句级通过率：基线 282/291 = 96.9% → 修复后同口径 224/224 = 100.0%（+3.1pp）；
- 轮内同口径对比：第三轮补强前 222/224 = 99.1%（B2045 新增、B2063 复现 2 条失败）→ 补强后 224/224 = 100.0%。
- 有 SQL 题目 70 → 61：11 题回归未生成 SQL（融合查询/归因分析类，Agent 判定研报或分析可作答），
  属生成行为更保守，非编译失败；B2031、B2080 则从基线空 SQL 变为生成 SQL 且全部通过。


- **9/9 通过**：B2007（10/10）、B2041（13/13）、B2049（1/1）、B2052（4/4）、B2053（2/2）、
  B2068（1/1）、B2074（5/5）、B2075（1/1）均从「基线未通过」转为「回归通过」；
  B2063 经第三轮补强（规则 9）后复跑通过（1/1）。


| 编号 | 问题类型 | 分类 | 错误 | 与基线对比 |
| --- | --- | --- | --- | --- |
| B2045 | 多意图 | A. 字段表归属（新增） | (1054) `t1.roe` 不存在于 `balance_sheet`（roe 属 core 表） | 基线通过 → 回归新增失败 |
| B2063 | 归因分析 | A. 字段表归属（复现） | (1054) `t1.operating_expense_cost_of_sales` 不存在于 core 表（属 income_sheet） | 基线未通过 → Agent 路径复现 |


第三轮补强（规则 9「字段-别名归属反查自检」）后重跑 B2045/B2063/B2051：**3 题全部通过**（B2045 1/1、B2063 1/1、B2051 2/2），Agent 口径语句级 **224/224 = 100.0%**，0 失败。


- **同口径修复后语句级编译通过率 100.0%（224/224），较基线 96.9%（282/291）提升 3.1pp**，
  这是与简历口径（291 句）最直接可比的修复后指标。
- 第三轮补强已落地：规则 9 强制「反查字段所属表 + 别名归属自检」，校验器新增子查询内表名白名单检查；
- 重跑验证两条口径均达 100%，目标全量 100% 达成。
- 与第十节（单发口径 52/52 = 100.0%）合并看：单发路径更保守（语句量少、B2035 本轮未生成 SQL），
- Agent 路径更接近生产形态；两条证据链互相印证修复有效性。


- `tools/data_scripts/sql_agent_regression.py` —— Agent 多轮累积口径全量回归脚本（断点续跑、同口径）
- `<结果数据>/sql_agent_regression.jsonl` / `.json` / `_summary.json` —— 回归明细与汇总（不入 git）
- `<结果数据>/sql_agent_regression_对比.md` —— 与基线逐题对比（不入 git）


针对全量回归暴露的 A 类字段-表归属问题（B2045/B2063/B2051）与 B2035 虚构子查询表，做第三轮闭环。


1. **提示词新增规则 9「字段-别名归属反查自检」**（已同步到 `docs/问题记录/提示词.txt`、`提示词配置文件（本地保留，不入库）`、Dify 已发布工作流 `bccf7a91-...`）：
   - 反查原则：SELECT 每个字段必须先到字段白名单反查所属表，再用该表别名取出；
   - 三步自检：字段属哪张表 → 挂在哪个别名下 → 别名对应表是否等于字段所属表；
   - 易错字段强记：`roe`/`net_profit_10k_yuan` 等 → core 表；`operating_expense_*`/`net_profit` 等 → income_sheet；`asset_*`/`liability_*`/`equity_*`（含 `asset_liability_ratio`）→ balance_sheet；
   - 跨表聚合场景：多表字段严禁全部挂到 FROM 第一张表别名下，必须 JOIN 后各挂所属表；
   - 附三条真实反例（B2045/B2063/B2051 的失败 SQL）。
2. **校验器扩展**（`tools/sql_validator.py`）：新增子查询内表名白名单检查（`_tables_at_level` + `_collect_subquery_tables`），覆盖 FROM/JOIN 派生表、WHERE IN (...) 等子查询内部的虚构表（拦截 B2035 类 `research_reports`）。自测：坏 SQL 拦截 11/11（9 原始 + 2 子查询用例），好 SQL 零误报（9 + 2）。


| 编号 | 单发口径 | Agent 多轮累积口径 |
| --- | --- | --- |
| B2045 | 1/1 通过 | 1/1 通过 |
| B2063 | 1/1 通过 | 1/1 通过 |
| B2051 | 1/1 通过（子问题 2 空 SQL 预期） | 2/2 通过 |
| B2035 | 未生成 SQL（空 SQL，非编译失败） | 5/5 通过（前轮） |

- **Agent 多轮累积口径：224/224 = 100.0%**（基线 96.9%，+3.1pp）。
- **单发口径：52/52 = 100.0%**（有 SQL 题目 52/80；B2035 本轮未生成 SQL，说明规则 9 让模型在无法可靠生成时更保守，不再产出虚构表 SQL）。


- 字段-表归属错误在两条口径下全部清零，语句级编译通过率双 100%；
- 校验器覆盖边界补齐（子查询内表名），虚构表类错误从「MySQL 编译兜底」升级为「静态拦截」；
- 剩余差异仅为「空 SQL vs 有 SQL」的生成行为（Agent 判定无需查询），非编译质量问题。


> 背景：#9 Agent 编排后端对照（自研 handwritten vs LangGraph StateGraph）。为验证"切换编排框架是否会引入 SQL 质量劣化"，
> 在 LangGraph 后端上同口径（golden v1，80 题 / 108 子问题 / 291 句基线）全量复跑 Agent 多轮累积回归。
> 完整设计对照见 `docs/评估报告/LangGraph对照.md`。


| 轮次 | 语句级编译通过率 | 失败点 | 修复手段 |
| --- | --- | --- | --- |
| 首跑（无守卫） | 116/127 = 91.3% | 11 条失败：B2011 全角逗号；B2047 未定义别名 / 非 SELECT 文本；B2060/B2073/B2076 编造 `*_yoy_growth` 字段 | — |
| 守卫 v1 | 104/107 = 97.2% | B2060/B2073 字段幻觉残留 | `tools/dify_guard.py` 静态校验 + MySQL 编译，失败带错误提示重问一次（`AGENT_DIFY_RETRY=1`） |
| 守卫 v2 | **108/108 = 100.0%** | 0 | 错误提示追加"可用字段参考 + yoy 字段白名单"，纠正 B2060/B2073 |


- `tools/dify_guard.py`：`sql_errors()`（静态校验 + 全角标点启发式 + MySQL 编译终审）+
  `call_dify_with_guard()`（校验失败 → 把错误与字段建议拼回问题重问，`AGENT_DIFY_RETRY` 默认 1 次）；
- 挂接点：`agents/planner.py::call_dify_chatflow`（自研 / LangGraph 同源），两条 Agent 链路共享同一守卫；
- 配置：`RAGConfig.AGENT_SQL_VALIDATE`（默认开）/ `AGENT_DIFY_RETRY`（默认 1）/ `MYSQL_*`（schema + 编译终审）；
- 单元测试：`tests/test_dify_guard.py` 8 例（离线，不依赖外部服务）。


- Dify 工作流 SQL 生成节点【再次强调】新增规则 5：4 张表真实 yoy/qoq 字段逐一列出
  （income_sheet 仅 `net_profit_yoy_growth`、`operating_revenue_yoy_growth`；费用科目无 yoy 字段、禁止编造任何变体）；
- 已同步写入 `提示词配置文件（本地保留，不入库）` 与 `docs/问题记录/提示词.txt`；
- **需重新导入该提示词配置文件到 Dify 后生效**；当前 100% 由守卫兜底达成，不依赖重新导入。


- LangGraph 在 **17 题**上未产生 SQL（其中 B2015/B2022 手工基线为 1/1 通过）；
- 在 **9 题**上新增了通过的 SQL（B2003/B2012/B2017/B2018/B2023/B2027/B2029/B2043/B2058，手工基线为空 SQL）；
- 两版都产生 SQL 的 **48 题**：handwritten 187/187 = 100%；LangGraph 终值 58/58 有 SQL 题全部通过；
- 结论：与既有 224/224（手工）口径一致，**同口径 80 题下 LangGraph 终值 108/108 = 100.0%**，
  首跑 91.3% 的失败全部是 Dify 生成 SQL 质量问题，**不能归因于 LangGraph 框架**；接入守卫后与手工基线持平。


---

**原生 SQL 链路全量复跑（2026-08-30，Dify 迁移下线后）**

> 背景：路线 3 已完成 Dify → 原生 SQL 链路迁移（`tools/native_financial.py`：SQL 生成 → 三层防线 → MySQL 执行 → 分析 → ECharts）。
> 为把证据链刷新到当前生产链路，按与上文完全相同的「Agent 多轮累积」口径（golden v1，80 题 / 108 子问题 / 291 句基线），
> 关闭查询缓存（`QUERY_CACHE_ENABLED=false`）后全量真实重跑。

**回归配置**：`AGENT_PLANNER_BACKEND=langgraph` + `AGENT_LANGGRAPH_MULTI_AGENT=true`（当前 `.env` 生产配置）；
每题同 user_id 逐子问题走 `RAGPipeline.agent_query`，财务工具调用走原生 SQL 链路并累积到 `conversation_state.sql`，
子问题异常重试 3 次（指数退避），空 SQL 不计入编译失败。

| 口径 | 基线 result_3_parallel | Dify Agent 多轮累积（历史） | LangGraph 对照（历史） | **原生 SQL 链路（2026-08-30）** |
| --- | --- | --- | --- | --- |
| 语句级编译通过率 | 282/291 = 96.9% | 224/224 = 100.0% | 108/108 = 100.0% | **102/102 = 100.0%** |
| 有 SQL 的题目 | 70/80 | 61/80 | 58/80 | **79/80** |
| 有 SQL 且全部语句通过 | 61/70 | 61/61 | 58/58 | **79/79** |
| 严格全题（空 SQL 视为未通过） | 61/80 | 61/80 | 58/80 | **79/80** |
| 空 SQL 题目 | 10 | 19 | 22 | **1（B2002，意图模糊）** |

- 语句级通过率：**102/102 = 100.0%**，与历史 Dify 口径（224/224）同结论，原生链路复现 100%，0 失败；
- 语句总量 224 → 102 属生成行为差异而非质量劣化：原生链路每次工具调用返回单条 SQL（`_generate_sql` 输出单语句），
  且 multi-agent 编排对多意图/融合查询更倾向合并查询；同口径「语句级编译通过率」下双 100%；
- 有 SQL 题目 61 → 79：原生链路对原本空 SQL 的 18 题生成了可通过 SQL（含 B2003/B2004/B2005 等），
  仅 B2002（国家医保目录新增中药，意图模糊）维持空 SQL，与基线一致；
- 全部 7 类题型（多意图/归因分析/融合查询/开放性/数据校验/意图模糊/复合）语句级通过率均 100.0%。

**产物**：
- 脚本：`tools/data_scripts/sql_full_regression_native.py`（断点续跑，`--limit/--only/--backend/--progress-every`，golden sha256 校验，强制关缓存）；
- 明细/汇总：`<结果数据>/sql_full_regression_native.jsonl` / `.json` / `_summary.json`（不入 git）；
- 评估报告：`docs/评估报告/评估报告.md`（`python -m eval report` 自动聚合，四条口径并列）。