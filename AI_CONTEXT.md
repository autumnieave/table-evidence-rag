# AI_CONTEXT.md —— 项目记忆锚点

> 生成时间：2026-09-09（项目现状深度扫描）；**更新：2026-09-10（B-13~B-31 落地后口径刷新；B-30 一致性校验已闭环；B-21A / B-23A / B-24A / B-25A 四件阶段 A 落地：CI 分层门禁、幻觉回查底稿、检索质量指标、一致性套件 + LLM-judge）**；证据来源见各小节引用的文件与行号。

## 1. 当前系统状态摘要

- 上市公司“智能问数”助手：用户用自然语言查财务数据（SQL 链路）与问研报观点（RAG 链路），答案带引用溯源，由 LangGraph supervisor-workers 多 Agent 编排（财务 / 研报子 Agent 并行）。
- 技术栈：Python 3.11 + FastAPI（REST / SSE）+ React 前端 + LangGraph + Qdrant + MySQL + SQLite 记忆；模型与 Embedding 走 DashScope（qwen3.5-plus / qwen3-rerank / text-embedding-v2）。
- 近 10 次提交主线：B-13 prompts 版本注册表 → B-14 三层结构 → B-15/16 few-shot 与动态检索 → B-17 输出契约 → B-18 兜底话术 → B-19/20 API 契约与覆盖率门禁 → B-22 对抗挑战集 v2（18 条/5 类，真实执行 + 人工抽审）→ B-26~B-29 挑战集暴露缺陷修复（supervisor 误拒答 / 注入先拒答 / 错别字归一化 / 研报预测荐股转述口径）→ B-31 库外指标统一出口（unsupported_metrics + refuse.metric_out_of_scope，C2007/C2016/C2017 闭环）→ B-30 指标-问题一致性校验（SELECT 指标列 ⊆ standard_fields，越界走 refuse.metric_mismatch）→ **B-33 金额列单位口径修复**（同单位两列误乘 ×10000 致区间筛选恒 0 行；financial v10 / 包级 v6；80 题回归 103/103）→ **B-36 答案先验校验**（「应有数据」三态 + 只读复算，B2053 同题 n=5 误拒答 4/5 → 0/5；**473 passed**）。
- 离线单测 **473 passed**（37 个 test_*.py + conftest，零外部依赖；`ruff check .` 零违规，规则集 E9/F63/F7/F82）；挑战集证据支持 `python -m eval challenge --rejudge` 零成本重判（不重跑 LLM），人工复核存 sidecar（训练结果数据/challenge_v2_review.json）；原始数据与 golden 基准本地保留、不入库。
- 评估入口（`python -m eval`）新增三件阶段 A 能力：`consistency`（同题 n 次真实重复生成，量化结构指纹 / 数值 IoU / 引用 Jaccard 三层面一致性，`--reuse` 可复用既有产物）、`llm-judge`（四判据判定 + 与规则信号交叉出「分歧样本清单」）、以及 `tools/data_scripts/hallucination_audit.py`（幻觉回查底稿：数值清单 CSV + 引用存在性比对）；CI 侧新增 `.github/workflows/ci-layered.yml`（L1 pytest+ruff 必过 / L2 默认关闭 / L3 nightly 占位）+ `scripts/ci_golden_subset.py` + `scripts/ci_compose_smoke.ps1`。
- 一致性基线（2026-09-10，10 题 × n=5）：数值 IoU 均值 **0.5111**（完全一致 0/10）、引用 Jaccard 均值 **0.7798**（7/10）、结构指纹一致率 **0.72**（3/10）、拒答判定不一致 4/10、SQL 输出契约通过率 **1.0**；judge 50 条 pass 33 / fail 17，与规则一致率 **0.8571**，分歧样本 **17 条**——**judge 未校准，不得作为对外结论**。检索层指标脚本 `eval/retrieval_metrics.py`（Recall@K / Precision@K / MRR）+ 30 题 × 10 片段预标注已就位，正式指标待人工审核（B-24B）。

## 2. 确定性锚点（已知稳定项）

- SQL 守卫族（tools/sql_validator.py / sql_guard.py / financial_mean_guard.py）——B-07~B-12 均以“规则守卫 + 单测”双重闭环，SQL 编译通过率 100% 基线稳定。
- L1 引用核验（pipelines/citation_validator.py + tests/test_citation_validator.py）——自动核验 + 报告证据链，文件可溯源 1080/1080 两次复验一致。
- 缓存与记忆基础设施（utils/query_cache.py、memory/store.py 的 SQLite 后端）——有独立单测（test_query_cache / test_memory / test_pipeline_memory）覆盖。
- 分块与表聚合（data/splitter.py、utils/helpers.py + test_splitter / test_table_agg_topk）——overlap=100 经 5 档对比实验固化，代码 / config / pipeline 三处口径统一。
- 离线评估入口（eval/runner + golden v1：80 题 / 108 子问题 / 291 句，不可变快照 + sha256 防篡改；golden v2 对抗挑战集 18 条/5 类，2026-09-10 按 B-29 方案 B 修订 C2018 后重固化）。
- prompts 版本注册表（prompts/registry.json，7 条登记：6 业务模块 + 包级；当前包级 2026-09-10-v6、financial v10、multi_agent v3、fallback v3、agent/rag/pipeline v1）+ 防漂移单测（tests/test_prompt_registry.py）。
- 出口守卫族（agents/langgraph_multi_agent.py：`_guard_injection_prefix` 注入先拒答、`_guard_research_advice` 研报预测/评级转述口径 + 免责）+ 单测 27 例。

## 3. 风险与债务清单

- 接口“绕道”均为构造 / 离线 / 评测场景直连具体类：pipelines/rag_pipeline.py:35、42，eval/retrieval_cmp.py:144，tools/data_scripts/rebuild_full_index.py:32；在线问答主路径经 IRetriever 接口，未见违规。
- 硬编码长 Prompt 集中在离线数据脚本（tools/data_scripts/pdf处理+校验入库.py:156、543，重抽取.py:92），未纳入 prompts/ 版本管理；app / agents / pipelines 在线代码无硬编码 system prompt。
- 缓存污染风险：/chat 缓存 key 不含 Prompt / Query 版本（app/api.py:86、124），QUERY_CACHE_VERSION 默认空；财务链路缓存 key 已含 FINANCIAL_PROMPT_VERSION，无此风险。
- Redis 记忆后端代码可用但无 docker-compose 编排（compose 仅 qdrant / backend / frontend），MEMORY_REDIS_URL 默认 localhost:6379/0 连通性需人工验证；评测与重建脚本集合默认值（research_reports_v3）与 config 默认（research_reports_v3_full）不一致，存在跑错集合风险。
- 测试健康：473 passed、0 失败 0 跳过；源码 TODO / FIXME / HACK 为 0 处；prompts 模块级版本号已补齐（B-13 起，registry.json 为唯一事实源）。
- **B-35（P1，已于 2026-09-12 闭环 `d14617b`）**：`ruff.toml` 原有 3 条 per-file-ignores 覆盖 **7 处 F821**（`llm/generator.py:105` 未导入 `interactive_main`；`pipelines/rag_pipeline.py` 983:12 / 983:25 / 988:65 / 990:77 误用 `question`（形参为 `user_input`）→ `/chat/clarify`（app/api.py:233）与 scripts/interactive.py:215 曾直接 NameError；`tools/data_scripts/pdf处理+校验入库.py` 888:9 / 889:9 缺 `import sys` 且被 `except Exception` 静默吞掉）。**7 处已全部修复、3 条 `per-file-ignores` 豁免已删净**，`ruff check .` 零豁免全绿、全量 pytest 无回归（`473 passed`）。
- **B-36（P0，已闭环 2026-09-10）**：B2053 类**系统性误拒答**已修——前置：`tools/native_financial.py` 的 `sql_ok_rows_empty` 分支补 `_append_sql`（故障现场 SQL 落盘）；新增 `eval/answer_keys.py` + `database/answer_keys/v1.json`（注册表已 gitignore）实现**先验三态**（B2053 `has_data` 附人工审核标准 SQL，只读复算 14 行、已存 sha256；B2004 / B2036 `no_data` 豁免），`eval/llm_judge.py` 增 `--answer-key` / `--no-prior` 与 5 列，`eval/consistency.py` 重复词表改为复用。B2053 真实复跑 n=5 **误拒答 4/5 → 0/5**；反向验证 B2004 / B2036 零误伤；80 题全量回归 **103/103 = 100%、有 SQL 69/69、41.7s/题**（B2053 恢复「14 家」）；`tests/test_answer_prior.py` **30 例**、全量 **473 passed**、ruff 全绿。**冒烟教训**：单测直连 `attach_priors` 全绿，但 CLI 冒烟 5/5 误报（`judge_rows` 输出行只留「答案字数」）→ **入口层数据流必须跑真实 CLI 冒烟**（`--no-judge` 零成本）。
- 挑战集复跑新发现的缺陷（2026-09-10）与处置：B-31 已闭环——C2016「每股公积金」的近似字段替换（net_asset_per_share 答 24.00 元）与 C2007 股价/总市值回落技术性 SQL 报错，均由「库外指标统一出口」修复（契约扩为五键 unsupported_metrics，tools/native_financial.py:unsupported_metrics_of + prompts/fallback.py:build_refuse_metric_out_of_scope）；复跑 auto 15/15 pass、0 fail、3 pending（仅 binding 恒 pending）。B-30（指标口径防线）已于同日闭环：**拦截层**=库外指标出口（unsupported_metrics + 库外词表 `_OUT_OF_SCOPE_QUESTION_TERMS` 兜底，词表已比对 golden 80 题与挑战集无其他命中）；**审计层**=`metric_field_consistency_error`（SELECT 指标列 vs standard_fields，只记录不拦截——首轮 80 题全量回归 94 次判定 6 次命中，逐条核验均为计划欠规范而非真实替换，硬拦截曾使 B2076 退化为仅研报回答），10 条单测；**80 题全量回归（审计版）语句级编译 99/99 = 100%、失败 0、平均 42.7s/题**，审计信号 92 次判定 1 次命中（计划欠规范），兜底出口 12 次（库外 7 + 词表兜底 1「市值」+ 空结果 3 + 建议 1）；同口径对照确认硬门禁版曾丢 B2031/B2033/B2035/B2076 的 SQL，审计版全部恢复。新发现指标库内外判定运行间波动（B2075/B2069）→ 登记 B-32；C2018 研报召回不稳定（3 次运行仅 1 次召回预测/评级段落）→ 归入 B-24 检索质量评估证据。
- **B-37（P2，未修，2026-09-10 登记）**：**分析层计数与 SQL 结果不一致**——B2053 第 5 次回答写「共有 13 家」而该轮 SQL 只读复算为 **14 行**，属分析层 LLM 未直接引用 SQL 行数的笔误（不属误拒答，B-36 已能区分）。修法：把结果行数作为硬约束注入分析提示词 + 自洽性核验（行数 vs 答案计数）+ B2053 复跑 n=5 验证一致率 100%。

## 4. 下一次迭代的检查建议

- 改动财务 / SQL 链路先回归 test_financial_mean_guard + test_sql_guard + test_sql_validator + test_metric_standardize；改编排回归 test_langgraph_planner / test_langgraph_multi_agent；合入前全量 `pytest tests/ -q`。
- 优先修复：为 /chat 缓存 key 并入 PROMPT_VERSION / QUERY_CACHE_VERSION，或对 RAG / Agent 模式默认关闭结果缓存。
- 统一评测与重建脚本的集合默认值为 research_reports_v3_full（或全部改读 config/rag_config.py），消除跑错集合风险。
- 将数据脚本长 Prompt 收口到 prompts/（至少登记版本与用途），并明确 Redis 是否启用——不启用则在文档标注“代码支持、环境未落地”。
- 指标库内外判定波动（B-32：B2075/B2069 全量判库外拒答、复跑可出数）与挑战集 pending 项（仅剩绑定 3 条，恒 pending）在 B-25 LLM-judge 与人工双回查对齐 ≥90% 后启用自动判定（§6.7.3）。
- 改动财务链路时除既有回归集外，另看 output_contract_stats.jsonl 的 kind=metric_consistency 命中率（计划↔SQL 字段差异，供人工判定是欠规范还是替换）。
- **下一优先项（按优先级）**（B-35 已于 2026-09-12 闭环 `d14617b`、B-36 已于 2026-09-10 闭环，均不再列入）：① **B-37 / B-32**（分析层计数硬约束；指标库内外判定波动 n=3 一致率基线）→ ② **B-34**（库外指标漏标样本固化）→ ③ 四件人工阶段 B：**B-21B**（CI secret 放行 / 月度 LLM 预算 / 启用哪几层）、**B-23B**（双人核对 313 数值与幻觉判定）、**B-24B**（审核 30 题预标注 + 定义「相关性」）、**B-25B**（17 条分歧样本先根因分类再定一致率阈值）。
- 改 judge 或一致性套件后：改 prompt 必须重跑并保留前一版产物（本轮已留 `prompt_v0_截断证据_judge_results.json` 作校准对比），否则一致率数字无法解释来源。
