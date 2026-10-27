# 行级自包含事实与 anchor_cells（阶段 2）

> 日期：2026-10-17　任务：B-60（数据管道重构 · 阶段 2）
> 代码：`src/data/row_facts.py`、`src/data/splitter.py`、`src/tools/data_scripts/rebuild_full_index.py`
> 测试：`tests/test_row_facts.py`、`tests/test_splitter_row_facts.py`

## 一、为什么做

阶段 1 的表格结构化解决了「一表一 chunk / 表头路径 / 元数据」问题，但检索单元仍是整表。
一张 40 行的财务表被整块嵌入时，用户问「2024 年归母净利润是多少」，向量只能匹配到「整张表」，
无法保证命中具体单元格；精排与引用也无法指出数字来自哪一行、哪一列。

阶段 2 把「表」拆成**可检索的单元格级事实**，并给每个事实留下**可回表的坐标**：

- 行级事实：进入向量索引，负责「被检索到」；
- `anchor_cells`：挂在整表父块上，负责「定位回原表单元格」；
- 标注迁移契约：负责「旧标注按坐标迁移到新结构」，不靠文本相似度猜。

## 二、设计边界（先定不做什么）

1. **按数据单元格生成，不按整行拼句**。
   一行里可能有多个「指标 × 期间」单元格；拼成一句会出现
   `营业收入 100 120 130`，无法回答具体期间。
2. **只对 5 类表生成事实**：
   `financial_statement / forecast / valuation / industry_summary / ranking`。
   `pipeline / other` 以文本、日期、标签为主（如研发管线阶段、事件日期），
   不属于数值事实，强行入索引会制造噪声。
3. **只生成数值或数值区间**（`31.18/21.33` 保留）。
   纯文本单元格不生成事实。
4. **不猜单位、不猜数量级、不取众数**。
   单位按「单元格/表头 → caption/title → footnote → 表级单值」回退；
   多来源冲突置 `multi`，缺失保持 `None`。

## 三、事实文本格式

```
{entity}｜{table_caption}｜{metric_path} > {period}: {value}{unit}
```

示例：

```
贵州茅台(600519)｜主要财务指标｜归母净利润 > 2024A: 3,367.89 百万元
```

- `entity`：优先取行内实体列（公司/证券简称/行业等）；缺失时回退文档级实体。
- `table_caption`：表标题。
- `metric_path`：由表头路径归一，保留「归母净利润」这类指标名。
- `period`：归一后的期间（见 §五）。
- `value`：原值文本（保留千分位与区间写法），不重算。
- `unit`：解析出的单位；缺失时不追加。

同一张表内按 `(entity, metric, period, value, unit)` 去重，避免合并单元格展开产生重复事实。

## 四、anchor_cells 字段

### 4.1 六项核心字段（固定）

| 字段 | 含义 |
| --- | --- |
| `source` | 来源标识（表 ID） |
| `row_index` | **真实行坐标**（0 基，指向原表行） |
| `col_index` | **真实列坐标**（0 基，指向原表列） |
| `metric` | 指标名 |
| `period` | 归一后的期间 |
| `value` | 单元格原值 |

### 4.2 附加定位字段

| 字段 | 用途 |
| --- | --- |
| `table_id` | 表级唯一 ID |
| `source_file` | 源文件名（跨集合迁移用） |
| `chunk_id` | 所属整表 chunk（`{table_id}#s{slice_index}`） |
| `header_path` | 表头路径（多级表头） |
| `unit` / `unit_source` | 单位与来源（cell/header/caption/footnote/table/multi/none） |
| `currency_raw` | 原始币种文本 |
| `unit_kind` | 单位类型（currency/percent/ratio/other） |
| `period_type` | `actual` / `forecast` / `unknown` |
| `entity` | 行实体（公司/行业） |
| `table_caption` | 表标题 |
| `table_type` / `table_scope` | 阶段 1 的表类型与范围 |

### 4.3 坐标语义（重要）

- `anchor_cells.row_index / col_index` 是**原表真实坐标**，可直接回表取单元格。
- 整表 chunk 上的 `row_index` 仍是**切片起点**（兼容既有父表回拉逻辑），
  真实单元格坐标只在 `anchor_cells` 内。
- 行事实 chunk 的 `chunk_id = {table_id}#r{row}c{col}`；
  整表父块为 `{table_id}#s{slice_index}`；正文块为 `{filename}#c{idx}`。

## 五、期间与单位口径

### 5.1 期间归一

| 原写法 | 归一 |
| --- | --- |
| `2024A` / `2024`（财务、预测、估值表） | `2024A` |
| `2025E` | `2025E` |
| `2025Q3` / `Q3 2025` | `2025Q3` |
| `2025H1` / `1H25` | `2025H1` |
| 行业汇总表的裸年份 | 保留 `2024`（不强行补 A） |
| 同比 / 环比 / 占比 / 增速列 | 不视为期间 |

`period_type`：`A` → actual，`E` → forecast，其余无法判定 → unknown。

### 5.2 单位回退链

1. 单元格/表头内的复合或比率单位（`元/股`、`%`、`倍`）优先；
2. 单元格/表头单位；
3. caption/title；
4. footnote；
5. 表级单值单位；
6. 冲突 → `multi`；全程缺失 → `None`。

不使用「众数」填补多单位表。

## 六、分块结构变化

`splitter.split_documents(..., table_mode="structured", row_fact_mode="chunks")`：

- 整表父块：保留原表格正文，新增 `anchor_cells / anchor_count / row_fact_count / chunk_id`；
- 行事实子块：`chunk_type="row_fact"`、`is_row_fact=True`、`is_table_row=False`、
  `parent_id=table_id`、`chunk_id={table_id}#r{row}c{col}`；
- `row_fact_mode="off"` 为默认值，不改变现有链路；`chunks` 必须搭配 `structured`，
  否则抛 `ValueError`。

摘要生成已排除 `chunk_type in ("table", "row_fact", "boilerplate")`，避免为表格与行事实生成无意义摘要。

## 七、离线全量干跑（零 API）

对 473 篇文档做完整分块（不调用 embedding）：

| 项 | 数值 |
| --- | ---: |
| 总 chunk | 181,543 |
| prose | 6,443 |
| table | 5,739 |
| row_fact | 169,242 |
| boilerplate | 119 |
| 平均行事实/表 | 29.49 |
| 行级 unit 非空 | 119,017（70.3%） |
| unit_source = cell | 119,017 |
| unit_source = none | 50,225 |
| 带 anchor 的整表块 | 3,085 / 5,739 |

事实类型分布：

| table_type | 行事实数 |
| --- | ---: |
| financial_statement | 64,886 |
| ranking | 45,380 |
| forecast | 38,869 |
| industry_summary | 15,568 |
| valuation | 4,539 |
| pipeline / other | 0（按设计排除） |

期间类型：actual 51,559 / forecast 40,374 / unknown 77,309。

**坐标抽检**：随机抽 20 条 anchor，按 `source_file + row_index + col_index` 回原表，
**20/20** 命中且单元格内容一致。

## 七之补、169,242 条的构成与分母（2026-10-17 复核）

### 7.1 只覆盖 5 类数值表

| table_type | row_fact | 物理表数 | 条/表 |
| --- | ---: | ---: | ---: |
| financial_statement | 64,886 | 597 | 108.7 |
| forecast | 38,869 | 531 | 73.2 |
| ranking | 45,380 | 542 | 83.7 |
| industry_summary | 15,568 | 633 | 24.6 |
| valuation | 4,539 | 202 | 22.5 |
| pipeline | 0 | 492 | 0 |
| other | 0 | 151 | 0 |
| **合计** | **169,242** | **3,148** | **53.8** |

- 5 类之和 **恰好等于 169,242**，`pipeline / other` 为 0，确认没有越界生成。
- **54 条/表的口径**：`169,242 / 3,148 = 53.8`，分母是**全部物理表**（含 643 张 0 事实的 pipeline/other）。
- 若只算真正生成事实的 5 类表（2,505 张）：`169,242 / 2,505 = 67.6` 条/表。
- 若按**表 chunk**（切片后 5,739 个）算：`169,242 / 5,739 = 29.5` 条/chunk。
- 三个分母都成立但含义不同，报告不得混用；「54 条/表」不是「每张数值表都有 54 条」。

### 7.2 unit 70.3% 的来源（纠正）

Qdrant 精确计数：

| unit_source | row_fact 条数 |
| --- | ---: |
| cell | 119,017 |
| header | 0 |
| caption | 0 |
| footnote | 0 |
| table | 0 |
| multi | 0 |
| none | 50,225 |

**结论**：行级 `unit` 的 70.3% **全部来自单元格自身**（含单元格内 `%`、`元/股` 等复合单位），
caption / footnote / 表级回退贡献为 **0**。

因此「行级 70.3% vs 表级 28.3%」**不是回退链带来的阶段 1 收益**，而是两个口径不同：

- 表级 `unit` 要求**整张表恰好一个单位**；多单位表按「不猜」置 `None`（28.3%）；
- 行级 `unit` 按**每个数值单元格**解析；同一张多单位表仍可逐格得到单位（70.3%）；
- 行事实只保留数值单元格，本身更可能携带 `%` / `元/股` 等显式单位。

回退链代码保留（cell/header → caption → footnote → 表级单值），但**本语料未触发**
caption/footnote 分支，不能把它们写成收益来源。

### 7.3 阶段 3 必须评估的稀释

row_fact **169,242** vs table chunk **5,739**，约 **29.5 : 1**。
若不做 `chunk_type` 过滤，向量/混合召回会被行事实淹没；多阶段检索必须：
1. 表级粗召回只查 `chunk_type=table`；
2. 行级精定位只查 `chunk_type=row_fact` 且 `table_id` 在候选表内；
3. 对比「单阶段整表」与「多阶段」的 top-10 真内容与 P95 延迟，确认收益不是靠数量堆出来的。

## 八、索引重建（v5_rowfacts）

- 集合：`research_reports_v5_rowfacts`（不覆盖 `v3_full / v3_structured / v4_contract`）
- 命令：
  `python -m tools.data_scripts.rebuild_full_index --collection research_reports_v5_rowfacts --table-mode structured --strip-boilerplate --row-fact-mode chunks --workers 4 --force`
- 点数：**181,543**（与离线分块一致）
- 上传方式：**边算边传**——每累计 500 条向量立即 upsert 并释放缓冲。
  首版「全量向量先攒内存、最后一次性上传」在 7.8GB 主机上必然 OOM：
  181,543 × 1536 维 Python float 约需 8–9GB。
- 实际过程：边算边传版本在 175,050 条（96.4%）时仍因主机内存压力中断，
  随后按 `chunk_id` 续跑，仅补 6,493 块（283.5s）。
- 总耗时：约 **50 分钟**（含一次 OOM 与续跑）；纯上传/嵌入阶段约 47 分钟。
- 调用量：181,543 条文本 / batch=25 = **7,262 次 embedding 请求**；
  平台账单未单独导出，成本未实测。
- payload index（11 个，均已建全）：
  `chunk_type / chunk_id / source_file / table_id / table_type / table_scope /
   unit_source / period / period_type`（keyword）、`row_index / col_index`（integer）。
  计数：`chunk_type / chunk_id / source_file` **181,543**；
  `table_id / table_type / table_scope / unit_source / row_index` **174,981**（表 + 行事实）；
  `period / period_type / col_index` **169,242**（行事实）。
- 过滤实测：按 `table_id` 精确计数 **14ms**，按 `source_file` **49ms**。
- 环境遗留：Qdrant 在重建期间出现一次后台 flush 被取消 + 临时段 rename
  `Permission denied`（与 B-57 同源），集合状态一度为 grey/yellow；
  数据完整性由点数、全库随机 anchor 抽检与向量自检索共同验证，环境问题单独登记。

## 九、验收

| 项 | 结果 |
| --- | --- |
| 单测 | 19 passed（row_facts 11 + splitter 3 + label_migration 5） |
| 离线分块可完成 | 473 篇 → 181,543 块 |
| anchor 可回表 | 20/20 |
| 索引点数一致 | **181,543 = 离线分块** |
| 索引内分层计数 | row_fact 169,242 / table 5,739 / prose 6,443 / boilerplate 119；带 anchor 表块 3,085；unit 非空 119,017 |
| payload 可过滤 | `table_id` 14ms / `source_file` 49ms |
| 全库随机 anchor 回表 | **20/20**（随机 ID 跨文件抽样，源 HTML 网格独立比对） |
| 旧标注可迁移 | 契约已实现；当前 L5 10 条无坐标，不适用（见迁移契约文档） |

## 十、边界与风险

- `table_type` 仍是规则分类 v1，未逐表人工核验；分类错误会直接影响「是否生成事实」。
- 行事实只覆盖 5 类表；`pipeline / other` 的数值（管线阶段、日期）不入索引。
- `ranking` 表产生 45,380 条事实（涨跌幅/排名为主），是否进入最终检索路径由阶段 3 的多阶段策略决定。
- 77,309 条事实期间为 `unknown`（多来自 ranking / 行业汇总），阶段 3 做期间过滤时只能作弱信号。
- 行级 `unit` 覆盖 70.3%，其余保持 `None`，不猜。
- 整表 chunk 的 `row_index` 是切片起点，不是单元格坐标；使用方必须读 `anchor_cells`。

## 十一、更正（2026-10-18）：`unit_source=cell` 不等于目标单元格自身有单位

阶段 3 冒烟发现一条反例：

```text
马应龙(600993)｜盈利预测与估值(人民币)｜市盈率(P/E) > 2025E: 17.62 百万元
```

P/E 不应带“百万元”。复核 `src/data/row_facts.py::_unit_for_cell` 后确认：

1. 依次查 `metric_cell + header + metric + value` → caption/title → footnote；
2. 三者都没有候选时，**回退到 `table.unit`**，并沿用 `table.unit_source`；
3. 因此当表级单位来自单元格时，行事实的 `unit_source` 仍显示 `cell`，但单位并非来自**该指标单元格**。

**更正 §7.2 的表述**：
- “caption/footnote 回退贡献为 0”仍成立（Qdrant 计数为 0）；
- “70.3% 全部来自单元格自身”**不成立**；
- 行级 70.3% 与表级 28.3% 的差异，既来自单元格显式 `%` / `元/股`，也来自表级单位回退；
- 多单位表（如同时含百万元、元/股、%、倍）会把这个回退放大为**语义错配**。

**处置（2026-10-18 已修，v6 重建后复评见 §十二）**：
- `_unit_for_cell` 增加显式 `unit_source="table_fallback"`，不沿用 `table.unit_source`；
- 多单位表禁止跨指标继承表级单位（按指标词/列头判断）；
- 修复后重建 `research_reports_v6_rowfacts` 再复评。

## 十二、修复与 v6 复评（2026-10-18）

### 12.1 单位修复

`src/data/row_facts.py` 已改：

- 表级回退显式写 `unit_source="table_fallback"`，不再沿用 `table.unit_source`；
- 多单位表（`len(table.unit_candidates) > 1`）禁止跨指标继承表级单位；
- 比率/估值/每股类指标（PE/PB/毛利率/ROE/EPS/同比/倍/每股）不继承金额单位。

`research_reports_v6_rowfacts` 的 row_fact 单位来源：

| unit_source | row_fact | 占比 |
| --- | ---: | ---: |
| cell | 78,172 | 46.2% |
| table_fallback | 23,163 | 13.7% |
| none | 67,899 | 40.1% |
| **非空合计** | **101,335** | **59.9%** |

与 v5 的 70.3% 不可直接比较：v5 把 23,163 条表级回退记成了 `cell`。修复后
`市盈率(P/E)` 134 条全部 `unit=None`，不再出现“P/E 17.62 百万元”。

### 12.2 指标/期间硬匹配

新增 `src/filters/row_match.py`：

- 期间精确匹配：`2025Q3` 不匹配 `2025A/2025E`；行事实缺期间时判不匹配；
- 指标归一与别名：营收→营业收入、投资性现金流量净额→投资活动现金流量净额、
  P/E→PE、研发费用占比→研发费用；
- 空约束不过滤；有约束时不匹配的行不进入候选。

`src/data/splitter.py` 的 row_fact payload 增加顶层 `metric/metric_name/metric_names`，
`src/core/multi_stage.py` 在 Qdrant 侧先按 `period + table_id` 过滤，再做指标硬匹配。

### 12.3 v6 重建

- 集合：`research_reports_v6_rowfacts`，**181,535 点**；
- 分层：row_fact **169,234** / table 5,739 / prose 6,443 / boilerplate 119；
- row_fact 比 v5 少 8 条，来自单位文本变化后的 `(entity, metric, period, value, unit)`
  去重，未发现丢表；table 数完全一致。
- 重建期间再次出现 B-57 同源的临时段 rename `Permission denied`，重启 Qdrant 后恢复；
- 但首次重启后 HNSW 近似检索异常：同内容、同向量、同 query 的点，exact 得分 0.654，
  默认 HNSW 连 top-1000 都进不去。已通过更新 HNSW 配置（m=17、单段优化）强制重建，
  最终 **181,535/181,535 向量全部索引、segments=2、green**。**HNSW 修复前的评测结果作废**。

### 12.4 27 + 4 题复评

口径：同一 query vector、同一实体闸门、同一精排；v5 为修复前旧基线，v6 为硬匹配后。

| 模式 | 集合 | 同主体 | 真内容 | 有行证据题数 | 空上下文 | trace P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| single | v5 | 2.15 | 1.74 | 0/27 | 8 | 2933.7ms |
| table_first | v5 | **2.78** | **2.33** | **22/27** | **5** | 1760ms |
| single | v6 | 2.19 | 1.78 | 0/27 | 11 | 1807.2ms |
| table_first | v6 | 2.44 | 2.00 | 4/27 | 10 | 556.6ms |

**解读**：v6 的 `table_first` 同主体/真内容下降，不是召回退化，而是旧 v5 的 22/27
“有行证据”包含大量指标/期间不匹配的行，错误行分把表排上来了。硬匹配后只剩 4/27
题有**约束成立**的行证据，指标精度提高、召回下降。这个代价必须如实保留。

4 条严重题（v6）：

| 题号 | 行候选 | row_hits | 指标命中 | 约束命中 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| B2005 片仔癀 | 200 | **3** | **3** | **3（无期间约束）** | 3 条均为利润总额，期间为 2023A/2025E/2027E；题面未限定期间 |
| B2015 方盛制药 | 15 | 0 | 0 | 0 | Q3 候选只有 `收入(YOY)`/`归母净利润(YOY)`，没有 `营业收入` 实际值 |
| B2069 佐力药业 | 15 | 0 | 0 | 0 | 候选表无 `投资活动现金流量净额`；佐力全库 Q3 行事实为 0 |
| B2065 佐力药业 | 0 | 0 | 0 | 0 | 佐力全库 `period=2025Q3` 行事实为 0 |

全库 `period=2025Q3` 共 2,113 条，指标以 `收入(YOY)`/`归母净利润(YOY)`/PE/市值为主；
精确匹配 `2025Q3 + 营业收入`、`2025Q3 + 研发费用`、`2025Q3 + 投资活动现金流量净额`
均为 **0 条**。因此 B2015/B2069/B2065 的 0 行证据首先是**研报语料边界**（季度实际值主要
不在研报表里），不是硬匹配代码误杀。季度实际值应走 SQL/财报路。

### 12.5 边界

- 本轮只验证 **约束命中覆盖率**，没有人工 gold 判定 `row_hits` 是否足以回答；不能把
  3/3 写成“行证据正确率 100%”。
- B2005 题面无期间，3 条行事实跨 2023A/2025E/2027E，不能据此认定“答案唯一”。
- 27 题只审计 top-10 结果中的 `row_hits`，不是全库行级 gold。
- HNSW 重建依赖 Qdrant 9p bind mount 的稳定性；B-57 仍是环境风险，不是已根治。
